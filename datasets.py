import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tool import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm
from gensim.models import Word2Vec, KeyedVectors
import random
import os

SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)



def extract_vgg16_features(x):
    from keras.preprocessing.image import img_to_array, array_to_img
    from keras.applications.vgg16 import preprocess_input, VGG16
    from keras.models import Model

    # im_h = x.shape[1]
    im_h = 224
    model = VGG16(include_top=True, weights='imagenet', input_shape=(im_h, im_h, 3))
    # if flatten:
    #     add_layer = Flatten()
    # else:
    #     add_layer = GlobalMaxPool2D()
    # feature_model = Model(model.input, add_layer(model.output))
    feature_model = Model(model.input, model.get_layer('fc1').output)
    print('extracting features...')
    x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((im_h,im_h))) for im in x])
    x = preprocess_input(x)  # data - 127. #data/255.#
    features = feature_model.predict(x)
    print('Features shape = ', features.shape)

    return features


def make_reuters_data(data_dir):
    np.random.seed(1234)
    from sklearn.feature_extraction.text import CountVectorizer
    from os.path import join
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        # did_to_cat = {k: did_to_cat[k] for k in list(did_to_cat.keys()) if len(did_to_cat[k]) > 1}
        for did in list(did_to_cat.keys()):
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(join(data_dir, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did in did_to_cat:
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    print((len(data), 'and', len(did_to_cat)))
    assert len(data) == len(did_to_cat)

    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    from sklearn.feature_extraction.text import TfidfTransformer
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
    x = x[:10000].astype(np.float32)
    print(x.dtype, x.size)
    y = y[:10000]
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
    print('todense succeed')

    p = np.random.permutation(x.shape[0])
    x = x[p]
    y = y[p]
    print('permutation finished')

    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], -1))
    np.save(join(data_dir, 'reutersidf10k.npy'), {'data': x, 'label': y})


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    print('MNIST samples', x.shape)
    return x, y


def load_fashion_mnist(): # ?????? ????????? ??????
    from keras.datasets import fashion_mnist  # this requires keras>=2.0.9
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    print('Fashion MNIST samples', x.shape)
    return x, y


def load_pendigits(data_path='./data/pendigits'):
    import os
    if not os.path.exists(data_path + '/pendigits.tra'):
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.tra -P %s' % data_path)
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.tes -P %s' % data_path)
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.names -P %s' % data_path)

    # load training data
    with open(data_path + '/pendigits.tra') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_train, labels_train = data[:, :-1], data[:, -1]
    print('data_train shape=', data_train.shape)

    # load testing data
    with open(data_path + '/pendigits.tes') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_test, labels_test = data[:, :-1], data[:, -1]
    print('data_test shape=', data_test.shape)

    x = np.concatenate((data_train, data_test)).astype('float32')
    y = np.concatenate((labels_train, labels_test))
    x /= 100.
    print('pendigits samples:', x.shape)
    return x, y


def load_usps(data_path='./data/usps'):
    import os
    if not os.path.exists(data_path+'/usps_train.jf'):
        if not os.path.exists(data_path+'/usps_train.jf.gz'):
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_train.jf.gz -P %s' % data_path)
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_test.jf.gz -P %s' % data_path)
        os.system('gunzip %s/usps_train.jf.gz' % data_path)
        os.system('gunzip %s/usps_test.jf.gz' % data_path)

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64') / 2.
    y = np.concatenate((labels_train, labels_test))
    print('USPS samples', x.shape)
    return x, y


def load_reuters(data_path='./data/reuters'):
    import os
    if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
        print('making reuters idf features')
        make_reuters_data(data_path)
        print(('reutersidf saved to ' + data_path))
    data = np.load(os.path.join(data_path, 'reutersidf10k.npy')).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], -1)).astype('float64')
    y = y.reshape((y.size,))
    print(('REUTERSIDF10K samples', x.shape))
    return x, y


def load_retures_keras():
    from keras.preprocessing.text import Tokenizer
    from keras.datasets import reuters
    max_words = 1000

    print('Loading data...')
    (x, y), (_, _) = reuters.load_data(num_words=max_words, test_split=0.)
    print(len(x), 'train sequences')

    num_classes = np.max(y) + 1
    print(num_classes, 'classes')

    print('Vectorizing sequence data...')
    tokenizer = Tokenizer(num_words=max_words)
    x = tokenizer.sequences_to_matrix(x, mode='binary')
    print('x_train shape:', x.shape)

    return x.astype(float), y


def load_imdb():
    from keras.preprocessing.text import Tokenizer
    from keras.datasets import imdb
    max_words = 1000

    print('Loading data...')
    (x1, y1), (x2, y2) = imdb.load_data(num_words=max_words)
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    print(len(x), 'train sequences')

    num_classes = np.max(y) + 1
    print(num_classes, 'classes')

    print('Vectorizing sequence data...')
    tokenizer = Tokenizer(num_words=max_words)
    x = tokenizer.sequences_to_matrix(x, mode='binary')
    print('x_train shape:', x.shape)

    return x.astype(float), y


def load_newsgroups():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.datasets import fetch_20newsgroups
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    vectorizer = TfidfVectorizer(max_features=2000, dtype=np.float64, sublinear_tf=True)
    x_sparse = vectorizer.fit_transform(newsgroups.data)
    x = np.asarray(x_sparse.todense())
    y = newsgroups.target
    print('News group data shape ', x.shape)
    print("News group number of clusters: ", np.unique(y).size)
    return x, y


def load_cifar10(data_path='./data/cifar10'):
    from keras.datasets import cifar10
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y)).reshape((60000,))

    # if features are ready, return them
    import os.path
    if os.path.exists(data_path + '/cifar10_features.npy'):
        return np.load(data_path + '/cifar10_features.npy'), y

    # extract features
    features = np.zeros((60000, 4096))
    for i in range(6):
        idx = range(i*10000, (i+1)*10000)
        print("The %dth 10000 samples" % i)
        features[idx] = extract_vgg16_features(x[idx])

    # scale to [0,1]
    from sklearn.preprocessing import MinMaxScaler
    features = MinMaxScaler().fit_transform(features)

    # save features
    np.save(data_path + '/cifar10_features.npy', features)
    print('features saved to ' + data_path + '/cifar10_features.npy')

    return features, y


def load_stl(data_path='./data/stl'):
    import os
    assert os.path.exists(data_path + '/stl_features.npy') or not os.path.exists(data_path + '/train_X.bin'), \
        "No data! Use %s/get_data.sh to get data ready, then come back" % data_path

    # get labels
    y1 = np.fromfile(data_path + '/train_y.bin', dtype=np.uint8) - 1
    y2 = np.fromfile(data_path + '/test_y.bin', dtype=np.uint8) - 1
    y = np.concatenate((y1, y2))

    # if features are ready, return them
    if os.path.exists(data_path + '/stl_features.npy'):
        return np.load(data_path + '/stl_features.npy'), y

    # get data
    x1 = np.fromfile(data_path + '/train_X.bin', dtype=np.uint8)
    x1 = x1.reshape((int(x1.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
    x2 = np.fromfile(data_path + '/test_X.bin', dtype=np.uint8)
    x2 = x2.reshape((int(x2.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
    x = np.concatenate((x1, x2)).astype(float)

    # extract features
    features = extract_vgg16_features(x)

    # scale to [0,1]
    from sklearn.preprocessing import MinMaxScaler
    features = MinMaxScaler().fit_transform(features)

    # save features
    np.save(data_path + '/stl_features.npy', features)
    print('features saved to ' + data_path + '/stl_features.npy')

    return features, y




# word2vec ?????? ??????
def save_model(data_file):
    pass
    # # ????????? ??????
    # print('Loading data...')
    # df_all = csv_reader(file_name) # csv ?????? load
    # df_preprocess = df_all.loc[:, ['total_score','tokenized_review']] # ????????? ???????????? ????????? ?????????
    # df_clean = df_preprocess.dropna(axis=0) # nan ?????? ?????? ??? ??????
    # df_reindex = df_clean.reset_index(drop=True) # ????????? ?????????
    # okt_review, label = df_reindex['tokenized_review'], df_reindex['total_score'] # ??????, ??????
    #
    # # ????????? ????????? ??????
    # x_list = okt_review.values.tolist()
    # y_list = label.values.tolist()
    #
    # # word2vec
    # model = Word2Vec(sentences=data_file, size=100, window=10, min_count=20, workers=4, sg=1) # ?????? ??????
    # model.save(f"model_mincnt_20_{file_name}")



# ????????? ?????? ????????? ???????????? ??????
def load_crawling_data():
    np_size = 100
    window_size = 30
    min_count_size = 50
    sg_type = 1 # 1 = skip-gram, 0 = CBOW
    hs_type = 0 # default??? 0, 1?????? softmax ?????? ??????
    iter_cnt = 100 # epoch??? ?????????  ???????????? ??????

    file_name = 'SCORE_1_TEST_DATA_30000'
    df = csv_reader(file_name)
    sentence_data = reviews_parcing(file_name)
    '''
    ?????? load(?????? ?????? ?????????)
    '''
    # model = Word2Vec.load('DEC_model_final')
    '''
    ?????? ????????? > ??????
    '''
    # model = Word2Vec(sentences=sentence_data, size=np_size, window=window_size, min_count=min_count_size, workers=4,
    #                  hs=hs_type, sg=sg_type, iter=iter_cnt, seed=SEED)

    model = Word2Vec(sentences=sentence_data, size=np_size, window=window_size, min_count=min_count_size, workers=4, sg=sg_type)

    model.save(f"model_{file_name}")
    '''
    ?????? vocab ??????
    '''
    # words = list(model.wv.vocab)
    '''
    ????????? ?????????
    '''
    # model_result_good = model.wv.most_similar("??????")  # ?????? ????????? ???????????? ??????
    # model_result_bad = model.wv.most_similar("??????")  # ?????? ????????? ???????????? ??????
    # print(model_result_good)
    # print(model_result_bad)


    # x = np.empty((np_size,), dtype='float64')
    x = np.zeros((np_size,), dtype='float64') # 0?????? ?????? ndarray(100,)
    y = np.array(df['total_score'], dtype='int64') # ?????? ndarray(5000,)??? ?????????
    nwords = 0.
    counter = 0.
    raw_reivews = []
    token_reviews = []
    y_list = []

    # GPU ver.
    idx_to_key = model.wv.index2word # ????????? ????????? ?????? ???????????? ?????? ?????????
    # print(idx_to_key)

    index2word_set = set(idx_to_key) # ????????? ?????? set ????????? ?????????
    # print(index2word_set)
    for idx in tqdm(range(len(sentence_data)), desc="?????????"):
        featureVec = np.zeros((np_size,), dtype='float64')
        for word in sentence_data[idx]:
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec, model.wv[word])
        is_all_zero = not np.any(featureVec)
        if is_all_zero:
            pass
        else:
            featureVec = np.divide(featureVec, nwords)
            x = np.append(x, featureVec, axis=0)
            # y??? ????????? x?????? ????????? ??????

            y_list.append(y[idx])
            raw_reivews.append(df['review'][idx])
            # ??????????????? ?????? ???????????? ??????????????? ??????
            token_reviews.append(df['tokenized_review'][idx])

        counter += 1

    y = np.array(y_list, dtype='int64')
    x = x[np_size:]
    x = x.reshape(y.size, -1)
    print(len(y), 'train sequences')
    print('x_train shape:', x.shape)
    print('y shape:', y.shape)
    print("raw_reviews shape:", len(raw_reivews))
    print("token_reviews shape:", len(token_reviews))

    # print(x.astype(float))
    return x.astype(float), y, raw_reivews, token_reviews

    # bert??? vector???
    # from transformers import BertTokenizer
    # import torch
    # bert??? ????????? ???????????? ?????? ?????????
    
    # sentences = df_for_vector['preprocessed_review']
    # sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
    # labels = df_for_vector['total_score'].values

    # ?????????
    # review_bert = ["[CLS]" + str(s) + "[SEP]" for s in df_for_vector.preprocessed_review]
    #
    # # ???????????????
    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    # tokenized_texts = [tokenizer.tokenize(s) for s in review_bert]
    # inputs = torch.tensor(tokenized_texts)
    # print(tokenized_texts)
    # print(inputs)



# ????????? ?????????>???????????? ???????????? ???????????? ?????? ??????????????????.
def load_data(dataset_name):
    if dataset_name == 'mnist':
        return load_mnist()
    elif dataset_name == 'fmnist':
        return load_fashion_mnist()
    elif dataset_name == 'usps':
        return load_usps()
    elif dataset_name == 'pendigits':
        return load_pendigits()
    elif dataset_name == 'reuters10k' or dataset_name == 'reuters':
        return load_reuters()
    elif dataset_name == 'stl':
        return load_stl()
    elif dataset_name == 'imdb':
        return load_imdb()
    elif dataset_name == 'crawling_reviews': # ????????? ?????? ?????? ???????????? ?????? ?????????
        return load_crawling_data()
    else:
        print('Not defined for loading', dataset_name)
        exit(0)




if __name__ == "__main__":

    load_crawling_data()

    # ????????? ?????? ??????
    '''
    df = csv_reader('all_hotels_review_data(final)')
    df.dropna(subset=['tokenized_review', 'total_score'], inplace=True)
    df = df.reset_index(drop=True) # 590??????
    
    # ????????? ??????
    df_shu = df.sample(frac=1).reset_index(drop=True)

    # ????????? ?????? ??????
    df_50 = df_shu.loc[:500000,:]
    df_100 = df_shu.loc[:1000000,:]
    
    # ????????? ??????
    csv_save(df_50,'test_csv_50')
    csv_save(df_100,'test_csv_100')
    csv_save(df, 'all_hotels_review_data(dropna)')
    '''

    # ????????? ?????? list??? ?????? ??????
    '''
    list_reviews = load_list('test_csv_50')
    model = Word2Vec(sentences=list_reviews, size=100, window=10, min_count=40, workers=4, sg=1)  # ?????? ??????
    model.save(f"model_csv_50_mincnt_50")
    '''



    # 3d ????????? ?????? ?????? csv ??????(save tsv files for visualization)
    '''
    model = Word2Vec.load('model_mincnt_40')
    df = pd.DataFrame(model.wv.vectors)
    df.to_csv('./model_mincnt_40_vectors.tsv', sep='\t', index=False)
    word_df = pd.DataFrame(model.wv.index2word)
    word_df.to_csv('./model_mincnt_40_metadata.tsv', sep='\t', index=False)
    '''

    # word2vec ?????? ??????
    '''
    save_model("all_hotels_review_data(final)")
    '''


    # word2vec ?????? ??????
    '''
    model_20 = Word2Vec.load('model_mincnt_20_all_hotels_review_data(final)')
    '''

    # ????????? ?????????
    '''
    x, y = load_crawleing_reviews('all_hotels_review_data(final)', model_20)
    print(f"???????????? data : {x}")
    print(f"label ??? : {y}")
    '''

    '''
    [word2vec]
    size : ??????????????? ?????? ???. ????????? ??? ????????? ??????
    window = ???????????? ????????? ????????? ????????? ????????????
    min_count = ?????? ?????? ?????? ??? ??????(????????? ?????? ????????? ???????????? ?????????)
    workers = ????????? ?????? ???????????? ???
    sg = 0??? CBOW, 1??? Skip-gram
    '''

    # ????????? ??????
    '''
    print('Vectorizing sequence data...')  # ????????? ?????????
    model = Word2Vec(sentences=tokenized_riviews, size=100, window=10, min_count=5, workers=4, sg=1) # ?????? ??????
    # model_result = model.wv.most_similar("??????") # ?????? ????????? ???????????? ??????
    '''

    # ?????? ??????
    '''
    raw_reivews = []
    y_list = []
    # ???????????? ?????? ????????? ????????? ?????? ?????? ??????

    print('Loading data...')
    df = csv_reader(file_name)
    df.dropna(subset=['tokenized_review', 'total_score'], inplace=True)
    df = df.reset_index(drop=True)
    y = np.array(df['total_score'], dtype='int64')

    # ?????? ???????????? ?????? ???????????? ????????????
    data = reviews_parcing(file_name)
    data = data[:y.size]
    print('y.size', y.size)

    print(data[:10])

    num_classes = np.max(y)
    print(num_classes, 'classes')


    # ?????? ??????
    # with gzip.open('pkl_list_test_review_data_50000.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #     data = data[:y.size]
    #
    # num_classes = np.max(y)
    # print(num_classes, 'classes')

    print('Vectorizing sequence data...')
    # model = Word2Vec.load('model_mincnt_20')
    np_size = 100
    model = Word2Vec(sentences=data, size=np_size, window=15, min_count=30, workers=4, sg=1)  # ?????? ??????iter=100

    x = np.empty((np_size,), dtype='float32')
    nwords = 0.
    counter = 0.
    # GPU ver.
    idx_to_key = model.wv.index2word

    # local ver.
    # idx_to_key = model.wv.index_to_key
    # key_to_idx = model.wv.key_to_index
    index2word_set = set(idx_to_key)
    for idx in tqdm(range(len(data)), desc="?????????"):
        featureVec = np.zeros((np_size,), dtype='float32')
        for word in data[idx]:
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec, model.wv[word])
        is_all_zero = not np.any(featureVec)
        if is_all_zero:
            pass
        else:
            featureVec = np.divide(featureVec, nwords)
            # print('featureVec :', featureVec)
            x = np.append(x, featureVec, axis=0)
            # y??? ????????? x?????? ????????? ??????

            y_list.append(y[idx])
            raw_reivews.append(df['review'][idx])

            # x[int(counter)] = featureVec
            # n_values = np.max(featureVec) + 1
            # x = np.eye(n_values)[x[counter]]
        counter += 1
            # print('x :', counter, x[int(counter)])
    # x = to_categorical(x, num_classes=x)
    y = np.array(y_list, dtype='int64')
    x = x[np_size:]
    x = x.reshape(y.size, -1)
    print(len(y), 'train sequences')
    print('x_train shape:', x.shape)
    print('y shape:', y.shape)
    print("raw_reviews shape:", len(raw_reivews))

    return x.astype(float), y, raw_reivews
    '''


