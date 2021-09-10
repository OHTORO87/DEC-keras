import pandas as pd
import numpy as np
from tqdm import tqdm
from konlpy.tag import Okt
okt = Okt()


# csv 파일 읽기
def csv_reader(file_name):
    # 경로 설정
    csv_location = "./data/{name}.csv".format(name= file_name)
    # 로컬 > data_frame > 변수에 할당후 결과 확인
    read_data = pd.read_csv(csv_location, index_col=[0])

    return read_data


# 형태소 분석 함수
def okt_morph(dataframe):
    # 범위 지정 가능
    df_pre = dataframe['preprocessed_review']
    df_corpus = df_pre[:]

    clean_words = []
    for i in tqdm(df_corpus):
        ok = okt.pos(i)
        # ok = okt.pos(i, stem=True)

        words = []
        for word in ok:
            if word[1] in ['Adjective', 'Verb', 'Noun', 'Adverb', 'Exclamation', 'Determiner', 'Unknown']:
                # 감정을 표현하는 품사만 따로 가져 오는 것이 유용하다고 판단.
                # 필요에 따라 추가시키기에도 좋다.

                # Exclamation : 감탄사는 감정의 가장 단순한 형태여서 꼭 필요하다고 생각
                # Determiner : 새호텔, 헌호텔 과 같은 리뷰가 점수에 영향이 있을것이라고 판단
                # Unknown : 미등록어는 널리쓰이는 유행어 같은 것을 놓치지 않기 위해 선택
                words.append(word[0])
        clean_words.append(words)

    return clean_words



# 문장에서 단어 벡터의 평균을 구하는 함수
def make_feat_vec(words, model, num_features):

    # 0으로 채운 배열로 초기화 한다(속도향상을 위해)
    feature_vec = np.zeros((num_features,), dtype="float32")

    nwords = 0
    # index2word는 모델 사전에 있는 단어명을 담은 리스트
    # 속도 향상을 위해 set 형태로 초기화
    index2word_set = set(model.wv.index2word)
    # 루프를 돌며 모델 사전에 포함이 되는 단어면 피쳐에 추가
    for word in words:
        if word in index2word_set:
            nwords += 1
            feature_vec = np.add(feature_vec, model[word])

    # 결과를 단어수로 나누어 평균을 구한다.
    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec



# okt 형태소 분류 자료 csv merge
def okt_csv(file_name):

    df_all = csv_reader(file_name)  # csv 파일 load
    df_preprocess = df_all.loc[:, ['review_id', 'score', 'review', 'preprocessed_review']]  # 점수, 전처리된 리뷰, 리뷰id(merge할때 필요)
    df_clean = df_preprocess.dropna(axis=0)  # nan 값이 있는 행 삭제
    df_reindex = df_clean.reset_index(drop=True)  # 인덱스 재정렬

    x, y = df_reindex['preprocessed_review'], df_reindex['score']  # x 리뷰, y 점수

    print('okt 형태소 분류...')  # 형태소 분류
    df_x1 = pd.DataFrame(x)  # 전처리 리뷰 데이터 프레임 변환
    tokenized_riviews = okt_morph(df_x1)  # 전처리된 리뷰 데이터 토크나이징

    dict_okt = {'okt_pos_review': tokenized_riviews}
    df_okt = pd.DataFrame(dict_okt)
    df_merge = pd.merge(df_reindex, df_okt, right_index=True, left_index=True)
    df_merge.to_csv(f"./data/{file_name}_Okt_version.csv", encoding='utf-8')




if __name__ == "__main__":
   test = okt_morph('naver_review_test_data')
   print(test[:10])
