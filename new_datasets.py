from tool import *
from tqdm import tqdm
from gensim.models import Word2Vec, KeyedVectors

# 크롤링 리뷰 벡터로 불러오는 함수
def load_crawling_data():
    # 설정값
    file_name = 'test_review_data_5000_Okt_stem'
    np_size = 100
    window_size = 15
    min_count = 30


    # 형태소 분석이 끝난 데이터 불러오기
    print('로딩 데이터...')
    raw_reviews = []
    y_list = []
    dataframe = csv_reader(file_name)
    dataframe.dropna(subset=['tokenized_review', 'total_score'], inplace=True)
    dataframe = dataframe.reset_index(drop=True)
    y = np.array(dataframe['total_score'], dtype='int64')
    # print(y)
    # print(len(y))

    # dataframe >> word2vec 입력 데이터 변환
    word2vec_input_data = reviews_parcing(file_name)
    word2vec_input_data = word2vec_input_data[:y.size]
    # print(word2vec_input_data[:10])
    # print(len(word2vec_input_data))

    # 입력 데이터 >> 벡터화
    print('Vectorizing sequence data...')
    model = Word2Vec(sentences=word2vec_input_data, size=np_size, window=window_size, min_count=min_count, workers=4, sg=1)  # 모델 학습iter=100
    x = np.empty((np_size,), dtype='float32')






if __name__ == "__main__":
    load_crawling_data()
