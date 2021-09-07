import pandas as pd
import glob
import os
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

        words = []
        for word in ok:
            if word[1] not in ['Josa', 'Eomi', 'Punctuation', 'Suffix']:  # 조사, 어미, 구두점이 있는 것은 포함 시키지 않음
                words.append(word[0])
        clean_words.append(words)

    return clean_words
