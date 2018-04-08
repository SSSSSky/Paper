import numpy as np
import pandas as pd
import os.path
import pywt

def get_data():  #i取0 -18
    path = '../feature/join_feature'
    files = os.listdir(path)
    files.sort()
    i=0
    for file in files:
        file_path =os.path.join(path, file)
        df1 = pd.read_csv(file_path, index_col='DateTime')
        if i == 0:
            join_df = df1
        else:
            join_df = pd.concat(join_df, df1)
        i += 1
    join_df.replace(np.inf, np.nan)
    join_df.replace(-np.inf, np.nan)
    join_df.dropna()
    return join_df

def wt(index_list, wavefunc, lv, m, n):
    coeff = pywt.wavedec(index_list, wavefunc, mode='sym', level=lv)
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0

    for i in range(m, n+1):
        cD = coeff[i]
        for j in range(len(cD)):
            Tr = np.sqrt(2*np.log(len(cD)))   #计算阈值
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) - Tr  # 向零收缩
            else:
                coeff[i][j] = 0  # 低于阈值置零

    #重构
    denoised_index = pywt.waverec(coeff, wavefunc)
    return denoised_index

def main():
    join_df = get_data()
    data = join_df.drop('DateTime', 'SecCode', axis=1)
    i = 0
    if(i < 67):
        wt(data.icol(i), 'db4', 4, 1, 4)
        i += 1
