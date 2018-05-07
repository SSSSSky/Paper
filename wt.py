import pywt
import numpy as np
import pandas as pd

# WaveletTransform
def wt(index_list, wavefunc, lv, m, n):
    coeff = pywt.wavedec(index_list, wavefunc, mode='sym', level=lv)
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0

    for i in range(m, n+1):
        cD = coeff[i]
        for j in range(len(cD)):
            Tr = np.sqrt( 2 * np.log(len(cD)))   #计算阈值
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) - Tr  # 向零收缩
            else:
                coeff[i][j] = 0  # 低于阈值置零

    #重构
    denoised_index = pywt.waverec(coeff, wavefunc)
    return denoised_index

def wavelettransform(dataset, inputs, types):
    df = pd.read_csv(dataset)
    print(df.head())
    print('start {} wavelettransform...'.format(types))
    data_df = wt(df[:, 0], 'db4', 3, 1, 3)
    wavecol = 1
    while(wavecol < inputs):
        print('start clomun {}'.format(wavecol))
        wave_df = wt(df[:, wavecol], 'db4', 3, 1, 3)
        data_df = np.vstack((data_df, wave_df))
        wavecol += 1
    #print(data_df.shape)
    data_df = data_df.T
    print('wt_data.shape:{}'.format(data_df.shape))
    print('finish {} wavelettransform...'.format(types))
    return data_df

def main():
    wavelettransform()