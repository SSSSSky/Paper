import numpy as np
import pandas as pd
from sklearn import preprocessing

def preprocess(dataset, start_date, end_date, datatype='alpha'):
    df = pd.read_csv(dataset, index_col=['SecCode', 'DateTime'])
    df[df == np.inf] = np.nan
    df[df == -np.inf] = np.nan
    df.reset_index(inplace=True)
    df = df.groupby('SecCode').ffill()  # 使用前时刻的值填充缺失值
    # 将SecCode和DateTime两列作为索引
    df.set_index(['SecCode', 'DateTime'], inplace=True)
    # df['br'][df['br'] > 30000] = 30000    # 去除br异常值
    # df['br'][df['br'] < -30000] = -30000

    md = df.median()
    MAD = (df - md).abs().median()
    up_bound = md + 3 * MAD
    lo_bound = md - 3 * MAD
    # print(up_bound)
    # print(lo_bound)
    for i in df.columns:
        df[i][df[i] > up_bound[i]] = up_bound[i]
        df[i][df[i] < lo_bound[i]] = lo_bound[i]

    #数据标准化
    dropped_df = df.dropna()
    scaler = preprocessing.StandardScaler().fit(dropped_df)

    # 平均值填充缺失值
    fill_na_dict = {}
    for i in range(len(df.columns)):
        fill_na_dict[df.columns[i]] = scaler.mean_[i]
    print(fill_na_dict)
    df.fillna(fill_na_dict, inplace=True)
    scaled_df = scaler.transform(df)
    print(scaled_df.shape)
    print(scaled_df)

    pd.DataFrame(data=scaled_df, index=scaled_df.index).to_csv(
        '../model1/input/{}_{}_{}.csv'.format(datatype, start_date, end_date))
    return scaled_df

def main():
    train_dataset = '../data/all_features/alpha_features_45_20100104_20150101.csv'
    valid_dataset = '../data/all_features/alpha_features_45_20150101_20170101.csv'
    test_dataset = '../data/all_features/alpha_features_45_20170101_20171231.csv'
    train_df = preprocess(train_dataset, '20100104', '20150101', 'alpha')
    valid_df = preprocess(valid_dataset, '20150101', '20170101', 'alpha')
    test_df = preprocess(test_dataset, '20170101', '20171231', 'alpha')
    
    train_dataset1 = '../data/all_features/technical_features_16_20100104_20150101.csv'
    valid_dataset1 = '../data/all_features/technical_features_16_20150101_20170101.csv'
    test_dataset1 = '../data/all_features/technical_features_16_20170101_20171231.csv'
    train_df1 = preprocess(train_dataset1, '20100104', '20150101', 'technical')
    valid_df1 = preprocess(valid_dataset1, '20150101', '20170101', 'technical')
    test_df1 = preprocess(test_dataset1, '20170101', '20171231', 'technical')