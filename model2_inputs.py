'''生成模型2的输入'''
import os

import numpy as np
import pandas as pd
from sklearn import preprocessing

# 按照model_2需要的输入格式生成训练数据X
def get_rnn_inputs():
    index_code = '000016'
    start_date = '20100104'
    end_date = '20150101'
    # start_date = '20150101'
    # end_date = '20170101'
    # start_date = '20170101'
    # end_date = '20171231'

    # train_encoding_df = pd.read_csv('model1_encoding_result/3_B_0_sigmoid_test_layer1_encoding.csv', dtype={'SecCode': np.str})
    train_encoding_df = pd.read_csv('../feature/features/technical_features_16_20100104_20150101.csv', dtype={'SecCode': np.str})
    train_encoding_df = train_encoding_df.loc[:, ['SecCode','DateTime','bias','boll','cci','price_efficiency','ma_ma_ratio']]
                                                     # ,'logreturn','rsi','mfi','ar','macd']]
    print(train_encoding_df.shape)
    # train_encoding_df = pd.read_csv('encoding_result/valid_layer3_encoding.csv', dtype={'SecCode': np.str})
    # train_encoding_df = pd.read_csv('encoding_result/test_layer3_encoding.csv', dtype={'SecCode': np.str})
    # train_encoding_df = train_encoding_df.reset_index().set_index(['SecCode', 'DateTime'])

    f_list = os.listdir('../feature/features/amp')
    f_list.sort()
    for f in f_list:  # 遍历属性文件
        f_splits = f.split('_')
        if int(start_date) <= int(f_splits[-1].split('.')[0]) < int(end_date)+1 and f_splits[0] == index_code:
            # df = pd.read_csv('../feature/features/amp/{}'.format(f), index_col='DateTime')
            df = pd.read_csv('../database/Cons_chg/{}_{}_{}'.format(index_code, f_splits[2], f_splits[-1].split('.')[0]), dtype=np.str, header=None)
            # cons_stocks_list = list(df.columns)  # constituent stocreks
            cons_stocks_list = list(df.iloc[:, 0])
            print(len(cons_stocks_list))
            te_df = train_encoding_df[(train_encoding_df.DateTime > f_splits[2]) & (train_encoding_df.DateTime < f_splits[-1].split('.')[0])]
            # te_df.reset_index(inplace=True)
            # print(te_df.groupby(by=['DateTime','SecCode']).count())
            te_df = te_df.set_index(['DateTime', 'SecCode'])
            df_columns = te_df.columns
            print(df_columns)
            te_df = te_df.unstack(1)
            concat_df = []
            for i in df_columns:
                concat_df.append(te_df[i].loc[:, cons_stocks_list])
            concat_df = pd.concat(concat_df, axis=1)
            print(concat_df.shape)
            print(concat_df.dropna().shape)
            # print(concat_df.head())
            # concat_df = concat_df.loc[:, cons_stocks_list]
            # print(concat_df.shape)
            concat_df.to_csv('model2_datasets/{}_{}.csv'.format(f_splits[2], f_splits[-1].split('.')[0]))

# 按照model_2需要的输入格式生成Y
def get_rnn_predicts():

    f_list = os.listdir('model2_predicts/')
    f_list.sort()

    for f in f_list:
        f_splits = f.split('_')
        df = pd.read_csv('model2_predicts/{}'.format(f), index_col='DateTime')
        print(df.head())
        cons_stocks_list = list(pd.read_csv('../database/Cons_chg/{}_{}_{}'.format(f_splits[0], f_splits[1], f_splits[2]),
                                            dtype=np.str, header=None).iloc[:, 0])
        df = df.iloc[:-241, :]              # 去除无效数据
        df = df.loc[:, cons_stocks_list]    # 调整成份股顺序
        print(df.head())
        df.to_csv('model2_predicts/{}'.format(f))


def get_y(test_y_path):
    test_y = []
    columns_y = None
    for f in os.listdir(test_y_path):
        df = pd.read_csv('{}/{}'.format(test_y_path, f), index_col='DateTime')
        test_y.append(df)
        columns_y = df.columns
    for i in range(len(test_y)):
        test_y[i].columns = columns_y
    test_y = pd.concat(test_y)
    # print(test_y)
    return test_y


def normalize_predicts():
    train_y = get_y('model2_predicts/train')
    train_y.fillna(0.0, inplace=True)
    valid_y = get_y('model2_predicts/valid')
    valid_y.fillna(0.0, inplace=True)
    test_y = get_y('model2_predicts/test')
    test_y.fillna(0.0, inplace=True)
    scaler = preprocessing.StandardScaler().fit(train_y)
    norm_train_y = scaler.transform(train_y)
    norm_valid_y = scaler.transform(valid_y)
    norm_test_y = scaler.transform(test_y)
    pd.DataFrame(data=norm_train_y, index=train_y.index, columns=train_y.columns).to_csv('model2_predicts/norm_train/norm_train.csv')
    pd.DataFrame(data=norm_valid_y, index=valid_y.index, columns=valid_y.columns).to_csv('model2_predicts/norm_valid/norm_valid.csv')
    pd.DataFrame(data=norm_test_y, index=test_y.index, columns=test_y.columns).to_csv('model2_predicts/norm_test/norm_test.csv')



def main():
    get_rnn_inputs()
    # get_rnn_predicts()
    # normalize_predicts()


if __name__ == '__main__':
    main()