import os
import numpy as np
import pandas as pd
from sklearn import preprocessing

def get_rnn_inputs(index_code, start_date, end_date, path, type, df):
    train_encoding_df = pd.read_csv(path, dtype={'SecCode': np.str})

    print(train_encoding_df.shape)
    print(train_encoding_df.head())
    train_encoding_df = train_encoding_df.reset_index().set_index(['SecCode', 'DateTime'])
    print(train_encoding_df.shape)

    f_list = os.listdir('../feature/features/amp')
    f_list.sort()
    for f in f_list:  # 遍历属性文件
        f_splits = f.split('_')
        # f: 000016_amp_20100104_20100630.csv
        # 文件分类的依据是成分股已经变更。如：0104变更，0701变更，所以0104-0630是一段
        if int(start_date) <= int(f_splits[-1].split('.')[0]) < int(end_date)+1 and f_splits[0] == index_code:
            # df = pd.read_csv('../feature/features/amp/{}'.format(f), index_col='DateTime')
            df = pd.read_csv('../feature/cons_chg/{}_{}'.format(f_splits[2], f_splits[-1].split('.')[0]), dtype=np.str, header=None)
            # cons_stocks_list = list(df.columns)  # constituent stocreks
            cons_stocks_list = list(df.iloc[:, 0])
            # 获得该时间段内的成分股列表
            print(len(cons_stocks_list))
            # 应当是50支股票
            te_df = train_encoding_df[(train_encoding_df.DateTime > f_splits[2]) & (train_encoding_df.DateTime < f_splits[-1].split('.')[0])]
            # 将该时间段的股票的特征选出来
            te_df = te_df.set_index(['DateTime', 'SecCode'])
            df_columns = te_df.columns
            #列明即每个特征名（自编码结果没有列名，应为0，1，2，3，4，5，6，7，8，9）
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
            if type == 'train':
                concat_df.to_csv('../Data/Model2/datasets/train/{}_{}.csv'.format(f_splits[2], f_splits[-1].split('.')[0]))
            elif type == 'valid':
                concat_df.to_csv('../Data/Model2/datasets/valid/{}_{}.csv'.format(f_splits[2], f_splits[-1].split('.')[0]))
            else:
                concat_df.to_csv('../Data/Model2/datasets/test/{}_{}.csv'.format(f_splits[2], f_splits[-1].split('.')[0]))



# 按照model_2需要的输入格式生成Y
def get_rnn_predicts():

    f_list = os.listdir('../feature/features/return')
    f_list.sort()
    # f_list里的文件应当是按时间段（19个时间段）分开的股票+时间+绝对收益率return
    for f in f_list:
        f_splits = f.split('_')
        df = pd.read_csv('../feature/features/return/{}'.format(f), index_col='DateTime')
        print(df.head())
        cons_stocks_list = list(pd.read_csv('../feature/cons_chg/{}_{}'.format(f_splits[2], f_splits[-1].split('.')[0]),
                                            dtype=np.str, header=None).iloc[:, 0])
        df = df.iloc[:-241, :]              # 去除无效数据
        df = df.loc[:, cons_stocks_list]    # 调整成份股顺序
        print(df.head())
        df.to_csv('model2_predicts/{}'.format(f))
#         要手动将预测数据分入train、valid、test文件中


def get_y(test_y_path):
    # 把每个时间段的股票每50只拼接起来
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
    train_y = get_y('../Data/Model2/predicts/train')
    train_y.fillna(0.0, inplace=True)
    valid_y = get_y('../Data/Model2/predicts/valid')
    valid_y.fillna(0.0, inplace=True)
    test_y = get_y('../Data/Model2/predicts/test')
    test_y.fillna(0.0, inplace=True)
    scaler = preprocessing.StandardScaler().fit(train_y)
    norm_train_y = scaler.transform(train_y)
    norm_valid_y = scaler.transform(valid_y)
    norm_test_y = scaler.transform(test_y)
    pd.DataFrame(data=norm_train_y, index=train_y.index, columns=train_y.columns).to_csv('../Data/Model2/predicts/norm_train/norm_train.csv')
    pd.DataFrame(data=norm_valid_y, index=valid_y.index, columns=valid_y.columns).to_csv('../Data/Model2/predicts/norm_valid/norm_valid.csv')
    pd.DataFrame(data=norm_test_y, index=test_y.index, columns=test_y.columns).to_csv('../Data/Model2/predicts/norm_test/norm_test.csv')



def main():
    df = pd.read_csv('../feature/features/amp/000016_amp_20100104_20100630.csv')
    df = df.loc['SecCode', 'DateTime']
    get_rnn_inputs('000016', '20100101', '20150101', '../Data/AutoEncoder/hidden_result/train/alpha_train_hidden_result.csv', 'train', df)
    get_rnn_inputs('000016', '20150101', '20170101', '../Data/AutoEncoder/hidden_result/valid/alpha_valid_hidden_result.csv', 'valid', df)
    get_rnn_inputs('000016', '20170101', '20171231', '../Data/AutoEncoder/hidden_result/test/alpha_test_hidden_result.csv', 'test', df)

    get_rnn_predicts()
    # normalize_predicts()


if __name__ == '__main__':
    main()

