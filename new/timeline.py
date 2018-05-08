import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller as ADF

# 检测是否是平稳序列并处理非平稳序列
def AdfTest(index_list):
    adftest = ADF(index_list)
    # 返回值依次为adf,pvalue,usedlag,nobs,critical values,icbest,regresults,resstore
    i = 0
    for key, value in adftest[4].items():
        if value < adftest[0]:
            i += 1
    # 假如adf值小于两个水平值，p值小于0.05，则判断为平稳序列
    if i <= 1 and adftest[1] < 0.01:
        return 1
    else:
        return 0

# 对一行数据进行平稳处理
def Timeline(index_list, maxdiffn):
    D_data = index_list.copy()
    stationary_result = AdfTest(D_data)
    if stationary_result == 1:
        print('Stationary series...')
    else:
        i = 0
        # 假如是平稳时间序列，则返回，如果不是，则差分，直到最大差分次数
        while i < maxdiffn:
            if AdfTest(D_data) == 1:
                break
            else:
                D_data = np.diff(D_data)
                print('Diff time: '+str(i+1))
    return D_data

# 对数据集进行平稳处理
def Timelinepreprocess(dataset, start_date, end_date, datatype, inputs):
    print('[INFO] start pca...')
    df = pd.read_csv('{}/{}_{}_{}.csv'.format(dataset, datatype, start_date, end_date),
                     index_col=['SecCode', 'DateTime'])
    i = df.shape[0] / 1000000
    j = 0
    fi_df = []
    while j < i:
        data_df = df.iloc[j:(j+1)*1000000, 0]
        data_df = Timeline(data_df, inputs)
        timecol = 1
        while timecol < len(df.columns) :
            print('start column {}'.format(timecol))
            time_df = Timeline(df.iloc[j:(j+1)*1000000, timecol], inputs)
            i = 0
            while i < len(df.iloc[j:(j+1)*1000000, 0]) - time_df.shape[0]:
                mean_df = time_df.mean()
                # timeline_df = np.append(timeline_df, mean_df)
                time_df = np.insert(time_df, 0, mean_df, 0)
                # print(timeline_df.shape)
                i += 1
            data_df = np.vstack(data_df, time_df)
            timecol += 1
            print(data_df.shape)
        fi_df = np.concatenate((fi_df, data_df), axis=0)
        print(fi_df.shape)

    fi_df = fi_df.T
    print('timeline_data.shape:{}'.format(fi_df.shape))
    print('[INFO] save pca result...')
    pd.DataFrame(data=fi_df, index=df.index, columns=df.columns).to_csv('../model1/tl_input/{}_{}_{}.csv').format(datatype, start_date, end_date)
    # return data_df

def main():
    dataset = '../model1/input'
    # Timelinepreprocess(dataset, '20100104', '20150101', 'technical', 8)
    # Timelinepreprocess(dataset, '20150101', '20170101', 'technical', 8)
    Timelinepreprocess(dataset, '20170101', '20171231', 'technical', 8)

    # Timelinepreprocess(dataset, '20100104', '20150101', 'alpha', 8)
    # Timelinepreprocess(dataset, '20150101', '20170101', 'alpha', 8)
    # Timelinepreprocess(dataset, '20170101', '20171231', 'alpha', 8)
    df = pd.read_csv('../model1/tl_input/technical_20170101_20171231')
    print(df.head())

if __name__ == '__main__':
    main()