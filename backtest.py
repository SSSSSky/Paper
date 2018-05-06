'''权重向量进行回测
1.将权重向量进行归一化，使每个时刻的权重之和为1
2.计算每日的超额收益'''
import os
import numpy as np
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt

# 将权重向量归一化和为1
def normalize_weights(daily_weights_df):
    # daily_weights_df = pd.read_csv(path, index_col='DateTime')
    # 使用0代替负权重后归一化权重
    norm_daily_weights = daily_weights_df
    norm_daily_weights[norm_daily_weights < 0] = 0.0
    norm_daily_weights = norm_daily_weights.div(norm_daily_weights.sum(axis=1), axis=0)

    # print(norm_daily_weights)
    return norm_daily_weights

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

# 根据预测结果获得实际真是收益
def caculate(train_predict_df, valid_predict_df, test_predict_df):
    # 将预测的收益率向量转换为权重值
    print('Normalizing weights...')
    train_norm_min_weights = normalize_weights(train_predict_df)
    valid_norm_min_weights = normalize_weights(valid_predict_df)
    test_norm_min_weights = normalize_weights(test_predict_df)

    print('Reading targets...')
    train_y = get_y('model2_predicts/train')
    valid_y = get_y('model2_predicts/valid')
    test_y = get_y('model2_predicts/test')

    print(train_predict_df.shape, train_y.shape)
    print(valid_predict_df.shape, valid_y.shape)
    print(test_predict_df.shape, test_y.shape)

    # 每天的预测收益率
    train_daily_predict = train_predict_df.iloc[range(0, (len(train_predict_df) // 241) * 241, 241)]
    # 每天的归一化权重
    train_norm_daily_weights = train_norm_min_weights.iloc[range(0, (len(train_norm_min_weights) // 241) * 241, 241)]
    # 每天的实际收益率
    train_daily_return = train_y.iloc[range(0, (len(train_y) // 241) * 241, 241)]
    # 根据预测结果的每天的实际真实收益
    train_daily_real_return = train_norm_daily_weights * train_daily_return
    train_real_return = (train_daily_real_return.sum(axis=1) + 1).cumprod()

    valid_daily_predict = valid_predict_df.iloc[range(0, (len(valid_predict_df) // 241) * 241, 241)]
    valid_norm_daily_weights = valid_norm_min_weights.iloc[range(0, (len(valid_norm_min_weights) // 241) * 241, 241)]
    valid_daily_return = valid_y.iloc[range(0, (len(valid_y) // 241) * 241, 241)]
    valid_daily_real_return = valid_norm_daily_weights * valid_daily_return
    valid_real_return = (valid_daily_real_return.sum(axis=1) + 1).cumprod()

    test_daily_predict = test_predict_df.iloc[range(0, (len(test_predict_df) // 241) * 241, 241)]
    test_norm_daily_weights = test_norm_min_weights.iloc[range(0, (len(test_norm_min_weights) // 241) * 241, 241)]
    test_daily_return = test_y.iloc[range(0, (len(test_y) // 241) * 241, 241)]
    test_daily_real_return = test_norm_daily_weights * test_daily_return
    test_real_return = (test_daily_real_return.sum(axis=1) + 1).cumprod()

    # 计算预测结果的MSE
    train_daily_mse = ((train_daily_predict - train_daily_return) ** 2).mean(axis=1).mean()
    valid_daily_mse = ((valid_daily_predict - valid_daily_return) ** 2).mean(axis=1).mean()
    test_daily_mse = ((test_daily_predict - test_daily_return) ** 2).mean(axis=1).mean()
    train_daily_rmse = train_daily_mse ** 0.5
    valid_daily_rmse = valid_daily_mse ** 0.5
    test_daily_rmse = test_daily_mse ** 0.5

    print('==================================')
    print('Train Daily MSE: {}  RMSE: {}'.format(train_daily_mse, train_daily_rmse))
    print('Valid Daily MSE: {}  RMSE: {}'.format(valid_daily_mse, valid_daily_rmse))
    print('Test Daily MSE: {}  RMSE: {}'.format(test_daily_mse, test_daily_rmse))

    return train_real_return, valid_real_return, test_real_return

def result(mode, train_index_real_return, valid_index_real_return, test_index_real_return):
    print(mode)

    print('Reading predict results...')
    train_predict_df = pd.read_csv('predict_weights/train_predict_weights_{}.csv'.format(mode),
                                   index_col='DateTime')
    valid_predict_df = pd.read_csv('predict_weights/valid_predict_weights_{}.csv'.format(mode),
                                   index_col='DateTime')
    test_predict_df = pd.read_csv('predict_weights/test_predict_weights_{}.csv'.format(mode),
                                  index_col='DateTime')

    train_real_return, valid_real_return, test_real_return = caculate(train_predict_df, valid_predict_df,
                                                                      test_predict_df)

    train_daily_alpha = np.array(list(train_real_return)) - np.array(list(train_index_real_return))
    valid_daily_alpha = np.array(list(valid_real_return)) - np.array(list(valid_index_real_return))
    test_daily_alpha = np.array(list(test_real_return)) - np.array(list(test_index_real_return))

    print('==================================')
    print('Train Cumulative alpha: {}'.format(train_daily_alpha[-1]))
    print('Train Max Alpha: {}'.format(np.nanmax(train_daily_alpha)))
    print('Train +%: {}%'.format(
        len(train_daily_alpha[train_daily_alpha > 0]) / len(train_daily_alpha) * 100))
    print('Valid cumulative alpha: {}'.format(valid_daily_alpha[-1]))
    print('Valid Max Alpha: {}'.format(np.nanmax(valid_daily_alpha)))
    print('Valid +%: {}%'.format(
        len(valid_daily_alpha[valid_daily_alpha > 0]) / len(valid_daily_alpha) * 100))
    print('Test Cumulative alpha: {}'.format(test_daily_alpha[-1]))
    print('Test Max Alpha: {}'.format(np.nanmax(test_daily_alpha)))
    print('Test +%: {}%'.format(len(test_daily_alpha[test_daily_alpha > 0]) / len(test_daily_alpha) * 100))

    return train_real_return, valid_real_return, test_real_return


def main():
    mode = 'wt_technical'

    # 与指数基准做对比
    cons = ts.get_apis()

    train_index_df = ts.bar('000016', conn=cons, asset='INDEX', start_date='20100104', end_date='20141211')
    train_index_df = train_index_df.sort_index()
    train_index_df['return'] = train_index_df.close / train_index_df.close.shift() - 1
    train_index_real_return = (train_index_df['return'] + 1).cumprod()

    valid_index_df = ts.bar('000016', conn=cons, asset='INDEX', start_date='20141212', end_date='20161208')
    valid_index_df = valid_index_df.sort_index()
    valid_index_df['return'] = valid_index_df.close / valid_index_df.close.shift() - 1
    valid_index_real_return = (valid_index_df['return'] + 1).cumprod()

    test_index_df = ts.bar('000016', conn=cons, asset='INDEX', start_date='20161209', end_date='20171228')
    test_index_df = test_index_df.sort_index()
    test_index_df['return'] = test_index_df.close / test_index_df.close.shift() - 1
    test_index_real_return = (test_index_df['return'] + 1).cumprod()

    ae_train_real_return, ae_valid_real_return, ae_test_real_return = result('ae_alpha', train_index_real_return, valid_index_real_return, test_index_real_return)
    wt_train_real_return, wt_valid_real_return, wt_test_real_return = result('wt_alpha', train_index_real_return, valid_index_real_return, test_index_real_return)

    #画图比较
    # plot training
    plt.figure(figsize=(16, 9), dpi=300)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('days', fontsize=18)
    plt.ylabel('net asset value', fontsize=18)
    ae, = plt.plot(list(ae_train_real_return), color='r')
    wt, = plt.plot(list(wt_train_real_return), color='b')
    index, = plt.plot(list(train_index_real_return), color='y')
    plt.legend([ae, wt, index], ['AutoEncoder', 'Wavelet Transform', 'Benchmark-000016'], loc='upper left', fontsize=16)
    plt.title('Train Net-Value Curve (20100104-20141212)', fontsize=18)
    plt.savefig('predict_weights/fig/net_value/train_2.jpeg')
    # plt.show()
    plt.close()

    # alpha curve
    plt.figure(figsize=(16, 9), dpi=300)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('days', fontsize=18)
    plt.ylabel('alpha', fontsize=18)
    ae_alpha, = plt.plot(np.array(ae_train_real_return) - np.array(train_index_real_return), color='r')
    wt_alpha, = plt.plot(np.array(wt_train_real_return) - np.array(train_index_real_return), color='b')
    plt.legend([ae_alpha, wt_alpha], ['AutoEncoder', 'Wavelet Transform'], loc='upper left', fontsize=16)
    plt.title('Train Alpha Curve (20100104-20141212)', fontsize=18)
    plt.savefig('predict_weights/fig/alpha/train_2.jpeg')
    # plt.show()
    plt.close()

    # plot validating
    plt.figure(figsize=(16, 9), dpi=300)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('days', fontsize=18)
    plt.ylabel('net asset value', fontsize=18)
    ae, = plt.plot(list(ae_valid_real_return), color='r')
    wt, = plt.plot(list(wt_valid_real_return), color='b')
    index, = plt.plot(list(valid_index_real_return), color='y')
    plt.legend([ae, wt, index], ['AutoEncoder', 'Wavelet Transform', 'Benchmark-000016'], loc='upper left', fontsize=16)
    plt.title('Valid Net-Value Curve (20141212-20161209)', fontsize=18)
    plt.savefig('predict_weights/fig/net_value/valid_2.jpeg')
    # plt.show()
    plt.close()

    plt.figure(figsize=(16, 9), dpi=300)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('days', fontsize=18)
    plt.ylabel('alpha', fontsize=18)
    ae_alpha, = plt.plot(np.array(ae_valid_real_return) - np.array(valid_index_real_return), color='r')
    wt_alpha, = plt.plot(np.array(wt_valid_real_return) - np.array(valid_index_real_return), color='b')
    plt.legend([ae_alpha, wt_alpha], ['AutoEncoder', 'Wavelet Transform'], loc='upper left', fontsize=16)
    plt.title('Valid Alpha Curve (20141212-20161209)', fontsize=18)
    plt.savefig('predict_weights/fig/alpha/valid_2.jpeg')
    # plt.show()
    plt.close()

    # plot testing
    plt.figure(figsize=(16, 9), dpi=300)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('days', fontsize=18)
    plt.ylabel('net asset value', fontsize=18)
    ae, = plt.plot(list(ae_test_real_return), color='r')
    wt, = plt.plot(list(wt_test_real_return), color='b')
    index, = plt.plot(list(test_index_real_return), color='y')
    plt.legend([ae, wt, index], ['AutoEncoder', 'Wavelet Transform', 'Benchmark-000016'], loc='upper left', fontsize=16)
    plt.title('Test Net-Value Curve (20161209-20171228)', fontsize=18)
    plt.savefig('predict_weights/fig/net_value/test_2.jpeg')
    # plt.show()
    plt.close()

    plt.figure(figsize=(16, 9), dpi=300)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('days', fontsize=18)
    plt.ylabel('alpha', fontsize=18)
    ae_alpha, = plt.plot(np.array(ae_test_real_return) - np.array(test_index_real_return), color='r')
    wt_alpha, = plt.plot(np.array(wt_test_real_return) - np.array(test_index_real_return), color='b')
    plt.legend([ae_alpha, wt_alpha], ['AutoEncoder', 'Wavelet Transform'], loc='upper left', fontsize=16)
    plt.title('Test Alpha Curve (20161209-20171228)', fontsize=18)
    plt.savefig('predict_weights/fig/alpha/test_2.jpeg')
    # plt.show()
    plt.close()





if __name__ == '__main__':
    main()
