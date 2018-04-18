'''权重向量进行回测
1.将权重向量进行归一化，使每个时刻的权重之和为1
2.计算每日的超额收益'''
import os
import numpy as np
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt

# 将权重向量归一化和为1
def normalize_weights(daily_weights_df ,method=1):
    # daily_weights_df = pd.read_csv(path, index_col='DateTime')

    daily_min = daily_weights_df.min(axis=1)
    daily_max = daily_weights_df.max(axis=1)

    # Method 1: 使用Min-Max对权重向量进行标准化后归一化权重
    if method == 1:
        norm_daily_weights = \
            daily_weights_df.sub(daily_min, axis=0).div(daily_max-daily_min, axis=0)
        # norm_daily_weights = norm_daily_weights.div(norm_daily_weights.sum(axis=1), axis=0)

    # Method 2: 使用0代替负权重后归一化权重
    if method == 2:
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

# 以市场中性方式归一化权重向量
def normalize_neutral_weights(path):
    daily_weights_df = pd.read_csv(path, index_col='DateTime')
    print(daily_weights_df.head())

    daily_mean = daily_weights_df.mean(axis=1)
    norm_daily_weights = daily_weights_df.sub(daily_mean, axis=0)
    return norm_daily_weights


def main():
    # model_name_list = ['1_A', '1_B', '1_C', '1_D', '3_A', '3_B', '4_A_layer1', '4_A_layer2',
    #                    '4_A_layer3', '4_A_layer4', '4_B_layer1', '4_B_layer2', '4_B_layer3', '4_B_layer4']
    model_name_list = ['10best_technical']
    # mode_list = ['dropout', 'cost_limited']
    mode_list = ['cost_limited']
    method_list = [1, 2]
    f = open('result.csv', 'w')
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

    f.write('model_name,mode,method,train_rmse,valid_rmse,test_rmse,train_cost,valid_cost,test_cost,'
            'train_alpha,valid_alpha,test_alpha,train_max_alpha,valid_max_alpha,test_max_alpha,'
            'train_profit_percent,valid_profit_percent,test_profit_percent\n')
    for model_name in model_name_list:
        for mode in mode_list:
            for method in method_list:
                print(model_name, mode, method)

                print('Reading predict results...')
                train_predict_df = pd.read_csv('..Data/Model2/predict_weights/train_predict_weights_{}_mode={}.csv'.format(model_name, mode), index_col='DateTime')
                valid_predict_df = pd.read_csv('..Data/Model2/predict_weights/valid_predict_weights_{}_mode={}.csv'.format(model_name, mode), index_col='DateTime')
                test_predict_df = pd.read_csv('..Data/Model2/predict_weights/test_predict_weights_{}_mode={}.csv'.format(model_name, mode), index_col='DateTime')

                # 将预测的收益率向量转换为权重值
                print('Normalizing weights...')
                train_norm_min_weights = normalize_weights(train_predict_df, method=method)
                valid_norm_min_weights = normalize_weights(valid_predict_df, method=method)
                test_norm_min_weights = normalize_weights(test_predict_df, method=method)
                # 按照市场中性归一化为权重
                # train_norm_min_weights = normalize_neutral_weights('predict_weights/train_predict_weights_4_B_layer1_mode=normal.csv')
                # valid_norm_min_weights = normalize_neutral_weights('predict_weights/valid_predict_weights_4_B_layer1_mode=normal.csv')
                # test_norm_min_weights = normalize_neutral_weights('predict_weights/test_predict_weights_4_B_layer1_mode=normal.csv')

                print('Reading targets...')
                train_y = get_y('../Data/Model2/predicts/train')
                valid_y = get_y('../Data/Model2/predicts/valid')
                test_y = get_y('../Data/Model2/predicts/test')

                print(train_predict_df.shape, train_y.shape)
                print(valid_predict_df.shape, valid_y.shape)
                print(test_predict_df.shape, test_y.shape)

                # 每天的预测收益率
                train_daily_predict = train_predict_df.iloc[range(0, (len(train_predict_df)//241)*241, 241)]
                # 每天的归一化权重
                train_norm_daily_weights = train_norm_min_weights.iloc[range(0, (len(train_norm_min_weights)//241)*241, 241)]
                # 每天的实际收益率
                train_daily_return = train_y.iloc[range(0, (len(train_y)//241)*241, 241)]
                # 根据预测结果的每天的实际真实收益
                train_daily_real_return = train_norm_daily_weights * train_daily_return
                train_real_return = (train_daily_real_return.sum(axis=1)+1).cumprod()

                valid_daily_predict = valid_predict_df.iloc[range(0, (len(valid_predict_df) // 241) * 241, 241)]
                valid_norm_daily_weights = valid_norm_min_weights.iloc[range(0, (len(valid_norm_min_weights) // 241) * 241, 241)]
                valid_daily_return = valid_y.iloc[range(0, (len(valid_y) // 241) * 241, 241)]
                valid_daily_real_return = valid_norm_daily_weights * valid_daily_return
                valid_real_return = (valid_daily_real_return.sum(axis=1)+1).cumprod()

                test_daily_predict = test_predict_df.iloc[range(0, (len(test_predict_df) // 241) * 241, 241)]
                test_norm_daily_weights = test_norm_min_weights.iloc[range(0, (len(test_norm_min_weights) // 241) * 241, 241)]
                test_daily_return = test_y.iloc[range(0, (len(test_y) // 241) * 241, 241)]
                test_daily_real_return = test_norm_daily_weights * test_daily_return
                test_real_return = (test_daily_real_return.sum(axis=1)+1).cumprod()

                # 计算预测结果的MAPE(原始值存在0的数据无法使用MAPE
                # train_min_mape = (((train_predict_df - train_y) / train_y).abs() * 100)
                # valid_min_mape = (((valid_predict_df - valid_y) / valid_y).abs() * 100)
                # test_min_mape = (((test_predict_df - test_y) / test_y).abs() * 100)
                # train_daily_mape = ((train_daily_predict - train_daily_return) / train_daily_return).abs() * 100
                # valid_daily_mape = ((valid_daily_predict - valid_daily_return) / valid_daily_return).abs() * 100
                # test_daily_mape = ((test_daily_predict - test_daily_return) / test_daily_return).abs() * 100

                # 计算预测结果的MSE
                train_daily_mse = ((train_daily_predict - train_daily_return)**2).mean(axis=1).mean()
                valid_daily_mse = ((valid_daily_predict - valid_daily_return)**2).mean(axis=1).mean()
                test_daily_mse = ((test_daily_predict - test_daily_return)**2).mean(axis=1).mean()
                train_daily_rmse = train_daily_mse**0.5
                valid_daily_rmse = valid_daily_mse**0.5
                test_daily_rmse = test_daily_mse**0.5

                print('==================================')
                print('Train Daily MSE: {}  RMSE: {}'.format(train_daily_mse, train_daily_rmse))
                print('Valid Daily MSE: {}  RMSE: {}'.format(valid_daily_mse, valid_daily_rmse))
                print('Test Daily MSE: {}  RMSE: {}'.format(test_daily_mse, test_daily_rmse))


                # 评估调仓成本
                train_min_cost = (train_norm_min_weights - train_norm_min_weights.shift()).abs().sum(axis=1).mean()
                train_daily_cost = (train_norm_daily_weights - train_norm_daily_weights.shift()).abs().sum(axis=1).mean()
                valid_min_cost = (valid_norm_min_weights - valid_norm_min_weights.shift()).abs().sum(axis=1).mean()
                valid_daily_cost = (valid_norm_daily_weights - valid_norm_daily_weights.shift()).abs().sum(axis=1).mean()
                test_min_cost = (test_norm_min_weights - test_norm_min_weights.shift()).abs().sum(axis=1).mean()
                test_daily_cost = (test_norm_daily_weights - test_norm_daily_weights.shift()).abs().sum(axis=1).mean()

                print('==================================')
                print('Train Min mean cost: {}'.format(train_min_cost))
                print('Train Daily mean cost: {}'.format(train_daily_cost))
                print('Valid Min mean cost: {}'.format(valid_min_cost))
                print('Valid Daily mean cost: {}'.format(valid_daily_cost))
                print('Test Min mean cost: {}'.format(test_min_cost))
                print('Test Daily mean cost: {}'.format(test_daily_cost))



                train_daily_alpha = np.array(list(train_real_return)) - np.array(list(train_index_real_return))
                valid_daily_alpha = np.array(list(valid_real_return)) - np.array(list(valid_index_real_return))
                test_daily_alpha = np.array(list(test_real_return)) - np.array(list(test_index_real_return))

                print('==================================')
                print('Train Cumulative alpha: {}'.format(train_daily_alpha[-1]))
                print('Train Max Alpha: {}'.format(np.nanmax(train_daily_alpha)))
                print('Train +%: {}%'.format(len(train_daily_alpha[train_daily_alpha > 0]) / len(train_daily_alpha) * 100))
                print('Valid cumulative alpha: {}'.format(valid_daily_alpha[-1]))
                print('Valid Max Alpha: {}'.format(np.nanmax(valid_daily_alpha)))
                print('Valid +%: {}%'.format(len(valid_daily_alpha[valid_daily_alpha > 0]) / len(valid_daily_alpha) * 100))
                print('Test Cumulative alpha: {}'.format(test_daily_alpha[-1]))
                print('Test Max Alpha: {}'.format(np.nanmax(test_daily_alpha)))
                print('Test +%: {}%'.format(len(test_daily_alpha[test_daily_alpha > 0]) / len(test_daily_alpha) * 100))

                # plot training
                plt.figure(figsize=(16, 9), dpi=300)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.xlabel('days', fontsize=18)
                plt.ylabel('net asset value', fontsize=18)
                my_strategy, = plt.plot(list(train_real_return), color='r')
                index, = plt.plot(list(train_index_real_return), color='b')
                plt.legend([my_strategy, index], ['Strategy', 'Benchmark-000016'], loc='upper left', fontsize=16)
                plt.title('Train Net-Value Curve (20100104-20141212)', fontsize=18)
                plt.savefig('..Data/Model2/predict_weights/fig/net_value/{}_{}_method={}_train.jpeg'.format(model_name, mode, method))
                # plt.show()
                plt.close()

                # alpha curve
                plt.figure(figsize=(16, 9), dpi=300)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.xlabel('days', fontsize=18)
                plt.ylabel('alpha', fontsize=18)
                alpha, = plt.plot(np.array(train_real_return) - np.array(train_index_real_return), color='y')
                plt.legend([alpha], ['alpha'], loc='upper left', fontsize=16)
                plt.title('Train Alpha Curve (20100104-20141212)', fontsize=18)
                plt.savefig('..Data/Model2/predict_weights/fig/alpha/{}_{}_method={}_train.jpeg'.format(model_name, mode, method))
                # plt.show()
                plt.close()

                # daily Turnover
                plt.figure(figsize=(16, 9), dpi=300)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.xlabel('days', fontsize=18)
                plt.ylabel('turnover', fontsize=18)
                turnover, = plt.plot(list((train_norm_daily_weights - train_norm_daily_weights.shift()).abs().sum(axis=1)))
                turnover_ma10, = plt.plot(list((train_norm_daily_weights - train_norm_daily_weights.shift()).abs().sum(axis=1).rolling(10).mean()))
                plt.legend([turnover, turnover_ma10], ['turnover', 'turnover-10'], loc='upper left', fontsize=16)
                plt.title('Train Turnover Curve (20100104-20141212)', fontsize=18)
                plt.savefig('..Data/Model2/predict_weights/fig/turnover/{}_{}_method={}_train.jpeg'.format(model_name, mode, method))
                # plt.show()
                plt.close()

                # plot validating
                plt.figure(figsize=(16, 9), dpi=300)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.xlabel('days', fontsize=18)
                plt.ylabel('net asset value', fontsize=18)
                my_strategy, = plt.plot(list(valid_real_return), color='r')
                index, = plt.plot(list(valid_index_real_return), color='b')
                plt.legend([my_strategy, index], ['Strategy', 'Benchmark-000016'], loc='upper left', fontsize=16)
                plt.title('Valid Net-Value Curve (20141212-20161209)', fontsize=18)
                plt.savefig('..Data/Model/2predict_weights/fig/net_value/{}_{}_method={}_valid.jpeg'.format(model_name, mode, method))
                # plt.show()
                plt.close()

                plt.figure(figsize=(16, 9), dpi=300)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.xlabel('days', fontsize=18)
                plt.ylabel('alpha', fontsize=18)
                alpha, = plt.plot(np.array(valid_real_return) - np.array(valid_index_real_return), color='y')
                plt.legend([alpha], ['alpha'], loc='upper left', fontsize=16)
                plt.title('Valid Alpha Curve (20141212-20161209)', fontsize=18)
                plt.savefig('..Data/Model2/predict_weights/fig/alpha/{}_{}_method={}_valid.jpeg'.format(model_name, mode, method))
                # plt.show()
                plt.close()

                plt.figure(figsize=(16, 9), dpi=300)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.xlabel('days', fontsize=18)
                plt.ylabel('turnover', fontsize=18)
                turnover, = plt.plot(list((valid_norm_daily_weights - valid_norm_daily_weights.shift()).abs().sum(axis=1)))
                turnover_ma10, = plt.plot(list((valid_norm_daily_weights - valid_norm_daily_weights.shift()).abs().sum(axis=1).rolling(10).mean()))
                plt.legend([turnover, turnover_ma10], ['turnover', 'turnover-10'], loc='upper left', fontsize=16)
                plt.title('Valid Turnover Curve (20141212-20161209)', fontsize=18)
                plt.savefig('..Data/Model2/predict_weights/fig/turnover/{}_{}_method={}_valid.jpeg'.format(model_name, mode, method))
                # plt.show()
                plt.close()

                # plot testing
                plt.figure(figsize=(16, 9), dpi=300)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.xlabel('days', fontsize=18)
                plt.ylabel('net asset value', fontsize=18)
                my_strategy, = plt.plot(list(test_real_return), color='r')
                index, = plt.plot(list(test_index_real_return), color='b')
                plt.legend([my_strategy, index], ['Strategy', 'Benchmark-000016'], loc='upper left', fontsize=16)
                plt.title('Test Net-Value Curve (20161209-20171228)', fontsize=18)
                plt.savefig('..Data/Model2/predict_weights/fig/net_value/{}_{}_method={}_test.jpeg'.format(model_name, mode, method))
                # plt.show()
                plt.close()

                plt.figure(figsize=(16, 9), dpi=300)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.xlabel('days', fontsize=18)
                plt.ylabel('alpha', fontsize=18)
                alpha, = plt.plot(np.array(test_real_return) - np.array(test_index_real_return), color='y')
                plt.legend([alpha], ['alpha'], loc='upper left', fontsize=16)
                plt.title('Test Alpha Curve (20161209-20171228)', fontsize=18)
                plt.savefig('..Data/Model2/predict_weights/fig/alpha/{}_{}_method={}_test.jpeg'.format(model_name, mode, method))
                # plt.show()
                plt.close()

                plt.figure(figsize=(16, 9), dpi=300)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.xlabel('days', fontsize=18)
                plt.ylabel('turnover', fontsize=18)
                turnover, = plt.plot(list((test_norm_daily_weights - test_norm_daily_weights.shift()).abs().sum(axis=1)))
                turnover_ma10, = plt.plot(list((test_norm_daily_weights - test_norm_daily_weights.shift()).abs().sum(axis=1).rolling(10).mean()))
                plt.legend([turnover, turnover_ma10], ['turnover', 'turnover-10'], loc='upper left', fontsize=16)
                plt.title('Test Turnover Curve (20161209-20171228)', fontsize=18)
                plt.savefig('..Data/Model2/predict_weights/fig/turnover/{}_{}_method={}_test.jpeg'.format(model_name, mode, method))
                # plt.show()
                plt.close()

                f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.
                        format(model_name, mode, method,
                               train_daily_rmse, valid_daily_rmse, test_daily_rmse,
                               train_daily_cost, valid_daily_cost, test_daily_cost,
                               train_daily_alpha[-1], valid_daily_alpha[-1], test_daily_alpha[-1],
                               np.nanmax(train_daily_alpha), np.nanmax(valid_daily_alpha), np.nanmax(test_daily_alpha),
                               len(train_daily_alpha[train_daily_alpha > 0]) / len(train_daily_alpha),
                               len(valid_daily_alpha[valid_daily_alpha > 0]) / len(valid_daily_alpha),
                               len(test_daily_alpha[test_daily_alpha > 0]) / len(test_daily_alpha)))
                f.flush()

if __name__ == '__main__':
    main()
