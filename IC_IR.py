'''因子有效性检验
input: 指定有效性检验时间段
1. 获取检验时间段内不同的时期（根据成份股变更情况）
2，获取不同时期内的成份股列表'''

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# def get_y(test_y_path):
#     test_y = []
#     columns_y = None
#     for f in os.listdir(test_y_path):
#         df = pd.read_csv('{}/{}'.format(test_y_path, f), index_col='DateTime')
#         test_y.append(df)
#         columns_y = df.columns
#     for i in range(len(test_y)):
#         test_y[i].columns = columns_y
#     test_y = pd.concat(test_y)
#     # print(test_y)
#     return test_y


# 自编码结果的IC值
def encoder_IC(start_date, end_date, index_code, model_name, mode='normal', data_set='valid', feature='alpha'):
    # 读取自编码器编码数据
    factors_df = pd.read_csv('../Data/AutoEncoder/{}/{}/{}_{}_hidden_result.csv'.
                             format(model_name, data_set, feature, data_set), index_col=['DateTime'],
                             dtype={'SecCode': np.str})
    factors_df = factors_df.reset_index().set_index(['SecCode', 'DateTime'])

    # 获取不同时期的成份股列表
    f_list = os.listdir('../feature/features/amp')
    for f in f_list:
        f_splits = f.split('_')
        if int(start_date) < int(f_splits[-1].split('.')[0]) < int(end_date) and f_splits[0] == index_code:
            df = pd.read_csv('../feature/features/amp/{}'.format(f), index_col='DateTime')
            cons_stocks_list = list(df.columns)  # constituent stocks
            # print(cons_stocks_list)
            s_date = f_splits[2]  # start date
            e_date = f_splits[-1].split('.')[0]  # end date
            print(s_date, e_date)
            for i in range(0, factors_df.shape[1]):  # for each factor
                factor_df = factors_df[str(i)]
                factor_df = factor_df.reset_index()[
                    (factor_df.reset_index().DateTime > s_date) & (factor_df.reset_index().DateTime < e_date)]
                factor_df = factor_df.set_index(['SecCode', 'DateTime'])
                factor_df = factor_df.loc[:, str(i)]
                factor_df = factor_df.unstack(0)
                factor_df = factor_df.loc[:, cons_stocks_list]
                print(factor_df.shape)

                for j in range(0, 10):  # for each delay
                    future_ret_df = pd.read_csv(
                        '../Data/Model2/predicts/predicts/{}_{}_{}_return_{}-{}min.csv'.
                            format(index_code, s_date, e_date, 241 * j, 241 * (j + 1)), index_col='DateTime')
                    if mode == 'normal':
                        IC = future_ret_df.corrwith(factor_df, axis=1, drop=True)
                        IC_MA = IC.rolling(241*10).mean()
                    elif mode == 'rank':
                        future_ret_rank = future_ret_df.rank(axis=1)
                        factor_rank = factor_df.rank(axis=1)
                        IC = future_ret_rank.corrwith(factor_rank, axis=1, drop=True)
                        IC_MA = IC.rolling(241*10).mean()
                    print(IC)
                    plt.figure(figsize=(16, 9))
                    plt.plot(np.array(IC))
                    plt.plot(np.array(IC_MA), color='r')
                    plt.hlines(0.05, 0, len(IC), colors='r')
                    plt.hlines(-0.05, 0, len(IC), colors='r')
                    plt.hlines(0.1, 0, len(IC), colors='c')
                    plt.hlines(-0.1, 0, len(IC), colors='c')
                    plt.title('{}_{}_{}_{}_{}_{}_delay{}'.format(index_code, mode, factors_df.shape[1], i, s_date, e_date, j+1))
                    plt.xlabel('min')
                    plt.ylabel('IC')
                    if not os.path.exists('factors_effect_fig/{}_ae_{}_{}_{}_{}'.
                                                  format(index_code, data_set, mode, factors_df.shape[1], i)):
                        os.mkdir('factors_effect_fig/{}_ae_{}_{}_{}_{}'.
                                                  format(index_code, data_set, mode, factors_df.shape[1], i))

                    plt.savefig('factors_effect_fig/{}_ae_{}_{}_{}_{}/{}_{}_delay{}'.
                                                  format(index_code, data_set, mode, factors_df.shape[1], i, s_date, e_date, j+1))
                    # plt.show()
                    plt.close()


# 自编码结果的IC值
def encoder_IC_1(start_date, end_date, index_code, model_name, mode='normal', data_set='test', feature='alpha'):
    # 读取自编码器编码数据
    factors_df = pd.read_csv('../Data/AutoEncoder/{}/{}/{}_{}_hidden_result.csv'.
                             format(model_name, data_set, feature, data_set), index_col=['DateTime'],
                             dtype={'SecCode': np.str})
    factors_df = factors_df.reset_index().set_index(['SecCode', 'DateTime'])

    factors_daily_IC_dict = {}
    for i in range(factors_df.shape[1]):
        factors_daily_IC_dict[i] = pd.Series()

    # 获取不同时期的成份股列表
    f_list = os.listdir('../feature/features/amp')
    for f in f_list:
        f_splits = f.split('_')
        if int(start_date) < int(f_splits[-1].split('.')[0]) < int(end_date) and f_splits[0] == index_code:
            df = pd.read_csv('../feature/features/amp/{}'.format(f), index_col='DateTime')
            cons_stocks_list = list(df.columns)  # constituent stocks
            # print(cons_stocks_list)
            s_date = f_splits[2]  # start date
            e_date = f_splits[-1].split('.')[0]  # end date
            print(s_date, e_date)
            for i in range(0, factors_df.shape[1]):  # for each factor
                factor_df = factors_df[str(i)]
                factor_df = factor_df.reset_index()[
                    (factor_df.reset_index().DateTime > s_date) & (factor_df.reset_index().DateTime < e_date)]
                factor_df = factor_df.set_index(['SecCode', 'DateTime'])
                factor_df = factor_df.loc[:, str(i)]
                factor_df = factor_df.unstack(0)
                factor_df = factor_df.loc[:, cons_stocks_list]
                print(factor_df.shape)

                for j in range(0, 1):  # for each delay
                    future_ret_df = pd.read_csv(
                        '../Data/Model2/predicts/predicts/{}_{}_{}_return_{}-{}min.csv'.
                            format(index_code, s_date, e_date, 241 * j, 241 * (j + 1)), index_col='DateTime')
                    if mode == 'normal':
                        IC_min = future_ret_df.corrwith(factor_df, axis=1, drop=True)
                        IC_daily = IC_min.iloc[range(0, (len(IC_min)//241)*241, 241)]
                    elif mode == 'rank':
                        future_ret_rank = future_ret_df.rank(axis=1)
                        factor_rank = factor_df.rank(axis=1)
                        IC_min = future_ret_rank.corrwith(factor_rank, axis=1, drop=True)
                        IC_daily = IC_min.iloc[range(0, (len(IC_min)//241)*241, 241)]
                    factors_daily_IC_dict[i] = factors_daily_IC_dict[i].append(IC_daily)
    for i in range(factors_df.shape[1]):
        plt.figure(figsize=(16, 9), dpi=300)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('days', fontsize=18)
        plt.ylabel('IC', fontsize=18)
        daily_IC, = plt.plot(list(factors_daily_IC_dict[i]))
        daily_IC_ma, = plt.plot(list(factors_daily_IC_dict[i].rolling(30).mean()))
        plt.legend([daily_IC, daily_IC_ma], ['IC', 'IC 30 days\' Moving Avg.'], loc='upper left', fontsize=16)
        # plt.plot(list(factors_daily_IC_dict[i]))
        # plt.plot(list(factors_daily_IC_dict[i].rolling(30).mean()))
        plt.hlines(factors_daily_IC_dict[i].mean(), 0, len(list(factors_daily_IC_dict[i])), colors='g')
        plt.hlines(0.05, 0, len(list(factors_daily_IC_dict[i])), colors='r', linestyles="dashed")
        plt.hlines(-0.05, 0, len(list(factors_daily_IC_dict[i])), colors='r', linestyles="dashed")
        plt.hlines(0.1, 0, len(list(factors_daily_IC_dict[i])), colors='c', linestyles="dashed")
        plt.hlines(-0.1, 0, len(list(factors_daily_IC_dict[i])), colors='c', linestyles="dashed")
        plt.title('{}_{}_factor{}'.format(mode, data_set, i), fontsize=18)
        # plt.savefig('../ai/factors_testing/encoding_effects/{}/{}_{}_{}_{}_IC={}_IR={}.jpeg'.
        #             format(mode, model_name, factors_df.columns[i], mode, data_set, factors_daily_IC_dict[i].mean(),
        #                    factors_daily_IC_dict[i].mean() / factors_daily_IC_dict[i].std()))
        plt.show()
        plt.close()


# 原始技术指标的IC值
def technical_IC(start_date, end_date, index_code, mode='normal', data_set='valid'):

    factors_df = pd.read_csv('../feature/all_features/technical_features_16_20100104_20150101.csv', index_col=['DateTime'],
                            dtype={'SecCode': np.str})
    #factors_df = pd.read_csv('../feature/all_features/technical_features_16_20150101_20170101.csv', index_col=['DateTime'],
             #               dtype={'SecCode': np.str})
    #factors_df = pd.read_csv('../feature/all_features/technical_features_16_20170101_20171231.csv', index_col=['DateTime'],
     #                       dtype={'SecCode': np.str})

    factors_df = factors_df.reset_index().set_index(['SecCode', 'DateTime'])

    factors_daily_IC_dict = {}
    for i in range(factors_df.shape[1]):
        factors_daily_IC_dict[i] = pd.Series()

    # 获取不同时期的成份股列表
    f_list = os.listdir('../feature/features/amp')
    for f in f_list:
        f_splits = f.split('_')
        if int(start_date) < int(f_splits[-1].split('.')[0]) < int(end_date) and f_splits[0] == index_code:
            df = pd.read_csv('../feature/features/amp/{}'.format(f), index_col='DateTime')
            cons_stocks_list = list(df.columns)  # constituent stocks

            s_date = f_splits[2]  # start date
            e_date = f_splits[-1].split('.')[0]  # end date
            print(s_date, e_date)
            for i in range(0, factors_df.shape[1]):  # for each factor
                print(factors_df.columns[i])
                factor_df = factors_df[factors_df.columns[i]]
                factor_df = factor_df.reset_index()[
                    (factor_df.reset_index().DateTime > s_date) & (factor_df.reset_index().DateTime < e_date)]
                factor_df = factor_df.set_index(['SecCode', 'DateTime'])
                factor_df = factor_df.loc[:, factors_df.columns[i]]
                factor_df = factor_df.unstack(0)
                factor_df = factor_df.loc[:, cons_stocks_list]
                print(factor_df.shape)

                for j in range(0, 1):
                    future_ret_df = pd.read_csv(
                        '../Data/Model2/predicts/predicts/{}_{}_{}_return_{}-{}min.csv'.
                            format(index_code, s_date, e_date, 241*j, 241*(j+1)), index_col='DateTime')
                    if mode == 'normal':
                        IC_min = future_ret_df.corrwith(factor_df, axis=1, drop=True)
                        IC_daily = IC_min.iloc[range(0, (len(IC_min) // 241) * 241, 241)]
                    elif mode == 'rank':
                        future_ret_rank = future_ret_df.rank(axis=1)
                        factor_rank = factor_df.rank(axis=1)
                        IC_min = future_ret_rank.corrwith(factor_rank, axis=1, drop=True)
                        IC_daily = IC_min.iloc[range(0, (len(IC_min) // 241) * 241, 241)]
                    factors_daily_IC_dict[i] = factors_daily_IC_dict[i].append(IC_daily)
    for i in range(factors_df.shape[1]):
        plt.figure(figsize=(16, 9), dpi=300)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('days', fontsize=15)
        plt.ylabel('IC', fontsize=15)
        daily_IC, = plt.plot(list(factors_daily_IC_dict[i].fillna(method='bfill')))
        daily_IC_ma, = plt.plot(list(factors_daily_IC_dict[i].fillna(method='bfill').rolling(30).mean()))
        plt.legend([daily_IC, daily_IC_ma], ['IC', 'IC 30 days\' Moving Avg.'], loc='upper left', fontsize=15)
        plt.hlines(factors_daily_IC_dict[i].mean(), 0, len(list(factors_daily_IC_dict[i])), colors='g')
        plt.hlines(0.05, 0, len(list(factors_daily_IC_dict[i])), colors='r', linestyles="dashed")
        plt.hlines(-0.05, 0, len(list(factors_daily_IC_dict[i])), colors='r', linestyles="dashed")
        plt.hlines(0.1, 0, len(list(factors_daily_IC_dict[i])), colors='c', linestyles="dashed")
        plt.hlines(-0.1, 0, len(list(factors_daily_IC_dict[i])), colors='c', linestyles="dashed")
        plt.title('{}_{}_{}'.format(mode, data_set, factors_df.columns[i]))
        # plt.savefig('../ai/factors_testing/technical_effects/{}/{}_{}_{}_IC={}_IR={}.jpeg'.
        #             format(mode, factors_df.columns[i], mode, data_set, factors_daily_IC_dict[i].mean(),
        #                    factors_daily_IC_dict[i].mean()/factors_daily_IC_dict[i].std()))
        plt.show()
        plt.close()

# Alpha因子的IC值
def alpha_IC(start_date, end_date, index_code, mode='normal', data_set='valid'):

    factors_df = pd.read_csv('../feature/all_features/alpha_features_45_20100104_20150101.csv', index_col=['DateTime'],
                             dtype={'SecCode': np.str})
    # factors_df = pd.read_csv('../feature/all_features/alpha_features_45_20150101_20170101.csv', index_col=['DateTime'],
    #               dtype={'SecCode': np.str})
    # factors_df = pd.read_csv('../feature/all_features/alpha_features_45_20170101_20171231.csv', index_col=['DateTime'],
    #                          dtype={'SecCode': np.str})
    factors_df = factors_df.reset_index().set_index(['SecCode', 'DateTime'])

    factors_daily_IC_dict = {}
    for i in range(factors_df.shape[1]):
        factors_daily_IC_dict[i] = pd.Series()

    # 获取不同时期的成份股列表
    f_list = os.listdir('../feature/features/amp')
    for f in f_list:
        f_splits = f.split('_')
        if int(start_date) < int(f_splits[-1].split('.')[0]) < int(end_date) and f_splits[0] == index_code:
            df = pd.read_csv('../feature/features/amp/{}'.format(f), index_col='DateTime')
            cons_stocks_list = list(df.columns)  # constituent stocks

            s_date = f_splits[2]  # start date
            e_date = f_splits[-1].split('.')[0]  # end date
            print(s_date, e_date)
            for i in range(0, factors_df.shape[1]):  # for each factor
                print(factors_df.columns[i])
                factor_df = factors_df[factors_df.columns[i]]
                factor_df = factor_df.reset_index()[
                    (factor_df.reset_index().DateTime > s_date) & (factor_df.reset_index().DateTime < e_date)]
                factor_df = factor_df.set_index(['SecCode', 'DateTime'])
                factor_df = factor_df.loc[:, factors_df.columns[i]]
                factor_df = factor_df.unstack(0)
                factor_df = factor_df.loc[:, cons_stocks_list]
                print(factor_df.shape)

                for j in range(0, 1):
                    future_ret_df = pd.read_csv(
                        '../Data/Model2/predicts/predicts/{}_{}_{}_return_{}-{}min.csv'.
                            format(index_code, s_date, e_date, 241*j, 241*(j+1)), index_col='DateTime')
                    if mode == 'normal':
                        IC_min = future_ret_df.corrwith(factor_df, axis=1, drop=True)
                        IC_daily = IC_min.iloc[range(0, (len(IC_min) // 241) * 241, 241)]
                    elif mode == 'rank':
                        future_ret_rank = future_ret_df.rank(axis=1)
                        factor_rank = factor_df.rank(axis=1)
                        IC_min = future_ret_rank.corrwith(factor_rank, axis=1, drop=True)
                        IC_daily = IC_min.iloc[range(0, (len(IC_min) // 241) * 241, 241)]
                    factors_daily_IC_dict[i] = factors_daily_IC_dict[i].append(IC_daily)

    for i in range(factors_df.shape[1]):
        plt.figure(figsize=(16, 9))
        plt.plot(list(factors_daily_IC_dict[i].fillna(method='bfill')))
        plt.plot(list(factors_daily_IC_dict[i].fillna(method='bfill').rolling(30).mean()))
        plt.hlines(factors_daily_IC_dict[i].mean(), 0, len(list(factors_daily_IC_dict[i])), colors='g')
        plt.hlines(0.05, 0, len(list(factors_daily_IC_dict[i])), colors='r', linestyles="dashed")
        plt.hlines(-0.05, 0, len(list(factors_daily_IC_dict[i])), colors='r', linestyles="dashed")
        plt.hlines(0.1, 0, len(list(factors_daily_IC_dict[i])), colors='c', linestyles="dashed")
        plt.hlines(-0.1, 0, len(list(factors_daily_IC_dict[i])), colors='c', linestyles="dashed")
        plt.title('{}_{}_{}'.format(mode, data_set, factors_df.columns[i]))
        # plt.savefig('../ai/factors_testing/alpha_effects/{}/{}_{}_{}_IC={}_IR={}.jpeg'.
        #             format(mode, factors_df.columns[i], mode, data_set, factors_daily_IC_dict[i].mean(),
        #                    factors_daily_IC_dict[i].mean()/factors_daily_IC_dict[i].std()))
        plt.show()
        plt.close()


def encoder_elastic_network(model_name='hidden_result', layer=1, data_set='train', feature='alpha'):
    # 读取自编码器编码数据
    factors_df = pd.read_csv('../Data/AutoEncoder/{}/{}/{}_{}_hidden_result.csv'.
                             format(model_name, data_set, feature, data_set), index_col=['DateTime'],
                             dtype={'SecCode': np.str})
    factors_df = factors_df.reset_index()
    print(factors_df.shape)
    print(factors_df.head())

    result_df = []
    for f in os.listdir('../Data/Model2/model2_predicts/{}'.format(data_set)):
        y = pd.read_csv('../Data/Model2/model2_predicts/{}/{}'.format(data_set, f), index_col='DateTime')
        y = y.stack(0)
        y = y.reset_index()
        y.columns = ['DateTime', 'SecCode', 'y']
        join_df = pd.merge(factors_df, y, on=['DateTime', 'SecCode'], how='inner')
        # print(join_df.shape)
        # print(join_df.head())
        result_df.append(join_df)
    result_df = pd.concat(result_df)
    scaler = StandardScaler()
    train = scaler.fit(result_df.iloc[:, 2:]).transform(result_df.iloc[:, 2:])
    # print(result_df.shape)
    # print(result_df.head())
    enet = ElasticNet(alpha=0.00001, l1_ratio=0.5)
    # lasso = Lasso(alpha=0.0001)
    # ridge = Ridge(alpha=0.0001)
    # rf = RandomForestRegressor(n_estimators=10, n_jobs=4, verbose=1)
    # rf.fit(train[:, :-1], train[:, -1])
    # print(rf.feature_importances_)
    # enet.fit(result_df.iloc[:, 2:-1], result_df.iloc[:, -1])
    # lasso.fit(result_df.iloc[:, 2:-1], result_df.iloc[:, -1])
    # ridge.fit(result_df.iloc[:, 2:-1], result_df.iloc[:, -1])
    enet.fit(train[:, :-1], train[:, -1])
    # lasso.fit(train[:, :-1], train[:, -1])
    # ridge.fit(train[:, :-1], train[:, -1])
    print(enet.coef_)
    # print(lasso.coef_)
    # print(ridge.coef_)


def technical_elastic_network(data_set='train'):
    factors_df = pd.read_csv('../feature/all_features/technical_features_16_20150101_20170101.csv', index_col=['DateTime'],
                             dtype={'SecCode': np.str})
    factors_df = factors_df.reset_index()
    factors_df.replace(np.inf, np.nan, inplace=True)
    factors_df.replace(-np.inf, np.nan, inplace=True)
    factors_df.dropna(inplace=True)
    print(factors_df.shape)
    print(factors_df.head())

    result_df = []
    for f in os.listdir('../Data/Model2/model2_predicts/{}'.format(data_set)):
        y = pd.read_csv('../Data/Model2/model2_predicts/{}/{}'.format(data_set, f), index_col='DateTime')
        y = y.stack(0)
        y = y.reset_index()
        y.columns = ['DateTime', 'SecCode', 'y']
        join_df = pd.merge(factors_df, y, on=['DateTime', 'SecCode'], how='inner')
        print(join_df.shape)
        # print(join_df.head())
        result_df.append(join_df)
    result_df = pd.concat(result_df)
    # result_df.dropna(inplace=True)
    scaler = StandardScaler()
    train = scaler.fit(result_df.iloc[:, 2:]).transform(result_df.iloc[:, 2:])
    print(result_df.shape)
    print(result_df.head())
    enet = ElasticNet(alpha=0.0001, l1_ratio=0.5)
    lasso = Lasso(alpha=0.0001)
    ridge = Ridge(alpha=0.0001)
    # enet.fit(result_df.iloc[:, 2:-1], result_df.iloc[:, -1])
    # lasso.fit(result_df.iloc[:, 2:-1], result_df.iloc[:, -1])
    # ridge.fit(result_df.iloc[:, 2:-1], result_df.iloc[:, -1])
    enet.fit(train[:, :-1], train[:, -1])
    lasso.fit(train[:, :-1], train[:, -1])
    ridge.fit(train[:, :-1], train[:, -1])
    print(enet.coef_)
    print(lasso.coef_)
    print(ridge.coef_)

def main():
    # encoder_IC_1('20150101', '20170101', '000016', mode='normal', model_name='4_B_0', layer=4, data_set='valid')
    # encoder_IC_1('20150101', '20170101', '000016', mode='rank', model_name='1_A_0', layer=1, data_set='valid')
    # encoder_IC_1('20100104', '20150101', '000016', mode='normal', model_name='4_B_0', layer=4, data_set='train')
    # encoder_IC_1('20100104', '20150101', '000016', mode='rank', model_name='4_B_0', layer=4, data_set='train')
    encoder_IC_1('20170101', '20171231', '000016', mode='normal', model_name='hidden_result', data_set='test')
    # encoder_IC_1('20170101', '20171231', '000016', mode='rank', model_name='4_B_0', layer=4, data_set='test')
    # encoder_IC('20150101',  '20170101', '000016', mode='normal', data_set='valid')
    # encoder_IC('20150101', '20170101', '000016', mode='rank', data_set='valid')
    # technical_IC('20100104', '20150101', '000016', mode='normal', data_set='train')
    # technical_IC('20100104', '20150101', '000016', mode='rank', data_set='train')
    # technical_IC('20150101', '20170101', '000016', mode='normal', data_set='valid')
    # technical_IC('20150101', '20170101', '000016', mode='rank', data_set='valid')
    # technical_IC('20170101', '20171231', '000016', mode='normal', data_set='test')
    # technical_IC('20170101', '20171231', '000016', mode='rank', data_set='test')
    # alpha_IC('20100104', '20150101', '000016', mode='normal', data_set='train')
    # alpha_IC('20100104', '20150101', '000016', mode='rank', data_set='train')
    # alpha_IC('20150101', '20170101', '000016', mode='normal', data_set='valid')
    # alpha_IC('20150101', '20170101', '000016', mode='rank', data_set='valid')
    # alpha_IC('20170101', '20171231', '000016', mode='normal', data_set='test')
    # alpha_IC('20170101', '20171231', '000016', mode='rank', data_set='test')
    # encoder_elastic_network(model_name='1_A_0', data_set='valid')
    # technical_elastic_network(data_set='valid')

if __name__ == '__main__':
    main()