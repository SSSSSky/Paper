import os
import pandas as pd
import os
import pandas as pd


def seperateData(path):
    df1 = pd.read_csv(path)
    train_df = df1[df1.DateTime < '20150101']
    valid_df = df1[(df1.DateTime >= '20150101') & (df1.DateTime < '20170101')]
    test_df = df1[df1.DateTime >= '20170101']
    # print('train_df:')
    # print(train_df.head())
    # print(train_df.tail())
    # print('valid_df:')
    # print(valid_df.head())
    # print(valid_df.tail())
    # print('test_df:')
    # print(test_df.head())
    # print(test_df.tail())
    return train_df, valid_df, test_df


def main():
    path = '../feature/join_feature'
    files = os.listdir(path)
    files.sort()
    q = 0
    for file in files:
        q += 1
        file_path = os.path.join(path, file)
        df1 = pd.read_csv(file_path)
        if q == 1:
            join_df = df1
        else:
            join_df = pd.concat([join_df, df1])
            print(join_df.shape)
            # join_df 就是全部的特征数据
    join_df = join_df.sort_values(by='DateTime')
    # 取20个技术指标
    technical_df = join_df.loc[:, ['SecCode', 'DateTime', 'amp', 'ar', 'atr', 'bias', 'boll',
                                   'br', 'cci', 'log_vol_chg_rate', 'logreturn', 'ma_ma_ratio',
                                   'macd', 'mfi', 'obv', 'pri_ma_ratio', 'price_efficiency',
                                   'roc', 'rsi', 'vma', 'vol_chg_rate', 'volume_relative_ratio']]
    #
    # # 取45个alpha指标
    # alpha_df = join_df.loc[:, ['SecCode', 'DateTime', 'alpha001', 'alpha002', 'alpha003', 'alpha004',
    #                            'alpha006', 'alpha007', 'alpha008', 'alpha009', 'alpha010', 'alpha012',
    #                            'alpha013', 'alpha014', 'alpha015', 'alpha016', 'alpha017', 'alpha018',
    #                            'alpha019', 'alpha020', 'alpha021', 'alpha022', 'alpha023', 'alpha024',
    #                            'alpha026', 'alpha028', 'alpha029', 'alpha030', 'alpha031', 'alpha033',
    #                            'alpha034', 'alpha035', 'alpha037', 'alpha038', 'alpha039', 'alpha040',
    #                            'alpha043', 'alpha044', 'alpha045', 'alpha046', 'alpha049', 'alpha051',
    #                            'alpha052', 'alpha053', 'alpha054', 'alpha055', 'alpha060']]
    #
    # print(technical_df.shape)
    # print(alpha_df.shape)
    # print(technical_df.head())
    # print(alpha_df.head())
    #
    pd.DataFrame(data=technical_df, index=technical_df.index, columns=technical_df.columns).to_csv('../feature/technical_all.csv')
    print('finish technical_all.csv...')
    # pd.DataFrame(data=alpha_df, index=alpha_df.index, columns=alpha_df.columns).to_csv('../feature/alpha_all.csv')

    technical_path = '../feature/technical_all.csv'
    # alpha_path = '../feature/alpha_all.csv'
    technical_train_df, technical_valid_df, technical_test_df = seperateData(technical_path)
    # alpha_train_df, alpha_valid_df, alpha_test_df = seperateData(alpha_path)
    pd.DataFrame(data=technical_train_df, index=technical_train_df.index, columns=technical_train_df.columns).to_csv('../Data/AutoEncoder/train/technical_train_df.csv')
    print('finish technical_train_df.csv ...')
    pd.DataFrame(data=technical_valid_df, index=technical_valid_df.index, columns=technical_valid_df.columns).to_csv('../Data/AutoEncoder/valid/technical_valid_df.csv')
    print('finish technical_valid_df.csv ...')
    pd.DataFrame(data=technical_test_df, index=technical_test_df.index, columns=technical_test_df.columns).to_csv('../Data/AutoEncoder/test/technical_test_df.csv')
    print('finish technical_test_df.csv ...')
    # pd.DataFrame(data=alpha_train_df, index=alpha_train_df.index, columns=alpha_train_df.columns).to_csv('../Data/AutoEncoder/train/alpha_train_df.csv')
    # print('finish alpha_train_df.csv ...')
    # pd.DataFrame(data=alpha_valid_df, index=alpha_valid_df.index, columns=alpha_valid_df.columns).to_csv('../Data/AutoEncoder/valid/alpha_valid_df.csv')
    # print('finish alpha_valid_df.csv ...')
    # pd.DataFrame(data=alpha_test_df, index=alpha_test_df.index, columns=alpha_test_df.columns).to_csv('../Data/AutoEncoder/test/alpha_test_df.csv')
    # print('finish alpha_test_df.csv ...')

if __name__ == '__main__':
    main()
