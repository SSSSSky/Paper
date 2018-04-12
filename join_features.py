import os
import pandas as pd

def sperate_time(df):
    train_df_index = df[(df.DataTime.split(' ')[0] == '20150101')].index.tolist()
    valid_df_index = df[(df.DataTime.split(' ')[0] == '20170101')].index.tolist()
    train_df = df[0:train_df_index]
    valid_df = df[train_df_index:valid_df_index]
    test_df = df[(df.DataTime.split(' ')[0] >= '20170101')]
    return train_df, valid_df, test_df

def main():
    path = '../feature/join_feature'
    files = os.listdir(path)
    files.sort()
    q = 0
    for file in files:
        file_path = os.path.join(path, file)
        df1 = pd.read_csv(file_path)
        if q == 0:
            join_df = df1
        else:
            pd.concat([join_df, df1])
            # join_df 就是全部的特征数据

    technical_df = join_df.loc[:, ['SecCode', 'DateTime', 'amp', 'ar',
                                   'atr', 'bias', 'boll', 'br', 'cci', 'log_vol_chg_rate', 'logreturn',
                                   'ma_ma_ratio', 'macd', 'mfi', 'obv', 'pri_ma_ratio', 'price_efficiency',
                                   'pvt', 'return', 'roc', 'rsi', 'vma', 'vol_chg_rate',
                                   'volume_relative_ratio']]

    alpha_df = join_df.loc[:, ['SecCode', 'DateTime', 'alpha001', 'alpha002', 'alpha003', 'alpha004',
                               'alpha006', 'alpha007', 'alpha008', 'alpha009', 'alpha010', 'alpha012',
                               'alpha013', 'alpha014', 'alpha015', 'alpha016', 'alpha017', 'alpha018',
                               'alpha019', 'alpha020', 'alpha021', 'alpha022', 'alpha023', 'alpha024',
                               'alpha026', 'alpha028', 'alpha029', 'alpha030', 'alpha031', 'alpha033',
                               'alpha034', 'alpha035', 'alpha037', 'alpha038', 'alpha039', 'alpha040',
                               'alpha043', 'alpha044', 'alpha045', 'alpha046', 'alpha049', 'alpha051',
                               'alpha052', 'alpha053', 'alpha054', 'alpha055', 'alpha060']]

    technical_train_df, technical_valid_df, technical_test_df = sperate_time(technical_df)
    alpha_train_df, alpha_valid_df, alpha_test_df = sperate_time(alpha_df)

    print(technical_train_df.iloc[0])
    print(technical_train_df.iloc[-1])
    print(technical_valid_df.iloc[0])
    print(technical_valid_df.iloc[-1])
    print(technical_test_df.iloc[0])
    print(technical_test_df.iloc[-1])

    print(alpha_train_df.iloc[0])
    print(alpha_train_df.iloc[-1])
    print(alpha_valid_df.iloc[0])
    print(alpha_train_df.iloc[-1])
    print(alpha_test_df.iloc[0])
    print(alpha_test_df.iloc[-1])

if __name__ == '__main__':
    main()
