import os
import pandas as pd

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

    # 取10个技术指标
    technical_df = join_df.loc[:, ['SecCode', 'DateTime', 'bias', 'boll', 'cci',
                                   'ma_ma_ratio', 'price_efficiency', 'logreturn', 'rsi', 'mfi', 'ar', 'macd']]

    # 取45个alpha指标
    alpha_df = join_df.loc[:, ['SecCode', 'DateTime', 'alpha001', 'alpha002', 'alpha003', 'alpha004',
                               'alpha006', 'alpha007', 'alpha008', 'alpha009', 'alpha010', 'alpha012',
                               'alpha013', 'alpha014', 'alpha015', 'alpha016', 'alpha017', 'alpha018',
                               'alpha019', 'alpha020', 'alpha021', 'alpha022', 'alpha023', 'alpha024',
                               'alpha026', 'alpha028', 'alpha029', 'alpha030', 'alpha031', 'alpha033',
                               'alpha034', 'alpha035', 'alpha037', 'alpha038', 'alpha039', 'alpha040',
                               'alpha043', 'alpha044', 'alpha045', 'alpha046', 'alpha049', 'alpha051',
                               'alpha052', 'alpha053', 'alpha054', 'alpha055', 'alpha060']]

    print(technical_df.shape)
    print(alpha_df.shape)

    pd.DataFrame(data=technical_df, index=technical_df.index, columns=technical_df.columns).to_csv('../feature/technical_all.csv')
    pd.DataFrame(data=alpha_df, index=alpha_df.index, columns=alpha_df.columns).to_csv('../feature/alpha_all.csv')


if __name__ == '__main__':
    main()
