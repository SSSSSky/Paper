import os
import pandas as pd
import os
import pandas as pd

def seperateData(path):
    df1 = pd.read_csv(path)
    #print(df1.head())
    train_df = df1[df1.DateTime < '20150101']
    valid_df = df1[(df1.DateTime >= '20150101') & (df1.DateTime < '20170101')]
    test_df = df1[df1.DateTime >= '20170101']
    print('train_df:')
    print(train_df.head())
    print(train_df.tail())
    print('valid_df:')
    print(valid_df.head())
    print(valid_df.tail())
    print('test_df:')
    print(test_df.head())
    print(test_df.tail())
    


def main():
    # path = '../feature/join_feature'
    # files = os.listdir(path)
    # files.sort()
    # q = 0
    # for file in files:
    #     q += 1
    #     file_path = os.path.join(path, file)
    #     df1 = pd.read_csv(file_path)
    #     if q == 1:
    #         join_df = df1
    #     else:
    #         join_df = pd.concat([join_df, df1])
    #         print(join_df.shape)
    #         # join_df 就是全部的特征数据
    # join_df = join_df.sort_values(by='DateTime')
    # # 取10个技术指标
    # technical_df = join_df.loc[:, ['SecCode', 'DateTime', 'bias', 'boll', 'cci',
    #                                'ma_ma_ratio', 'price_efficiency', 'logreturn', 'rsi', 'mfi', 'ar', 'macd']]
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
    # pd.DataFrame(data=technical_df, index=technical_df.index, columns=technical_df.columns).to_csv('../feature/technical_all.csv')
    # pd.DataFrame(data=alpha_df, index=alpha_df.index, columns=alpha_df.columns).to_csv('../feature/alpha_all.csv')


    technical_path = '../feature/technical_all.csv'
    alpha_path = '../feature/alpha_all.csv'
    seperateData(technical_path)
    #seperateData(alpha_path)



if __name__ == '__main__':
    main()
