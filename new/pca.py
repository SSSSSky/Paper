
import pandas as pd
from sklearn.decomposition import PCA

# PCA处理，com是降维后的维数
def deal_with_pca(df, com):
    pca = PCA(n_components= com)
    principalCompoents = pca.fit_transform(df)
    return principalCompoents

def pca(dataset, start_date, end_date, datatype, inputs):
    print('[INFO] start pca...')
    df = pd.read_csv('{}/{}_{}_{}.csv'.format(dataset, datatype, start_date, end_date), index_col=['SecCode', 'DateTime'])
    data_df = deal_with_pca(df, inputs)
    print('pca.shape:{}'.format(data_df.shape))
    print('[INFO] save pca result...')
    pd.DataFrame(data=data_df, index=df.index, columns=df.columns).to_csv('../model1/pca_input/{}_{}_{}.csv').format(datatype, start_date, end_date)
    #return data_df

def main():
    dataset = '../model1/input'
    pca(dataset, '20100104', '20150101', 'technical')
    pca(dataset, '20150101', '20170101', 'technical')
    pca(dataset, '20170101', '20171231', 'technical')

    pca(dataset, '20100104', '20150101', 'alpha')
    pca(dataset, '20150101', '20170101', 'alpha')
    pca(dataset, '20170101', '20171231', 'alpha')
    df = pd.read_csv('../model1/pca_input/technical_20170101_20171231')
    print(df.head())

if __name__ == '__main__':
    main()