import pandas as pd
import pymongo
from pymongo import MongoClient


# 获取数据库连接
# local: mongodb://localhost:27017/
# remote: mongodb://172.16.17.139:27017/
def get_database_conn(host_name='mongodb://172.16.17.139:27017/',
                      db_name='stock_basic', collection_name='constituent_stocks'):
    client = MongoClient(host_name)
    db = client[db_name]
    collection = db[collection_name]
    return client, db, collection

# 获取变更成分股
def get_index_stocks():
    client, _, collection = get_database_conn(db_name='stock_basic', collection_name='index_cons_change')
    query = collection.find({'Indexcd': '000016'})
    index_stocks = []
    date_chg = []
    for q in query:
        index_stocks.append(q['Stkcd'])
        date_chg.append(q['Trddt'])
    client.close()
    chg_df = pd.DataFrame({'SecCode' : index_stocks, 'DateTime' : date_chg})
    print(chg_df.shape)
    print(chg_df.head(51))
    i = chg_df.shape[0] / 50
    j = 0
    while (j < i-1):
        one_chg_df = chg_df[j:j+50]
        print(one_chg_df.shape)
        print(one_chg_df)
    # pd.DataFrame(data=chg_df, index=chg_df.index).to_csv('../Data/AutoEncoder/tl_hidden_result/train/techinical_train_hidden_result_timeline.csv')

def main():
    get_index_stocks()

if __name__ == '__main__':
    main()

