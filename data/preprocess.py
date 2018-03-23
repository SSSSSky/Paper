'''
Data Preprocess
数据预处理
fq_factor(): 计算复权因子，存入MongoDB
'''
import sys

import numpy as np
import pymongo

import data_api


def index_cons_change(index_list=None, start_date='20100101', end_date='20171231', host_name='mongodb://172.16.17.139:27017/'):
    '''计算指数成份股变更信息，存入数据库
    '''
    dt_client, _, dt_col = data_api.get_database_conn(db_name='stock_basic', collection_name='calendar')
    cons_client, _, cons_col = data_api.get_database_conn(db_name='stock_basic', collection_name='constituent_stocks')
    cons_change_client, _, cons_change_col = data_api.get_database_conn(db_name='stock_basic',
                                                                        collection_name='index_cons_change')

    # 获取当前成份股列表中所有指数
    if index_list is None:
        index_query = cons_col.find(projection={'Indexcd': True, '_id': False})
        index_list = index_query.distinct('Indexcd')

    # 记录前一天的成份股信息
    index_cons_dict = {}
    for i in index_list:
        index_cons_dict[i] = {None}

    # 查询指定日期范围
    dt_query = dt_col.find({'$and': [{'trade_date': {'$gte': start_date}},
                                     {'trade_date': {'$lte': end_date}}]},
                           projection={'trade_date': True, '_id': False}).distinct('trade_date')

    # 每个交易日遍历所有指数，对比与前一日的成份股信息
    for dt_q in dt_query:
        for index in index_list:
            cons_query = cons_col.find({'Indexcd': index, 'Enddt': dt_q},
                                       projection={'Stkcd': True, '_id':False}).distinct('Stkcd')
            # 新的成份股信息与前一日不同
            if len(set(cons_query) - index_cons_dict[index]) != 0:
                print('[WARN] CHANGE_DATE: {}, CHANGE_INDEX: {}, LENGTH: {}'.format(dt_q, index, len(set(cons_query))))
                for c in cons_query:
                    cons_change_col.insert_one({'Indexcd': index, 'Stkcd': c, 'Trddt': dt_q})
                index_cons_dict[index] = set(cons_query)
            else:
                print('[INFO] NO CHANGE: {} {}'.format(dt_q, index))

    # 关闭数据库连接
    dt_client.close()
    cons_client.close()
    cons_change_client.close()


def fq_factor(host_name='mongodb://172.16.17.139:27017/'):
    '''计算指数成份股的复权因子
    :return:
    '''
    td_client, _, trading_date_collection = data_api.get_database_conn(host_name=host_name, collection_name='calendar')
    cons_client, _, constituent_collection = data_api.get_database_conn(host_name=host_name, collection_name='constituent_stocks')
    rs_client, _, rationed_shares_collection = data_api.get_database_conn(host_name=host_name, collection_name='rationed_shares')
    sd_client, _, stock_daily_collection = data_api.get_database_conn(host_name=host_name, db_name='stock_high_freq',
                                                                      collection_name='stock_daily')
    fq_client, _, fq_factor_collection = data_api.get_database_conn(host_name=host_name, collection_name='fq_daily_factor')

    # 获取成分股列表
    stocks_list = constituent_collection.find({'$or':[{'Indexcd': '000300'}, {'Indexcd': '000016'},
                                                      {'Indexcd': '000905'}, {'Indexcd': '000906'}]},
                                              projection={'_id': False, 'Stkcd': True}).distinct('Stkcd')
    cons_client.close()

    # 获取交易日列表
    dates_list = trading_date_collection.find({'trade_date': {'$gte': '20100101'}},
                                              projection={'_id': False, 'trade_date': True})\
                                        .sort('trade_date').distinct('trade_date')
    td_client.close()

    for s in stocks_list:
        print('[INFO] {}'.format(s))
        fq_fac = 1.0    # 每只股票初始化复权因子为1.0
        for d in range(0, len(dates_list)):
            # fq_fac = 1.0
            # 获取除权除息日配股信息
            date = dates_list[d]
            rs_query = rationed_shares_collection.find({'Stkcd': s, 'Exdistdt': date},
                                                       projection={'Disttyp': True, 'Amount': True,
                                                                   'Roprc': True})
            if rs_query.count() != 0:
                # 获取股权登记日收盘价
                prev_date = dates_list[d-1]
                sd_price = stock_daily_collection.find({'Stkcd': s, 'Trddt': {'$lte': prev_date}},
                                                       projection={'_id': False, 'Trddt': True, 'Clsprc': True})\
                                                       .sort('Trddt', pymongo.DESCENDING).next()
                prev_close_price = sd_price['Clsprc']
                print('[INFO] prev_close: {} {}'.format(sd_price['Trddt'], prev_close_price))
                # 获取登记日价格信息
                rationed_shares_ratio = 0   # 配股比例
                rationed_price = 0          # 配股价格
                dividend_stocks_ratio = 0   # 送股比例
                dividend_currency = 0       # 分红
                for q in rs_query:
                    if q['Disttyp'] == 'CA':
                        dividend_currency = q['Amount']
                        print('[INFO] CA: {}'.format(dividend_currency))
                    elif q['Disttyp'] == 'RO':
                        rationed_shares_ratio = q['Amount']
                        rationed_price = q['Roprc']
                        print('[INFO] RO: {}'.format(rationed_shares_ratio))
                    elif q['Disttyp'] == 'SD':
                        dividend_stocks_ratio = q['Amount']
                        print('[INFO] SD: {}'.format(dividend_stocks_ratio))
                    elif q['Disttyp'] == 'DS' or q['Disttyp'] == 'GQ' or q['Disttyp'] == 'SN':
                        print('[WARNING] Stock: {} Date: {} Disttyp: {}'.format(s, dates_list[d], q['Disttyp']))
                cur_fac = prev_close_price * (1+rationed_shares_ratio+dividend_stocks_ratio)/\
                         (prev_close_price-dividend_currency+rationed_price*rationed_shares_ratio)
                fq_fac *= cur_fac
            fq_factor_collection.insert_one({'Stkcd': s, 'Trddt': dates_list[d], 'fq_factor_daily': fq_fac})
            print('[INFO] Stock: {} Date: {} fq_factor: {}'.format(s, date, fq_fac))
    rs_client.close()
    sd_client.close()
    fq_client.close()


def fill_suspended(host_name='mongodb://localhost:27017/'):
    '''
    填充停牌导致的日行情数据缺失
    :return:
    '''
    td_client, _, trading_date_collection = data_api.get_database_conn(host_name=host_name, collection_name='calendar')
    cons_client, _, constituent_collection = data_api.get_database_conn(host_name=host_name, collection_name='constituent_stocks')
    sd_client, _, stock_daily_collection = data_api.get_database_conn(host_name=host_name, db_name='stock_high_freq',
                                                                      collection_name='stock_daily')
    # 获取成分股列表
    stocks_list = constituent_collection.find({'$or': [{'Indexcd': '000300'}, {'Indexcd': '000016'},
                                                       {'Indexcd': '000905'}, {'Indexcd': '000906'}]},
                                              projection={'_id': False, 'Stkcd': True}).distinct('Stkcd')
    cons_client.close()
    # 获取交易日列表
    dates_list = trading_date_collection.find({'trade_date': {'gte': '20100101'}} ,
                                              projection={'_id': False, 'trade_date': True}).distinct('trade_date')
    td_client.close()
    # 填充停牌数据
    for s in stocks_list:
        is_suspended = False    # 前一日是否停牌
        fill = None             # 填充数据
        for d in range(0, len(dates_list)):
            query = stock_daily_collection.find({'Stkcd': s, 'Trddt': dates_list[d]})
            if query.count() == 0:
                if is_suspended and fill is not None:    # 前一日停牌，直接使用之前的填充值进行填充
                    fill['Trddt'] = dates_list[d]
                    stock_daily_collection.insert_one(fill)
                else:               # 前一日未停牌，查询填充值
                    fill = stock_daily_collection.find({'Stkcd': s, 'Trddt':{'$lt': dates_list[d]}})\
                                                 .sort('Trddt', pymongo.DESCENDING).limit(1).next()
                    fill['Opnprc'] = fill['Hiprc'] = fill['Loprc'] = fill['Clsprc'] = \
                        fill['Dnshrtrd'] = fill['Dnvaltrd'] = fill['Dretwd'] = fill['Dretnd'] = np.nan
    sd_client.close()

# def fill_min_suspended(host_name='mongodb://172.16.17.139:27017/'):
#     time_client, _, time_col = data_api.get_database_conn(db_name='stock_high_freq', collection_name='index_1min')
#     time_col.find({'SecCode': '000001'}, projection={'_id': False, 'TDate': True, 'MinTime': True})
#     for q in query:

def main(argv):
    print(argv)
    if argv[1] == 'fq':
        if len(argv) == 3:
            fq_factor(host_name=argv[2])
        else:
            fq_factor()
    if argv[1] == 'fill-na':
        if len(argv) == 3:
            fill_suspended(host_name=argv[2])
        else:
            fill_suspended()
    if argv[1] == 'index_cons':
        index_cons_change()

if __name__ == '__main__':
    main(sys.argv)