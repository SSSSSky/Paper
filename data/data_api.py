'''
Data_API
数据查询接口
'''
import datetime
import time

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

# 确定后只留一种格式

def date2date(date):
    '''日期格式转换YYYYMMDD'''
    if type(date) == float:
        pass
    elif '-' in date and len(date) == 10:
        return date[0:4]+date[5:7]+date[8:10]
    # elif '-' not in date and len(date) == 8:
    #     return date[0:4]+'-'+date[4:6]+'-'+date[6:8]

# 获取指数成分股代码
def get_index_stocks(symbol, date):
    '''
    :param symbol: stock symbol
    :param date: search date
    :return: constituent stocks of the index in the date.
    '''
    if '-' in date:
        date = date2date(date)
    client, _, collection = get_database_conn(db_name='stock_basic', collection_name='constituent_stocks')
    query = collection.find({'Indexcd': symbol, 'Enddt': date})
    index_stocks = []
    for q in query:
        index_stocks.append(q['Stkcd'])
    client.close()
    return index_stocks


# 获取股票分红数据
def get_stock_profit_sharing(collection, symbol, date=None):
    if date is None:
        query = collection.find({'Symbol': symbol})
        if query.count() == 0:
            return None
        else:
            profit_sharing_list = []
            for q in query:
                profit_sharing_list.append(q)
            return profit_sharing_list
    else:
        if '-' in date:
            date = date2date(date)
        query = collection.find({'Symbol': symbol, 'ExDividendDate': date})
        if query.count() == 0:
            return None
        else:
            print('Total {} times of Profit Sharing.'.format(query.count()))
            return query.next()

# 获取复权因子，供get_price调用
def get_fq_factor(symbol, start_unix, end_unix):
    '''
    :param symbol: 要获取复权因子的股票列表
    :param start_unix: 开始unix时间
    :param end_unix: 结束unix时间
    :return: 股票/日期复权因子dataframe
    '''
    fq_client, _, fq_col = get_database_conn(db_name='stock_basic', collection_name='fq_daily_factor')
    # 获取当前时间
    def my_date(t):
        return time.strftime('%Y%m%d', time.localtime(t / 1000))
    query = fq_col.find({'Stkcd':{'$in': symbol}, '$and': [{'Trddt': {'$gte': my_date(start_unix)}},
                                                           {'Trddt': {'$lte': my_date(end_unix)}}]},
                        projection={'_id': False})
    result = []
    for q in query:
        result.append(q)
    fq_client.close()
    result_df = pd.DataFrame(result)
    return result_df

# 获取股票指定日期范围内的日级行情数据
def get_price(symbol, start_date, end_date, fre='d', fields=None, bar_count=0, fq=False):
    '''
    :param symbol: stock symbol
    :param start_date: search start_date YYYY-MM-DD
    :param end_date: search end_date YYYY-MM-DD
    :param fre: frequency(d: day/ 1m: 1min)
    :param fields: selected fields
    :param fq: fq or not
    :return: pandas dataframe of stocks price/volume info.
    '''
    # format of date
    start_time = '0930'
    end_time = '1500'
    if start_date is None:
        pass
    elif '-' in start_date:
        if len(start_date.split(' ')) == 1:
            start_date = date2date(start_date)
        else:
            start_time = start_date.split(' ')[1]
            start_date = date2date(start_date.split(' ')[0])
    else:
        if len(start_date.split(' ')) == 1:
            pass
        else:
            start_time = start_date.split(' ')[1]
            start_date = start_date.split(' ')[0]
    if end_date is None:
        pass
    elif '-' in end_date:
        if len(end_date.split(' ')) == 1:
            end_date = date2date(end_date)
        else:
            end_time = end_date.split(' ')[1]
            end_date = date2date(end_date.split(' ')[0])
    else:
        if len(end_date.split(' ')) != 1:
            end_time = end_date.split(' ')[1]
            end_date = end_date.split(' ')[0]
    # 日级数据
    if fre == 'd':
        client, _, collection = get_database_conn(db_name='stock_high_freq', collection_name='stock_daily')
        if start_date is None:
            dt_client, _, dt_col = get_database_conn(collection_name='calendar')
            start_date = dt_col.find({'trade_date': {'$lte': end_date}}) \
                               .sort('trade_date', pymongo.DESCENDING) \
                               .skip(bar_count - 1).next()['trade_date']
        if len(symbol) == 1:
            query = collection.find({'Stkcd': symbol[0], '$and': [{'Trddt': {'$gte': start_date}},
                                                                  {'Trddt': {'$lte': end_date}}]})
        else:
            query = collection.find({'Stkcd': {'$in': symbol}, '$and': [{'Trddt': {'$gte': start_date}},
                                                                        {'Trddt': {'$lte': end_date}}]})
        result = []
        for q in query:
            result.append(q)
        client.close()
        result_df = pd.DataFrame(result)
        print(result_df.shape)
        if len(symbol) == 1:
            result_df = result_df.set_index('Trddt')
            return result_df
        else:
            result_df = result_df.set_index(['Trddt', 'Stkcd'])
            result_panel = result_df.to_panel()
            if fields is None:
                fields = ['Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'Dnshrtrd', 'Dnvaltrd', 'Dretnd']
            return result_panel.loc[fields, :, :]

    elif fre == '1min':           # 分钟级数据
        client, _, collection = get_database_conn(db_name='stock_high_freq', collection_name='stock_1min')
        if start_date is None:
            start_date, start_time = get_dis_mins(end_date+' '+end_time, bar_count)
        start_date = time.mktime(datetime.datetime.strptime(start_date + ' ' + start_time, '%Y%m%d %H%M').timetuple())*1000
        end_date = time.mktime(datetime.datetime.strptime(end_date+' '+end_time, '%Y%m%d %H%M').timetuple())*1000
        if len(symbol) == 1:
            query = collection.find({'SecCode': symbol[0], '$and': [{'UNIX': {'$gte': start_date}},
                                                                  {'UNIX': {'$lte': end_date}}]},
                                    projection={'_id': False, 'Market': False})
        else:
            query = collection.find({'SecCode': {'$in': symbol}, '$and': [{'UNIX': {'$gte': start_date}},
                                                                        {'UNIX': {'$lte': end_date}}]},
                                    projection={'_id': False, 'Market': False})
        result = []
        for q in query:
            result.append(q)
        client.close()
        result_df = pd.DataFrame(result)
        if len(symbol) == 1:
            result_df = result_df.set_index('TDate', 'MinTime')
            return result_df
        else:
            def my_datetime(t):
                return time.strftime('%Y%m%d %H%M', time.localtime(t/1000))
            result_df['DateTime'] = result_df['UNIX'].apply(my_datetime)

            # 数据去重(同股票同时间)
            result_df.drop_duplicates(['DateTime', 'SecCode'], inplace=True)

            # 复权
            if fq:
                fq_df = get_fq_factor(symbol, start_date, end_date)
                result_df = \
                    pd.merge(result_df, fq_df, how='inner', left_on=['SecCode', 'TDate'], right_on=['Stkcd', 'Trddt'])
                result_df.StartPrc = result_df.StartPrc * result_df.fq_factor_daily
                result_df.EndPrc = result_df.EndPrc * result_df.fq_factor_daily
                result_df.HighPrc = result_df.HighPrc * result_df.fq_factor_daily
                result_df.LowPrc = result_df.LowPrc * result_df.fq_factor_daily

            result_df = result_df.set_index(['DateTime', 'SecCode'])
            result_panel = result_df.to_panel()
            result_panel['MinRet'] = result_panel.EndPrc / result_panel.EndPrc.shift() - 1
            if fields is None:
                fields = ['StartPrc', 'HighPrc', 'LowPrc', 'EndPrc', 'MinTq', 'MinTm', 'MinRet']
            return result_panel.loc[fields, :, :]


# 判断指定日期或前后dis个间隔日期是否为交易日
def is_trading_date(collection, date, dis=None):
    if dis is None:
        query = collection.find({'trade_date': date})
        if query.count == 0:
            return False
        else:
            return True
    else:
        query = collection.find({'trade_date': {'$lt': date}})
        record_num = query.count()
        if record_num < dis:
            print('Date out of available range!')
        else:
            return query.limit(1).skip(record_num-dis).next()['trade_date']

# 计算end_datetime前dis_mins分钟的时间
def get_dis_mins(end_datetime, dis_mins):
    '''
    :param end_datetime: 结束日期时间
    :param dis_mins: 距离结束时间的时间间隔
    :return: 计算得到的起始时间
    '''
    end_date = end_datetime.split()[0]
    end_time = end_datetime.split()[1]
    end_date_time = \
        datetime.datetime(int(end_date[0:4]), int(end_date[4:6]), int(end_date[6:8]), int(end_time[0:2]), int(end_time[2:4]))
    unix = time.mktime(end_date_time.timetuple()) * 1000
    client, _, col = get_database_conn(db_name='stock_high_freq', collection_name='index_1min')
    query = col.find({'UNIX':{'$lte':unix}, 'SecCode':'000001'}, projection={'TDate': True, 'MinTime': True})\
        .sort([('TDate', pymongo.DESCENDING), ('MinTime', pymongo.DESCENDING)]).skip(dis_mins-1).limit(1)
    start_datetime = query.next()
    return start_datetime['TDate'], start_datetime['MinTime']

# 计算end_date前dis_dates日的交易日期
def get_dis_dates(end_date, dis_dates):
    dt_client, _, dt_col = get_database_conn(collection_name='calendar')
    dt_query = dt_col.find({'trade_date': end_date})
    if dt_query.count() == 0:
        dt_query = dt_col.find({'trade_date': {'$lte': end_date}},
                               projection={'_id': False, 'trade_date': True})\
                         .sort('trade_date', direction=pymongo.DESCENDING).skip(dis_dates-1).limit(1)
    else:
        dt_query = dt_col.find({'trade_date': {'$lte': end_date}},
                               projection={'_id': False, 'trade_date': True}) \
                         .sort('trade_date', direction=pymongo.DESCENDING).skip(dis_dates).limit(1)
    return dt_query.next()['trade_date']

# 获取指定日期范围间的交易日
def get_trading_dates(start_date, end_date):
    dt_client, _, dt_col = get_database_conn(collection_name='calendar')
    dt_query = dt_col.find({'$and':[{'trade_date': {'$gte': start_date}}, {'trade_date': {'$lte': end_date}}]},
                           projection={'_id': False, 'trade_date': True}).distinct('trade_date')
    return dt_query
