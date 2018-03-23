'''gen_feature
特征生成并存储到本地
'''

from alpha101 import *
from data_api import *
from technical import *


def main():
    index_code = '000016'
    # dt_client, _, dt_col = get_database_conn(db_name='stock_basic', collection_name='calendar')
    index_cons_client, _, index_cons_col = get_database_conn(db_name='stock_basic', collection_name='index_cons_change')
    # 查询指数所有成份股变更日期
    ind_cons_chg_dt_query = index_cons_col.find({'Indexcd': index_code},
                                                projection={'_id': False, 'Trddt': True}).distinct('Trddt')

    # 获取变更日期之间时间段的分钟级数据
    for i in range(0, len(ind_cons_chg_dt_query)):
        start_date = ind_cons_chg_dt_query[i]
        if i+1 < len(ind_cons_chg_dt_query):
            end_date = ind_cons_chg_dt_query[i+1]
        else:
            end_date = '20171231'
        print('[INFO] from {} to {}'.format(start_date, end_date))
        # 将起始日期向前多取一天，保证计算特征时数据足够
        if i > 0:
            start_date = get_dis_dates(start_date, 1)
        end_date = get_dis_dates(end_date, 1)
        tic = time.time()
        print('[INFO] initializing wq_101 object...')
        alpha = wq_101(index_code, start_date, end_date, fre='1min', fq=True)
        toc = time.time()
        print('[INFO] takes {} seconds to initialize wq_101 object.'.format(toc - tic))
        print('[INFO] wq_101 calculating features...')
        alpha.get_all_features('features', '../feature/features')
        tic = time.time()
        print('[INFO] initializing technical_indicator object...')
        ti = technical_indicator(index=alpha.index, start_date=start_date, end_date=end_date,
                                 security=alpha.security, price=alpha.price)
        toc = time.time()
        print('[INFO] takes {} seconds to initialize technical_indicator object.'.format(toc - tic))
        print('[INFO] technical_indicator calculating features...')
        ti.cal_all_features('features', '../feature/features')



if __name__ == '__main__':
    main()
