import os
import re

import numpy as np
import pandas as pd

from data_api import get_index_stocks
from data_api import get_price


# 定义计算技术指标的类 23个指标
class technical_indicator(object):
    def __init__(self, index=None, start_date=None, end_date=None, fre='1min', fq=True, cal_days=None, security=None, price=None):
        self.cal_days = cal_days
        if security is None:
            self.security = get_index_stocks(index, end_date)  # 获取index指数对应的成分股
        else:
            self.security = security
        if price is None:
            self.price = get_price(self.security, start_date, end_date, fre=fre, fq=fq)
        else:
            self.price = price
        if index is not None:
            self.index = index
        if start_date is not None:
            self.start_date = start_date
        if end_date is not None:
            self.end_date = end_date
        # 获取历史数据
        if fre == 'd':
            self.open = self.price.loc['Opnprc', :, :]
            self.high = self.price.loc['Hiprc', :, :]
            self.low = self.price.loc['Loprc', :, :]
            self.close = self.price.loc['Clsprc', :, :]
            self.volume = self.price.loc['Dnshrtrd', :, :]
            self.returns = self.price.loc['Dretnd', :, :]
        elif fre == '1min':
            self.open = self.price.loc['StartPrc', :, :]
            self.high = self.price.loc['HighPrc', :, :]
            self.low = self.price.loc['LowPrc', :, :]
            self.close = self.price.loc['EndPrc', :, :]
            self.volume = self.price.loc['MinTq', :, :]
            self.returns = self.price.loc['MinRet', :, :]

###############################################################################################################
    def get_logreturn(self, n=1):
        '''N日对数收益率
        :param period: 计算周期
        '''
        if self.cal_days is None:
            return (np.log(self.close) - np.log(self.close.shift(n)))
        else:
            return (np.log(self.close) - np.log(self.close.shift(n))).iloc[-self.cal_days:, :]

    def get_return(self, n=1):
        '''N日绝对收益率
        :param period: 计算周期
        '''
        if self.cal_days is None:
            return (self.close / self.close.shift(n) - 1)
        else:
            return (self.close / self.close.shift(n) - 1).iloc[-self.cal_days:, :]

    def get_log_vol_chg_rate(self, n=5):
        '''N日对数成交量变化率
        :param period: 计算周期
        '''
        if self.cal_days is None:
            return np.log(self.volume) - np.log(self.volume.shift(n))
        else:
            return (np.log(self.volume) - np.log(self.volume.shift(n))).iloc[-self.cal_days:, :]

    def get_vol_chg_rate(self, n=1):
        '''N日成交量绝对变化率
        :param period: 计算周期
        '''
        if self.cal_days is None:
            return self.volume / self.volume.shift(n) - 1
        else:
            return (self.volume / self.volume.shift(n) - 1).iloc[-self.cal_days:, :]

    def get_volume_relative_ratio(self, n=5):
        '''计算成交量量比
        今日成交量/过去N日平均每日成交量
        :param period: 平均成交量计算天数
        '''
        vol_avg = self.volume.rolling(n).mean()
        if self.cal_days is None:
            return self.volume / vol_avg.shift()
        else:
            return (self.volume / vol_avg.shift()).iloc[-self.cal_days:, :]

    def get_amp(self, n=5):
        '''计算N日振幅
        :param period: n日对数收益率
        '''
        if self.cal_days is None:
            return (self.high - self.low) / self.close.shift()
        else:
            return ((self.high - self.low) / self.close.shift()).iloc[-self.cal_days:, :]

    def get_price_efficiency(self, n=5):
        '''计算N日价格轨迹效率
        :param period: 计算周期，N日价格轨迹效率
        '''
        abs_dis = self.close - self.close.shift(n)
        path_dis = (self.close - self.close.shift()).abs().rolling(n).sum()
        if self.cal_days is None:
            return abs_dis / path_dis
        else:
            return (abs_dis / path_dis).iloc[-self.cal_days:, :]

    def get_pri_ma_ratio(self, n=5):
        '''计算价格与均线的比值
        :param n: 均线计算参数
        '''
        if self.cal_days is None:
            return self.close / self.close.rolling(n).mean()
        else:
            return (self.close / self.close.rolling(n).mean()).iloc[-self.cal_days:, :]

###############################################################################################################
    def get_macd(self, fp=12, sp=26, s_p=9):
        '''MACD 指数平滑异动移动平均线
        :param close: 收盘价序列
        :param fp: 快速移动平均线周期
        :param sp: 慢速移动平均线周期
        :param s_p: 离差移动平均线周期
        '''
        ema_f = pd.ewma(self.close, span=fp, adjust=False)   # 快速移动平均线
        ema_s = pd.ewma(self.close, span=sp, adjust=False)   # 慢速移动平均线
        dif = ema_s - ema_f                             # 离差值
        dea = pd.ewma(dif, span=s_p, adjust=False)      # 离差移动平均线
        macd = (dif - dea) * 2
        if self.cal_days is None:
            # return macd.iloc[-self.cal_days:, :], dif[-self.cal_days:, :], dea[-self.cal_days:, :]
            return macd
        else:
            # return macd, dif, dea
            return macd.iloc[-self.cal_days:, :]

    def get_atr(self, n=14):
        '''ATR 平均真实波动幅度
        :param n: TR的n日简单平均
        '''
        val1 = (self.high - self.low).abs()
        val2 = (self.high - self.close.shift()).abs()
        val3 = (self.close.shift() - self.low).abs()
        TR = val1
        TR[TR < val2] = val2
        TR[TR < val3] = val3
        if self.cal_days is None:
            ATR = TR.rolling(n).mean()
        else:
            ATR = TR.rolling(n).mean().iloc[-self.cal_days:, :]
        return ATR

    def get_rsi(self, n=14):
        '''RSI 相对强弱指数
        :param n: 均线计算周期
        '''
        momentum = self.close - self.close.shift()
        U = momentum[momentum > 0]
        D = momentum[momentum < 0]
        U = U.fillna(0)
        D = D.fillna(0)
        U_sma = pd.ewma(U, com=n)
        D_sma = pd.ewma(D, com=n)
        RS = U_sma / D_sma
        if self.cal_days is None:
            RSI = 100 - 100 / (1 + RS)
        else:
            RSI = (100 - 100 / (1 + RS)).iloc[-self.cal_days:, :]
        return RSI

    def get_cci(self, n=14):
        '''CCI 顺势指标
        :param n: 移动平均计算周期
        '''
        TP = (self.high + self.low + self.close) / 3
        MA = self.close.rolling(n).mean()
        MD = (MA - self.close).abs().rolling(n).mean()
        if self.cal_days is None:
            CCI = (TP - MA) / MD / 0.015
        else:
            CCI = ((TP - MA) / MD / 0.015).iloc[-self.cal_days:, :]
        return CCI

    # OBV 能量潮
    # 变体: 返回OBV每日变化量
    # @param: mode计算模式: 0 常规OBV定义; 1 变体OBV
    def get_obv(self, mode=0):
        if self.cal_days is None:
            OBV = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low) * self.volume
        else:
            OBV = (((self.close - self.low) - (self.high - self.close)) / (self.high - self.low) * self.volume).iloc[-self.cal_days:, :]
        return OBV

    def get_roc(self, n=12):
        '''ROC 变动率指标
        :param n: 间隔天数n
        '''
        if self.cal_days is None:
            ROC = (self.close - self.close.shift(n)) / self.close.shift(n)
        else:
            ROC = ((self.close - self.close.shift(n)) / self.close.shift(n)).iloc[-self.cal_days:, :]
        return ROC

    def get_bias(self, n=6):
        '''# BIAS 乖离率
        :param n: 均值计算周期
        :min_period: n
        '''
        if self.cal_days is None:
            bias = (self.close / self.close.rolling(n).mean() - 1) * 100
        else:
            bias = ((self.close / self.close.rolling(n).mean() - 1) * 100).iloc[-self.cal_days:, :]
        return bias

    def get_vma(self, n=6):
        '''VMA 量均线
        :param n: 成交量计算周期
        :min_period: n
        '''
        if self.cal_days is None:
            return self.volume.rolling(n).mean()
        else:
            return (self.volume.rolling(n).mean()).iloc[-self.cal_days:, :]

    def get_pvt(self, mode=0):
        '''PVT 量价趋势
        :param mode: 0 原指标 1 每天变化值
        :min_period: 2
        '''
        pvt = (self.close / self.close.shift() - 1) * self.volume
        if mode == 0:
            if self.cal_days is None:
                return pvt
            else:
                return pvt.iloc[-self.cal_days:, :]
        else:
            return pvt.sum()

    def get_mfi(self, n=14):
        '''MFI 资金流量指标
        :param n:
        '''
        typical_price = (self.high + self.low + self.close) / 3
        money_flow = typical_price * self.volume
        cond1 = (typical_price - typical_price.shift()) > 0
        cond2 = (typical_price - typical_price.shift()) < 0
        positive_money_flow = money_flow[cond1]
        negative_money_flow = money_flow[cond2]
        positive_money_flow = positive_money_flow.fillna(0)
        negative_money_flow = negative_money_flow.fillna(0)
        positive_money_flow = positive_money_flow.rolling(n).sum()
        negative_money_flow = negative_money_flow.rolling(n).sum()
        money_ratio = positive_money_flow / negative_money_flow
        if self.cal_days is None:
            MFI = 100 - 100 / (1 + money_ratio)
        else:
            MFI = (100 - 100 / (1 + money_ratio)).iloc[-self.cal_days:, :]
        return MFI

    def get_ma_ma_ratio(self, n1=5, n2=20):
        '''计算均线之间的距离
        :param n1: 短期均线周期
        :param n2: 长期均线周期
        :min_period: n2
        '''
        MA1 = self.close.rolling(n1).mean()
        MA2 = self.close.rolling(n2).mean()
        if self.cal_days is None:
            return MA1 / MA2
        else:
            return (MA1/MA2).iloc[-self.cal_days:, :]

    def get_boll(self, n=20):
        '''布林带
        :param n: 移动平均计算周期
        :return: 价格偏离移动平均线多少个标准差
        :min_period: n
        '''
        MA = self.close.rolling(n).mean()
        MD = (((self.close - MA)**2).rolling(n).mean())**0.5
        if self.cal_days is None:
            return (self.close / MA - 1) / MD
        else:
            return ((self.close / MA - 1) / MD).iloc[-self.cal_days:, :]

    def get_ar(self):
        '''AR人气指标
        :min_period：26'''
        val1 = (self.high - self.open).rolling(26).sum()
        val2 = (self.open - self.low).rolling(26).sum()
        if self.cal_days is None:
            return val1 / val2 * 100
        else:
            return (val1 / val2 * 100).iloc[-self.cal_days:, :]

    def get_br(self):
        '''BR意愿指标
        :min_period：26'''
        val1 = (self.high - self.close.shift()).rolling(26).sum()
        val2 = (self.close.shift() -self.low).rolling(26).sum()
        if self.cal_days is None:
            return val1 / val2 * 100
        else:
            return (val1 / val2 * 100).iloc[-self.cal_days:, :]

    def cal_all_features(self, path):
        func_names = self.__dir__()
        pattern = r'get_'
        for fn in func_names:
            if re.match(pattern, fn) is not None:
                print('[INFO] {} calculating...'.format(fn))
                alpha = getattr(self, fn)()
                # 创建指标文件名
                if not os.path.exists('{}/{}'.format(path, fn[fn.index('_')+1: ])):
                    os.mkdir('{}/{}'.format(path, fn[fn.index('_')+1: ]))
                # 创建指标文件
                alpha.to_csv('{}/{}/{}_{}_{}_{}.csv'.format(path, fn[fn.index('_')+1: ],
                                                            self.index, fn[fn.index('_')+1: ],
                                                            self.start_date, self.end_date))
