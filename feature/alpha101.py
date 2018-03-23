'''
World Quant Alpha 101
世坤投资Alpha 101因子库
'''
import os
import re

import numpy as np
import pandas as pd
from numpy import abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata

from data_api import get_index_stocks
from data_api import get_price


# 定义计算alpha值的类
class wq_101(object):
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
    #######################################################################################################################################
    # 计算alpha时会使用的函数
    def ts_sum(self, df, window=10):
        return df.rolling(window).sum()

    def sma(self, df, window=10):
        return df.rolling(window).mean()

    def stddev(self, df, window=10):
        return df.rolling(window).std()

    def correlation(self, x, y, window=10):
        return x.rolling(window).corr(y)

    def covariance(self, x, y, window=10):
        return x.rolling(window).cov(y)

    def rolling_rank(self, na):
        return rankdata(na)[-1]

    def ts_rank(self, df, window=10):
        return df.rolling(window).apply(self.rolling_rank)

    def rolling_prod(self, na):
        return na.prod(na)

    def product(self, df, window=10):
        return df.rolling(window).apply(self.rolling_prod)

    def ts_min(self, df, window=10):
        return df.rolling(window).min()

    def ts_max(self, df, window=10):
        return df.rolling(window).max()

    def delta(self, df, period=1):
        return df.diff(period)

    def delay(self, df, period=1):
        return df.shift(period)

    def rank(self, df):
        return df.rank(axis=1, pct=True)

    def scale(self, df, k=1):
        return df.mul(k).div(np.abs(df).sum())

    def ts_argmax(self, df, window=10):
        return df.rolling(window).apply(np.argmax) + 1

    def ts_argmin(self, df, window=10):
        return df.rolling(window).apply(np.argmin) + 1

    def decay_linear(self, df, period=10):
        if df.isnull().values.any():
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)  # 有改动
            df.fillna(value=0, inplace=True)
        na_lwma = np.zeros_like(df)
        na_lwma[:period, :] = df.ix[:period, :]
        na_series = df.as_matrix()
        divisor = df.as_matrix()
        y = (np.arange(period) + 1) / ((np.arange(period) + 1).sum())
        for row in range(period + 1, df.shape[0]):
            x = na_series[row - period + 1:row + 1, :]
            na_lwma[row, :] = (np.dot(x.T, y))  # 有改动
        return pd.DataFrame(na_lwma, index=df.index, columns=df.columns)

    #######################################################################################################################################

    #   每个因子的计算公式：
    #   alpha001:(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5)
    def alpha001(self):
        inner = self.close.copy()
        inner[self.returns < 0] = self.stddev(self.returns, 20)
        if self.cal_days is None:
            alpha = (self.rank(self.ts_argmax(inner ** 2, 5)))
        else:
            alpha = (self.rank(self.ts_argmax(inner ** 2, 5)).iloc[-self.cal_days:, :])
        return alpha

    #  alpha002:(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    def alpha002(self):
        df = -1 * self.correlation(self.rank(self.delta(log(self.volume), 2)),
                                   self.rank((self.close - self.open) / self.open), 6)
        if self.cal_days is None:
            alpha = ((df.replace([-np.inf, np.inf], 0).fillna(value=0)))
        else:
            alpha = ((df.replace([-np.inf, np.inf], 0).fillna(value=0)).iloc[-self.cal_days:, :])
        return alpha

    # alpha003:(-1 * correlation(rank(open), rank(volume), 10))
    def alpha003(self):
        df = -1 * self.correlation(self.rank(self.open), self.rank(self.volume), 10)
        if self.cal_days is None:
            alpha = ((df.replace([-np.inf, np.inf], 0).fillna(value=0)))
        else:
            alpha = ((df.replace([-np.inf, np.inf], 0).fillna(value=0)).iloc[-self.cal_days:, :])
        return alpha

    # alpha004: (-1 * Ts_Rank(rank(low), 9))
    def alpha004(self):
        if self.cal_days is None:
            alpha = ((-1 * self.ts_rank(self.rank(self.low), 9)))
        else:
            alpha = ((-1 * self.ts_rank(self.rank(self.low), 9)).iloc[-self.cal_days:, :])
        return alpha

    #  alpha006: (-1 * correlation(open, volume, 10))
    def alpha006(self):
        df = -1 * self.correlation(self.open, self.volume, 10)
        if self.cal_days is None:
            alpha = (df.replace([-np.inf, np.inf], 0).fillna(value=0))
        else:
            alpha = (df.replace([-np.inf, np.inf], 0).fillna(value=0).iloc[-self.cal_days:, :])
        return alpha

    # alpha007: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))
    def alpha007(self):
        adv20 = self.sma(self.volume, 20)
        alpha = -1 * self.ts_rank(abs(self.delta(self.close, 7)), 60) * sign(self.delta(self.close, 7))
        alpha[adv20 >= self.volume] = -1
        if self.cal_days is not None:
            alpha = (alpha.iloc[-self.cal_days:, :])
        return alpha

    # alpha008: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10))))
    def alpha008(self):
        if self.cal_days is None:
            alpha = ((-1 * (self.rank(((self.ts_sum(self.open, 5) * self.ts_sum(self.returns, 5)) -
                                       self.delay((self.ts_sum(self.open, 5) * self.ts_sum(self.returns, 5)), 10))))))

        else:
            alpha = ((-1 * (self.rank(((self.ts_sum(self.open, 5) * self.ts_sum(self.returns, 5)) -
                                       self.delay((self.ts_sum(self.open, 5) * self.ts_sum(self.returns, 5)),
                                                  10))))).iloc[-self.cal_days:, :])
        return alpha

    # alpha009:((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?delta(close, 1) : (-1 * delta(close, 1))))
    def alpha009(self):
        delta_close = self.delta(self.close, 1)
        cond_1 = self.ts_min(delta_close, 5) > 0
        cond_2 = self.ts_max(delta_close, 5) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        if self.cal_days is not None:
            alpha = (alpha.iloc[-self.cal_days:, :])
        return alpha

    # alpha010: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0)? delta(close, 1) : (-1 * delta(close, 1)))))
    def alpha010(self):
        delta_close = self.delta(self.close, 1)
        cond_1 = self.ts_min(delta_close, 4) > 0
        cond_2 = self.ts_max(delta_close, 4) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        if self.cal_days is not None:
            alpha = (alpha.iloc[-self.cal_days:, :])
        return alpha

    #  alpha012:(sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    def alpha012(self):
        if self.cal_days is None:
            alpha = ((sign(self.delta(self.volume, 1)) * (-1 * self.delta(self.close, 1))))
        else:
            alpha = ((sign(self.delta(self.volume, 1)) * (-1 * self.delta(self.close, 1))).iloc[-self.cal_days:, :])
        return alpha

    # alpha013:(-1 * rank(covariance(rank(close), rank(volume), 5)))
    def alpha013(self):
        if self.cal_days is None:
            alpha = ((-1 * self.rank(self.covariance(self.rank(self.close), self.rank(self.volume), 5))))
        else:
            alpha = ((-1 * self.rank(self.covariance(self.rank(self.close), self.rank(self.volume), 5))).iloc[-self.cal_days:, :])
        return alpha

    #  alpha014:((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    def alpha014(self):
        df = self.correlation(self.open, self.volume, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        if self.cal_days is None:
            alpha = ((-1 * self.rank(self.delta(self.returns, 3)) * df))
        else:
            alpha = ((-1 * self.rank(self.delta(self.returns, 3)) * df).iloc[-self.cal_days:, :])
        return alpha

    # alpha015:(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    def alpha015(self):
        df = self.correlation(self.rank(self.high), self.rank(self.volume), 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        if self.cal_days is None:
            alpha = ((-1 * self.ts_sum(self.rank(df), 3)))
        else:
            alpha = ((-1 * self.ts_sum(self.rank(df), 3)).iloc[-self.cal_days:, :])
        return alpha

    #  alpha016:(-1 * rank(covariance(rank(high), rank(volume), 5)))
    def alpha016(self):
        if self.cal_days is None:
            alpha = ((-1 * self.rank(self.covariance(self.rank(self.high), self.rank(self.volume), 5))))
        else:
            alpha = ((-1 * self.rank(self.covariance(self.rank(self.high), self.rank(self.volume), 5))).iloc[-self.cal_days:, :])
        return alpha

    # alpha017: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))
    def alpha017(self):
        adv20 = self.sma(self.volume, 20)
        if self.cal_days is None:
            alpha = ((-1 * (self.rank(self.ts_rank(self.close, 10)) *
                            self.rank(self.delta(self.delta(self.close, 1), 1)) *
                            self.rank(self.ts_rank((self.volume / adv20), 5)))))
        else:
            alpha = ((-1 * (self.rank(self.ts_rank(self.close, 10)) *
                            self.rank(self.delta(self.delta(self.close, 1), 1)) *
                            self.rank(self.ts_rank((self.volume / adv20), 5)))).iloc[-self.cal_days:, :])
        return alpha

    # alpha018: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))
    def alpha018(self):
        df = self.correlation(self.close, self.open, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        if self.cal_days is None:
            alpha = ((-1 * (self.rank((self.stddev(abs((self.close - self.open)), 5) + (self.close - self.open)) +
                                      df))))
        else:
            alpha = ((-1 * (self.rank((self.stddev(abs((self.close - self.open)), 5) + (self.close - self.open)) +
                                      df))).iloc[-self.cal_days:, :])
        return alpha

    #  alpha019:((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))
    def alpha019(self):
        if self.cal_days is None:
            alpha = ((((-1 * sign((self.close - self.delay(self.close, 7)) + self.delta(self.close, 7))) *
                       (1 + self.rank(1 + self.ts_sum(self.returns, 250))))))
        else:
            alpha = ((((-1 * sign((self.close - self.delay(self.close, 7)) + self.delta(self.close, 7))) *
                       (1 + self.rank(1 + self.ts_sum(self.returns, 250))))).iloc[-self.cal_days:, :])
        return alpha

    # alpha020: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -delay(low, 1))))
    def alpha020(self):
        if self.cal_days is None:
            alpha = ((-1 * (self.rank(self.open - self.delay(self.high, 1)) *
                            self.rank(self.open - self.delay(self.close, 1)) *
                            self.rank(self.open - self.delay(self.low, 1)))))
        else:
            alpha = ((-1 * (self.rank(self.open - self.delay(self.high, 1)) *
                            self.rank(self.open - self.delay(self.close, 1)) *
                            self.rank(self.open - self.delay(self.low, 1)))).iloc[-self.cal_days:, :])
        return alpha

    # alpha012: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1))))
    def alpha021(self):
        cond_1 = self.sma(self.close, 8) + self.stddev(self.close, 8) < self.sma(self.close, 2)
        cond_2 = self.sma(self.volume, 20) / self.volume < 1
        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index,
                             columns=self.close.columns)
        alpha[cond_1 | cond_2] = -1
        if self.cal_days is not None:
            alpha = (alpha.iloc[-self.cal_days:, :])
        return alpha

    # alpha022:(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
    def alpha022(self):
        df = self.correlation(self.high, self.volume, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        if self.cal_days is None:
            alpha = ((-1 * self.delta(df, 5) * self.rank(self.stddev(self.close, 20))))
        else:
            alpha = ((-1 * self.delta(df, 5) * self.rank(self.stddev(self.close, 20))).iloc[-self.cal_days:, :])
        return alpha

    # alpha023: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    def alpha023(self):
        cond = self.sma(self.high, 20) < self.high
        alpha = pd.DataFrame(np.zeros_like(self.close), index=self.close.index,
                             columns=self.close.columns)
        alpha[cond] = -1 * self.delta(self.high, 2)
        if self.cal_days is not None:
            alpha = (alpha.iloc[-self.cal_days:, :])
        return alpha

    # alpha024: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close,100))) : (-1 * delta(close, 3)))
    def alpha024(self):
        cond = self.delta(self.sma(self.close, 100), 100) / self.delay(self.close, 100) <= 0.05
        alpha = -1 * self.delta(self.close, 3)
        alpha[cond] = -1 * (self.close - self.ts_min(self.close, 100))
        if self.cal_days is not None:
            alpha = (alpha.iloc[-self.cal_days:, :])
        return alpha

    #   alpha026:(-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    def alpha026(self):
        df = self.correlation(self.ts_rank(self.volume, 5), self.ts_rank(self.high, 5), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        if self.cal_days is None:
            alpha = ((-1 * self.ts_max(df, 3)))
        else:
            alpha = ((-1 * self.ts_max(df, 3)).iloc[-self.cal_days:, :])
        return alpha

    # alpha028:scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    def alpha028(self):
        adv20 = self.sma(self.volume, 20)
        df = self.correlation(adv20, self.low, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        if self.cal_days is None:
            alpha = ((self.scale(((df + ((self.high + self.low) / 2)) - self.close))))
        else:
            alpha = ((self.scale(((df + ((self.high + self.low) / 2)) - self.close))).iloc[-self.cal_days:, :])
        return alpha

    # alpha029:(min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
    def alpha029(self):
        if self.cal_days is None:
            alpha = (((self.ts_min(self.rank(self.rank(
                self.scale(
                    log(self.ts_sum(self.rank(self.rank(-1 * self.rank(self.delta((self.close - 1), 5)))), 2))))),5) +
                       self.ts_rank(self.delay((-1 * self.returns), 6), 5))))
        else:
            alpha = (((self.ts_min(self.rank(self.rank(
                self.scale(
                    log(self.ts_sum(self.rank(self.rank(-1 * self.rank(self.delta((self.close - 1), 5)))), 2))))),5) +
                       self.ts_rank(self.delay((-1 * self.returns), 6), 5))).iloc[-self.cal_days:, :])
        return alpha

    # alpha030:(((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    def alpha030(self):
        delta_close = self.delta(self.close, 1)
        inner = sign(delta_close) + sign(self.delay(delta_close, 1)) + sign(self.delay(delta_close, 2))
        if self.cal_days is None:
            alpha = ((((1.0 - self.rank(inner)) * self.ts_sum(self.volume, 5)) / self.ts_sum(self.volume, 20)))
        else:
            alpha = ((((1.0 - self.rank(inner)) * self.ts_sum(self.volume, 5)) / self.ts_sum(self.volume, 20)).iloc[-self.cal_days:, :])
        return alpha

    # alpha031:((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 *delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    def alpha031(self):
        adv20 = self.sma(self.volume, 20)
        df = self.correlation(adv20, self.low, 12)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        if self.cal_days is None:
            alpha = ((((self.rank(
                self.rank(self.rank(self.decay_linear((-1 * self.rank(self.rank(self.delta(self.close, 10)))), 10)))) +
                        self.rank((-1 * self.delta(self.close, 3)))) + sign(self.scale(df)))))
        else:
            alpha = ((((self.rank(
                self.rank(self.rank(self.decay_linear((-1 * self.rank(self.rank(self.delta(self.close, 10)))), 10)))) +
                        self.rank((-1 * self.delta(self.close, 3)))) + sign(self.scale(df)))).iloc[-self.cal_days:, :])
        return alpha

    # alpha033: rank((-1 * ((1 - (open / close))^1)))
    def alpha033(self):
        if self.cal_days is None:
            alpha = (self.rank(-1 + (self.open / self.close)))
        else:
            alpha = (self.rank(-1 + (self.open / self.close)).iloc[-self.cal_days:, :])
        return alpha

    # alpha034: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    def alpha034(self):
        inner = self.stddev(self.returns, 2) / self.stddev(self.returns, 5)
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        if self.cal_days is None:
            alpha = ((self.rank(2 - self.rank(inner) - self.rank(self.delta(self.close, 1)))))
        else:
            alpha = ((self.rank(2 - self.rank(inner) - self.rank(self.delta(self.close, 1)))).iloc[-self.cal_days:, :])
        return alpha

    # alpha035:((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))
    def alpha035(self):
        if self.cal_days is None:
            alpha = ((((self.ts_rank(self.volume, 32) *
                        (1 - self.ts_rank(self.close + self.high - self.low, 16))) *
                       (1 - self.ts_rank(self.returns, 32)))))
        else:
            alpha = ((((self.ts_rank(self.volume, 32) *
                        (1 - self.ts_rank(self.close + self.high - self.low, 16))) *
                       (1 - self.ts_rank(self.returns, 32)))).iloc[-self.cal_days:, :])
        return alpha

    # alpha037:(rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    def alpha037(self):
        if self.cal_days is None:
            alpha = ((self.rank(self.correlation(self.delay(self.open - self.close, 1), self.close, 200)) + self.rank(
                self.open - self.close)))
        else:
            alpha = ((self.rank(self.correlation(self.delay(self.open - self.close, 1), self.close, 200)) + self.rank(
                self.open - self.close)).iloc[-self.cal_days:, :])
        return alpha

    # alpha038: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    def alpha038(self):
        inner = self.close / self.open
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        if self.cal_days is None:
            alpha = ((-1 * self.rank(self.ts_rank(self.open, 10)) * self.rank(inner)))
        else:
            alpha = ((-1 * self.rank(self.ts_rank(self.open, 10)) * self.rank(inner)).iloc[-self.cal_days:, :])
        return alpha

    # alpha039:((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))
    def alpha039(self):
        adv20 = self.sma(self.volume, 20)
        if self.cal_days is None:
            alpha = ((((-1 * self.rank(
                self.delta(self.close, 7) * (1 - self.rank(self.decay_linear(self.volume / adv20, 9))))) * (
                               1 + self.rank(self.ts_sum(self.returns, 250))))))
        else:
            alpha = ((((-1 * self.rank(
                self.delta(self.close, 7) * (1 - self.rank(self.decay_linear(self.volume / adv20, 9))))) * (
                               1 + self.rank(self.ts_sum(self.returns, 250))))).iloc[-self.cal_days:, :])
        return alpha

    # alpha040: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    def alpha040(self):
        if self.cal_days is None:
            alpha = ((-1 * self.rank(self.stddev(self.high, 10)) * self.correlation(self.high, self.volume, 10)))
        else:
            alpha = ((-1 * self.rank(self.stddev(self.high, 10)) * self.correlation(self.high, self.volume, 10)).iloc[-self.cal_days:,
                     :])
        return alpha

    # alpha43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    def alpha043(self):
        adv20 = self.sma(self.volume, 20)
        if self.cal_days is None:
            alpha = ((self.ts_rank(self.volume / adv20, 20) * self.ts_rank((-1 * self.delta(self.close, 7)), 8)))
        else:
            alpha = ((self.ts_rank(self.volume / adv20, 20) * self.ts_rank((-1 * self.delta(self.close, 7)), 8)).iloc[-self.cal_days:,
                     :])
        return alpha

    # alpha04: (-1 * correlation(high, rank(volume), 5))
    def alpha044(self):
        df = self.correlation(self.high, self.rank(self.volume), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        if self.cal_days is None:
            alpha = ((-1 * df))
        else:
            alpha = ((-1 * df).iloc[-self.cal_days:, :])
        return alpha

    # alpha045: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *rank(correlation(sum(close, 5), sum(close, 20), 2))))

    def alpha045(self):
        df = self.correlation(self.close, self.volume, 2)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        if self.cal_days is None:
            alpha = ((-1 * (self.rank(self.sma(self.delay(self.close, 5), 20)) * df *
                            self.rank(
                                self.correlation(self.ts_sum(self.close, 5), self.ts_sum(self.close, 20), 2)))))
        else:
            alpha = ((-1 * (self.rank(self.sma(self.delay(self.close, 5), 20)) * df *
                            self.rank(self.correlation(self.ts_sum(self.close, 5), self.ts_sum(self.close, 20), 2)))).iloc[
                     -self.cal_days:, :])
        return alpha

    # alpha046: ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?(-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) * (close - delay(close, 1)))))

    def alpha046(self):
        inner = ((self.delay(self.close, 20) - self.delay(self.close, 10)) / 10) - (
                (self.delay(self.close, 10) - self.close) / 10)
        alpha = (-1 * self.delta(self.close))
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        if self.cal_days is not None:
            alpha = (alpha.iloc[-self.cal_days:, :])
        return alpha

    # alpha049:(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))

    def alpha049(self):
        inner = (((self.delay(self.close, 20) - self.delay(self.close, 10)) / 10) - (
                (self.delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * self.delta(self.close))
        alpha[inner < -0.1] = 1
        if self.cal_days is not None:
            alpha = (alpha.iloc[-self.cal_days:, :])
        return alpha

    # alpha051:(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))

    def alpha051(self):
        inner = (((self.delay(self.close, 20) - self.delay(self.close, 10)) / 10) - (
                (self.delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * self.delta(self.close))
        alpha[inner < -0.05] = 1
        if self.cal_days is not None:
            alpha = (alpha.iloc[-self.cal_days:, :])
        return alpha

    # alpha052: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume, 5))

    def alpha052(self):
        if self.cal_days is None:
            alpha = (((((-1 * self.delta(self.ts_min(self.low, 5), 5)) *
                        self.rank(
                            ((self.ts_sum(self.returns, 240) - self.ts_sum(self.returns, 20)) / 220))) * self.ts_rank(
                self.volume, 5))))
        else:
            alpha = (((((-1 * self.delta(self.ts_min(self.low, 5), 5)) *
                        self.rank(((self.ts_sum(self.returns, 240) - self.ts_sum(self.returns, 20)) / 220))) * self.ts_rank(
                self.volume, 5))).iloc[-self.cal_days:, :])
        return alpha

    # alpha053:(-1 * delta((((close - low) - (high - close)) / (close - low)), 9))

    def alpha053(self):
        inner = (self.close - self.low).replace(0, 0.0001)
        if self.cal_days is None:
            alpha = ((-1 * self.delta((((self.close - self.low) - (self.high - self.close)) / inner), 9)))
        else:
            alpha = ((-1 * self.delta((((self.close - self.low) - (self.high - self.close)) / inner), 9)).iloc[-self.cal_days:, :])
        return alpha

    # alpha054:((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))

    def alpha054(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        if self.cal_days is None:
            alpha = ((-1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))))
        else:
            alpha = ((-1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))).iloc[-self.cal_days:, :])
        return alpha

    # alpha055: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))

    def alpha055(self):
        divisor = (self.ts_max(self.high, 12) - self.ts_min(self.low, 12)).replace(0, 0.0001)
        inner = (self.close - self.ts_min(self.low, 12)) / (divisor)
        df = self.correlation(self.rank(inner), self.rank(self.volume), 6)
        if self.cal_days is None:
            alpha = ((-1 * df.replace([-np.inf, np.inf], 0).fillna(value=0)))
        else:
            alpha = ((-1 * df.replace([-np.inf, np.inf], 0).fillna(value=0)).iloc[-self.cal_days:, :])
        return alpha

    # alpha060: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))))

    def alpha060(self):
        divisor = (self.high - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        if self.cal_days is None:
            alpha = ((-((2 * self.scale(self.rank(inner))) - self.scale(self.rank(self.ts_argmax(self.close, 10))))))
        else:
            alpha = (
                (-((2 * self.scale(self.rank(inner))) - self.scale(self.rank(self.ts_argmax(self.close, 10))))).iloc[
                -self.cal_days:, :])
        return alpha

    def get_all_features(self, path):
        func_names = self.__dir__()
        pattern = r'alpha[0-9]{3}'
        for fn in func_names:
            if re.match(pattern, fn) is not None:
                print('[INFO] {} calculating...'.format(fn))
                alpha = getattr(self, fn)()
                if not os.path.exists('{}/{}'.format(path, fn)):
                    os.mkdir('{}/{}'.format(path, fn))
                alpha.to_csv('{}/{}/{}_{}_{}_{}.csv'.format(path, fn, self.index, fn, self.start_date, self.end_date))
