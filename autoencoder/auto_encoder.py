'''
AutoEncoder 自编码器
'''
import tensorflow as tf
import os.path
import pandas as pd


class Autoencoder(object):

    # tf.nn.softplus(features, name=None)	计算softplus：log(exp(features) + 1)
    # tf.train.AdamOptimizer() 实现Adam算法的优化器
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        # matmul 矩阵相乘
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost
        # pow 幂次方
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden],
            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

def Concat(i):
    rootdir = '../feature/features'
    j = 0
    q = 0

    for filelist in os.listdir(rootdir):
        path = os.path.join(rootdir, filelist)
        for filename in os.listdir(path):
            if j == i:
                path2 = os.path.join(path, filename)
                df1 = pd.read_csv(path2, index_col="DataTime")
                df1 = df1.unstack(-1).reset_index()
                df1.columns = ['SecCode', 'DateTime', filelist]
                if q == 0:
                    df2 = df1
                else:
                    df2 = pd.merge(df1, df2, on=['SecCode', 'DateTime'], how = 'inner')
                    df2.dropna(axis=0, inplace=True)
                j = 0
            else:
                break
            j += 1
        q += 1

    return df2

def main():

    # df1 = pd.read_csv('../feature/features/return/000016_return_20100104_20100630.csv', index_col='DateTime')
    # df2 = pd.read_csv('../feature/features/atr/000016_atr_20100104_20100630.csv', index_col='DateTime')
    # df3 = pd.read_csv('../feature/features/macd/000016_macd_20100104_20100630.csv', index_col='DateTime')
    # df1 = df1.unstack(-1).reset_index()
    # df1.columns = ['SecCode', 'DateTime', 'return']
    # print(df1.shape[0])
    # df2 = df2.unstack(-1).reset_index()
    # df2.columns = ['SecCode', 'DateTime', 'ATR']
    # print(df2.shape[0])
    # df3 = df3.unstack(-1).reset_index()
    # df3.columns = ['SecCode', 'DateTime', 'MACD']
    # join_df = pd.merge(df1, df2, on=['SecCode', 'DateTime'], how='inner')
    # join_df = pd.merge(join_df, df3, on=['SecCode', 'DateTime'], how='inner')
    # print(join_df.shape[0])
    # join_df.dropna(axis=0, inplace=True)
    # print(join_df.shape[0])
    # print(join_df.columns)
    # df1 = pd.read_csv('../feature/features/return/000016_return_20100104_20100630.csv', index_col='DateTime')
    # df2 = pd.read_csv('../feature/features/atr/000016_atr_20100104_20100630.csv', index_col='DateTime')
    # df3 = pd.read_csv('../feature/features/macd/000016_macd_20100104_20100630.csv', index_col='DateTime')
    # df4 = pd.read_csv('../feature/features/logreturn/000016_logreturn_20100104_20100630.csv', index_col='DateTime')
    # df5 = pd.read_csv('../feature/features/log_vol_chg_rate/000016_log_vol_chg_rate_20100104_20100630.csv',
    #                   index_col='DateTime')
    # df6 = pd.read_csv('../feature/features/vol_chg_rate/000016_vol_chg_rate_20100104_20100630.csv',
    #                   index_col='DateTime')
    # df7 = pd.read_csv('../feature/features/volume_relative_ratio/000016_volume_relative_ratio_20100104_20100630.csv',
    #                   index_col='DateTime')
    # df8 = pd.read_csv('../feature/features/amp/000016_amp_20100104_20100630.csv', index_col='DateTime')
    # df9 = pd.read_csv('../feature/features/price_efficiency/000016_price_efficiency_20100104_20100630.csv',
    #                   index_col='DateTime')
    # df10 = pd.read_csv('../feature/features/pri_ma_ratio/000016_pri_ma_ratio_20100104_20100630.csv',
    #                    index_col='DateTime')
    # df11 = pd.read_csv('../feature/features/atr/000016_atr_20100104_20100630.csv', index_col='DateTime')
    # df12 = pd.read_csv('../feature/features/rsi/000016_rsi_20100104_20100630.csv', index_col='DateTime')
    # df13 = pd.read_csv('../feature/features/cci/000016_cci_20100104_20100630.csv', index_col='DateTime')
    # df14 = pd.read_csv('../feature/features/obv/000016_obv_20100104_20100630.csv', index_col='DateTime')
    # df15 = pd.read_csv('../feature/features/roc/000016_roc_20100104_20100630.csv', index_col='DateTime')
    # df16 = pd.read_csv('../feature/features/bias/000016_bias_20100104_20100630.csv', index_col='DateTime')
    # df17 = pd.read_csv('../feature/features/vma/000016_vma_20100104_20100630.csv', index_col='DateTime')
    # df18 = pd.read_csv('../feature/features/pvt/000016_pvt_20100104_20100630.csv', index_col='DateTime')
    # df19 = pd.read_csv('../feature/features/mfi/000016_mfi_20100104_20100630.csv', index_col='DateTime')
    # df20 = pd.read_csv('../feature/features/ma_ma_ratio/000016_ma_ma_ratio_20100104_20100630.csv', index_col='DateTime')
    # df21 = pd.read_csv('../feature/features/boll/000016_boll_20100104_20100630.csv', index_col='DateTime')
    # df22 = pd.read_csv('../feature/features/ar/000016_ar_20100104_20100630.csv', index_col='DateTime')
    # df23 = pd.read_csv('../feature/features/br/000016_br_20100104_20100630.csv', index_col='DateTime')

    join_df = Concat(0)
    print(join_df.shape[0])
    print(join_df.columns)

    # ae = Autoencoder(3, 10, tf.nn.tanh)
    # cost = ae.partial_fit(join_df.loc[:, ['return', 'ATR', 'MACD']])
    # print(cost)
    # print(ae.transform(join_df.loc[:, ['return', 'ATR', 'MACD']]))

if __name__ == '__main__':
    main()
