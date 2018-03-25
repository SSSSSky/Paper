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
    filelists = os.listdir(rootdir)
    filelists.sort()
    q = 0

    for filelist in filelists:
        path = os.path.join(rootdir, filelist)
        files = os.listdir(path)
        files.sort()
        file_path = os.path.join(path, files[i])
        df1 = pd.read_csv(file_path, index_col='DateTime')
        df1.fillna(0)
        df1 = df1.unstack(-1).reset_index()
        df1.columns = ['SecCode', 'DateTime', filelist]
        if q == 0:
            df2 = df1
        else:
            df2 = pd.merge(df2, df1, on=['SecCode', 'DateTime'], how = 'inner')
            df2.dropna(axis=0, inplace=True)
        q += 1

    df2.fillna(0)
    df2.replace(inf, 0)
    df2.replace(-inf,0)
    return df2

def main():

    join_df = Concat(0)
    join_df.to_csv('../feature/20100104_20100630.csv')
    join_df1 = Concat(1)
    join_df1.to_csv('../feature/20100104_20100630.csv')
    join_df2 = Concat(2)
    join_df2.to_csv('../feature/20100104_20100630.csv')
    join_df3 = Concat(3)
    join_df3.to_csv('../feature/20100630_20100728.csv')
    join_df4 = Concat(4)
    join_df4.to_csv('../feature/20100728_20101231.csv')
    join_df5 = Concat(5)
    join_df5.to_csv('../feature/20101231_20110630.csv')
    join_df6 = Concat(6)
    join_df6.to_csv('../feature/20110630_20111230.csv')
    join_df7 = Concat(7)
    join_df7.to_csv('../feature/20111230_20120730.csv')
    join_df8 = Concat(8)
    join_df8.to_csv('../feature/20120730_20130130.csv')
    join_df9 = Concat(9)
    join_df9.to_csv('../feature/20130130_20130730.csv')
    join_df10 = Concat(10)
    join_df10.to_csv('../feature/20130730_20131230.csv')
    join_df11 = Concat(11)
    join_df11.to_csv('../feature/20131230_20140613.csv')
    join_df12 = Concat(12)
    join_df12.to_csv('../feature/20141212_20150520.csv')
    join_df13 = Concat(13)
    join_df13.to_csv('../feature/20150520_20150612.csv')
    join_df14 = Concat(14)
    join_df14.to_csv('../feature/20150612_20151211.csv')
    join_df15 = Concat(15)
    join_df15.to_csv('../feature/20151211_20160608.csv')
    join_df16 = Concat(16)
    join_df16.to_csv('../feature/20160608_20161209.csv')
    join_df17 = Concat(17)
    join_df17.to_csv('../feature/20161209_20170609.csv')
    join_df18 = Concat(18)
    join_df18.to_csv('../feature/20170609_20171208.csv')
    join_df19 = Concat(19)
    join_df19.to_csv('../feature/20171208_20171229.csv')

    print(join_df.shape[0])
    print(join_df.columns)
    
    ae = Autoencoder(67, 10, tf.nn.tanh)
    cost = ae.partial_fit(join_df.loc[:, ['alpha001', 'alpha002', 'alpha003', 'alpha004',
       'alpha006', 'alpha007', 'alpha008', 'alpha009', 'alpha010', 'alpha012',
       'alpha013', 'alpha014', 'alpha015', 'alpha016', 'alpha017', 'alpha018',
       'alpha019', 'alpha020', 'alpha021', 'alpha022', 'alpha023', 'alpha024',
       'alpha026', 'alpha028', 'alpha029', 'alpha030', 'alpha031', 'alpha033',
       'alpha034', 'alpha035', 'alpha037', 'alpha038', 'alpha039', 'alpha040',
       'alpha043', 'alpha044', 'alpha045', 'alpha046', 'alpha049', 'alpha051',
       'alpha052', 'alpha053', 'alpha054', 'alpha055', 'alpha060', 'amp', 'ar',
       'atr', 'bias', 'boll', 'br', 'cci', 'log_vol_chg_rate', 'logreturn',
       'ma_ma_ratio', 'macd', 'mfi', 'obv', 'pri_ma_ratio', 'price_efficiency',
       'pvt', 'return', 'roc', 'rsi', 'vma', 'vol_chg_rate',
       'volume_relative_ratio']])
    print(cost)
    print(ae.transform(join_df.loc[:, ['alpha001', 'alpha002', 'alpha003', 'alpha004',
       'alpha006', 'alpha007', 'alpha008', 'alpha009', 'alpha010', 'alpha012',
       'alpha013', 'alpha014', 'alpha015', 'alpha016', 'alpha017', 'alpha018',
       'alpha019', 'alpha020', 'alpha021', 'alpha022', 'alpha023', 'alpha024',
       'alpha026', 'alpha028', 'alpha029', 'alpha030', 'alpha031', 'alpha033',
       'alpha034', 'alpha035', 'alpha037', 'alpha038', 'alpha039', 'alpha040',
       'alpha043', 'alpha044', 'alpha045', 'alpha046', 'alpha049', 'alpha051',
       'alpha052', 'alpha053', 'alpha054', 'alpha055', 'alpha060', 'amp', 'ar',
       'atr', 'bias', 'boll', 'br', 'cci', 'log_vol_chg_rate', 'logreturn',
       'ma_ma_ratio', 'macd', 'mfi', 'obv', 'pri_ma_ratio', 'price_efficiency',
       'pvt', 'return', 'roc', 'rsi', 'vma', 'vol_chg_rate',
       'volume_relative_ratio']]))

if __name__ == '__main__':
    main()
