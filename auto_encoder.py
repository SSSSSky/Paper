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

def main():

    path = '../feature/join_feature'
    files = os.listdir(path)
    files.sort()
    q = 0
    for file in files:
        file_path = os.path.join(path, file)
        df1 = pd.read_csv(file_path)
        if q == 0:
            join_df = df1
        else:
            pd.concat([join_df, df1])
    
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
