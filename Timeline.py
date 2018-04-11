'''
SingleSparseAutoEncoder and Timeline
'''
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
import matplotlib
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox

matplotlib.use('Agg')

# 检测是否是平稳序列并处理非平稳序列
def AdfTest(index_list):
    adftest = ADF(index_list)
    # 返回值依次为adf,pvalue,usedlag,nobs,critical values,icbest,regresults,resstore
    i = 0
    for key, value in adftest[4].items():
        if value < adftest[0]:
            i += 1
    # 假如adf值小于两个水平值，p值小于0.05，则判断为平稳序列
    if i <= 1 and adftest[1] < 0.05:
        print('Stationary series...')
        return index_list
    # 假如不是平稳序列，就对其进行差分
    else:
        D_data = np.diff(index_list)
        # result = AdfTest(D_data)
        return D_data

# 检测是否是非白噪声序列
# def whitenoise(index_list):
#     whitenoisetest = acorr_ljungbox(index_list, lags=1)
#     # lags 为滞后数,返回统计量和p值
#     # 假如p值小于0.05，则判断为非白噪声序列
#     if whitenoisetest[1] < 0.05:
#         print('No white noise...')
#         return index_list
#     else:


def Timeline(index_list):
    stationary_result = AdfTest(index_list)
    return stationary_result

#AutoEncoder
def run_sparse_auto_encoder(n_input=16, n_hidden_1=5, batch_size=2048, transfer=tf.nn.sigmoid, epoches=500, rho=0.5,
                            beta=3.0, lamda=0.001, alpha=0.00005, decay=1.0, path='', model_name='autoencoder',
                            device='0'):
    print(
        'model_name:{}\nn_input:{}\nn_hidden_1:{}\nbatch_size:{}\ntransfer:{}\nepochs:{}\nrho:{}\nbeta:{}\nlamda:{}\nalpha:{}\ndecay:{}\ndevice:{}'.format(
            model_name, n_input, n_hidden_1, batch_size, transfer.__name__, epoches, rho, beta, lamda, alpha, decay,
            device))
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    N_INPUT = n_input
    N_HIDDEN_1 = n_hidden_1
    N_OUTPUT_1 = N_INPUT

    EPOCHES = epoches
    RHO = rho  # 稀疏性参数
    BETA = tf.constant(beta)  # 稀疏系数
    LAMBDA = tf.constant(lamda)  # 正则化系数
    ALPHA = tf.constant(alpha)  # 学习率
    BATCH_SIZE = batch_size
    DECAY = tf.constant(decay)  # 学习率衰减

    # AutoEncoder 1
    w_model_one_init = np.sqrt(6. / (N_INPUT + N_HIDDEN_1))
    model_one_weights = {
        "hidden": tf.Variable(
            tf.random_uniform([N_INPUT, N_HIDDEN_1], minval=-w_model_one_init, maxval=w_model_one_init)),
        "out": tf.Variable(
            tf.random_uniform([N_HIDDEN_1, N_OUTPUT_1], minval=-w_model_one_init, maxval=w_model_one_init))
    }
    model_one_bias = {
        "hidden": tf.Variable(tf.random_uniform([N_HIDDEN_1], minval=-w_model_one_init, maxval=w_model_one_init)),
        "out": tf.Variable(tf.random_uniform([N_OUTPUT_1], minval=-w_model_one_init, maxval=w_model_one_init))
    }

    model_one_X = tf.placeholder("float", [None, N_INPUT])
    epoch = tf.placeholder("float")

    def model_one(X):
        hidden = transfer(tf.add(tf.matmul(X, model_one_weights["hidden"]), model_one_bias["hidden"]))
        out = transfer(tf.add(tf.matmul(hidden, model_one_weights["out"]), model_one_bias["out"]))
        return [hidden, out]

    def KLD(p, q):
        invrho = tf.subtract(tf.constant(1.), p)
        invrhohat = tf.subtract(tf.constant(1.), q)
        addrho = tf.add(tf.multiply(p, tf.log(tf.div(p, q))), tf.multiply(invrho, tf.log(tf.div(invrho, invrhohat))))
        return tf.reduce_sum(addrho)

    # model one
    model_one_hidden, model_one_out = model_one(model_one_X)
    # loss
    model_one_cost_J = tf.reduce_mean(tf.pow(tf.subtract(model_one_out, model_one_X), 2))
    # cost sparse
    input_size = tf.shape(model_one_X)[0]
    model_one_rho_hat = tf.div(tf.reduce_sum(model_one_hidden), tf.to_float(N_HIDDEN_1 * input_size))
    model_one_cost_sparse = tf.multiply(BETA, KLD(RHO, model_one_rho_hat))
    # cost reg
    model_one_cost_reg = tf.multiply(LAMBDA, tf.add(tf.nn.l2_loss(model_one_weights["hidden"]),
                                                    tf.nn.l2_loss(model_one_weights["out"])))
    # cost function
    model_one_cost = tf.add(tf.add(model_one_cost_J, model_one_cost_reg), model_one_cost_sparse)
    train_op_1 = tf.train.AdamOptimizer(ALPHA * DECAY ** epoch).minimize(model_one_cost)

    # =======================================================================================
    # 训练集
    path = '../feature/join_feature'
    files = os.listdir(path)
    files.sort()
    q = 0
    for file in files:
        file_path = os.path.join(path, file)
        df1 = pd.read_csv(file_path)
        if q == 0:
            join_df = df1
        # else:
        #     pd.concat([join_df, df1])
    result_df = join_df.loc[:, ['alpha001', 'alpha002', 'alpha003', 'alpha004',
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
                                'volume_relative_ratio']]
    print(result_df.shape)

    # 去除nan值和inf值
    result_df[result_df == np.inf] = np.nan
    result_df[result_df == -np.inf] = np.nan
    md = result_df.median()
    MAD = (result_df - md).abs().median()
    up_bound = md + 3 * MAD
    lo_bound = md - 3 * MAD
    print(up_bound)
    print(lo_bound)
    for i in result_df.columns:
        result_df[i][result_df[i] > up_bound[i]] = up_bound[i]
        result_df[i][result_df[i] < lo_bound[i]] = lo_bound[i]
    droped_result_df = result_df.dropna()

    scaler = preprocessing.StandardScaler().fit(droped_result_df)

    # 平均值填充缺失值
    fill_na_dict = {}
    for i in range(len(result_df.columns)):
        fill_na_dict[result_df.columns[i]] = scaler.mean_[i]
    print(fill_na_dict)
    result_df.fillna(fill_na_dict, inplace=True)
    scaled_result_df = scaler.transform(result_df)
    print(scaled_result_df.shape)

    print('start Timeline...')
    data_df = Timeline(scaled_result_df[:,0])
    columns = 1
    while(columns < 67):
        print('start clomun {}'.format(columns))
        timeline_df = Timeline(scaled_result_df[:,columns])
        print(timeline_df.shape)
        i = 0
        while i < scaled_result_df.shape[0] - timeline_df.shape[0]:
            mean_df = timeline_df.mean()
            timeline_df = np.append(timeline_df,mean_df)
            print(timeline_df.shape)
            i += 1
        data_df = np.vstack((data_df, timeline_df))
        print(data_df.shape)
        columns += 1
    print(data_df.shape)
    data_df = data_df.T
    print(data_df.shape)
    print('finish Timeline...')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        model_path = '../model1/Autoencoder_{}_act={}_lambda={}_alpha={}_decay={}_beta={}_RHO={}_epoch={}_batch={}_{}_{}.ckpt' \
            .format(model_name, transfer.__name__, lamda, alpha, decay, beta, rho, epoches, batch_size, N_INPUT,
                    n_hidden_1)
        init = tf.global_variables_initializer()
        sess.run(init)
        trainig_losses = []
        validating_losses = []
        print('train model 1 ...')
        for i in range(EPOCHES):
            for start, end in zip(range(0, len(data_df), BATCH_SIZE),
                                  range(BATCH_SIZE, len(data_df), BATCH_SIZE)):
                input_ = data_df[start: end]
                sess.run(train_op_1, feed_dict={model_one_X: input_, epoch: i})
            cost = sess.run(model_one_cost, feed_dict={model_one_X: data_df})
            # valid_cost = sess.run(model_one_cost, feed_dict={model_one_X: scaled_valid_df})
            sparse_cost = sess.run(model_one_cost_sparse, feed_dict={model_one_X: data_df})
            cost_J = sess.run(model_one_cost_J, feed_dict={model_one_X: data_df})
            # trainig_losses.append(cost)
            # validating_losses.append(valid_cost)
            print('{} Epoch {}: cost = {}  cost_J = {} sparse_cost = {}'
                  .format(model_name, i, cost, cost_J, sparse_cost))
        print('finish model 1 ...')

def main():
    run_sparse_auto_encoder(n_input=67, n_hidden_1=10, epoches=400, batch_size=2048, rho=0.1, beta=1.0, alpha=1e-4,
                            lamda=1.0, transfer=tf.nn.sigmoid, decay=1.0, path='../feature/join_feature',
                            model_name='1_D_sigmoid', device='1')


if __name__ == '__main__':
    main()
