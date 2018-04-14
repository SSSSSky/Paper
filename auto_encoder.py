'''
SingleSparseAutoEncoder 单层稀疏自编码器
'''
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def preprocess(df):
    df[df == np.inf] = np.nan
    df[df == -np.inf] = np.nan
    df.reset_index(inplace=True)
    df = df.groupby('SecCode').ffill()  # 使用前时刻的值填充缺失值
    df.set_index(['SecCode', 'DateTime'], inplace=True)
    # df['br'][df['br'] > 30000] = 30000    # 去除br异常值
    # df['br'][df['br'] < -30000] = -30000

    md = df.median()
    MAD = (df - md).abs().median()
    up_bound = md + 3 * MAD
    lo_bound = md - 3 * MAD
    # print(up_bound)
    # print(lo_bound)
    for i in df.columns:
        df[i][df[i] > up_bound[i]] = up_bound[i]
        df[i][df[i] < lo_bound[i]] = lo_bound[i]

    #数据标准化
    dropped_df = df.dropna()
    scaler = preprocessing.StandardScaler().fit(dropped_df)

    # 平均值填充缺失值
    fill_na_dict = {}
    for i in range(len(df.columns)):
        fill_na_dict[df.columns[i]] = scaler.mean_[i]
    print(fill_na_dict)
    df.fillna(fill_na_dict, inplace=True)
    scaled_df = scaler.transform(df)
    print(scaled_df.shape)

    return scaled_df


def run_sparse_auto_encoder(n_input=16, n_hidden_1=5, batch_size=2048, transfer=tf.nn.sigmoid,epoches=500, rho=0.5, beta=3.0, lamda=0.001, alpha=0.00005, decay=1.0,
                            train_dataset='', valid_dataset='', test_dataset='', model_name='autoencoder', device='0'):
    print(
        'model_name:{}\nn_input:{}\nn_hidden_1:{}\nbatch_size:{}\ntransfer:{}\nepochs:{}\nrho:{}\nbeta:{}\nlamda:{}\nalpha:{}\ndecay:{}\ndevice:{}'.format(model_name, n_input, n_hidden_1, batch_size, transfer.__name__, epoches, rho, beta, lamda, alpha, decay, device))
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    print('training_sets: {}'.format(train_dataset))
    print('validating_sets: {}'.format(valid_dataset))
    print('testing_sets: {}'.format(test_dataset))

    N_INPUT = n_input
    N_HIDDEN_1 = n_hidden_1
    N_OUTPUT_1 = N_INPUT

    EPOCHES = epoches
    RHO = rho  # 稀疏性参数
    BETA = tf.constant(beta)        # 稀疏系数
    LAMBDA = tf.constant(lamda)     # 正则化系数
    ALPHA = tf.constant(alpha)      # 学习率
    BATCH_SIZE = batch_size
    DECAY = tf.constant(decay)      # 学习率衰减

    # AutoEncoder 1
    w_model_one_init = np.sqrt(6. / (N_INPUT + N_HIDDEN_1))
    model_one_weights = {
        "hidden": tf.Variable(tf.random_uniform([N_INPUT, N_HIDDEN_1], minval = -w_model_one_init, maxval = w_model_one_init)),
        "out": tf.Variable(tf.random_uniform([N_HIDDEN_1, N_OUTPUT_1], minval = -w_model_one_init, maxval = w_model_one_init))
    }
    model_one_bias = {
        "hidden": tf.Variable(tf.random_uniform([N_HIDDEN_1], minval = -w_model_one_init, maxval = w_model_one_init)),
        "out": tf.Variable(tf.random_uniform([N_OUTPUT_1], minval = -w_model_one_init, maxval = w_model_one_init))
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
    train_op_1 = tf.train.AdamOptimizer(ALPHA*DECAY**epoch).minimize(model_one_cost)

    # 在tensorboard上画出收敛曲线
    # tf.summary.scalar('model_one_cost', model_one_cost)
    # tf.summary.scalar('model_one_cost_J', model_one_cost_J)
    # tf.summary.scalar('model_one_cost_reg', model_one_cost_reg)
    # tf.summary.scalar('model_one_cost_sparse', model_one_cost_sparse)
    # tf.summary.histogram('model_one_X', model_one_X)
    # tf.summary.histogram('model_one_hidden', model_one_hidden)
    # tf.summary.histogram('model_one_out', model_one_out)


    # =======================================================================================
    # 训练集
    train_df = pd.read_csv(train_dataset, index_col=['SecCode', 'DateTime'])
    print(train_df.shape)
    scaled_train_df = preprocess(train_df)
    print(scaled_train_df.shape)

    # 验证集
    valid_df = pd.read_csv(valid_dataset, index_col=['SecCode', 'DateTime'])
    print(valid_df.shape)
    scaled_valid_df = preprocess(valid_df)
    print(scaled_valid_df.shape)

    # 预测集
    test_df = pd.read_csv(test_dataset, index_col=['SecCode', 'DateTime'])
    print(test_df.shape)
    scaled_test_df = preprocess(test_df)
    print(scaled_test_df.shape)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        model_path = '../model1/Autoencoder_{}_act={}_lambda={}_alpha={}_decay={}_beta={}_RHO={}_epoch={}_batch={}_{}_{}.ckpt'\
            .format(model_name, transfer.__name__, lamda, alpha, decay, beta, rho, epoches, batch_size, N_INPUT, n_hidden_1)
        init = tf.global_variables_initializer()
        sess.run(init)
        training_losses = []
        validating_losses = []

        print('start training sparse_autoencoder ...')
        for i in range(EPOCHES):
            for start, end in zip(range(0, len(scaled_train_df), BATCH_SIZE), range(BATCH_SIZE, len(scaled_train_df), BATCH_SIZE)):
                input_ = scaled_train_df[start: end]
                sess.run(train_op_1, feed_dict={model_one_X: input_, epoch: i})
            cost = sess.run(model_one_cost, feed_dict={model_one_X: scaled_train_df})
            valid_cost = sess.run(model_one_cost, feed_dict={model_one_X: scaled_valid_df})
            sparse_cost = sess.run(model_one_cost_sparse, feed_dict={model_one_X: scaled_train_df})
            cost_J = sess.run(model_one_cost_J, feed_dict={model_one_X: scaled_train_df})
            training_losses.append(cost)
            validating_losses.append(valid_cost)
            print('{} Epoch {}: cost = {}  cost_J = {} sparse_cost = {}'
                  .format(model_name, i, cost, cost_J, sparse_cost))
        print('finish sparse_autoencoder ...')

        print('get hidden result...')
        # 获得隐藏层的数据，即所需要的信息
        train_hidden_result = sess.run(model_one_hidden, feed_dict={scaled_train_df})
        valid_hidden_result = sess.run(model_one_hidden, feed_dict={scaled_valid_df})
        test_hidden_result = sess.run(model_one_hidden, feed_dict={scaled_test_df})

        print('train_hidden_result.shape: {}'.format(train_hidden_result.shape))
        print('valid_hidden_result.shape: {}'.format(valid_hidden_result.shape))
        print('test_hidden_result.shape: {}'.format(test_hidden_result.shape))


        # 保存自编码器中间隐藏层输出的结果
        # print('save auencoder result...')
        # pd.DataFrame(data=train_hidden_result, index=train_df.index).to_csv('../Data/AutoEncoder/hidden_result/train_hidden_result.csv')
        # pd.DataFrame(data=valid_hidden_result, index=valid_df.index).to_csv('../Data/AutoEncoder/hidden_result/valid_hidden_result.csv')
        # pd.DataFrame(data=test_hidden_result, index=test_df.index).to_csv('../Data/AutoEncoder/hidden_result/test_hidden_result.csv')

def main():
    run_sparse_auto_encoder(n_input=22, n_hidden_1=10, epoches=1000, batch_size=2048, rho=0.1, beta=1.0, alpha=1e-4, lamda=1.0, transfer=tf.nn.sigmoid, decay=1.0,
                            train_dataset='../Data/AutoEncoder/train/technical_train_df.csv',
                            valid_dataset='../Data/AutoEncoder/valid/technical_valid_df.csv',
                            test_dataset='../Data/AutoEncoder/test/technical_test_df.csv',
                            model_name='SparseAutoEncoder', device='1')
    # run_sparse_auto_encoder(n_input=22, n_hidden_1=10, epoches=1000, batch_size=2048, rho=0.1, beta=1.0, alpha=1e-4, lamda=1.0, transfer=tf.nn.sigmoid, decay=1.0,
    #                         train_dataset='../Data/AutoEncoder/train/alpha_train_df.csv',
    #                         valid_dataset='../Data/AutoEncoder/valid/alpha_valid_df.csv',
    #                         test_dataset='../Data/AutoEncoder/test/alpha_test_df.csv',
    #                         model_name='SparseAutoEncoder', device='1')

if __name__ == '__main__':
    main()
