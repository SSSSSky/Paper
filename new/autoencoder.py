'''
AutoEncoder 自编码器
'''
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing


def run_sparse_auto_encoder(n_input=16, n_hidden_1=12, n_hidden_2=8, n_hidden_3=5, n_hidden_4=5, batch_size=2048,
                            epoches=500, rho=0.1, beta=3.0, lamda=0.001, alpha=0.000001,
                            model_name='autoencoder', feature='alpha', device='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    N_INPUT = n_input
    N_HIDDEN_1 = n_hidden_1
    N_OUTPUT_1 = N_INPUT
    N_HIDDEN_2 = n_hidden_2
    N_OUTPUT_2 = N_HIDDEN_1
    N_HIDDEN_3 = n_hidden_3
    N_OUTPUT_3 = N_HIDDEN_2
    N_HIDDEN_4 = n_hidden_4
    N_OUTPUT_4 = N_HIDDEN_3

    EPOCHES = epoches
    RHO = rho  # 稀疏性参数
    BETA = tf.constant(beta)        # 稀疏系数
    LAMBDA = tf.constant(lamda)     # 正则化系数
    ALPHA = tf.constant(alpha)      # 学习率
    BATCH_SIZE = batch_size

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

    # AutoEncoder 2
    w_model_two_init = np.sqrt(6. / (N_HIDDEN_1 + N_HIDDEN_2))
    model_two_weights = {
        "hidden": tf.Variable(
            tf.random_uniform([N_HIDDEN_1, N_HIDDEN_2], minval=-w_model_two_init, maxval=w_model_two_init)),
        "out": tf.Variable(
            tf.random_uniform([N_HIDDEN_2, N_OUTPUT_2], minval=-w_model_two_init, maxval=w_model_two_init))
    }
    model_two_bias = {
        "hidden": tf.Variable(tf.random_uniform([N_HIDDEN_2], minval=-w_model_two_init, maxval=w_model_two_init)),
        "out": tf.Variable(tf.random_uniform([N_OUTPUT_2], minval=-w_model_two_init, maxval=w_model_two_init))
    }

    # AutoEncoder 3
    w_model_three_init = np.sqrt(6. / (N_HIDDEN_2 + N_HIDDEN_3))
    model_three_weights = {
        "hidden": tf.Variable(
            tf.random_uniform([N_HIDDEN_2, N_HIDDEN_3], minval=-w_model_three_init, maxval=w_model_three_init)),
        "out": tf.Variable(
            tf.random_uniform([N_HIDDEN_3, N_OUTPUT_3], minval=-w_model_three_init, maxval=w_model_three_init))
    }
    model_three_bias = {
        "hidden": tf.Variable(tf.random_uniform([N_HIDDEN_3], minval=-w_model_three_init, maxval=w_model_three_init)),
        "out": tf.Variable(tf.random_uniform([N_OUTPUT_3], minval=-w_model_three_init, maxval=w_model_three_init))
    }

    # AutoEncoder 4
    w_model_four_init = np.sqrt(6. / (N_HIDDEN_3 + N_HIDDEN_4))
    model_four_weights = {
        "hidden": tf.Variable(
            tf.random_uniform([N_HIDDEN_3, N_HIDDEN_4], minval=-w_model_four_init, maxval=w_model_four_init)),
        "out": tf.Variable(
            tf.random_uniform([N_HIDDEN_4, N_OUTPUT_4], minval=-w_model_four_init, maxval=w_model_four_init))
    }
    model_four_bias = {
        "hidden": tf.Variable(tf.random_uniform([N_HIDDEN_4], minval=-w_model_four_init, maxval=w_model_four_init)),
        "out": tf.Variable(tf.random_uniform([N_OUTPUT_4], minval=-w_model_four_init, maxval=w_model_four_init))
    }

    model_one_X = tf.placeholder("float", [None, N_INPUT])
    model_two_X = tf.placeholder("float", [None, N_HIDDEN_1])
    model_three_X = tf.placeholder("float", [None, N_HIDDEN_2])
    model_four_X = tf.placeholder("float", [None, N_HIDDEN_3])

    def model_one(X):
        hidden = tf.sigmoid(tf.add(tf.matmul(X, model_one_weights["hidden"]), model_one_bias["hidden"]))
        out = tf.sigmoid(tf.add(tf.matmul(hidden, model_one_weights["out"]), model_one_bias["out"]))
        return [hidden, out]

    def model_two(X):
        hidden = tf.sigmoid(tf.add(tf.matmul(X, model_two_weights["hidden"]), model_two_bias["hidden"]))
        out = tf.sigmoid(tf.add(tf.matmul(hidden, model_two_weights["out"]), model_two_bias["out"]))
        return [hidden, out]

    def model_three(X):
        hidden = tf.sigmoid(tf.add(tf.matmul(X, model_three_weights["hidden"]), model_three_bias["hidden"]))
        out = tf.sigmoid(tf.add(tf.matmul(hidden, model_three_weights["out"]), model_three_bias["out"]))
        return [hidden, out]

    def model_four(X):
        hidden = tf.sigmoid(tf.add(tf.matmul(X, model_four_weights["hidden"]), model_four_bias["hidden"]))
        out = tf.sigmoid(tf.add(tf.matmul(hidden, model_four_weights["out"]), model_four_bias["out"]))
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
    train_op_1 = tf.train.AdamOptimizer(ALPHA).minimize(model_one_cost)
    # =======================================================================================

    # model two
    model_two_hidden, model_two_out = model_two(model_two_X)
    # loss
    model_two_cost_J = tf.reduce_mean(tf.pow(tf.subtract(model_two_out, model_two_X), 2))
    # cost sparse
    input_size = tf.shape(model_two_X)[0]
    model_two_rho_hat = tf.div(tf.reduce_sum(model_two_hidden), tf.to_float(N_HIDDEN_2 * input_size))
    model_two_cost_sparse = tf.multiply(BETA, KLD(RHO, model_two_rho_hat))
    # cost reg
    model_two_cost_reg = tf.multiply(LAMBDA, tf.add(tf.nn.l2_loss(model_two_weights["hidden"]),
                                                    tf.nn.l2_loss(model_two_weights["out"])))
    # cost function
    model_two_cost = tf.add(tf.add(model_two_cost_J, model_two_cost_reg), model_two_cost_sparse)
    train_op_2 = tf.train.AdamOptimizer(ALPHA).minimize(model_two_cost)
    # =======================================================================================

    # model three
    model_three_hidden, model_three_out = model_three(model_three_X)
    # loss
    model_three_cost_J = tf.reduce_mean(tf.pow(tf.subtract(model_three_out, model_three_X), 2))
    # cost sparse
    input_size = tf.shape(model_three_X)[0]
    model_three_rho_hat = tf.div(tf.reduce_sum(model_three_hidden), tf.to_float(N_HIDDEN_3 * input_size))
    model_three_cost_sparse = tf.multiply(BETA, KLD(RHO, model_three_rho_hat))
    # cost reg
    model_three_cost_reg = tf.multiply(LAMBDA, tf.add(tf.nn.l2_loss(model_three_weights["hidden"]),
                                                    tf.nn.l2_loss(model_three_weights["out"])))
    # cost function
    model_three_cost = tf.add(tf.add(model_three_cost_J, model_three_cost_reg), model_three_cost_sparse)
    train_op_3 = tf.train.AdamOptimizer(ALPHA).minimize(model_three_cost)
    # =======================================================================================

    # model four
    model_four_hidden, model_four_out = model_four(model_four_X)
    # loss
    model_four_cost_J = tf.reduce_mean(tf.pow(tf.subtract(model_four_out, model_four_X), 2))
    # cost sparse
    input_size = tf.shape(model_four_X)[0]
    model_four_rho_hat = tf.div(tf.reduce_sum(model_four_hidden), tf.to_float(N_HIDDEN_4 * input_size))
    model_four_cost_sparse = tf.multiply(BETA, KLD(RHO, model_four_rho_hat))
    # cost reg
    model_four_cost_reg = tf.multiply(LAMBDA, tf.add(tf.nn.l2_loss(model_four_weights["hidden"]),
                                                      tf.nn.l2_loss(model_four_weights["out"])))
    # cost function
    model_four_cost = tf.add(tf.add(model_four_cost_J, model_four_cost_reg), model_four_cost_sparse)
    train_op_4 = tf.train.AdamOptimizer(ALPHA).minimize(model_four_cost)

    # =======================================================================================
    # 训练集
    # result_df = pd.read_csv('../feature/features/technical_features_16_20150101.csv', index_col=['SecCode', 'DateTime'])
    train_df = pd.read_csv('../model1/{}/{}_20100104_20150101'.format(model_name, feature), index_col=['SecCode', 'DateTime'])
    print(train_df.shape)


    # 验证集
    # valid_df = pd.read_csv('../feature/features/technical_features_16_20150101_20170101.csv',
    #                        index_col=['SecCode', 'DateTime'])
    valid_df = pd.read_csv('../model1/{}/{}_20150101_20170101'.format(model_name, feature), index_col=['SecCode', 'DateTime'])

    # =======================================================================================
    # 测试集
    # result_df = pd.read_csv('../feature/features/technical_features_16_20150101.csv', index_col=['SecCode', 'DateTime'])
    test_df = pd.read_csv('../model1/{}/{}_20150101_20171231'.format(model_name, feature), index_col=['SecCode', 'DateTime'])
    print(test_df.shape)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        model_path = 'model/Autoencoder_{}_lambda={}_alpha={}_beta={}_RHO={}_epoch={}_batch={}_{}_{}_{}_{}_{}.ckpt'\
            .format(model_name, lamda, alpha, beta, rho, epoches, batch_size, N_INPUT, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4)
        init = tf.global_variables_initializer()
        sess.run(init)

        print('train model 1 ...')
        for i in range(EPOCHES):
            for start, end in zip(range(0, len(train_df), BATCH_SIZE), range(BATCH_SIZE, len(train_df), BATCH_SIZE)):
                input_ = train_df[start: end]
                sess.run(train_op_1, feed_dict={model_one_X: input_})
            cost = sess.run(model_one_cost, feed_dict={model_one_X: train_df})
            valid_cost = sess.run(model_one_cost, feed_dict={model_one_X: valid_df})
            sparse_cost = sess.run(model_one_cost_sparse, feed_dict={model_one_X: train_df})
            cost_J = sess.run(model_one_cost_J, feed_dict={model_one_X: train_df})
            print('Epoch {}: cost = {} valid_c = {} cost_J = {} sparse_cost = {}'
                  .format(i, cost, valid_cost, cost_J, sparse_cost))
        print('finish model 1 ...')

        print('calculate 1st encoder result ...')
        scaled_result_df1 = \
            sess.run(tf.sigmoid(tf.add(tf.matmul(model_one_X, model_one_weights['hidden']), model_one_bias['hidden'])),
                     feed_dict={model_one_X: train_df})
        scaled_valid_df1 = \
            sess.run(tf.sigmoid(tf.add(tf.matmul(model_one_X, model_one_weights['hidden']), model_one_bias['hidden'])),
                     feed_dict={model_one_X: valid_df})
        print(len(scaled_result_df1))
        print(len(scaled_valid_df1))
        print('finish 1st encoder result ...')

        print('train model 2 ...')
        for j in range(EPOCHES):
            for start, end in zip(range(0, len(scaled_result_df1), BATCH_SIZE), range(BATCH_SIZE, len(scaled_result_df1), BATCH_SIZE)):
                input_ = scaled_result_df1[start:end]
                sess.run(train_op_2, feed_dict={model_two_X: input_})

            cost = sess.run(model_two_cost, feed_dict={model_two_X: scaled_result_df1})
            valid_cost = sess.run(model_two_cost, feed_dict={model_two_X: scaled_valid_df1})
            sparse_cost = sess.run(model_two_cost_sparse, feed_dict={model_two_X: scaled_result_df1})
            cost_J = sess.run(model_two_cost_J, feed_dict={model_two_X: scaled_result_df1})
            print('Epoch {}: cost = {} valid_c = {} cost_J = {} sparse_cost = {}'
                  .format(j, cost, valid_cost, cost_J, sparse_cost))
        print('finish model 2 ...')

        print('calculate 2nd encoder result ...')
        scaled_result_df2 = \
            sess.run(tf.sigmoid(tf.add(tf.matmul(model_two_X, model_two_weights['hidden']), model_two_bias['hidden'])),
                     feed_dict={model_two_X: scaled_result_df1})
        scaled_valid_df2 = \
            sess.run(tf.sigmoid(tf.add(tf.matmul(model_two_X, model_two_weights['hidden']), model_two_bias['hidden'])),
                     feed_dict={model_two_X: scaled_valid_df1})
        print(len(scaled_result_df2))
        print(len(scaled_valid_df2))
        print('finish 2nd encoder result ...')

        print('train model 3 ...')
        for j in range(EPOCHES):
            for start, end in zip(range(0, len(scaled_result_df2), BATCH_SIZE),
                                  range(BATCH_SIZE, len(scaled_result_df2), BATCH_SIZE)):
                input_ = scaled_result_df2[start:end]
                sess.run(train_op_3, feed_dict={model_three_X: input_})

            cost = sess.run(model_three_cost, feed_dict={model_three_X: scaled_result_df2})
            valid_cost = sess.run(model_three_cost, feed_dict={model_three_X: scaled_valid_df2})
            sparse_cost = sess.run(model_three_cost_sparse, feed_dict={model_three_X: scaled_result_df2})
            cost_J = sess.run(model_three_cost_J, feed_dict={model_three_X: scaled_result_df2})
            print('Epoch {}: cost = {} valid_c = {} cost_J = {} sparse_cost = {}'
                  .format(j, cost, valid_cost, cost_J, sparse_cost))
        print('finish model 3 ...')

        print('calculate 3rd encoder result ...')
        scaled_result_df3 = \
            sess.run(tf.sigmoid(tf.add(tf.matmul(model_three_X, model_three_weights['hidden']), model_three_bias['hidden'])),
                     feed_dict={model_three_X: scaled_result_df2})
        scaled_valid_df3 = \
            sess.run(tf.sigmoid(tf.add(tf.matmul(model_three_X, model_three_weights['hidden']), model_three_bias['hidden'])),
                     feed_dict={model_three_X: scaled_valid_df2})
        print(len(scaled_result_df3))
        print(len(scaled_valid_df3))
        print('finish 3rd encoder result ...')

        print('train model 4 ...')
        for j in range(EPOCHES):
            for start, end in zip(range(0, len(scaled_result_df3), BATCH_SIZE),
                                  range(BATCH_SIZE, len(scaled_result_df3), BATCH_SIZE)):
                input_ = scaled_result_df3[start:end]
                sess.run(train_op_4, feed_dict={model_four_X: input_})

            cost = sess.run(model_four_cost, feed_dict={model_four_X: scaled_result_df3})
            valid_cost = sess.run(model_four_cost, feed_dict={model_four_X: scaled_valid_df3})
            sparse_cost = sess.run(model_four_cost_sparse, feed_dict={model_four_X: scaled_result_df3})
            cost_J = sess.run(model_four_cost_J, feed_dict={model_four_X: scaled_result_df3})
            print('Epoch {}: cost = {} valid_c = {} cost_J = {} sparse_cost = {}'
                  .format(j, cost, valid_cost, cost_J, sparse_cost))
        print('finish model 4 ...')

        print('calculate 4th encoder result ...')
        scaled_result_df4 = \
            sess.run(
                tf.sigmoid(tf.add(tf.matmul(model_four_X, model_four_weights['hidden']), model_four_bias['hidden'])),
                feed_dict={model_four_X: scaled_result_df3})
        scaled_valid_df4 = \
            sess.run(
                tf.sigmoid(tf.add(tf.matmul(model_four_X, model_four_weights['hidden']), model_four_bias['hidden'])),
                feed_dict={model_four_X: scaled_valid_df3})
        print(len(scaled_result_df4))
        print(len(scaled_valid_df4))
        print('finish 4th encoder result ...')

        print('save model ...')
        saver.save(sess, model_path)

        print('save encode result ...')
        print('save train_layer1_encoding.csv')
        pd.DataFrame(data=scaled_result_df1, index=train_df.index).to_csv('../model1/result/{}_train_layer1_encoding.csv'.format(model_name))
        print('save train_layer2_encoding.csv')
        pd.DataFrame(data=scaled_result_df2, index=train_df.index).to_csv('../model1/result/{}_train_layer2_encoding.csv'.format(model_name))
        print('save train_layer3_encoding.csv')
        pd.DataFrame(data=scaled_result_df3, index=train_df.index).to_csv('../model1/result/{}_train_layer3_encoding.csv'.format(model_name))
        print('save train_layer4_encoding.csv')
        pd.DataFrame(data=scaled_result_df4, index=train_df.index).to_csv('../model1/result/{}_train_layer4_encoding.csv'.format(model_name))
        print('save valid_layer1_encoding.csv')
        pd.DataFrame(data=scaled_valid_df1, index=valid_df.index).to_csv('../model1/result/{}_valid_layer1_encoding.csv'.format(model_name))
        print('save valid_layer2_encoding.csv')
        pd.DataFrame(data=scaled_valid_df2, index=valid_df.index).to_csv('../model1/result/{}_valid_layer2_encoding.csv'.format(model_name))
        print('save valid_layer3_encoding.csv')
        pd.DataFrame(data=scaled_valid_df3, index=valid_df.index).to_csv('../model1/result/{}_valid_layer3_encoding.csv'.format(model_name))
        print('save valid_layer4_encoding.csv')
        pd.DataFrame(data=scaled_valid_df4, index=valid_df.index).to_csv('../model1/result/{}_valid_layer4_encoding.csv'.format(model_name))
        print('finish save encode result ...')

def main():
    # run_sparse_auto_encoder(n_input=45, n_hidden_1=30, n_hidden_2=15, n_hidden_3=10, n_hidden_4=5, epoches=500,
    run_sparse_auto_encoder(n_input=16, n_hidden_1=12, n_hidden_2=10, n_hidden_3=8, n_hidden_4=5, epoches=500, batch_size=2048,
                            # model_name='alpha_autoencoder', device='0')
                            model_name = 'input', feature = 'alpha', device='1')
    run_sparse_auto_encoder(n_input=16, n_hidden_1=12, n_hidden_2=10, n_hidden_3=8, n_hidden_4=5, epoches=500, batch_size=2048,
                            # model_name='alpha_autoencoder', device='0')
                            model_name = 'input', feature = 'technical', device='1')
    run_sparse_auto_encoder(n_input=16, n_hidden_1=12, n_hidden_2=10, n_hidden_3=8, n_hidden_4=5, epoches=500, batch_size=2048,
                            # model_name='alpha_autoencoder', device='0')
                            model_name = 'wt_input', feature = 'alpha', device='1')
    run_sparse_auto_encoder(n_input=16, n_hidden_1=12, n_hidden_2=10, n_hidden_3=8, n_hidden_4=5, epoches=500, batch_size=2048,
                            # model_name='alpha_autoencoder', device='0')
                            model_name = 'wt_input', feature = 'technical', device='1')

if __name__ == '__main__':
    main()
