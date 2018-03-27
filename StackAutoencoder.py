import tensorflow as tf
import numpy as np
import pandas as pd
import os.path

N_INPUT = 153039 * 70
N_HIDDEN_1 = 10
N_OUTPUT_1 = N_INPUT
N_HIDDEN_2 = 15
N_OUTPUT_2 = N_HIDDEN_1
N_OUTPUT = 10
# BATCH_SIZE = 16
# EPOCHES = 10
RHO = .1
BETA = tf.constant(3.0)
LAMBDA = tf.constant(.0001)

w_model_one_init = np.sqrt(6. / (N_INPUT + N_HIDDEN_1))

model_one_weights = {
    "hidden": tf.Variable(tf.random_uniform([N_INPUT, N_HIDDEN_1], minval = -w_model_one_init, maxval = w_model_one_init)),
    "out": tf.Variable(tf.random_uniform([N_HIDDEN_1, N_OUTPUT_1], minval = -w_model_one_init, maxval = w_model_one_init))
}
model_one_bias = {
    "hidden": tf.Variable(tf.random_uniform([N_HIDDEN_1], minval = -w_model_one_init, maxval = w_model_one_init)),
    "out": tf.Variable(tf.random_uniform([N_OUTPUT_1], minval = -w_model_one_init, maxval = w_model_one_init))
}

w_model_two_init = np.sqrt(6. / (N_HIDDEN_1 + N_HIDDEN_2))

model_two_weights = {
    "hidden": tf.Variable(tf.random_uniform([N_HIDDEN_1, N_HIDDEN_2], minval = -w_model_two_init, maxval = w_model_two_init)),
    "out": tf.Variable(tf.random_uniform([N_HIDDEN_2, N_OUTPUT_2], minval = -w_model_two_init, maxval = w_model_two_init))
}
model_two_bias = {
    "hidden": tf.Variable(tf.random_uniform([N_HIDDEN_2], minval = -w_model_two_init, maxval = w_model_two_init)),
    "out": tf.Variable(tf.random_uniform([N_OUTPUT_2], minval = -w_model_two_init, maxval = w_model_two_init))
}

w_model_init = np.sqrt(6. / (N_HIDDEN_2 + N_OUTPUT))

model_weights = {
    "out": tf.Variable(tf.random_uniform([N_HIDDEN_2, N_OUTPUT], minval = -w_model_init, maxval = w_model_init))
}
model_bias = {
    "out": tf.Variable(tf.random_uniform([N_OUTPUT], minval = -w_model_init, maxval = w_model_init))
}


model_one_X = tf.placeholder("float", [None, N_INPUT])
model_two_X = tf.placeholder("float", [None, N_HIDDEN_1])
Y = tf.placeholder("float", [None, N_OUTPUT])

def model_one(X):
    hidden = tf.sigmoid(tf.add(tf.matmul(X, model_one_weights["hidden"]), model_one_bias["hidden"]))
    out = tf.sigmoid(tf.add(tf.matmul(hidden, model_one_weights["out"]), model_one_bias["out"]))
    return [hidden, out]

def model_two(X):
    hidden = tf.sigmoid(tf.add(tf.matmul(X, model_two_weights["hidden"]), model_two_bias["hidden"]))
    out = tf.sigmoid(tf.add(tf.matmul(hidden, model_two_weights["out"]), model_two_bias["out"]))
    return [hidden, out]

def model(X):
    hidden_1 = tf.sigmoid(tf.add(tf.matmul(X, model_one_weights["hidden"]), model_one_bias["hidden"]))
    hidden_2 = tf.sigmoid(tf.add(tf.matmul(hidden_1, model_two_weights["hidden"]), model_two_bias["hidden"]))
    out = tf.add(tf.matmul(hidden_2, model_weights["out"]), model_bias["out"])
    return out

def KLD(p, q):
    invrho = tf.subtract(tf.constant(1.), p)
    invrhohat = tf.subtract(tf.constant(1.), q)
    addrho = tf.add(tf.multiply(p, tf.log(tf.div(p, q))), tf.multiply(invrho, tf.log(tf.div(invrho, invrhohat))))
    return tf.reduce_sum(addrho)

# model one
model_one_hidden, model_one_out = model_one(model_one_X)
# loss
model_one_cost_J = tf.reduce_sum(tf.pow(tf.subtract(model_one_out, model_one_X), 2))
# cost sparse
model_one_rho_hat = tf.div(tf.reduce_sum(model_one_hidden), N_HIDDEN_1)
model_one_cost_sparse = tf.multiply(BETA, KLD(RHO, model_one_rho_hat))
# cost reg
model_one_cost_reg = tf.multiply(LAMBDA, tf.add(tf.nn.l2_loss(model_one_weights["hidden"]), tf.nn.l2_loss(model_one_weights["out"])))
# cost function
model_one_cost = tf.add(tf.add(model_one_cost_J, model_one_cost_reg), model_one_cost_sparse)
train_op_1 = tf.train.AdamOptimizer().minimize(model_one_cost)
# =======================================================================================

# model two
model_two_hidden, model_two_out = model_two(model_two_X)
# loss
model_two_cost_J = tf.reduce_sum(tf.pow(tf.subtract(model_two_out, model_two_X), 2))
# cost sparse
model_two_rho_hat = tf.div(tf.reduce_sum(model_two_hidden), N_HIDDEN_2)
model_two_cost_sparse = tf.multiply(BETA, KLD(RHO, model_two_rho_hat))
# cost reg
model_two_cost_reg = tf.multiply(LAMBDA, tf.add(tf.nn.l2_loss(model_two_weights["hidden"]), tf.nn.l2_loss(model_two_weights["out"])))
# cost function
model_two_cost = tf.add(tf.add(model_two_cost_J, model_two_cost_reg), model_two_cost_sparse)
train_op_2 = tf.train.AdamOptimizer().minimize(model_two_cost)
# =======================================================================================

# final model
model_out = model(model_one_X)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = model_out, logits = Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(model_out, 1)
# =======================================================================================

def get_data():
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
    return join_df

with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    join_df = get_data()

    cost, opt = sess.run(train_op_1, feed_dict={model_one_X: join_df})
    print('cost:')
    print(cost)
    print('finish model one ...')

    input_ = sess.run(tf.sigmoid(tf.add(tf.matmul(join_df, model_one_weights["hidden"]), model_one_bias["hidden"])))
    cost, opt = sess.run(train_op_2, feed_dict = {model_two_X: input_})
    print('cost:')
    print(cost)
    print('finish model two ...')

    # for i in range(EPOCHES):
    #     for start, end in zip(range(0, len(join_df), BATCH_SIZE), range(BATCH_SIZE, len(join_df), BATCH_SIZE)):
    #         input_ = join_df[start:end]
    #         cost, opt = sess.run(train_op_1, feed_dict = {model_one_X: input_})
    #         print('cost:')
    #         print(cost)
    # print('finish model one ...')
    #
    # for i in range(EPOCHES):
    #     for start, end in zip(range(0, len(join_df), BATCH_SIZE), range(BATCH_SIZE, len(join_df), BATCH_SIZE)):
    #         input_ = join_df[start:end]
    #         input_ = sess.run(tf.sigmoid(tf.add(tf.matmul(input_, model_one_weights["hidden"]), model_one_bias["hidden"])))
    #         sess.run(train_op_2, feed_dict = {model_two_X: input_})
    # print('finish model two ...')
    #
    # for i in range(EPOCHES):
    #     for start, end in zip(range(0, len(join_df), BATCH_SIZE), range(BATCH_SIZE, len(join_df), BATCH_SIZE)):
    #         input_ = join_df[start:end]
    #         sess.run(train_op, feed_dict = {model_one_X: input_, Y: join_df[start:end]})
    # print('finish model ...')
