import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pandas as pd
from sklearn import preprocessing
import os

print('init parameters')
epochs = 500
input_size = 250    # 50*5
output_size = 50    # 50
hidden_size = 128   # LSTM隐层size，经验决定
num_steps = 240     # 时间步长240min
train_batch_size = 512
valid_batch_size = 256
keep_prob = 0.95
decay = 0.99

train_test_batch_size = 256
valid_test_batch_size = 128
test_batch_size = 64

alpha = 1e-6
# model_name = '4_B_layer4'
model_name = '5best_technical'
# mode = 'normal'
# mode = 'dropout'
mode = 'cost_limited'
cost_lambda = 0.001   # 调仓成本惩罚系数
device = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = device

class Data():
    def __init__(self, train_X_path=None, valid_X_path=None, test_X_path=None,
                 train_y_path=None, valid_y_path=None, test_y_path=None):
        print('[INFO] reading train_X ...')
        self.train_X = []
        columns_X = None
        for f in os.listdir(train_X_path):
            df = pd.read_csv('{}/{}'.format(train_X_path, f), index_col='DateTime')
            self.train_X.append(df)
            columns_X = df.columns
        for i in range(len(self.train_X)):
            self.train_X[i].columns = columns_X
        self.train_X = pd.concat(self.train_X)
        # self.train_X = pd.read_csv(train_X_path, index_col='DateTime')

        print('[INFO] reading train_y...')
        self.train_y = []
        columns_y = None
        for f in os.listdir(train_y_path):
            df = pd.read_csv('{}/{}'.format(train_y_path, f), index_col='DateTime')
            self.train_y.append(df)
            columns_y = df.columns
        for i in range(len(self.train_y)):
            self.train_y[i].columns = columns_y
        self.train_y = pd.concat(self.train_y)
        # self.train_y = pd.read_csv(train_y_path, index_col='DateTime')

        print('[INFO] reading valid_X...')
        self.valid_X = []
        columns_X = None
        for f in os.listdir(valid_X_path):
            df = pd.read_csv('{}/{}'.format(valid_X_path, f), index_col='DateTime')
            self.valid_X.append(df)
            columns_X = df.columns
        for i in range(len(self.valid_X)):
            self.valid_X[i].columns = columns_X
        self.valid_X = pd.concat(self.valid_X)
        # self.valid_X= pd.read_csv(valid_X_path, index_col='DateTime')

        print('[INFO] reading valid_y...')
        self.valid_y = []
        columns_y = None
        for f in os.listdir(valid_y_path):
            df = pd.read_csv('{}/{}'.format(valid_y_path, f), index_col='DateTime')
            self.valid_y.append(df)
            columns_y = df.columns
        for i in range(len(self.valid_y)):
            self.valid_y[i].columns = columns_y
        self.valid_y = pd.concat(self.valid_y)
        # self.valid_y = pd.read_csv(valid_y_path, index_col='DateTime')

        print('[INFO] reading test_X...')
        self.test_X = []
        columns_X = None
        for f in os.listdir(test_X_path):
            df = pd.read_csv('{}/{}'.format(test_X_path, f), index_col='DateTime')
            self.test_X.append(df)
            columns_X = df.columns
        for i in range(len(self.test_X)):
            self.test_X[i].columns = columns_X
        self.test_X = pd.concat(self.test_X)
        # self.test_X = pd.read_csv(test_X_path, index_col='DateTime')

        print('[INFO] reading test_y...')
        self.test_y = []
        columns_y = None
        for f in os.listdir(test_y_path):
            df = pd.read_csv('{}/{}'.format(test_y_path, f), index_col='DateTime')
            self.test_y.append(df)
            columns_y = df.columns
        for i in range(len(self.test_y)):
            self.test_y[i].columns = columns_y
        self.test_y = pd.concat(self.test_y)
        # self.test_y = pd.read_csv(test_y_path, index_col='DateTime')
        # self.train_X.fillna(0, inplace=True)    # 填充缺失值（填充为0有待考证）
        # self.train_y = self.train_y.iloc[481:, :]
        # self.train_y.fillna(0, inplace=True)

        # Attention：填充缺失值，仅限在技术指标时使用（自编码结果不需要）
        self.train_X[self.train_X == np.inf] = np.nan
        self.train_X[self.train_X == -np.inf] = np.nan
        self.valid_X[self.valid_X == np.inf] = np.nan
        self.valid_X[self.valid_X == -np.inf] = np.nan
        self.test_X[self.test_X == np.inf] = np.nan
        self.test_X[self.test_X == -np.inf] = np.nan
        self.train_X.fillna(method='ffill', inplace=True)
        self.valid_X.fillna(method='ffill', inplace=True)
        self.test_X.fillna(method='ffill', inplace=True)
        self.train_X.fillna(1e-6, inplace=True)
        # scaler = preprocessing.StandardScaler()
        # scaler.fit(self.train_X)
        # self.train_X = scaler.transform(self.train_X)
        # print(self.train_X.shape)
        # self.train_X.dropna(inplace=True)
        # print(self.train_X.shape)
        self.valid_X.fillna(0.0, inplace=True)
        # self.valid_X = scaler.transform(self.valid_X)
        self.test_X.fillna(0.0, inplace=True)
        # self.test_X = scaler.transform(self.test_X)

        print(self.train_X.shape, self.train_y.shape)
        self.train_y.fillna(0.0, inplace=True)
        print(self.valid_X.shape, self.valid_y.shape)
        self.valid_y.fillna(0.0, inplace=True)
        print(self.test_X.shape, self.test_y.shape)
        # self.test_X.fillna(0.0, inplace=True)
        self.test_y.fillna(0.0, inplace=True)

print('reading datasets')
train_X_path = '../Data/Model2/datasets/train'
valid_X_path = '../Data/Model2/datasets/valid'
predict_X_path = '../Data/Model2/datasets/test'

train_y_path = '../Data/Model2/predicts/train'.format(model_name)
valid_y_path = '../Data/Model2/predicts/valid'.format(model_name)
predict_y_path = '../Data/Model2/predicts/test'.format(model_name)



data = Data(train_X_path=train_X_path, valid_X_path=valid_X_path, test_X_path=predict_X_path,
            train_y_path=train_y_path, valid_y_path=valid_y_path, test_y_path=predict_y_path)

def gen_train_batch(data):
    # 此处需要对X和y进行对齐（因为长度可能不一致）
    train_X_length = len(data.train_X)

    batch_partition_len = train_X_length // train_batch_size
    data_X = np.zeros([train_batch_size, batch_partition_len, input_size])
    data_y = np.zeros([train_batch_size, batch_partition_len, output_size])
    for i in range(train_batch_size):
        data_X[i, :, :] = data.train_X.iloc[i*batch_partition_len:(i+1)*batch_partition_len, :]
        data_y[i, :, :] = data.train_y.iloc[i*batch_partition_len:(i+1)*batch_partition_len, :]

    epoch_size = batch_partition_len - num_steps + 1

    for i in range(epoch_size):
        X = data_X[:, i: i+num_steps, :]
        y = data_y[:, i: i+num_steps, :]
        yield (X, y)

def gen_valid_batch(data):
    valid_X_length = len(data.valid_X)

    batch_partition_len = valid_X_length // valid_batch_size
    data_X = np.zeros([valid_batch_size, batch_partition_len, input_size])
    data_y = np.zeros([valid_batch_size, batch_partition_len, output_size])
    for i in range(valid_batch_size):
        data_X[i, :, :] = data.valid_X.iloc[i*batch_partition_len: (i+1)*batch_partition_len, :]
        data_y[i, :, :] = data.valid_y.iloc[i*batch_partition_len: (i+1)*batch_partition_len, :]

    epoch_size = batch_partition_len - num_steps + 1

    for i in range(epoch_size):
        X = data_X[:, i: i+num_steps, :]
        y = data_y[:, i: i+num_steps, :]
        yield (X, y)

# 使用训练集做回测测试
def gen_train_test_batch(data):
    train_X_length = len(data.train_X)

    data_X = np.zeros([train_test_batch_size, num_steps, input_size])

    epoch_size = train_X_length - num_steps + 1
    for i in range(epoch_size // train_test_batch_size + 1):
        if i == epoch_size // train_test_batch_size:
            print(i)
            last_batch_size = epoch_size - (epoch_size // train_test_batch_size) * train_test_batch_size
            print(last_batch_size)
            data_last_X = np.zeros([last_batch_size, num_steps, input_size])
            for j in range(last_batch_size):
                print(j, i * train_test_batch_size + j, i * train_test_batch_size + j + num_steps)
                data_last_X[j, :, :] = data.train_X.iloc[
                                       i * train_test_batch_size + j: i * train_test_batch_size + j + num_steps, :]
            yield (data_last_X, last_batch_size)
        else:
            for j in range(train_test_batch_size):
                data_X[j, :, :] = data.train_X.iloc[
                                  i * train_test_batch_size + j: i * train_test_batch_size + j + num_steps, :]
            yield (data_X, train_test_batch_size)

# 使用验证集做回测测试
def gen_valid_test_batch(data):
    valid_X_length = len(data.valid_X)

    data_X = np.zeros([valid_test_batch_size, num_steps, input_size])

    epoch_size = valid_X_length - num_steps + 1
    for i in range(epoch_size // valid_test_batch_size + 1):
        if i == epoch_size // valid_test_batch_size:
            print(i)
            last_batch_size = epoch_size - (epoch_size // valid_test_batch_size) * valid_test_batch_size
            print(last_batch_size)
            data_last_X = np.zeros([last_batch_size, num_steps, input_size])
            for j in range(last_batch_size):
                print(j, i * valid_test_batch_size + j, i * valid_test_batch_size + j + num_steps)
                data_last_X[j, :, :] = data.valid_X.iloc[i * valid_test_batch_size + j: i * valid_test_batch_size + j + num_steps, :]
            yield (data_last_X, last_batch_size)
        else:
            for j in range(valid_test_batch_size):
                data_X[j, :, :] = data.valid_X.iloc[i * valid_test_batch_size + j: i * valid_test_batch_size + j + num_steps, :]
            yield (data_X, valid_test_batch_size)

def gen_test_batch(data):
    test_X_length = len(data.test_X)

    # batch_partition_len = test_X_length
    data_X = np.zeros([test_batch_size, num_steps, input_size])
    # data_y = np.zeros([test_batch_size, num_steps, output_size])

    epoch_size = test_X_length - num_steps + 1
    for i in range(epoch_size // test_batch_size + 1):
        # 最后一个batch
        if i == epoch_size // test_batch_size:
            last_batch_size = epoch_size - (epoch_size // test_batch_size)*test_batch_size
            print(last_batch_size)  # 最后一个batch大小
            data_last_X = np.zeros([last_batch_size, num_steps, input_size])
            for j in range(last_batch_size):
                print(j, i*test_batch_size+j, i*test_batch_size+j+num_steps)
                data_last_X[j, :, :] = data.test_X.iloc[i*test_batch_size+j: i*test_batch_size+j+num_steps, :]
            yield (data_last_X, last_batch_size)
        else:
            for j in range(test_batch_size):
                data_X[j, :, :] = data.test_X.iloc[i*test_batch_size+j: i*test_batch_size+j+num_steps, :]
            yield (data_X, test_batch_size)

def gen_epochs(data, epochs):
    for i in range(epochs):
        yield gen_train_batch(data)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

DECAY = tf.constant(decay)
ALPHA = tf.constant(alpha)

Epoch = tf.placeholder("float")
X = tf.placeholder("float", [None, num_steps, input_size])
y = tf.placeholder("float", [None, num_steps, output_size])

batch_size = tf.placeholder(tf.int32, [])
# keep_prob = tf.placeholder(tf.float32)

# 设置单层LSTM
lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)

# 设置dropout
if mode == 'dropout' or mode == 'cost_limited':
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)

# Double-LSTM
# mlstm_cell = rnn.MultiRNNCell([lstm_cell] * 2, state_is_tuple=True)

# 初始化状态
init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
# init_state = lstm_cell.zero_state(Batch_Size, dtype=tf.float32)

# outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
# h_state = outputs[:, -1, :]

outputs = []
state = init_state
with tf.variable_scope('RNN'):
    for step in range(num_steps):
        if step > 0:
            tf.get_variable_scope().reuse_variables()
        cell_output, state = lstm_cell(X[:, step, :], state)
        outputs.append(cell_output)
last_state = outputs[-1]

# define
W = tf.Variable(tf.truncated_normal([hidden_size, output_size], dtype=tf.float32))
b = tf.Variable(tf.constant(0.0, shape=[output_size]), dtype=tf.float32)

logits = [tf.matmul(output, W) + b for output in outputs]
last_logit = tf.matmul(last_state, W) + b
# y_pred = tf.matmul(last_state, W) + b
y_as_list = tf.unstack(y, num=num_steps, axis=1)
losses = [tf.reduce_mean(tf.square(logit - pred)) for logit, pred in zip(logits, y_as_list)]
total_loss = tf.reduce_mean(losses)

if mode == 'cost_limited':
    tmp1 = [(logit - tf.expand_dims(tf.reduce_min(logit, 1), 1)) for logit in logits]
    tmp2 = [(tf.reduce_max(logit, 1) - tf.reduce_min(logit, 1)) for logit in logits]
    norm_weight = [(t1 / tf.expand_dims(t2, 1)) for t1, t2 in zip(tmp1, tmp2)]
    # norm_weight = [(logit - tf.reduce_min(logit, 1) / (tf.reduce_max(logit, 1) - tf.reduce_min(logit, 1))) for logit in logits]
    cost_losses = [tf.reduce_mean(tf.abs(weight - weight_prev)) for weight, weight_prev in zip(norm_weight[1:], norm_weight[:-1])]
    total_cost_loss = tf.reduce_mean(cost_losses)
    loss = tf.add(total_loss, tf.multiply(cost_lambda, total_cost_loss))    # 普通loss+调仓成本loss
    train_op = tf.train.AdamOptimizer(ALPHA*DECAY*Epoch).minimize(loss)
else:
    train_op = tf.train.AdamOptimizer(ALPHA*DECAY*Epoch).minimize(total_loss)

training_losses = []
validating_losses = []
train_predicts = []
valid_predicts = []
test_predicts = []
with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    model_path = '../model2/model2_{}_mode={}_numsteps={}.ckpt'.format(model_name, mode, num_steps)
    sess.run(tf.global_variables_initializer())

    for i, epoch in enumerate(gen_epochs(data, epochs)):
        training_loss = 0.0
        validating_loss = 0.0
        testing_loss = 0.0
        training_cost_loss = 0.0
        validating_cost_loss = 0.0

        for step, (X_win, y_win) in enumerate(epoch):
            sess.run(train_op, feed_dict={X: X_win, y: y_win, batch_size: train_batch_size, Epoch: step})
            # print(sess.run(logits[0], feed_dict={X: X_win, y: y_win, batch_size: train_batch_size}))
            training_loss_= sess.run(total_loss, feed_dict={X: X_win, y: y_win, batch_size: train_batch_size})
            training_loss += training_loss_
            # print(training_loss_)
            if mode == 'cost_limited':
                cost_loss_ = sess.run(total_cost_loss, feed_dict={X: X_win, y: y_win, batch_size: train_batch_size})
                training_cost_loss += cost_loss_
                # print(cost_loss_)
        training_losses.append(training_loss)

        for step, (valid_X, valid_y) in enumerate(gen_valid_batch(data)):
            validating_loss_ = sess.run(total_loss, feed_dict={X: valid_X, y: valid_y, batch_size: valid_batch_size})
            validating_loss += validating_loss_
            if mode == 'cost_limited':
                cost_loss_ = sess.run(total_cost_loss, feed_dict={X: valid_X, y: valid_y, batch_size: valid_batch_size})
                validating_cost_loss += cost_loss_
        validating_losses.append(validating_loss)

        if mode == 'cost_limited':
            print('{} {} Epoch {} train_loss = {} train_cost_loss = {} valid_loss = {} valid_cost_loss = {}'
                  .format(model_name, mode, i, training_loss, training_cost_loss, validating_loss, validating_cost_loss))
        else:
            print('{} {} Epoch {} train_loss = {} valid_loss = {}'.format(model_name, mode, i, training_loss, validating_loss))

        # Early Stopping
        if i > 50 and validating_losses[-1] > validating_losses[-2]:
            break

    print('save model ...')
    saver.save(sess, model_path)

    print('predicting on traininig sets...')
    for step, (train_X, valid_b_s) in enumerate(gen_train_test_batch(data)):
    # 第一个预测周期需要预测整个num_step
        if step == 0:
            weights = sess.run(logits, feed_dict={X: train_X, batch_size: valid_b_s})
            print(len(weights), len(weights[0]), len(weights[0][0]), len(weights[0:(num_steps-1)]))
            for w in weights[0:(num_steps-1)]:
                train_predicts.append(w[0])

        weights = sess.run(last_logit, feed_dict={X: train_X, batch_size: valid_b_s})
        for w in weights:
            train_predicts.append(w)
        if step % 10 == 0:
            print('predicting {}...'.format(step))

    print('predicting on validating sets...')
    for step, (valid_X, valid_b_s) in enumerate(gen_valid_test_batch(data)):
    # 第一个预测周期需要预测整个num_step
        if step == 0:
            weights = sess.run(logits, feed_dict={X: valid_X, batch_size: valid_b_s})
            print(len(weights), len(weights[0]), len(weights[0][0]), len(weights[0:(num_steps-1)]))
            for w in weights[0:(num_steps-1)]:
                valid_predicts.append(w[0])

        weights = sess.run(last_logit, feed_dict={X: valid_X, batch_size: valid_b_s})
        for w in weights:
            valid_predicts.append(w)
        if step % 10 == 0:
            print('predicting {}...'.format(step))

    print('predicting on testing sets...')
    for step, (test_X, test_b_s) in enumerate(gen_test_batch(data)):
    # 第一个预测周期需要预测整个num_step
        if step == 0:
            weights = sess.run(logits, feed_dict={X: test_X, batch_size: test_b_s})
            print(len(weights), len(weights[0]), len(weights[0][0]), len(weights[0:(num_steps-1)]))
            for w in weights[0:(num_steps-1)]:
                test_predicts.append(w[0])

        weights = sess.run(last_logit, feed_dict={X: test_X, batch_size: test_b_s})
        for w in weights:
            test_predicts.append(w)
        if step % 10 == 0:
            print('predicting {}...'.format(step))

print(len(train_predicts))
print(len(valid_predicts))
print(len(test_predicts))

train_predicts_df = pd.DataFrame(data=train_predicts, index=data.train_X.index, columns=data.train_y.columns)
valid_predicts_df = pd.DataFrame(data=valid_predicts, index=data.valid_X.index, columns=data.valid_y.columns)
test_predicts_df = pd.DataFrame(data=test_predicts, index=data.test_X.index, columns=data.test_y.columns)
train_predicts_df.to_csv('..Data/Model2/predict_weights/train_predict_weights_{}_mode={}.csv'.format(model_name, mode))
valid_predicts_df.to_csv('..Data/Model2/predict_weights/valid_predict_weights_{}_mode={}.csv'.format(model_name, mode))
test_predicts_df.to_csv('..Data/Model2/predict_weights/test_predict_weights_{}_mode={}.csv'.format(model_name, mode))

