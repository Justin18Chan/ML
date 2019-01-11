from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

# 数组创建为10维二进制向量:
trainimgs = mnist.train.images
trainlabels = mnist.train.labels
testimgs = mnist.test.images
testlabels = mnist.test.labels
print(trainimgs.shape) # # (55000, 784)
# images数据第一维表示数据量, 第二维784表示图片28*28,标签10表示10个分类.
ntrain = trainimgs.shape[0] # (55000, 784)[0]训练集样本数
ntest = testimgs.shape[0] # (55000, 10)[0]测试集样本数
dim = trainimgs.shape[1] # (10000, 784)[1]训练集每张图片像素点个数
nclasses = trainlabels.shape[1] # (10000, 10)[1]样本分类个数

# 查看MNIST数据集结构
samplesIdx = [100, 101, 102] # 查看第100,101,102个样本

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(testimgs[samplesIdx[0]].reshape([28,28]), cmap='gray')
xx, yy = np.meshgrid(np.linspace(0,28,28), np.linspace(0,28,28))
X = xx
Y = yy
Z = 100*np.ones(X.shape)

img = testimgs[samplesIdx[0]].reshape([28,28])
ax = fig.add_subplot(122, projection='3d')
ax.set_zlim(0, 200)

offset = 200
for i in samplesIdx:
    img = testimgs[i].reshape([28,28]).transpose()
    ax.contourf(X, Y, img, 200, zdir='Z', offset=offset, cmp='gay')
    offset -= 100

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# 使plt.show()函数不阻塞
plt.ion()
plt.show()
# plt暂停显示2s
plt.pause(2)

for i in samplesIdx:
    print("Sample: {0} - Class: {1} - Label Vector: {2}".format(i, np.nonzero(testlabels[i])[0], testlabels[i]))


# 设置RNN超参数,形成的RNN组成如下:
#  一个可以将28*28维输入转化成128维隐藏层的输入层
# 中间循环神经网络LSTM
# 一个可以将128维的LSTM输出转化为10维代表类标签的输出层.
n_input = 28 #
n_steps = 28
n_hidden = 128
n_classes = 10

learning_rate = 0.001
trainning_iters = 100000
batch_size = 100
display_step = 10000

# 构建递归网络, 设置权重偏差
# 当前数据输入形状为: (batch_size, n_steps, n_input)[100x28x28]
x = tf.placeholder(dtype='float', shape=[None, n_steps, n_input], name='x')
y = tf.placeholder(dtype='float', shape=[None, n_classes], name='y')
weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
biases = {'out':tf.Variable(tf.random_normal([n_classes]))}

# 也可以对神经网络进行堆叠,将不同神经网络放进list中,其它步骤一样,如下
# cells = []
# for _ in range(num_layers):
#     cell = tf.contrib.rnn.LSTMCell(LSTM_CELL_SIZE)
#     cells.append(cell)
# stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells)
# outputs, state = tf.nn.dynamic_rnn(cells, inputs=x, dtype=tf.float32)

# 本例子只定义一个lstm单元
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

# 创建一个动态lstm循环神经网络
outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs=x, dtype=tf.float32)

# 设置RNN输出矩阵
# RNN输出的将是一个[100,28,128]矩阵,我们使用线性激活将其映射到[?,10]矩阵中
# 使用tf.split(axis, value, num_split, name=None)根据axis维度将一个张量切割成num_solit个张量.
output = tf.reshape(tf.split(outputs, 28, axis=1, num=None, name='split')[-1],[-1,128])
pred = tf.matmul(output, weights['out']) + biases['out']

# 定义成本函数, 优化器, 设置准确性和评估方法
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 开始训练数据
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 创建tf.train.Saver()用来保存训练好的神经网络,
    saver = tf.train.Saver()
    sess.run(init)
    step = 1 
    # 保持循环, 直到最大迭代次数
    while step*batch_size < trainning_iters:
        # 以batch_x 的形式读取一批100*784图像
        #  batch_y 是[100*10]标签矩阵
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # 我们将图像的每一行视为一个序列
        # 重塑数据以获得28个元素的28个序列, 因此, batch_x是[100*28*28]
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))

        # 运行优化操作
        sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})

        if step % display_step == 0:
            # 计算批次精度
            acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
            # 计算批量损失
            loss = sess.run(cost, feed_dict={x:batch_x, y:batch_y})
            print("Iter" + str(step*batch_size) + ", Minibatch loss=" + \
            "{:.6f}".format(loss) + ", Trainning Accuracy=" + \
            "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # 计算128个mnist测试图像的准确性
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy: ", sess.run(accuracy, feed_dict={x:test_data, y:test_label}))

    # 开始保存训练模型
    save_path = saver.save(sess, "./RNN/SLTM.ckpt")
    print("Save to path:", save_path)

sess.close()


# 提取保存的神经网络数据,注意貌似不能在同个py中即存储右提取.
# 先建立临时的W,bias容器
# W = tf.Variable(tf.random_normal([128, 10]), dtype=tf.float32)
# b = tf.Variable(tf.random_normal([10]), dtype=tf.float32)

# # 注意不需要初始化变量

# saver = tf.train.Saver()
# with tf.Session() as sess:
#     # 提取变量
#     saver.restore(sess,  "./RNN/SLTM.ckpt")
#     print("weights", sess.run(W))
#     print("biases", sess.run(b))
