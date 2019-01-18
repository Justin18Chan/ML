# coding:utf-8

import argparse
import sys
import os
import io
import importlib
import time
import numpy as np
# 包含了dict,list,tuple,set以外的一些特殊容器类型,如:
# OrderedDict 排序字典---字典的子类
# Counter 哈希对象家计数器---字典的子类
# deque 双向队列
# 等
import collections
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.legacy_seq2seq as seq2seq

importlib.reload(sys)
# sys.setdefaultencoding('utf-8')
tf.reset_default_graph()  

# 设置超参数
BEGIN_CHAR = '^'
END_CHAR = '$'
UNKNOWN_CHAR = '*'
MAX_LENGTH = 100
MIN_LENGTH = 10
max_words = 3000   # 最大单词个数
epochs = 5
poetry_file = 'poetry.txt'
save_dir = 'log'


class Data:
    """数据类,用于数据提取与处理"""
    def __init__(self):
        self.batch_size = 64
        self.poetry_file = poetry_file
        self.load()
        self.create_batches()

    def load(self):
        def handle(line):
            """处理每一行字串,保证是句号结尾或者MAX_LENGTH长度."""
            if len(line) > MAX_LENGTH:
                index_end = line.rfind('。', 0, MAX_LENGTH)
                index_end = index_end if index_end > 0 else MAX_LENGTH
                line = line[:index_end + 1]
            return BEGIN_CHAR + line + END_CHAR

        self.poetrys = [line.strip().replace(' ', '').split(':')[1] for line in
                        io.open(self.poetry_file, encoding='utf-8')]
        self.poetrys = [handle(line) for line in self.poetrys if len(line) > MIN_LENGTH]
        # 所有字
        words = [] # 定义列表words用来存储每个词
        for poetry in self.poetrys:
            words += [word for word in poetry]
        counter = collections.Counter(words) # 字典计数器, key是单词名,value是出现次数,注意key是不重复的.
        count_pairs = sorted(counter.items(), key=lambda x: -x[1]) # counter.items()返回的是key_value对元组, 按照-x[1]即出现次数的降序排列.返回排序后的列表数组.
        words, _ = zip(*count_pairs) #把键值对解压

        # 取出现频率最高的词的数量组成字典，不在字典中的字用'*'代替
        words_size = min(max_words, len(words))
        self.words = words[:words_size] + (UNKNOWN_CHAR,) # words最后一个词是*即不在字典中的词.
        self.words_size = len(self.words)

        # 字映射成id
        self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}
        self.unknow_char = self.char2id_dict.get(UNKNOWN_CHAR)
        self.char2id = lambda char: self.char2id_dict.get(char, self.unknow_char)
        self.id2char = lambda num: self.id2char_dict.get(num)
        self.poetrys = sorted(self.poetrys, key=lambda line: len(line))
        self.poetrys_vector = [list(map(self.char2id, poetry)) for poetry in self.poetrys]

    def create_batches(self):
        """创建数据批次
        poetrys_vector: 数据输入向量
        batch_size: 每个批次数据大小
        n_size: 生成批次数量
        """
        self.n_size = len(self.poetrys_vector) // self.batch_size
        self.poetrys_vector = self.poetrys_vector[:self.n_size * self.batch_size] # 取整
        self.x_batches = []
        self.y_batches = []
        for i in range(self.n_size):
            batches = self.poetrys_vector[i * self.batch_size: (i + 1) * self.batch_size]
            length = max(map(len, batches))
            for row in range(self.batch_size):
                if len(batches[row]) < length:
                    r = length - len(batches[row])
                    batches[row][len(batches[row]): length] = [self.unknow_char] * r
            xdata = np.array(batches)
            ydata = np.copy(xdata)
            ydata[:, :-1] = xdata[:, 1:]
            self.x_batches.append(xdata)
            self.y_batches.append(ydata)


class Model:
    """模型类
    lstm建立模型
    data : 原始数据
    model: 模型算法(lstm,rnn,gru),默认lstm, 
    infer: 是否批次训练
    """
    def __init__(self, data, model='lstm', infer=False):
        self.rnn_size = 128 # 隐层神经元个数
        self.n_layers = 2 # 隐层数

        if infer:
            self.batch_size = 1
        else:
            self.batch_size = data.batch_size

        if model == 'rnn':
            cell_rnn = rnn.BasicRNNCell
        elif model == 'gru':
            cell_rnn = rnn.GRUCell
        elif model == 'lstm':
            cell_rnn = rnn.BasicLSTMCell

        cell = cell_rnn(self.rnn_size, state_is_tuple=False)
        self.cell = rnn.MultiRNNCell([cell] * self.n_layers, state_is_tuple=False) # 多隐层RNN

        self.x_tf = tf.placeholder(tf.int32, [self.batch_size, None])
        self.y_tf = tf.placeholder(tf.int32, [self.batch_size, None])

        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'): # 设置神经层网络共享变量作用域
            softmax_w = tf.get_variable("softmax_w", [self.rnn_size, data.words_size]) #返回给定名称的变量
            softmax_b = tf.get_variable("softmax_b", [data.words_size])
            with tf.device("/cpu:0"): # 指定cpu运行, cpu不区分设备号,设0即可. gpu区分设备号'/gpu:0' 和'/gpu:0'表示两张不同的显卡.
                embedding = tf.get_variable("embedding", [data.words_size, self.rnn_size])
                inputs = tf.nn.embedding_lookup(embedding, self.x_tf) # tf.nn.embedding_lookup查找张量embedding,对应的索引self.x_tf

        """
        静态rnn必须提前将图展开,执行时是固定长度,并且最大长度有限制.
        总之, 能用动态就用动态.
        """
        outputs, final_state = tf.nn.dynamic_rnn(
            self.cell, inputs, initial_state=self.initial_state, scope='rnnlm')

        self.output = tf.reshape(outputs, [-1, self.rnn_size])
        self.logits = tf.matmul(self.output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        self.final_state = final_state
        pred = tf.reshape(self.y_tf, [-1])
        # seq2seq 权重交叉熵损失值
        loss = seq2seq.sequence_loss_by_example([self.logits],
                                                [pred],
                                                [tf.ones_like(pred, dtype=tf.float32)],)

        self.cost = tf.reduce_mean(loss)
        self.learning_rate = tf.Variable(0.0, trainable=False)
        # 返回需要训练的变量列表
        tvars = tf.trainable_variables()
        # tf.all_variables() # 返回所有变量的列表
        # tf.clip_by_global_norm求导过程中,在突然变得特别陡峭的求导函数中,加上一些判断,如果过于陡峭就适当减小求导步伐.
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5) 

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


def train(data, model):
    """训练数据类"""
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # model_file = tf.train.latest_checkpoint(save_dir) # 会报路径错误
        # saver.restore(sess, model_file) # 会报路径错误,移到with末尾就没报了

        n = 0
        for epoch in range(epochs): #轮次epochs==5
            # tf.assign(A,B) 把A的值更新为B
            sess.run(tf.assign(model.learning_rate, 0.002 * (0.97 ** epoch))) # 通过这个函数可以动态修改学习率
            pointer = 0
            for batche in range(data.n_size):
                n += 1
                feed_dict = {model.x_tf: data.x_batches[pointer], model.y_tf: data.y_batches[pointer]}
                pointer += 1
                train_loss, _, _ = sess.run([model.cost, model.final_state, model.train_op], feed_dict=feed_dict)
                sys.stdout.write('\r')
                info = "{}/{} (epoch {}) | train_loss {:.3f}" \
                    .format(epoch * data.n_size + batche,
                            epochs * data.n_size, epoch, train_loss)
                sys.stdout.write(info)
                sys.stdout.flush()
                # save
                if (epoch * data.n_size + batche) % 1000 == 0 or (epoch == epochs-1 and batche == data.n_size-1):
                    checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=n)
                    sys.stdout.write('\n')
                    print("model saved to {}".format(checkpoint_path))
            sys.stdout.write('\n')
        model_file = tf.train.latest_checkpoint(save_dir)
        saver.restore(sess, model_file)

def sample(data, model, head=u''):
    def to_word(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sa = int(np.searchsorted(t, np.random.rand(1) * s))
        return data.id2char(sa)

    for word in head:
        if word not in data.words:
            return u'{} 不在字典中'.format(word)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        model_file = tf.train.latest_checkpoint(save_dir)
        # print(model_file)
        saver.restore(sess, model_file)

        if head:
            print('生成藏头诗 ---> ', head)
            poem = BEGIN_CHAR
            for head_word in head:
                poem += head_word
                x = np.array([list(map(data.char2id, poem))])
                state = sess.run(model.cell.zero_state(1, tf.float32))
                feed_dict = {model.x_tf: x, model.initial_state: state}
                [probs, state] = sess.run([model.probs, model.final_state], feed_dict)
                word = to_word(probs[-1])
                while word != u'，' and word != u'。':
                    poem += word
                    x = np.zeros((1, 1))
                    x[0, 0] = data.char2id(word)
                    [probs, state] = sess.run([model.probs, model.final_state],
                                              {model.x_tf: x, model.initial_state: state})
                    word = to_word(probs[-1])
                poem += word
            return poem[1:]
        else:
            poem = ''
            head = BEGIN_CHAR
            x = np.array([list(map(data.char2id, head))])
            state = sess.run(model.cell.zero_state(1, tf.float32))
            feed_dict = {model.x_tf: x, model.initial_state: state}
            [probs, state] = sess.run([model.probs, model.final_state], feed_dict)
            word = to_word(probs[-1])
            while word != END_CHAR:
                poem += word
                x = np.zeros((1, 1))
                x[0, 0] = data.char2id(word)
                [probs, state] = sess.run([model.probs, model.final_state],
                                          {model.x_tf: x, model.initial_state: state})
                word = to_word(probs[-1])
            return poem


def main():
    msg = """
            Usage:
            Training: 
                python poetry_gen.py --mode train # 执行此语句训练数据
            Sampling:
                python poetry_gen.py --mode sample --head 明月别枝惊鹊 # 执行此语句测试数据
            """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='sample',
                        help=u'usage: train or sample, sample is default')
#     parser.add_argument('--mode', type=str, default='train', help=u'usage: train or sample, sample is default')
    parser.add_argument('--head', type=str, default='', help='生成藏头诗')

    args = parser.parse_args()

    if args.mode == 'sample':
        infer = True  # True
        data = Data()
        model = Model(data=data, infer=infer)
        print(sample(data, model, head=args.head))
    elif args.mode == 'train':
        infer = False
        data = Data()
        model = Model(data=data, infer=infer)
        print(train(data, model))
    else:
        print(msg)


if __name__ == '__main__':
    main()
