# coding: utf-8
import os
import time
import random
import jieba
import nltk 
import sklearn
from sklearn.naive_bayes import MultinomialNB
# from sklearn.naive_bayes import BernoulliNB
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt


def make_word_set(word_file):
    """生成停用词集合"""
    words_set = set()
    with open(word_file, 'rb') as fp:
        for line in fp.readlines():
            word = line.strip().decode('utf-8')
            if len(word) > 0 and word not in words_set:
                words_set.add(word)
    return words_set

def text_processing(folder_path, test_size=0.2):
    """文本处理"""
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    # 遍历文件夹
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)
        # 读取文件
        j = 1
        for file in files:
            if j > 100:
                break
            with open(os.path.join(new_folder_path, file), 'rb') as fp:
                raw = fp.read()
            # jieba.enable_parallel(4) # 开启并行分词,windows系统不支持
            word_cut = jieba.cut(raw, cut_all=False)
            word_list = list(word_cut)
            # jieba.disable_parallel() # 关闭并行分词,windows系统不支持
            data_list.append(word_list)
            class_list.append(folder)
            j += 1

    # 粗暴地划分训练集和测试集
    data_class_list = list(zip(data_list, class_list))
    # 注意, python3中zip返回的是一个zip可迭代对象,没有len(), 且使用list等迭代一遍后指针后移,就无法取到数据值.
    random.shuffle(data_class_list)
    index = int(len(data_class_list)*test_size) + 1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)
    # 或者使用sklearn自带模块划分
    # train_data_list, test_data_list, train_class_list, test_class_list =  sklearn.cross_validation.train_test_split(data_list, class_list, test_size=test_size)

    # 统计词频,放入all_words_list
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            # if all_words_dict.has_key(word): python3已经没有dict.has_key('key'),可以使用dict.__contains__()
            if all_words_dict.__contains__(word):
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
                
    # key函数利用词频进行降序排列
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f:f[1], reverse=True)
    # print(len(all_words_tuple_list)) #9989
    # print(all_words_tuple_list[0]) # ('，', 3630)
    all_words_list = list(zip(*all_words_tuple_list))[0]
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list

def words_dict(all_words_list, deleteN, stopwords_set=set()):
    # 选取特征词
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1<len(all_words_list[t])<5:
            feature_words.append(all_words_list[t])
            n += 1
    return feature_words

def text_features(train_data_list, test_data_list, feature_words, flag='nltk'):
    # 本文特征
    def text_features(text, feature_words):
        text_words = set(text)
        ####-------------------------------------------------------------------
        if flag == 'nltk':
            ### nltk 特征dict
            features = {word: 1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            ### sklearn 特征list
            features = [1 if word in text_words else 0 for word in feature_words]
        else:
            features = []
        ####-------------------------------------------------------------------
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list,test_feature_list

def text_classifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag='nltk'):
    """分类,并输出准确率"""
    ####-------------------------------------------------------------------------
    if flag == 'nltk':
        ### 使用nltk分类器
        train_flist = list(zip(train_feature_list, train_class_list))
        test_flist = list(zip(test_feature_list, test_class_list))
        # classifier = nltk.NaiveBayesClassifier.train(train_flist)
        classifier = nltk.classify.NaiveBayesClassifier.train(train_flist)
        test_accuracy = nltk.classify.accuracy(classifier, test_flist)
    elif flag == 'sklearn':
        ### sklearn 分类器
        classifier = MultinomialNB().fit(train_feature_list,train_class_list)
        test_accuracy = classifier.score(test_feature_list, test_class_list)
    else:
        test_accuracy = []
    return test_accuracy


print("======================开始处理=======================")

#### 文本预处理
folder_path = r'C:\study\机器学习与人工智能\NPL\NLP\课件资料\Lecture_2\Lecture_2语言模型到朴素贝叶斯\Naive-Bayes-Text-Classifier\Database\SogouC\Sample'
all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = text_processing(folder_path, test_size=0.2)

# 生成停用词集合
stopwords_file = r'C:\study\机器学习与人工智能\NPL\NLP\课件资料\Lecture_2\Lecture_2语言模型到朴素贝叶斯\Naive-Bayes-Text-Classifier\stopwords_cn.txt'
stopwords_set = make_word_set(stopwords_file)
## 文本特征提取和分类
# flag = 'nltk'
flag = 'sklearn'
deleteNs = range(0, 1000, 20) # 分段去除低维度特征
test_accuracy_list = []
for deleteN in deleteNs:
    # feature_words = words_dict(all_words_list, deleteN)
    feature_words = words_dict(all_words_list, deleteN, stopwords_set)
    train_feature_list, test_feature_list = text_features(train_data_list, test_data_list, feature_words, flag)
    test_accuracy = text_classifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag)
    test_accuracy_list.append(test_accuracy)
print(test_accuracy_list)

#  结果评价
plt.figure()
plt.plot(deleteNs, test_accuracy_list)
plt.title('Relationship of deleteNs and test_accuracy')
plt.xlabel('deleteNs')
plt.ylabel('test_accuracy')
plt.show()
plt.savefig('result.png')

print("finished")

