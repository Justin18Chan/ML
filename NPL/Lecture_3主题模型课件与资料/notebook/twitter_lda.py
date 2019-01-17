import gensim 
from gensim import corpora, models, similarities
import pandas as pd
import numpy as np
import re

docs_list = []
with open(r'C:\Users\Justin\Documents\tmp\twitter.txt') as fp:
    for line in fp.readlines():
        if line.strip():
            docs_list.append(line)
docs = pd.DataFrame(docs_list)
# print(docs.values)

# 文本预处理
def clean_text(text):
    # 注意传入的text必须是str类型.
    text = text.replace('\n', ' ') # 新行是不需要的
    text = re.sub(r'-', ' ', text) # 把 "-" 的两个单词，分开。（比如：july-edu ==> july edu）
    text = re.sub(r'[\?|&|,|.|\'|:|!|’]', ' ', text) # 把标点符号都用空格替代
    text = re.sub(r'[0-2]?[0-9]:[0-6]?[0-9]:?[0-6]?[0-9]?', '', text) # 时间,没有意义
    pure_text = ''
    for letter in text:
        if letter.isalpha() or letter == ' ':
            pure_text += letter
    text = ' '.join(word for word in pure_text.split() if len(word) > 1)
    return text
docs = docs[0]
docs = docs.apply(lambda doc:clean_text(doc))
# print(docs)

# 这里直接写一个Stop words table
# 这些词在不同语境中指代意义完全不同，但是在不同主题中的出现概率是几乎一致的。所以要去除，否则对模型的准确性有影响.
stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours', 
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their', 
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once', 
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you', 
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will', 
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be', 
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself', 
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both', 
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn', 
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about', 
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn', 
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']

word_list = [[word for word in doc.lower().split() if word not in stoplist] for doc in docs]

# 生成语料对象
dictionary = corpora.Dictionary(word_list)
wordbag = [dictionary.doc2bow(word) for word in word_list]
# print(dictionary.token2id)

# 构建lda模型
lda = gensim.models.ldamodel.LdaModel(corpus=wordbag, id2word=dictionary, num_topics=5)
print(lda.print_topic(1, topn=5)) # 打印第N个主题的top N个关键字
print("=========================================")
print(lda.print_topics(num_topics=3,num_words=5)) # 打印前N个主题,并按顺序排列
