import numpy as np
import pandas as pd
import re

f = open(r"C:\study\机器学习与人工智能\NPL\NLP\课件资料\主题模型课件与资料\Lecture_3主题模型课件与资料\input\HillaryEmails.csv",'rb')
df = pd.read_csv(f) # pandas.read_csv()无法直接读取中文路径,可以先用open读取句柄
# 原邮件数据中有很多nan值, 直接扔掉.
# print(df.shape) #(7945, 22)
df = df[['Id','ExtractedBodyText']].dropna()
# print(df.shape) #(6742, 2)

# 文本预处理
def clean_email_text(text):
    text = text.replace('\n', ' ') # 新行是不需要的
    print('type====',type(text))
    text = re.sub(r'-', ' ', text) # 把 "-" 的两个单词，分开。（比如：july-edu ==> july edu）
    text = re.sub(r'\d+/\d+/\d+', '', text) #日期,对主体模型没有什么意义
    text = re.sub(r'[0-2]?[0-9]:[0-6]?[0-9]:?[0-6]?[0-9]?', '', text) # 时间,没有意义
    text = re.sub(r'[\w]+@[\w]+.[\w]+.+', '', text) #邮件地址,没意义
    # text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text) #网址，没意义
    text = re.sub(r'https?:.+', '', text) #网址,没有意义
    pure_text = ''
    # 以防还有其他特殊字符(数字)等等, 我们直接把text loop一遍,过滤掉.
    for letter in text:
        if letter.isalpha() or letter == ' ':
            pure_text += letter
    #  再把那些去除特殊字符后落单的单词,直接排除.
    # 我们就只剩下有意义的单词了.
    text = ' '.join(word for word in pure_text.split() if len(word) > 1)
    return text

docs = df['ExtractedBodyText']
docs = docs.apply(lambda s: clean_email_text(s))
# print(docs.head(1).values) # ['Thursday March PM Latest How Syria ....']
# 取出所有邮件内容
doclist = docs.values

# LDA模型构建
# 好，我们用Gensim来做一次模型构建
# 首先，我们得把我们刚刚整出来的一大波文本数据
# [[一条邮件字符串]，[另一条邮件字符串], ...]
# 转化成Gensim认可的语料库形式：
# [[一，条，邮件，在，这里],[第，二，条，邮件，在，这里],[今天，天气，肿么，样],...]
# 引入库：
from gensim import corpora, models, similarities
import gensim

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

# 汉语人工分词
# 这里，英文的分词，直接就是对着空白处分割就可以了。
# 中文的分词稍微复杂点儿，具体可以百度：CoreNLP, HaNLP, 结巴分词，等等
# 分词的意义在于，把我们的长长的字符串原文本，转化成有意义的小元素：
texts = [[word for word in doc.lower().split() if word not in stoplist] for doc in doclist]
# print(texts[0]) # ['thursday', 'march', 'pm', 'latest', 'syria',...]

#建立语料库
#Dictionary的用法可以查看http://blog.sina.com.cn/s/blog_6877dad30102xc7n.html
dictionary = corpora.Dictionary(texts) # 生成语料Dictionary对象, 
corpus = [dictionary.doc2bow(text) for text in texts] #Dictionary.doc2bow将文档转成词袋, 返回的词袋是词标签和出现次数的映射字典

# 建议模型
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)
print(lda.print_topic(1, topn=5)) # 查看第10号主题中的top5 
# print(lda.print_topics(num_topics=20, num_words=5)) # 打印全部主题

# 最后通过以下两种方法,可以把文本/单词,分类成其中主题的一个.
# 但同样需要经过文本预处理+词袋化.
# lda.get_document_topics(bow)
# lda.get_term_topics(word_id)
