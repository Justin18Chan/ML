# -*- coding: UTF-8 -*-
from __future__ import unicode_literals

"""
词性标注
jieba.posseg.POSTokenizer(tokenizer=None) 新建自定义分词器，tokenizer 参数可指定内部使用的 jieba.Tokenizer 分词器。jieba.posseg.dt 为默认词性标注分词器。
标注句子分词后每个词的词性，采用和 ictclas 兼容的标记法。
具体的词性对照表参见计算所汉语词性标记集.
"""

import jieba.posseg as pseg
words = pseg.cut("我爱自然语言处理")
# for word, flag in words:
    # print('%s %s' % (word, flag))

"""
并行分词
原理：将目标文本按行分隔后，把各行文本分配到多个 Python 进程并行分词，然后归并结果，从而获得分词速度的可观提升. 
    基于python自带的 multiprocessing 模块，目前暂不支持 Windows.

用法：
jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数
jieba.disable_parallel() # 关闭并行分词模式
实验结果：在 4 核 3.4GHz Linux 机器上，对金庸全集进行精确分词，获得了 1MB/s 的速度，是单进程版的 3.3 倍。
注意：并行分词仅支持默认分词器 jieba.dt 和 jieba.posseg.dt。
"""
# import sys
# import time
# import jieba

# jieba.enable_parallel() # windows不支持
# content = open(r"C:\\Users\\Justin\\Documents\\NLP\\西游记.txt",'rb').read()
# t1 = time.time()
# words = "/ ".join(jieba.cut(content))
# t2 = time.time()
# tm_cost = t2-t1
# print('并行分词速度为 %s bytes/second' % (len(content)/tm_cost))

# jieba.disable_parallel()
# content = open(r"C:\\Users\\Justin\\Documents\\NLP\\西游记.txt",'rb').read()
# t1 = time.time()
# words = "/ ".join(jieba.cut(content))
# t2 = time.time()
# tm_cost = t2-t1
# print('非并行分词速度为 %s bytes/second' % (len(content)/tm_cost))

"""=============================================================================="""

# Tokenize：返回词语在原文的起止位置
# 注意，输入参数只接受 unicode
# import jieba
# print("这是默认模式的tokenize")
# result = jieba.tokenize(u'自然语言处理真的很自然,非常有用')
# for tk in result:
#     print("%s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))

# print("\n-----------我是神奇的分割线------------\n")

# print("这是搜索模式的tokenize")
# result = jieba.tokenize(u'自然语言处理真的很自然,非常有用', mode='search')
# for tk in result:
#     print("%s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))

"""================================================================================"""

"""
ChineseAnalyzer for Whoosh 搜索引擎
from jieba.analyse import ChineseAnalyzer

Whoosh搜索引擎使用的基本步骤:
1. 添加索引
生成schema, schema有两个field,一个是title,一个是content.
用关键词参数来映射 filed name 与field type，这些名字与类型将定义在索引的对象以及可搜索的对象
from whoosh.fields import Schema, STORED, ID, KEYWORD, TEXT
schema  = Schema(title=TEXT(stored=True), content=TEXT(stored=True), path=ID(stored=True), tags=KEYWORD(stored=True), icon=STORED)
2. 创建索引index:
（create_in函数，生成index文件夹，里面包含二进制文件）
import os.path
from whoosh.index import create_in
if not os.path.exists("index"):
    os.mkdir("index")
ix = create_in("index", schema)

3. 编辑和删除索引
delete_document(docnum)方法 (docnum是索引查询结果的每条结果记录的docnum属性)
delete_by_term(field_name, termtext)方法  (特别适合删除ID,或KEYWORD字段)
delete_by_query(query)

添加操作使用writer的以下方法:
add_document
update_document
e.g.
writer = ix.writer()
writer.add_document(title=u"My document", content=u"This is my document!",
                    path=u"/a",tags=u"first short", icon=u"/icons/star.png")
writer.commit()

4. 查询索引
创建search对象:
searcher = ix.searcher()
用完一定要关闭, searcher.close()
生成查询对象,有三种方式:
构建query对象:
    from whoosh.query import *
    myquery = And([Term("content",u"apple"),Term("content","bear")])
构建查询分析器:
    from whoosh.qparser import QueryParser
    parser = QueryParser("content", ix.schema)
    myquery = parser.parse(querystring)
以query对象为参数调用searcher的search方法, 得到查询result:
    默认的search方法的results最多仅返回10个匹配的文档,若要得到全部的结果,可把limit=None
    result = searcher.search(query, limit=20)
    results = searcher.find方法:
    search_str = "document"
    results = searcher.find("content", search_str)
    print(results[0])
"""


import jieba.analyse as analyse
import sys,os
sys.path.append("../")
from whoosh.index import create_in,open_dir
from whoosh.fields import Schema, STORED, ID, KEYWORD, TEXT
from whoosh.qparser import QueryParser

# 添加索引
analyzer = analyse.ChineseAnalyzer()
schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT(stored=True, analyzer=analyzer))
    
if not os.path.exists("tmp"):
    os.mkdir("tmp")
# 创建索引
ix = create_in("tmp", schema) # for create new index
#ix = open_dir("tmp") # for read only
writer = ix.writer()
# 添加索引
writer.add_document(
    title="document1",
    path="/a",
    content="This is the first document we’ve added!"
)

writer.add_document(
    title="document2",
    path="/b",
    content="The second one 你 中文测试中文 is even more interesting! 吃水果"
)

writer.add_document(
    title="document3",
    path="/c",
    content="买水果然后来世博园。"
)

writer.add_document(
    title="document4",
    path="/c",
    content="工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"
)

writer.add_document(
    title="document4",
    path="/c",
    content="咱俩交换一下吧。"
)

writer.commit()

# 查询索引
searcher = ix.searcher()
parser = QueryParser("content", schema=ix.schema)

for keyword in ("水果世博园","你","first","中文","交换机","交换"):
    print("["+keyword+"]"+"的结果为如下：")
    q = parser.parse(keyword)
    results = searcher.search(q)
    for hit in results:
        print(hit.highlights("content"))
    print("\n--------------我是神奇的分割线--------------\n")

for t in analyzer("我的好朋友是李明;我爱北京天安门;IBM和Microsoft; I have a dream. this is intetesting and interested me a lot"):
    print(t.text)
    
