#encoding:utf-8
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# from sklearn.naive_bayes import BernoulliNB

"""
语种检测分类器
"""
class LanguageDetector():

    def __init__(self, classifier=MultinomialNB()):
        """
        使用朴素贝叶斯多项式分布分类器MultinomialNB
        文本数据特征矢量化器CountVectorizer
        """
        self.classifier = classifier
        self.vectorizer = CountVectorizer(ngram_range=(1,2), max_features=2000, preprocessor=self._remove_noise)

    def _remove_noise(self, document):
        """
        去噪
        """
        noise_pattern = re.compile("|".join(["http\S+", "\@\w+", "\#\w+"]))
        clean_text = re.sub(noise_pattern, "", document) # re.sub()正则匹配成功后用空字符替换
        return clean_text

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)

# 测试数据data.csv包含English, French, German, Spanish, Italian 和 Dutch 6种语言
in_f = open(r'C:\study\机器学习与人工智能\NPL\NLP\课件资料\Lecture_2\Lecture_2\Language-Detector\data.csv','rb')
lines = in_f.readlines()
in_f.close()
dataset = [(line.strip()[:-3], line.strip()[-2:]) for line in lines]
# 使用zip(*)将元组数据分隔开
x, y = zip(*dataset)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

language_detector = LanguageDetector()
language_detector.fit(x_train, y_train)
print(language_detector.predict('This is an English sentence'))
print(language_detector.score(x_test, y_test))
