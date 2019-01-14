"""
NLP 自然语言处理基础-----> 字符串常规操作
"""

# str.strip[[chars]用于移除字符串str中头尾指定的字符(默认为空格或者换行符)或字节序列.
# 注意 这个方法只能用户删除头尾的字符,不能删除中间部分的字符.
s = " hello, world! "
print(s.strip())
print(s.lstrip(' hello, '))
print(s.rstrip('! '))

# 连接字符串,+
sStr1 = 'strcat'
sStr2 = 'append'
sStr1 += sStr2
print(sStr1)

# 查找字符 < 0 未找到, str.index('strx')返回查找到的索引位置
sStr1 = "strchr"
sStr2 = 'r'
nPos = sStr1.index(sStr2)
print(nPos)

# lower/upper()字符串的大小写切换
sStr1 = 'JCstrlwr'
sStr1 = sStr1.upper()
#sStr1 = sStr1.lower()
print(sStr1)

# 切片方式翻转字符串
sStr1 = 'abcdefg'
sStr1 = sStr1[::-1]
print(sStr1)

# 查找字符串str.find(strx), 返回找到的字符串第一个元素出现位置
sStr1 = 'abcdefg'
sStr2 = 'cde'
print(sStr1.find(sStr2))

# 分割字符串
sStr1 = 'ab,cde,fgh,ijk'
sStr2 = ','
sStr1 = sStr1[sStr1.find(sStr2) + 1:]
print(sStr1)
#或者str.split(sep)
s = 'ab,cde,fgh,ijk'
print(s.split(','))

import re
from collections import Counter
# 统计字符串中出现频次最多的字母
# version1
def get_max_value_v1(text):
    text = text.lower()
    result = re.findall('[a-zA-Z]', text) # 去掉列表中的符号符
    count = Counter(result) # Counter({'l':3, 'o':2, 'd':1, 'h':1, 'r':1, 'e':1, 'w':1})
    count_list = list(count.values())
    max_value = max(count_list) # 出现次数最多字母的次数
    max_list = []
    for k, v in count.items():
        if v == max_value: # 匹配出现次数最多的字母
            max_list.append(k)
    max_list = sorted(max_list)
    return max_list[0]

v1 = get_max_value_v1('jfaolifjalfnkfaljfs')
print(v1)

# version2
def get_max_value_v2(text):
    count = Counter([x for x in text.lower() if x.isalpha()])
    m = max(count.values())
    return sorted([key for (key, value) in count.items() if value == m])[0]

print(get_max_value_v2('jfaolifjalfnkfaljfs'))

# version3
import string
def get_max_value_v3(text):
    text = text.lower()
    return max(string.ascii_lowercase, key=text.count)

print(get_max_value_v3('jfaolifjalfnkfaljfs'))

# 带入key函数中, 各个元素返回布尔值, 相当于[False, False, False, True, True, True]
# key函数要求返回值为True, 有多个符合的值, 则挑选第一个,不是最大的一个.
v = max(range(6), key = lambda x : x>2)
print(v)

# 带入key函数中, 各个元素返回自身的值, 最大的值为5, 返回5.
v = max([3,5,2,1,4,3,0], key = lambda x : x)
print(v)

# 带入key函数,各个字符串返回最后一个字符, 其中'ah'的h要大于'bf'中的f,因此返回'ah'
v = max('ah', 'bf', key=lambda x:x[-1])
print(v)

# 带入key函数，各个字符串返回第一个字符，其中'bf'的b要大于'ah'中的a，因此返回'bf'
v = max('ah', 'bf', key=lambda x: x[0])
print(v)

# 带入key函数,返回各个字符在'Hello World'中出现的次数, 出现次数最多的字符为'l',因此输出'l'
text = 'Hello World!'
v = max('abcdefghijklmnopqrstuvwxyz', key=text.count)
print(v)

# 统计一个字符串中字符出现的次数,输出出现最多字母的次数
sentence = 'The Mississippi River'
def count_chars(s):
    s = s.lower()
    count = list(map(s.count, s))
    return (max(count))
print(count_chars(sentence))
