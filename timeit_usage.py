# -*- coding:utf-8 -*-
# Author:Justin Chan
# Python:3.6

import timeit

timeittimeitt = timeit.Timer('x=range(10000)')
print(t.timeit())

print(timeit.timeit('x=range(1000)'))

t2 = timeit.Timer('sum(x)','x = (i for i in range(1000))')
print(t2.timeit())

def test1(x):
    n = 0
    for i in range(x):
        n += i
    return n

def test2(x):
    return sum(range(x))

def test3(x):
    return sum(y for y in range(x))

if __name__ == '__main__':
    from timeit import Timer
    t1 = Timer('test1(1000)','from __main__ import test1')
    t2 = Timer('test2(1000)', 'from __main__ import test2')
    t3 = Timer('test3(1000)', 'from __main__ import test3')
    print(t1.timeit(100000))
    print(t2.timeit(100000))
    print(t3.timeit(100000))

