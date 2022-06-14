"""
testing if python object passes by reference
"""
import os, psutil
import numpy as np
class Foo:
    def __init__(self, val):
        self.val = val

class A:
    def __init__(self, foo):
        self.foo = foo
class B:
    def __init__(self, foo):
        self.a = A(foo)
        self.foo = foo

foo = Foo(np.abs(np.random.rand(10000)))
print('after foo: ')
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

a = A(foo)
print('after a: ')
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

b = B(foo)
print('after b: ')
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

# before changing the value
print('before changing values:')
print('a.foo sum: ')
print(a.foo.val.sum())
print('b.foo sum: ')
print(b.foo.val.sum())
print('b.a.foo sum: ')
print(b.a.foo.val.sum())

# check whether changing the value of foo, a and b will be changed
foo.val = np.zeros(100000)
print('a.foo sum: ')
print(a.foo.val.sum())
print('b.foo sum: ')
print(b.foo.val.sum())
print('b.a.foo sum: ')
print(b.a.foo.val.sum())
