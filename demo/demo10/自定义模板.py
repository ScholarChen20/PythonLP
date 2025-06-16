import math    #定义数学类型模板
print(id(math))
print(type(math))
print(math)
print(math.pi)
print('—————-------—————')
print(dir(math))
print(math.pow(2,4),type(math.pow(2,4)))
print(math.ceil(9.0001))
print(math.floor(9.9999))



print('——————————————————————')
from math import pow
from math import pi
print(pow(2,9))
print(pi)

def add(a,b):
    return a+b
def div(a,b):
    return a/b
