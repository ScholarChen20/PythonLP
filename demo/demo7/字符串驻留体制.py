#字符串的驻留体制
import sys

s='hello'
s1='world'
s2='python'
print(s,id(s))
print(s1,id(s1))
print(s2,id(s2))

print('——————————————————————————————')
a='nihao'
b='nihao'
print(a is b)
print(id(a))
print(id(b))
print('——————————————————————————————')
c='nihao'
a=sys.intern(c)
print(a,id(a))
print(c,id(c))
