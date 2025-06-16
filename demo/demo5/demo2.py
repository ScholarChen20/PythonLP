#集合间的关系

s={1,2,3,4,5,6,7,8}
s1={1,2,3,4,5,6,7}
s2={1,2,3,4}
print(s==s1)
print(s1!=s2)

print('——————-子集关系————————')
print(s1.issubset(s))
print(s1.issubset(s2))
print(s.issubset(s2))
print('——————-超集关系————————')
print(s.issuperset(s1))
print(s1.issuperset(s))
print(s2.issuperset(s1))
print('——————-交集关系————————')
print(s.isdisjoint(s1))
print(s.isdisjoint(s2))
print(s1.isdisjoint(s1))