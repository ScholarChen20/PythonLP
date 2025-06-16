#集合的相关操作
s={1,2,3,4,5,6}
print(s)
print(3 in s)
print(30 in s)
print(2 not in s)


print('---———————添加--————————')
s.add(45)
print(s)
s.update({10,20,30})
print(s)
s.update([80,90])
print(s)
print('--————————删除————————')

s.remove(1)
print(s)
s.discard(2)
print(s)
s.pop()
print(s)
s.clear()
print(s)