import numpy as np

# x=np.array([[1,2],[3,4],[5,6]])
# y=x[[0,1,2],[0,1,0]]
# print(y)

x=np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
# print("原始数组：\n",x)
raw=np.array([[0,0],[3,3]])
column=np.array([[0,2],[0,2]])

# y=x[raw,column]
# print(y)
#可以借助切片 : 或 … 与索引数组组合
b=x[1:3,1:3]
c=x[1:3,[1,2]]
d=x[...,1:]
# print(b)
# print(c)
# print(d)

# print(x[x>5])

a=np.array([1,2.0+6j,5,3.5+5j])
# print(a[np.iscomplex(a)])
t=np.arange(32).reshape((8,4))
print(t)
print('\n')
print(t[[4,2,1,7]])
print('\n')
print(t[[-4,-2,-1,-7]])
# np.ix_ 笛卡儿积，两个集合各元素相乘
print('\n')
print(t[np.ix_([1,2,3,4],[0,3,1,2])])