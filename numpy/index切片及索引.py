import numpy as np
x=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
# print('我们的数组是')
# print(x)
# print('\n')
# rows=np.array([[0,0],[3,3]])
# cols=np.array([[0,2],[2,0]])
# y=x[rows,cols]
# print(y)

b=np.array([[1,2,3],[4,5,6],[7,8,9]])
c=b[1:3,1:3]
d=b[1:3,[1,2]]
e=b[...,1:]
# print(c)
# print(d)
# print(e)

# x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])
# print ('我们的数组是：')
# print (x)
# print ('\n')
# # 现在我们会打印出大于 5 的元素
# print  ('大于 5 的元素是：')
# print (x[x >  5])
#
# y=np.arange(32).reshape(8,4)
# print (y)
x=np.array([[0,1,2],[3,4,5],[5,6,7]])
print(x[...,1])
print(x[1,...])
print(x[...,1:])