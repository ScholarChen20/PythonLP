import numpy as np
# 广播(Broadcast)是 numpy 对不同形状(shape)的数组进行数值计算的方式， 对数组的算术运算通常在相应的元素上进行
a=np.array([[0,0,0],
            [1,2,3],
            [4,5,6],
            [7,8,9]])
b=np.array([1,1,1])

c=a+b
d=np.tile(b,(4,1))
# d=np.dot(a,b)
print(c)
print('\n')
print(a+d)