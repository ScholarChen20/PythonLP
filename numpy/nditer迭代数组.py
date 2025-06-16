import numpy as np
import matplotlib.pyplot as plt

x=np.array([0,20,40,60])
y=np.array([100,200,300,400])
#
#
# plt.plot(x,y)
# plt.show()

a=np.arange(0,60,5)
a=a.reshape(3,4)
print('原始数组是:',a)
b=a.T
print("转至数组是:")
print(b)
print('\n')
print("c风格排序：")
c=b.copy(order='C')
print(c)
for x in np.nditer(c):
    print(x,end=",")
print('\n')
print("F顺序排列：")
c = b.copy(order = 'F')
print(c)
for x in np.nditer(c):
    print(x,end=',')
