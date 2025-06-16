import numpy as np
a=np.array([1,2,3,4,5],ndmin=2)
b=np.array([1,2,3,4,5],dtype=complex)
# print(b)

student=np.dtype([('name','S20'),('age','i1'),('marks','i2')])
# print(student)
a=np.array([('abc',21,50),('xysz',18,74.0)],dtype=student)
# print(a)

a=np.arange(24)
b=a.reshape(2,4,3)
# print(a.ndim)
# print(b.ndim)

c=np.array([[1,2,3],[4,5,6]])
# print(c.shape)
# c.shape=(3,2)
# print(c)

x=np.ones([2,2],dtype=int)
# print(x)
arr=np.array([[1,2,3],[4,5,6],[7,8,9]])
zeros_arr=np.zeros_like(arr)
ones_arr=np.ones_like(arr)
# print(zeros_arr)
# print(ones_arr)

s=b'hello world'
a=np.frombuffer(s,dtype='S1')
# print(a)
list=range(5)
it=iter(list)

x=np.fromiter(it,dtype=float)
# print(x)
x1=np.arange(10,20,2)
# print(x1)

x2=np.linspace(1,10,10)
# print(x2)

a=np.arange(10)
s=slice(2,7,2)
print(a[s])

print(a[2:5])
print(a[2:])