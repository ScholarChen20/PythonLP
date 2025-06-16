a=20
b=100
c=a+b
print(c)

d=a.__add__(b)   #类似于a与b想加
print(d)

class Student:
    def __init__(self,name):
        self.name=name
    def __add__(self, other):
        return self.name+other.name
    def __len__(self):
        return len(self.name)

stu1=Student('Jack')
stu2=Student('李四')
s=stu1+stu2
print(s)

s1=stu1.__add__(stu2)
print(s1)


lst=[11,22,33,44]
print(len(lst))
print(lst.__len__())
print(len(stu1))
