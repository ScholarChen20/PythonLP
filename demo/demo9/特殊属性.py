class A:
    pass
class B:
    pass
class C(A,B):
    def __init__(self,name,age):
        self.name=name
        self.age=age
class D(A):
    pass
x=C('Jack',20)   #x是C类型的实例对象
print(x.__dict__)  #实例对象的属性
print(C.__dict__)
print('——————————————————')
print(C.__class__) #输出对象的所有类
print(C.__bases__)   #输出对象的父类
print(C.__base__ )    #类的基类，一般指第一个父类
print(C.__mro__)    #输出C类的层次结构
print(A.__subclasses__())  #输出对象所属的类