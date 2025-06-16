class Student:
    pass
print(Student)
print(type(Student))


print('------1-------')
class Teacher:
    teacher_place='福建'
    #初始化方法
    def __init__(self,name,age):
        self.name=name
        self.age=age
    #实例方法
    def eat(self):
        print('老师在吃饭')
    #静态方法
    @staticmethod
    def drink():
        print('老师在喝水')
    #类方法
    @classmethod
    def cm(cls):
        print('老师在干嘛？')

print('---------函数---------')
def fun():
    print('你好帅！')
fun()

class Facuity:
    place='福建中医药大学人文与管理学院'
    date='1958'
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def show(self):
        print(self.name+'请前往'+self.place+'报道')
fcu1=Facuity('陈伟',20)
fcu1.show()