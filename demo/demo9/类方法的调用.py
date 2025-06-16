class Student:
    native_place='福建'
    #初始化方法
    def __init__(self,name,age):
        self.name=name
        self.age=age
    #实例方法
    def eat(self):
        print('吃饭')
    #静态方法
    @staticmethod
    def drink():
        print('老师在喝水,静态方法！')
    #类方法
    @classmethod
    def cm(cls):
        print('老师在干嘛？类方法！')
stu1=Student('张三',20)
stu2=Student('李四',33)
print(stu1.native_place)
print(stu2.native_place)
print('-----------修改后----------')
Student.native_place='广东'
print(stu1.native_place)
print(stu2.native_place)

print('——————————————静态方法调用——————————————')
stu1.drink()
print('——————————————类方法调用——————————————')
stu1.cm()