class Student:
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
stu1=Student('卓伟攀',21)
stu1.eat()
print(stu1.name)
print(stu1.age)