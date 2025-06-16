class Student:
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def __str__(self):
        return '姓名：{0},今年{1}岁'.format(self.name,self.age)

stu=Student('张三',20)
print(id(stu))
print(stu)
