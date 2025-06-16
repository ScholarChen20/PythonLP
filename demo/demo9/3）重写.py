class Person(object):
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def show(self):
        print('姓名:{0},年龄:{1}'.format(self.name,self.age))

class Student(Person):
    def __init__(self,name,age,stu_no):
        super().__init__(name, age)
        self.stu_no=stu_no
    def show(self):
        super().show()
        print('学号:{0}'.format(self.stu_no))

stu=Student('张三',20,'1001')
stu.show()