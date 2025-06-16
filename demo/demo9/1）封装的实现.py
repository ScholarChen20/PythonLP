class Student:
    def __init__(self,name,age):
        self.name=name
        self.__age=age
    def show(self):
        print(self.name,',',self.__age)
stu=Student('张三',22)
stu.show()
#在类外使用name与age属性
print(stu.name)
#print(stu.age)
#print(stu.Student.__age)                 #在类之外可以使实例对象的属性显现出来