class Student:
    native_place='福建'
    #初始化方法
    def __init__(self,name,age):
        self.name=name
        self.age=age
    #实例方法
    def eat(self):
        print(self.name+'在吃饭')
stu1=Student('张三',20)
stu2=Student('李四',33)
stu1.eat()
stu2.eat()
print('————————————动态绑定属性——————————————')
stu1.gender='女'
stu1.phone='10086'
print(stu1.name,stu1.age,stu1.gender,stu1.phone)
print(stu2.name,stu2.age)
print('————————————动态绑定方法————————————————')
def show():
    print('定义在类之外，是函数')
stu1.show=show
stu1.show()
#stu2.show()    这语句无法执行，因为类中没有实例函数show()