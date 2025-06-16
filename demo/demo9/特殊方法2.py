class Person(object):
    def __init__(self,name,age):
        print('init被调用执行了，self的id值为：{0}'.format(id(self)))
        self.name=name
        self.age=age
    def __new__(cls, *args, **kwargs):
        print('new被调用执行了，cls的id值为：{0}'.format(id(cls)))
        obj=super().__new__(cls)
        print('创建的对象的值为：{0}'.format(id(obj)))
        return obj

print('object这个类的id值为：{0}'.format(id(object)))
print('Person这个类的id值为：{0}'.format(id(Person)))


p1=Person('张三',20)
print('p1这个的实例对象的id值为：{0}'.format(id(p1)))