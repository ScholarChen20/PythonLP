age=input('请输入你的年龄：')
print(type(age))
if int(age)>=18:
    print('是成年人,年龄为{0}'.format(age))
else:
    print('未成年，年龄为{0}'.format(age))

name='张三'
print('姓名是%s'%name)
print('姓名是{0}'.format(name))