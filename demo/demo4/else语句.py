###else语句
print ('________________方法1————————————————————')
for i in range (3):
    pwd = input ('输入密码：')
    if pwd == '8888':
        print ('密码正确！')
        break
    else:
        print ('密码错误，请再次输入')
else:
    print ('3次密码输入错误！')

a = 1
while a <= 3:
    pwd = input ('请输入密码：')
    if pwd == "1234":
        print ('密码正确')
        break
    else:
        print ('密码错误')
else:
    print ('三次输入错误！')
