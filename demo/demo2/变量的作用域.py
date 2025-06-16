def add(a,b):
    c=a+b
    print(c)
add(1,2)

print('—————作用域1———————')
name='张三'
def fun():
    print(name)
fun()
print('—————作用域2———————')
def dun():
    global age          #当函数内部变量定义时，局部变量使用global定义时，这个变量实际上变成了全局变量
    age=20
    print(age)
dun()
print('age=',age)