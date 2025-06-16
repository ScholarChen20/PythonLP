##计算1~100之间的偶数和
print('——————————方法1——————————')
a=1
sum=0
while a<=100:
    if not bool(a%2):
        sum+=a
    a+=1
print('1~100之间的偶数和是：',sum)


print('——————————方法2—————————')
a = 1
sum = 0
while a <= 100:
    if a % 2==0:
        sum += a
    a += 1
print('1~100之间的偶数和是：', sum)
