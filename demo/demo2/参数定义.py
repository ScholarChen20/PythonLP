def fun(*args):
     print(args)
fun(10,20,30)

def dun(**args):
     print(args)
dun(a=10,b=20)

print('————————————')
def sun(a,b,c):
     print('a=',a)
     print('b=',b)
     print('c=',c)
sun(10,20,30)
lst=[10,20,30]
sun(*lst)
print('————————————')
asg={'a':10,'b':20,'c':30}
sun(**asg)