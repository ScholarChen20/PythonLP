try:
    a=int(input('输入第一个整数:'))
    b=int(input('输入第二个整数:'))
    result=a/b
except BaseException as e:
    print('出错了')
    print(e)
else:
    print('结果是：',result)

