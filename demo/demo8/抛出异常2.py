try:
    n1=int(input('输入第一个整数:'))
    n2=int(input('输入第二个整数:'))
    result=n1/n2
except BaseException as e:
    print('出错了',e)
else:
    print('结果是:',result)
finally:
    print('谢谢你的使用！')