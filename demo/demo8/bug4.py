try:
    a = int (input ('第一个整数是:'))
    b = int (input ('第二个整数是:'))
    result = a / b
    print ('结果是：',result)
except ZeroDivisionError:
    print('输入数字不能为0！')
except ValueError:
    print('不能输入字符串！')
print("程序结束！")



