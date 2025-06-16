def fun(num):
    odd = []
    even = []
    for i in num:
        if i % 2:
            odd.append(i)
        else:
            even.append(i)
    return odd, even


lst = [1, 2, 3, 4, 5, 6, 7, 8, 13, 15, 17, 23, 100]
print(fun(lst))
'''
1）:如果函数没有返回值，return可以省略不写
2）:函数的返回值如果是一个，则返回其类型
3）：函数的返回值如果是多个，返回的结果为元组。
'''
