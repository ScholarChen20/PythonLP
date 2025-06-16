def fun(a, b=10):
    c = a + b
    d = a * b
    return c, d


s = fun(100)
m = fun(10, 20)
print('————情况1————')
print(s)
print('————情况2————')
print(m)
