def show(lst1,lst2):
    print('lst1',lst1)
    print('lst2',lst2)
    lst1=100
    lst2.append(100)
    print('lst1', lst1)
    print('lst2', lst2)
    return lst1,lst2
print('---函数调用前--')
a=11
b=[10,20,30]
print('a',a)
print('b',b)
show(a,b)

print('————————调用函数后————————')
print('a',a)
print('b',b)
