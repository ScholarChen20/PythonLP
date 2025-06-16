######列表的增加操作
lst=[10,20,30,40]
lst.append(100)
lst.append('hello')
print(lst)
arr=['a','ddd']
lst.extend(arr)
print(lst)

arr.insert(1,'Hello')
print(arr)

lst[1:]=arr
print(lst)