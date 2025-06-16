lst=[70,20,80,90,60]
print('排序前列表：',lst)
#开始排序，调用列表对象sort方法
lst.sort()
print('排序后列表：',lst)

print('------降序排列-----')
lst.sort(reverse=True)
print(lst)
print('------升序排列-----')
lst.sort(reverse=False)
print(lst)

print('——————————————使用内置函数sorted排序，产生一个新的列表对象——————————————————————')
lat=[100,40,33,45,68,24]
new_list=sorted(lat)
print(lat)
print(new_list)

desc_list=sorted(lat,reverse=True)
print(desc_list)