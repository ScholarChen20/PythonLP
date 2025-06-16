#
print('-—————字符串的替换———————')
s='Hello,Python'
s1='hello,Python,Python,Python'
print(s.replace('Python','Java'))
print(s1.replace('Python','Java',2))

print('-——————字符串的合并——————')
lst=['hello','world','python']
print(''.join(lst))
print('&'.join(lst))
lst1=('hello','python')
print('|'.join(lst1))
print('*'.join('Java'))