#
s='Hello World Python'
print('—————————————居中———————————————')
print(s.center(30,'_'))
print('——————————————左对齐——————————————')
print(s.ljust(20,'*'))
print('——————————————右对齐—————————————')
print(s.rjust(20,'*'))
print('——————————————右对齐，0填充—————————————')
print(s.zfill(30))
