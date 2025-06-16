s = 'hello,world'
print('1:', s.isidentifier())
print('2:', '$'.isidentifier())
print('-—————空白字符—————')
print('3:', '   '.isspace())
print('-————字母———')
print('4:', 'hello,world'.isalpha())
print('张三'.isalpha())

print('---十进制数字------')
print('135'.isdecimal())
print('10010'.isdecimal())
print('213df'.isdecimal())

print('-—————数字——————')
print('12329'.isnumeric())
print('sas12'.isnumeric())
print('Ⅱ'.isnumeric())

print('-———————字母和数字—————')
print('123hjds'.isalnum())
print('张三12'.isalnum())
