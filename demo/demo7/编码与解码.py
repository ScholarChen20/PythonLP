str = '天涯共此时'
print(str.encode(encoding='GBK'))
print('——————————————————————————')
byte = str.encode(encoding='GBK')
print(byte.decode(encoding='GBK'))
