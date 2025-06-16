from typing import TextIO

print(520)
print(98.5)
##
print('Hello World')
print("Hello World")
##
print(3 + 6)

fp = open('D:/text.txt', 'a+')
print('Hello World', file=fp)
fp.close()

print('Hello', 'World', 'Python')
