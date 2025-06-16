import sys
import math
import urllib.request
import time
print('-----1-----')
print(sys.path)
print(sys.getsizeof(12))
print(sys.getsizeof(24))
print('-----2-----')
print(time.time())
print(time.struct_time)
print(time.localtime(time.time()))
print('-----3-----')
print(urllib.request.urlopen('http://www.baidu.com').read())
print('-----4------')
print(math.pi)
