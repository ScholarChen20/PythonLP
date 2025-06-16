file=open('a.txt', 'r')
print(file.readline())
file.close()

file=open('b.txt', 'w')
print(file.write('我爱Python'))
file.close()

file=open('b.txt', 'a')
print(file.write('美丽中国'))
file.close()

src_file=open('panno.png', 'rb')
target_file=open('1.png', 'wb')
target_file.write(src_file.read())
src_file.close()
target_file.close()