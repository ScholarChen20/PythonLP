file=open('a.txt', 'r')
print(file.readlines())
file.close()

file=open('a.txt', 'a')
lst=['java','python','c++']
print(file.writelines(lst))
file.close()

file=open('a.txt', 'r')
file.seek(2)
print(file.readline())
print(file.tell())
file.close()

file=open('c.txt', 'w')
print(file.write('Hello'))
file.flush()
print(file.write('World'))
file.close()