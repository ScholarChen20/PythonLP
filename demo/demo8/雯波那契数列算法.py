def fic(n):
    if n==1:
        return 1
    elif n==2:
        return 2
    else:
        return fic(n-1)+fic(n-2)

print('——————————————————')
print(fic(4))
print('—————————求这函数各项的值—————————')
for i in range(1,7):
    print(fic(i))
print ('—————————求这函数各项的值—————————')
for i in range(1,10):
    sum=fic(i)
print(sum)