def fac(n):
    if n==1:
        return 1
    else:
        return n*fac(n-1)

print(fac(4))
print(fac(6))