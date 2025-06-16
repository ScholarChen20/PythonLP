for i in range(3):
    for j in range(1, 11):
        if j % 2 == 0:
            break
        print(j,end='\t')
    print()


for i in range(3):
    for j in range(1, 11):
        if j % 2 == 0:
            ##break
            continue
        print(j,end='\t')
    print()
