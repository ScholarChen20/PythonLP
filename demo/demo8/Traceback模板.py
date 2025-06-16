import traceback
try:
    print('-------------------------')
    print(1/0)
except:
    traceback.print_exc()
