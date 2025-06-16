with open ('a.txt','r') as file:
    print(file.readline())


class MyContentMgr(object):
    def __enter__(self):
        print(' 1')
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(' 2')
    def show(self):
        print(' 3')
with MyContentMgr() as file:
    file.show()