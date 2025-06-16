class Disk:
    pass
class Cpu:
    pass
class Computer:
    def __init__(self,cpu,disk):
        self.cpu=cpu
        self.disk=disk
cpu1=Cpu()
cpu2=cpu1
print(cpu1,id(cpu1))
print(cpu2,id(cpu2))

disk=Disk()
computer=Computer(cpu1,disk)
print('——————————-------')
import copy
computer2=copy.copy(computer)
print(computer,computer.cpu,computer.disk)
print(computer2,computer2.cpu,computer2.disk)
