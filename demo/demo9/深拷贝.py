class CPU:
    pass
class Disk:
    pass
class Computer:
    def __init__(self,cpu,disk):
        self.cpu=cpu
        self.disk=disk
cpu1=CPU()
disk=Disk()
computer=Computer(cpu1,disk)
import copy
computer2=copy.deepcopy(computer)
print(computer,computer.cpu,computer.disk)
print(computer2,computer2.cpu,computer2.disk)