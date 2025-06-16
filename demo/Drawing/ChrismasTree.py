import turtle as t
import random as r
screen=t.Screen()

screen.screensize(bg='black')
screen.setup(750,690)
circle = t.Turtle ()
circle.shape ('circle')
circle.color ('red')
circle.speed ('fastest')
circle.up ()

square = t.Turtle ()
square.shape ('square')
square.color ('green')
square.speed ('fastest')
square.up ()

circle.goto(0,280)
circle.stamp()

k=0
for i in range(1,13):
    y=i*30
    for j in range(i-k):
        x=30*j
        square.goto(x, -y+280)
        square.stamp()
        square.goto(-x,-y+280)
        square.stamp()

    if i%4==0:
            x=30*(j+1)
            circle.color('red')
            circle.goto(-x,-y+280)
            circle.stamp()
            circle.goto(x,-y+280)
            circle.stamp()
            k+=3

    if i%4==3:
            x=30*(j+1)
            circle.color('yellow')
            circle.goto(-x,-y+280)
            circle.stamp()
            circle.goto(x,-y+280)
            circle.stamp()

square.color('brown')
for i in range(13,17):
    y=30*i
    for j in range(3):
        x=30*j
        square.goto(x,-y+280)
        square.stamp()
        square.goto(-x,-y+280)
        square.stamp()

t.goto(-16,-300)
t.color("dark red","red")
t.write('Merry Chrismas',align="center",font=("Comic Sans MS",40,"bold"))


def drawsnow ():  # 定义画雪花的方法
    t.ht ()  # 隐藏笔头，ht=hideturtle
    t.pensize (2)  # 定义笔头大小
    for i in range (150):  # 画多少雪花
        t.pencolor ("white")  # 定义画笔颜色为白色，其实就是雪花为白色
        t.pu ()  # 提笔，pu=penup
        t.setx (r.randint (-350, 350))  # 定义x坐标，随机从-350到350之间选择
        t.sety (r.randint (-100, 350))  # 定义y坐标，注意雪花一般在地上不会落下，所以不会从太小的纵座轴开始
        t.pd ()  # 落笔，pd=pendown
        dens = 6  # 雪花瓣数设为6
        snowsize = r.randint (1, 10)  # 定义雪花大小
        for j in range (dens):  # 就是6，那就是画5次，也就是一个雪花五角星
            t.fd (int (snowsize))
            t.backward (int (snowsize))
            t.right (int(360/dens))

drawsnow ()
t.done()





