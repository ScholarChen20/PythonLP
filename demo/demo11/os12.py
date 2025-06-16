import os.path
print(os.path.abspath('clac.py'))      #提取文件路径
print(os.path.exists('c.txt'),os.path.exists('clac.py'))  #判断文件是否存在
print(os.path.join('E:\\python','c.txt'))  #联接文件名和扩展名
print(os.path.split('demo11.py'))  #分离文件名和扩展名
print(os.path.basename('D:\\Python Project\\demo10\\clac.py'))  #目录中提取文件名
print(os.path.dirname('D:\\Python Project\\demo10\\clac.py'))   #路径中提取文件路径
print(os.path.isdir('D:\\Python Project\\demo11\\demo1py'))    #判断是否有路径