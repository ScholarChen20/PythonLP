import os.path
print(os.path.abspath('clac.py'))      #��ȡ�ļ�·��
print(os.path.exists('c.txt'),os.path.exists('clac.py'))  #�ж��ļ��Ƿ����
print(os.path.join('E:\\python','c.txt'))  #�����ļ�������չ��
print(os.path.split('demo11.py'))  #�����ļ�������չ��
print(os.path.basename('D:\\Python Project\\demo10\\clac.py'))  #Ŀ¼����ȡ�ļ���
print(os.path.dirname('D:\\Python Project\\demo10\\clac.py'))   #·������ȡ�ļ�·��
print(os.path.isdir('D:\\Python Project\\demo11\\demo1py'))    #�ж��Ƿ���·��