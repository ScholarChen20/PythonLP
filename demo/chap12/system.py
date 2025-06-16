#coding=gbk
import os
filename='student.txt'

"""�˵�"""
def menu():
    print ('=======================20�Ź�ѧ���ɼ�����ϵͳ=========================')
    print ('----------------------------����ģ��----------------------------')
    print ('\t\t\t\t\t\t1.¼��ѧ����Ϣ')
    print ('\t\t\t\t\t\t2.����ѧ����Ϣ')
    print ('\t\t\t\t\t\t3.ɾ��ѧ����Ϣ')
    print ('\t\t\t\t\t\t4.�޸�ѧ����Ϣ')
    print ('\t\t\t\t\t\t5.����')
    print ('\t\t\t\t\t\t6.ͳ��ѧ��������')
    print ('\t\t\t\t\t\t7.��ʾ����ѧ����Ϣ')
    print ('\t\t\t\t\t\t0.�˳�ϵͳ')
    print ('-------------------------------------------------------------')

"""������"""
def main ():
    while True:
        menu()
        choice=int(input('��ѡ��:'))
        if choice in [0,1,2,3,4,5,6,7]:
            if choice==0:
                answer = input ('��ȷ��Ҫ�˳�ϵͳ?y/n:')
                if answer == 'y' or answer == 'Y':
                    print ('лл���ʹ��')
                    break
                else:
                    continue
            elif choice==1:
                insert()
            elif choice==2:
                search ()
            elif choice==3:
                delete()
            elif choice==4:
                modify()
            elif choice==5:
                sort()
            elif choice==6:
                total()
            elif choice==7:
                show()
        else:
            print('����ȷ�������֣�')

"""����ѧ����Ϣ���ļ�"""
def save(lst):
    try:
        stu_txt=open(filename,'a',encoding='utf-8')
    except:
        stu_txt=open(filename,'w',encoding='utf-8')
    for item in lst:
        stu_txt.write(str(item)+'\n')
    stu_txt.close()

"""����ѧ����Ϣ"""
def insert():
    stu_lst = []
    while True:
        no=int(input('������ѧ�ţ�����1001��:'))
        if not no:
            break
        name=input('����������:')
        if not name:
            break
        try:
            c = int (input ('������C���Գɼ�:'))
            python = int (input ('������Python�ĳɼ�:'))
            java = int (input ('������Java�ĳɼ�:'))
        except:
            print('������Ч�����������룡')
            continue
        student={'id':no,'name':name,'C����':c,'Python':python,'Java':java}
        stu_lst.append(student)
        save (stu_lst)
        print ('��Ϣ¼��ɹ���')
        stu_lst.clear()
        choice=input('�Ƿ������y/n:')
        if choice=='y' or choice=='Y':
            continue
        else:
            break

"""����ѧ����Ϣ������ѧ��d����������ѯ��"""
def search():
    search_qurry=[]
    while True:
        id=''
        name=''
        if os.path.exists(filename):
            choice = int (input ('Id��ѯ�밴1�����ֲ�ѯ�밴2:'))
            if choice == 1:
                id=int(input('������ѧ��id:'))
            elif choice==2:
                name=input('������ѧ������:')
            else:
                print('���������������룡')
                search()
            with open(filename,'r',encoding='utf-8') as file:
                student_lst=file.readlines()
                for item in student_lst:
                    d = dict (eval (item))
                    if id!='':
                        if d['id']==id:
                            search_qurry.append(d)
                    elif name!='':
                        if d['name']==name:
                            search_qurry.append(d)
            show_student(search_qurry)
            search_qurry.clear()
        a=input('�Ƿ�������ң�y/n:')
        if a=='y':
            continue
        else:
            break

"""ѧϰ�ɼ���Ϣ�����ʽ�����"""
def show_student(lst):
    if len(lst)==0:
        print('�б����޴�ѧ������Ϣ')
        return
    student_title='{:^6}\t{:^12}\t{:^8}\t{:^10}\t{:^10}\t{:^8}'
    print(student_title.format('ID','����','C���Գɼ�','Python�ɼ�','Java�ɼ�','����'))
    student_data='{:^6}\t{:^12}\t{:^8}\t{:^8}\t{:^8}\t{:^8}' # ��ʽ�����
    for item in lst:
        print(student_data.format(item.get('id'),item.get('name'), item.get('C����'),item.get('Python')
                                  ,item.get('Java'),round((int(item.get('C����'))+int(item.get('Python'))+int(item.get('Java'))-180)/30,2)))
    print ('\n')

"""ɾ��ѧ����Ϣ"""
def delete():
    while True:
        student_id=int(input('������ѧ����id:'))
        if student_id:
            if os.path.exists(filename):
                with open(filename,'r',encoding='utf-8') as file:
                    student_old = file.readlines()
            else:
                student_old=[]
            flag=False
            if student_old:
                with open(filename,'w',encoding='utf-8') as files:
                    for item in student_old:
                        d = dict (eval (item))
                        if d['id']!=student_id:
                            files.write(str(d)+'\n')
                        else:
                            flag=True
                    if flag:
                        print (f'ѧ��Ϊ{student_id}��ѧ����Ϣ��ɾ����')
                    else:
                        print (f'û���ҵ�idΪ{student_id}��ѧ����Ϣ')
            else:
                print('��ѧ����¼')
                break
            show()
            choice = input ('�Ƿ������y/n:')
            if choice == 'y':
                continue
            else:
                break

"""��ʾ����ѧ����Ϣ"""
def show():
    student_lst=[]
    if os.path.exists(filename):
        with open(filename,'r',encoding='utf-8') as file:
            student=file.readlines()
            for item in student:
                student_lst.append(eval(item))
            if student_lst:
                show_student(student_lst)
    else:
        print('��δ����ѧ�����ݣ�')

"""�޸�ѧ����Ϣ"""
def modify():
    show()
    if os.path.exists(filename):
        with open(filename,'r',encoding='utf-8') as file:
            student_lst=file.readlines()
    else:
        return
    student_id=int(input('������ѧ��id:'))
    with open(filename,'w',encoding='utf-8') as file1:
        for item in student_lst:
            d = dict (eval (item))
            if d ['id'] == student_id:
                print(f'�Ѿ��ҵ�idΪ{student_id}��ѧ��')
                while True:
                    try:
                        d ['name'] = input ('������ѧ������:')
                        d ['C����'] = int (input ('������C���Գɼ�:'))
                        d ['Python'] = int (input ('������Python�ĳɼ�:'))
                        d ['Java'] = int (input ('������Java�ĳɼ�:'))
                    except:
                        print('�������Ϣ�����������룡��')
                    else:
                        break
                file1.write(str(d)+'\n')
                print('�޸���Ϣ�ɹ���������������')
            else:
                file1.write(str(d)+'\n')
        switch = input ('�Ƿ�Ҫ�޸���Ϣ��y/n:')
        if switch == 'y':
            modify()

"""�����������"""
def sort():
   show()
   if os.path.exists(filename):
       with open(filename,'r',encoding='utf-8')as file:
           student_lst=file.readlines()
       student_new=[]
       for item in student_lst:
           d=dict(eval(item))
           student_new.append(d)
   else:
       return
   switch=input('����ʽ��0.����1.����')
   if switch=='0':
       switch_bool=False
   elif switch=='1':
       switch_bool=True
   else:
       print('�������')
       sort()
   choice=input('��ѡ������ʽ��1.C���Գɼ���2.Python�ɼ���3.Java�ɼ���4.���㣩')
   if choice=='1':
       student_new.sort(key=lambda x:int(x['C����']),reverse=switch_bool)
   elif choice=='2':
       student_new.sort (key=lambda x: int (x ['Python']), reverse=switch_bool)
   elif choice=='3':
       student_new.sort (key=lambda x: int (x ['Java']), reverse=switch_bool)
   elif choice=='4':
       student_new.sort (key=lambda x: round((int (x ['C����'])+int (x ['Python'])+int (x ['Java'])),2), reverse=switch_bool)
   else:
       print('ѡ�������������')
       sort()
   show_student(student_new)

"""ͳ��ѧ��������"""
def total():
    if os.path.exists(filename):
        with open(filename,'r',encoding='utf-8') as file:
            students=file.readlines()
            if students:
                print('ϵͳ����{}��ѧ��'.format(len(students)))
            else:
                print('ϵͳ����ѧ����¼��')
    else:
        print('��δ����ѧ����Ϣ��')

if __name__ == '__main__':
    main()

