###########字典的增删改查操作
scores={'陈文钦':100,'陈伟':30,'陈家隆':20}
print('陈文钦' in scores)
print('蓝晨' not in scores)

print('————————————————————————————')
del scores['陈伟']
print(scores)
scores['宸宸宸']=100
print(scores)
scores['宸宸宸']=98
print(scores)