###########字典生成式
name=['陈文钦','陈伟','陈加龙']
print(name)
scores=[100,20,60,100,30]
print('——————————————————————————————————————')
d={name.upper():scores for name,scores in zip(name,scores)}
print(d)