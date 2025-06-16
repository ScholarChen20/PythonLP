lst=[{'rating':[9.7,50],'type':['剧情','犯罪'],'title':'肖申克的救赎','actors':['蒂姆.罗兵斯','莫根.弗里曼']},
     {'rating':[9.6,50],'type':['剧情','爱情'],'title':'霸王别姬','actors':['张国荣','张丰毅','葛优']}]
name=input('输入电影演员：')
for item in lst:
    act_lst=item['actors']
    for actors in act_lst:
        if name in actors:
            print (name, '出演了', item ['title'])
