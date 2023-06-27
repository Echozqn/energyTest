import json

def getdata(pathContext, pathLabel):  # 获取某一个数据集的数据
        context = []
        label = []
        for i,line in enumerate(open(pathContext)):
            f = json.loads(line)
            
            f["context"] = " ".join(f["context"].replace("\r"," ").replace("\n"," ").split())
            f["question"] = " ".join(f["question"].replace("\r"," ").replace("\n"," ").split())
            f["answerA"] = " ".join(f["answerA"].replace("\r"," ").replace("\n"," ").split())
            f["answerB"] = " ".join(f["answerB"].replace("\r"," ").replace("\n"," ").split())
            f["answerC"] = " ".join(f["answerC"].replace("\r"," ").replace("\n"," ").split())
            context.append(f)
        return context, label
type = "dev"
pathContext = type+'.jsonl'
pathLabel = type+'-labels.lst'
context, label = getdata(pathContext,pathLabel)
with open(type+'.src', 'w',encoding="utf-8") as ff:
    for i in context:
        ff.write(i["context"]+i["question"]+'\n')

# with open("new_"+type+'.jsonl', 'w',encoding="utf-8") as ff:
#     for i in context:
#         ff.write(json.dumps(i)+'\n')
