import json

def getdata(pathContext):  # 获取某一个数据集的数据
        context = []
        label = []
        for i,line in enumerate(open(pathContext)):
            f = json.loads(line)
            n = {}
            c = f["question"]["choices"]
            question = f["question"]["stem"]
            answer = f["answerKey"]
            n["question"] = " ".join(question.replace("\r"," ").replace("\n"," ").split())
            n["answerA"] = " ".join(c[0]["text"].replace("\r"," ").replace("\n"," ").split())
            n["answerB"] = " ".join(c[1]["text"].replace("\r"," ").replace("\n"," ").split())
            n["answerC"] = " ".join(c[2]["text"].replace("\r"," ").replace("\n"," ").split())
            n["answerD"] = " ".join(c[3]["text"].replace("\r"," ").replace("\n"," ").split())
            n["correct"] = answer
            context.append(n)
        return context, label
type = "test"
pathContext = type+'.jsonl'
context, label = getdata(pathContext)
with open(type+'.src', 'w',encoding="utf-8") as ff:
    for i in context:
        ff.write(i["question"]+'\n')

with open("new_"+type+'.jsonl', 'w',encoding="utf-8") as ff:
    for i in context:
        ff.write(json.dumps(i)+'\n')
