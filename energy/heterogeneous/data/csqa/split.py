import json

def getdata(Context):  # 获取某一个数据集的数据
        context = []
        src = []
        tgt = []
        for i,line in enumerate(open(Context+'.jsonl')):
            f = json.loads(line)
            context.append(f)
        for i,line in enumerate(open(Context+'.src')):
            src.append(line[:-1])
        for i,line in enumerate(open(Context+'.tgt')):
            tgt.append(line[:-1])
        return context, src, tgt

qids = []
for line in open('inhouse_split_qids.txt'):
    qids.append(line[:-1])
context, src, tgt = getdata("train")
train_context = []
train_src = []
train_tgt = []

test_context = []
test_src = []
test_tgt = []

for i in range(len(context)):
    if context[i]["id"] in qids:
        train_context.append(context[i])
        train_src.append(src[i])
        train_tgt.append(tgt[i])
    else:
        test_context.append(context[i])
        test_src.append(src[i])
        test_tgt.append(tgt[i])

with open('new_train.jsonl', 'w',encoding="utf-8") as ff:
    for i in train_context:
        ff.write(json.dumps(i)+'\n')
with open('new_train.src', 'w',encoding="utf-8") as ff:
    for i in train_src:
        ff.write(i+'\n')
with open('new_train.tgt', 'w',encoding="utf-8") as ff:
    for i in train_tgt:
        ff.write(i+'\n')

with open('new_test.jsonl', 'w',encoding="utf-8") as ff:
    for i in test_context:
        ff.write(json.dumps(i)+'\n')
with open('new_test.src', 'w',encoding="utf-8") as ff:
    for i in test_src:
        ff.write(i+'\n')
with open('new_test.tgt', 'w',encoding="utf-8") as ff:
    for i in test_tgt:
        ff.write(i+'\n')

