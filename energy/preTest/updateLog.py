import os
import re


def gao(file):
    print(file)
    all = file.split('.')[-2].split('_')[-2]
    per = file.split('.')[-2].split('_')[-1]
    with open(file,"r",encoding="utf-8") as f:
        data = f.readlines()
        for i in range(len(data)):
            if data[i].count("Loss: "):

                matches = data[i].split("Loss: ")[-1].strip('\n')
                gen = float(matches) * (int(all)//int(per))
                data[i]=data[i].replace(matches,str(gen))

    with open(file,"w",encoding="utf-8") as f:
        f.writelines(data)


for root, dirs, files in os.walk('./log'):
    # 遍历当前目录下的所有文件
    for file in files:
        # 检查文件扩展名是否为JSON
        if file.endswith('.log'):
            gao('./log/'+file)
