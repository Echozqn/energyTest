import os
from Common import Constant
from Common import publicFunction

publicFunction.remove(Constant.CSV_FILE_NAME)
model_dataset = ['resnet50_cifar10','albert_obqa','NeuMF_Movielens-1M', 'vgg19_cifar10', 'bert_sentiment140']
batches = [256, 32,512, 128, 64]
publicFunction.remove(Constant.CSV_FILE_NAME)
GPU = "T4"
for i in range(0, len(model_dataset)):
    batch = batches[i]
    file = model_dataset[i]
    file_name = f"{GPU}_{file}_{batch}_{batch}.log"
    cmd = f"python3 {file}.py {batch} {batch} {file_name} 3 {GPU}"
    # os.system(cmd)
    print(cmd)
