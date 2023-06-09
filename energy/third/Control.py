import os
import time

from Common import Constant
from Common import publicFunction

model_datasets = ['resnet50_cifar10','albert_obqa','NeuMF_Movielens-1M', 'vgg19_cifar10', 'bert_sentiment140']
batches = [512, 16, 1024, 128, 64]
publicFunction.writeCSV(Constant.CSV_FILE_NAME,["frequency","model",'dataset',"batch_size","device_batch_size","energy","exec_time"])
for i in range(len(model_datasets)):
    model_dataset = model_datasets[i]
    batch = batches[i]
    for fre in range(800, 1601,100):
        lock_fre = f"nvidia-smi -lgc {fre}"
        print(lock_fre)
        # os.system(lock_fre)

        dmon_name = f"./dmon/dmon_{fre}_{model_dataset}_{batch}_{batch}.log"
        dmon_cmd = f"nvidia-smi dmon -f {dmon_name} &"
        print(dmon_cmd)
        # os.system(dmon_cmd)


        # time.sleep(5)
        file_name = f"{fre}_{model_dataset}_{batch}_{batch}.log"
        cmd = f"python3 {model_dataset}.py {batch} {batch} {file_name} 3 {fre}"
        print(cmd)
        # os.system(cmd)

        # time.sleep(5)
        kill_cmd = "pgrep -f \"nvidia-smi dmon -f\" | xargs kill"
        print(kill_cmd)
        # os.system(kill_cmd)

"""
python3 resnet50_cifar10.py 512 512 A100_resnet50_cifar10_512_512.log 3 A100


nvidia-smi dmon -f ./dmon/dmon_noLockFrequency_resnet50_cifar10_512_512.log &
python3 resnet50_cifar10.py 512 512 noLockFrequency_resnet50_cifar10_512_512.log 3 noLockFrequency
pgrep -f "nvidia-smi dmon -f" | xargs kill
"""