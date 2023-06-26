import os
import time

from Common import Constant
from Common import publicFunction

publicFunction.remove(Constant.CSV_FILE_NAME)
model_dataset = 'resnet50_cifar10'
batch = 512
publicFunction.remove(Constant.CSV_FILE_NAME)
for fre in range(900, 1601,100):
    lock_fre = f"nvidia-smi -lgc {fre}"
    print(lock_fre)
    os.system(lock_fre)

    dmon_name = f"./dmon/dmon_{fre}_{model_dataset}_{batch}_{batch}.log"
    dmon_cmd = f"nvidia-smi dmon -f {dmon_name} &"
    print(dmon_cmd)
    os.system(dmon_cmd)
    time.sleep(5)
    file_name = f"{fre}_{model_dataset}_{batch}_{batch}.log"
    cmd = f"python3 {model_dataset}.py {batch} {batch} {file_name} 3 {fre}"
    os.system(cmd)
    print(cmd)

    kill_cmd = "pgrep -f \"nvidia-smi dmon -f\" | xargs kill"
    print(kill_cmd)
    os.system(kill_cmd)
