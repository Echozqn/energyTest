import os
from Common import Constant
from Common import publicFunction

publicFunction.remove(Constant.CSV_FILE_NAME)
model_dataset = 'resnet50_cifar10'
# batches = [64,128,256,512]
batches = [128]
""""
python3 resnet50_cifar10.py 128 128 resnet50_cifar10_128_1100_no.log 3 4
"""
publicFunction.remove(Constant.CSV_FILE_NAME)
for batch in batches:
    for fre in range(1100,1101,100):
        for percent in range(40,101,10):
            # lgc
            lock_fre = f"sudo nvidia-smi -lgc {fre} -i 4"
            print(lock_fre)
            # os.system(lock_fre)

            # set percent
            set_percent = f"echo set_active_thread_percentage 1625198 {percent} | nvidia-cuda-mps-control"
            print(set_percent)
            # os.system(set_percent)

            file = model_dataset
            file_name = f"{file}_{batch}_{fre}_{percent}.log"
            cmd = f"python3 {file}.py {batch} {batch} {file_name} 3 4"
            # os.system(cmd)
            print(cmd)

