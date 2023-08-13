import os
from Common import Constant
from Common import publicFunction

publicFunction.remove(Constant.CSV_FILE_NAME)
model_dataset = 'resnet50_cifar10'
batches = [64,128,256,512]
publicFunction.remove(Constant.CSV_FILE_NAME)
for batch in batches:
    for fre in range(800,1501,100):
        for percent in range(40,101,10):
            # lgc
            lock_fre = f"sudo nvidia-smi -lgc {fre} -i 4"
            print(lock_fre)
            os.system(lock_fre)

            # set percent
            set_percent = f"echo set_active_thread_percentage 1503813 {percent} | nvidia-cuda-mps-control"
            print(set_percent)
            os.system(set_percent)

            file = model_dataset
            file_name = f"{file}_{batch}_{fre}_{percent}.log"
            cmd = f"python3 {file}.py {batch} {batch} {file_name} 3 4"
            os.system(cmd)
            print(cmd)
