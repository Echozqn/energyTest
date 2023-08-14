import os
import subprocess
import time
import pynvml

from Common import Constant
from Common import publicFunction

# Initialize NVML
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(4)
start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
batch = 128
fre = 1100
lock_fre = f"sudo nvidia-smi -lgc {fre} -i 4"
print(lock_fre)
os.system(lock_fre)



# set percent
set_percent = f"echo set_active_thread_percentage 1654877 {45} | nvidia-cuda-mps-control"
print(set_percent)
os.system(set_percent)
file = 'resnet50_cifar10'
file_name1 = f"{file}_{batch}_{fre}_{45}_job1.log"
cmd1 = f"python3 {file}.py {batch} {batch} {file_name1} 3 A100 4"
process1 = subprocess.Popen(cmd1, shell=True)
print(cmd1)

time.sleep(1)

# set percent
set_percent = f"echo set_active_thread_percentage 1654877 {45} | nvidia-cuda-mps-control"
print(set_percent)
os.system(set_percent)
file_name2 = f"{file}_{batch}_{fre}_{45}_job2.log"
cmd2 = f"python3 {file}.py {batch} {batch} {file_name2} 3 A100 4"
print(cmd2)

# Start the two processes and wait for them to complete
process2 = subprocess.Popen(cmd2, shell=True)

process1.wait()
process2.wait()


end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
energy = (end_energy - start_energy) / 1000
print(f"energy = {energy}")
pynvml.nvmlShutdown()