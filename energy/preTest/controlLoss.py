import os
import publicFunction
import Constant

file = 'resnet50_cifar10'
small_batch = 256
while small_batch >= 32:
    file_name = f"{file}_{str(256)}_{str(small_batch)}_version3_loss.log"
    cmd = f"python3 {file}.py 256 {str(small_batch)} {file_name} 20"
    print(cmd)
    # os.system(cmd)
    small_batch //= 2
"""
python3 resnet50_cifar10.py 1024 512 resnet50_cifar10_1024_512_version3_loss.log 2
"""