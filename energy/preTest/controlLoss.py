import os
import publicFunction
import Constant

for i in range(10):
    file = 'resnet50_cifar10'
    small_batch = 256
    while small_batch >= 32:
        file_name = f"{file}_{str(256)}_{str(small_batch)}_{i}_loss.log"
        cmd = f"python3 {file}.py 256 {str(small_batch)} {file_name} 15"
        print(cmd)
        os.system(cmd)
        small_batch //= 2
