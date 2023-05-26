import os
batch = 512
small_batch = batch
model_dataset = ['resnet50_cifar10','vgg19_cifar10']
for file in model_dataset:
    while True:
        if small_batch < 32:break
        file_name = f"{file}_{batch}_{small_batch}.log"
        cmd = f"python3 {file}.py {batch} {small_batch} {file_name}"
        print(cmd)
        os.system(cmd)
        small_batch //= 2