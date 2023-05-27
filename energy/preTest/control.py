import os

model_dataset = ['resnet50_cifar10', 'vgg19_cifar10', 'BERT_SST2']
batches = [512, 512, 32]
small_batches = [32, 32, 4]
for i in range(1, len(model_dataset)):
    file = model_dataset[i]
    batch = batches[i]
    small_batch = batch
    while True:
        if small_batch < small_batches[i]: break
        file_name = f"{file}_{batch}_{small_batch}.log"
        cmd = f"python3 {file}.py {batch} {small_batch} {file_name} 5"
        print(cmd)
        os.system(cmd)
        small_batch //= 2
