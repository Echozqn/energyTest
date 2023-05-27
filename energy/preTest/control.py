import os
import publicFunction
import Constant

publicFunction.remove(Constant.CSV_FILE_NAME)
model_dataset = ['resnet50_cifar10', 'vgg19_cifar10', 'BERT_SST2']
batches = [512, 512, 32]
small_batches = [32, 32, 4]

for i in range(0, len(model_dataset)):
    batch = batches[i]
    while batch >= small_batches[i]:
        file = model_dataset[i]
        small_batch = batch
        while small_batch >= small_batches[i]:
            file_name = f"{file}_{batch}_{small_batch}.log"
            cmd = f"python3 {file}.py {batch} {small_batch} {file_name} 3"
            print(cmd)
            os.system(cmd)
            small_batch //= 2
        batch //= 2
