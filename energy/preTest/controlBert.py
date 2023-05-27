import os
import publicFunction
import Constant

publicFunction.remove(Constant.CSV_FILE_NAME)
model_dataset = ['bert_quora']
batches = [32]
small_batches = [4]
publicFunction.remove(Constant.CSV_FILE_NAME)
for i in range(0, len(model_dataset)):
    batch = small_batches[i]
    while batch <= batches[i]:
        file = model_dataset[i]
        small_batch = small_batches[i]
        while small_batch <= batch:
            file_name = f"{file}_{batch}_{small_batch}.log"
            cmd = f"python3 {file}.py {batch} {small_batch} {file_name} 3"
            print(cmd)
            os.system(cmd)
            small_batch *= 2
        batch *= 2
