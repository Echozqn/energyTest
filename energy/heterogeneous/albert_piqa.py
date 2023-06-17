import logging
import sys
import pynvml

import torch
from torch.utils.data import DataLoader
from transformers import AlbertForSequenceClassification, AutoTokenizer
from datasets import load_dataset

from Common import Constant
from Common import publicFunction


batch_size = int(sys.argv[1])
device_batch_size = int(sys.argv[2])
file_name = f"{Constant.Log_DIR_NAME}/{sys.argv[3]}"
num_epochs = int(sys.argv[4])
GPU = sys.argv[5]
publicFunction.remove(file_name)
# 配置日志记录器
logging.basicConfig(filename=file_name, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


# 参数设定
model_name = 'albert-base-v2'
learning_rate = 1e-5

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AlbertForSequenceClassification.from_pretrained(model_name)

dataset = load_dataset("piqa")
# print(dataset)

def encode(examples):
    return tokenizer(examples['goal'], truncation=True, padding='max_length')

train_dataset = dataset['train'].map(encode, batched=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
optim = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
logging.info("开始训练")
for epoch in range(num_epochs):
    model.train()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    for batch in train_dataloader:
        inputs = {k: v for k, v in batch.items() if k != 'label'}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optim.step()
        optim.zero_grad()
    end_time.record()
    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time)
    energy_info = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - energy_before
    energy_usage = energy_info / 1000
    logging.info(f"Epoch {epoch + 1} elapsed time: {elapsed_time} ms, energy Usage: {energy_usage} J")

end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
energy = (end_energy - start_energy) / 1000
logging.info(f"Total energy Usage: {energy} J")
publicFunction.writeCSV(Constant.CSV_FILE_NAME,[GPU,"albert-base-v2",'piqa',batch_size,device_batch_size,format(energy/num_epochs,'.2f')])
pynvml.nvmlShutdown()
