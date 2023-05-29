import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_dataset
import pynvml
import logging
import Constant
import publicFunction
import sys

batch_size = int(sys.argv[1])
device_batch_size = int(sys.argv[2])
file_name = f"{Constant.Log_DIR_NAME}/{sys.argv[3]}"
num_epochs = int(sys.argv[4])
publicFunction.remove(file_name)
# 配置日志记录器
logging.basicConfig(filename=file_name, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# 加载BERT模型和tokenizer
logging.info("加载BERT模型和tokenizer")
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 加载SST-2数据集
logging.info("加载SST-2数据集")
dataset = load_dataset('glue', 'sst2')
train_dataset = dataset['train']


def preprocess_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)


# 数据预处理和编码
logging.info("数据预处理和编码")
train_dataset = train_dataset.map(preprocess_function, batched=True)

accumulation_steps = batch_size // device_batch_size  # 梯度累积的步数
# 数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=device_batch_size, shuffle=True)

# GPU设置（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

# 训练和评估
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)


def compute_accuracy(logits, labels):
    _, predicted_labels = torch.max(logits, dim=1)
    correct_predictions = (predicted_labels == labels).sum().item()
    return correct_predictions / labels.size(0)


def train(epoch):
    # 开始计时
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

    model.train()
    total_loss, total_accuracy = 0, 0
    for i, batch in enumerate(train_dataloader):
        # print(batch)
        input_ids = torch.stack(batch['input_ids'], dim=-1).to(device)

        attention_mask = torch.stack(batch['attention_mask'], dim=-1).to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # 反向传播和梯度累积
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        total_accuracy += compute_accuracy(logits, labels)

    average_loss = total_loss / len(train_dataloader) * accumulation_steps

    # 结束计时
    end_time.record()
    # 计算时间差（以毫秒为单位）
    elapsed_time = start_time.elapsed_time(end_time)
    # 计算功耗
    energy_info = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - energy_before
    energy_usage = energy_info / 1000  # 转换为瓦特
    logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")
    logging.info(f"Epoch {epoch + 1} elapsed time: {elapsed_time} ms, energy Usage: {energy_usage} J")


# 初始化 pynvml
pynvml.nvmlInit()
# 计算能耗
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 假设只有一个GPU，索引为0
start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
logging.info("开始训练")
# 训练和评估循环
for epoch in range(num_epochs):
    train(epoch)
end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
energy = (end_energy - start_energy) / 1000
logging.info(f"Total energy Usage: {energy} J")
publicFunction.writeCSV(Constant.CSV_FILE_NAME,["BERT",'SST-2',batch_size,device_batch_size,format(energy/num_epochs,'.2f')])
# 释放 pynvml 资源
pynvml.nvmlShutdown()
