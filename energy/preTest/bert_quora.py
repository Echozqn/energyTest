import logging
import sys

import pynvml
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, Trainer, TrainingArguments,BertTokenizerFast
from datasets import load_dataset
import Constant
import publicFunction

batch_size = int(sys.argv[1])
device_batch_size = int(sys.argv[2])
file_name = f"{Constant.Log_DIR_NAME}/{sys.argv[3]}"
num_epochs = int(sys.argv[4])
publicFunction.remove(file_name)
# 配置日志记录器
logging.basicConfig(filename=file_name, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_hidden_states=False)
# 从hugginface的数据集库中下载quora数据集
# 也可以全部读取完用dataset.train_test_split(test_size = 0.1)这样来读取
train_dataset = load_dataset('quora', split='train[:3%]')

def preprocess_function(examples):
    # Tokenize the texts
    result = tokenizer([examples['questions'][i]['text'][0] for i in range(len(examples['questions']))],
                       [examples['questions'][i]['text'][1] for i in range(len(examples['questions']))],
                       padding=True, truncation=True, max_length=32)

    # Map labels to IDs
    result["label"] = [(1 if l else 0) for l in examples["is_duplicate"]]
    return result


logging.info("数据预处理和编码")
train_dataset = train_dataset.map(preprocess_function, batched=True,load_from_cache_file=False)
accumulation_steps = batch_size // device_batch_size  # 梯度累积的步数
# 数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=device_batch_size, shuffle=True)
# GPU设置（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# 训练和评估
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)


def train(epoch):
    # 开始计时
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    model.train()
    total_loss, total_accuracy = 0, 0
    for i, batch in enumerate(train_dataloader):
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

    average_loss = total_loss / (len(train_dataloader) * accumulation_steps)

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
# 释放 pynvml 资源
pynvml.nvmlShutdown()
