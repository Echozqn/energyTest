import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pynvml
import logging
import torch.utils.data as data
import sys
from Common import Constant
from Common import publicFunction
import wandb  # 添加此行

# Initialize wandb
wandb.init(project="aws-V100-frequency")

batch_size = int(sys.argv[1])
device_batch_size = int(sys.argv[2])
file_name = f"{Constant.Log_DIR_NAME}/{sys.argv[3]}"
num_epochs = int(sys.argv[4])
GPU = sys.argv[5]
GPU_index = 4
publicFunction.remove(file_name)
# 配置日志记录器
logging.basicConfig(filename=file_name, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

wandb.config.batch_size = batch_size  # 记录超参数
wandb.config.device_batch_size = device_batch_size
wandb.config.num_epochs = num_epochs
wandb.config.GPU = GPU_index


# 加载预训练的ResNet模型
logging.info("加载预训练的ResNet模型")
resnet50 = models.resnet50(pretrained=True)

# 替换最后一层全连接层
num_features = resnet50.fc.in_features
num_classes = 10  # CIFAR-10数据集有10个类别
resnet50.fc = nn.Linear(num_features, num_classes)

# 将模型转换为GPU上的可训练状态
logging.info("将模型转换为GPU上的可训练状态")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(f"cuda:{GPU_index}" if torch.cuda.is_available() else "cpu")
resnet50 = resnet50.to(device)

# 定义损失函数和优化器
logging.info("定义损失函数和优化器")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet50.parameters(), lr=1e-4)

# 定义数据预处理
logging.info("定义数据预处理")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


accumulation_steps = batch_size // device_batch_size  # 梯度累积的步数
# 加载训练集和测试集
logging.info("加载训练集和测试集")
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_dataset = data.Subset(train_dataset, range(len(train_dataset)//256 * 256)) # 2
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=device_batch_size, shuffle=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=device_batch_size, shuffle=False)

# 初始化 pynvml
pynvml.nvmlInit()

# 训练过程
total_loss = 0.0


# 计算能耗
before_time = time.time()
handle = pynvml.nvmlDeviceGetHandleByIndex(int(GPU_index))  # 通过GPU索引获取handle
start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
logging.info("开始训练")
for epoch in range(num_epochs):
    # 开始计时
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = resnet50(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和梯度累积
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    # 打印训练损失
    avg_loss = total_loss / len(train_loader) * accumulation_steps
    logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")
    total_loss = 0.0

    wandb.log({"avg_loss": avg_loss})  # 记录每个 epoch 的平均损失

    # 结束计时
    end_time.record()
    torch.cuda.synchronize()

    # 计算时间差（以毫秒为单位）
    elapsed_time = start_time.elapsed_time(end_time)

    # 计算功耗
    energy_info = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - energy_before
    energy_usage = energy_info / 1000  # 转换为瓦特
    logging.info(f"Epoch {epoch + 1} elapsed time: {elapsed_time} ms, energy Usage: {energy_usage} J")

    wandb.log({"elapsed_time": elapsed_time, "energy_usage": energy_usage})  # 记录每个 epoch 的时间和能耗


wandb.finish()  # 最后关闭 wandb
after_time = time.time()
exec_time = after_time - before_time
end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
energy = (end_energy - start_energy) / 1000
logging.info(f"Total energy Usage: {energy} J")
publicFunction.writeCSV(Constant.CSV_FILE_NAME,[GPU,"resnet50",'cifar10',batch_size,device_batch_size,format(energy,'.2f'),format(exec_time,'.2f')])
# 释放 pynvml 资源
pynvml.nvmlShutdown()

"""
nsys profile python3 resnet50_cifar10.py 512 512 A100_resnet50_cifar10_512_512.log 3 A100
"""
