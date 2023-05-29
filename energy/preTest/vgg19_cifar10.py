import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
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


# 加载预训练的ResNet模型
logging.info("加载预训练的VGG19模型")
vgg = models.vgg19(pretrained=True)

# 替换最后一层全连接层
num_features = vgg.classifier[6].in_features
num_classes = 10  # CIFAR-10数据集有10个类别
vgg.classifier[6] = nn.Linear(num_features, num_classes)


# 将模型转换为GPU上的可训练状态
logging.info("将模型转换为GPU上的可训练状态")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = vgg.to(device)

# 定义损失函数和优化器
logging.info("定义损失函数和优化器")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(vgg.parameters(), lr=1e-4)

# 定义数据预处理
logging.info("定义数据预处理")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


accumulation_steps = batch_size // device_batch_size  # 梯度累积的步数
# 加载训练集和测试集
logging.info("加载训练集和测试集")
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=device_batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=device_batch_size, shuffle=False)

# 初始化 pynvml
pynvml.nvmlInit()

# 训练过程
num_epochs = 5
total_loss = 0.0


# 计算能耗
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 假设只有一个GPU，索引为0
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
        outputs = vgg(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和梯度累积
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    # 打印训练损失
    avg_loss = total_loss / (len(train_loader)* accumulation_steps)
    logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")
    total_loss = 0.0

    # 结束计时
    end_time.record()
    torch.cuda.synchronize()

    # 计算时间差（以毫秒为单位）
    elapsed_time = start_time.elapsed_time(end_time)

    # 计算功耗
    energy_info = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - energy_before
    energy_usage = energy_info / 1000  # 转换为瓦特
    logging.info(f"Epoch {epoch + 1} elapsed time: {elapsed_time} ms, energy Usage: {energy_usage} J")

end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
energy = (end_energy - start_energy) / 1000
logging.info(f"Total energy Usage: {energy} J")
publicFunction.writeCSV(Constant.CSV_FILE_NAME,["vgg19",'cifar10',batch_size,device_batch_size,format(energy/num_epochs,'.2f')])
# 释放 pynvml 资源
pynvml.nvmlShutdown()
