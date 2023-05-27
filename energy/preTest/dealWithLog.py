import os
import csv
import re

import publicFunction

log_folder = './log'
output_csv = 'elapsed_time.csv'
total_energy_usage = 0.0

# 获取log文件夹下的所有文件
log_files = [f for f in os.listdir(log_folder) if os.path.isfile(os.path.join(log_folder, f))]

def getEnergy(line):# r"elapsed time: (\d+\.\d+)"
    match = re.findall(r'elapsed time: (\d+\.\d+) ms', line)
    if match:
        energy_usage = float(match[0])  # 提取功耗值
        return energy_usage

# 遍历每个文件并提取信息
data = []
publicFunction.writeCSV(output_csv,["model","dataset","large_batch","small_batch","epoch_time"])
for file_name in log_files:
    file_path = os.path.join(log_folder, file_name)
    with open(file_path, 'r') as file:
        lines = file.readlines()
        elapsed_time = []
        for line in lines:
            cur = getEnergy(line)
            if cur is not None:
                elapsed_time.append(cur)
        if len(elapsed_time) == 0: continue
        epoch_time = sum(elapsed_time)/len(elapsed_time)
        # 分割字符串并提取各部分
        parts = file_name.split("_")
        model = parts[0]  # 提取模型名称
        dataset = parts[1]  # 提取数据集名称
        large_batch = int(parts[2])  # 提取大批量大小并将其转换为整数
        small_batch = int(parts[3].split(".")[0])  # 提取小批量大小并将其转换为整数
        data.append([model,dataset,large_batch,small_batch,format(epoch_time,'.2f')])


# 对二维数组进行排序
data = sorted(data, key=lambda x: tuple(x[i] for i in range(len(x))))

for item in data:
    print(item)
    publicFunction.writeCSV(output_csv,item)
