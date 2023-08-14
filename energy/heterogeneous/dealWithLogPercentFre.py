import os
import csv
import re

from Common import Constant
from Common import publicFunction


def dealWithLogTimeEnergy(log_folder, output_csv):
    # 获取log文件夹下的所有文件
    log_files = [f for f in os.listdir(log_folder) if os.path.isfile(os.path.join(log_folder, f))]
    def getVal(re_match,line):  # r"elapsed time: (\d+\.\d+)"
        match = re.findall(re_match, line)
        if match:
            usage = float(match[0])  # 提取功耗值
            return usage


    # 遍历每个文件并提取信息
    data = []
    publicFunction.writeCSV(output_csv, ["model", "dataset", "batch", "frequency", "resource", "epoch_time", "epoch_energy"])
    for file_name in log_files:
        file_path = os.path.join(log_folder, file_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            elapsed_time = []
            usage_energy = []
            for line in lines:
                re_time = r'elapsed time: (\d+\.\d+) ms'
                re_energy = r'energy Usage: (\d+\.\d+) J'
                cur_time = getVal(re_time,line)
                cur_energy = getVal(re_energy,line)
                if cur_time is not None:
                    elapsed_time.append(cur_time)
                if cur_energy is not None:
                    usage_energy.append(cur_energy)
            if len(elapsed_time) == 0 or len(usage_energy) == 0: continue
            epoch_time = (sum(elapsed_time) - elapsed_time[0]) / (len(elapsed_time) - 1)
            epoch_energy =  (sum(usage_energy) - usage_energy[0]) / (len(usage_energy) - 1)
            # 分割字符串并提取各部分
            parts = file_name.split("_")
            model = parts[0]  # 提取模型名称
            dataset = parts[1]  # 提取数据集名称
            batch = int(parts[2])  # 提取大批量大小并将其转换为整数
            frequency = int(parts[3])
            resource = parts[4].split(".")[0]  # 提取小批量大小并将其转换为整数
            data.append([model, dataset, batch, frequency ,resource, format(epoch_time/1000, '.2f'), format(epoch_energy, '.2f')])

    # 对二维数组进行排序
    data = sorted(data, key=lambda x: tuple(x[i] for i in range(len(x))))
    for item in data:
        print(item)
        publicFunction.writeCSV(output_csv, item)



log_folder = './logPercentFre'
output_csv = './csv/PercentFrequency.csv'
dealWithLogTimeEnergy(log_folder,output_csv)
