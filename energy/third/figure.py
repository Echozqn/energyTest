import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Common import Constant
from Common import publicFunction

# 读取CSV文件
csv_data = pd.read_csv('./csv/model_energy.csv')
models = ["resnet50", "albert", "vgg19"]

# 设置图表主题风格
sns.set_theme()

def pic1():
    for model in models:
        data = csv_data[csv_data['model'] == model]

        # 提取所需的列数据
        frequency = data['frequency']
        energy = data['energy']
        exec_time = data['exec_time']

        directory = f"./pic/{model}"
        os.makedirs(directory, exist_ok=True)

        for val in range(0, 11):
            delta = float(format(val * 0.1, ".2f"))
            y = energy * delta + (1 - delta) * exec_time * 250

            # 创建一个新的figure，设置图表尺寸
            plt.figure(figsize=(10, 6))

            # 绘制折线图
            plt.plot(frequency, y, label=model)
            plt.xlabel('Frequency')
            plt.ylabel(f'Energy * {delta} + {(1 - delta)} * Execution Time')
            plt.title('Performance')
            plt.grid(True)

            # 添加美化效果
            plt.rcParams.update({
                'font.size': 14,
                'lines.linewidth': 2,
                'axes.linewidth': 1.5,
                'xtick.major.width': 1.5,
                'ytick.major.width': 1.5
            })

            # 添加图例
            plt.legend(loc='best')

            plt.savefig(f"{directory}/{delta}.png")
            plt.clf()
            plt.show()

def pic2():
    for model in models:
        data = csv_data[csv_data['model'] == model]
        frequency = data['frequency']
        energy = data['energy']

        # 创建一个新的figure，设置图表尺寸
        plt.figure(figsize=(10, 6))

        plt.plot(frequency, energy, label=model)
        plt.xlabel('Frequency')
        plt.ylabel('Energy')

        plt.rcParams.update({
            'font.size': 14,
            'lines.linewidth': 2,
            'axes.linewidth': 1.5,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5
        })

        # 添加图例
        plt.legend(loc='best')

        directory = "./pic/energy"
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{directory}/{model}_energy.png")
        plt.clf()
        plt.show()

pic1()
pic2()
