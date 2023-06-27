import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('./csv/model_energy.csv')

# 提取所需的列数据
frequency = data['frequency']
energy = data['energy']
exec_time = data['exec_time']

# 设置默认的delta值
delta = 0.7
for val in range(0, 11):
    delta = float(format(val * 0.1, ".2f"))
    y = energy * delta + (1 - delta) * exec_time * 250

    # 绘制折线图
    plt.plot(frequency, y)
    plt.xlabel('Frequency')
    plt.ylabel(f'Energy * {delta} + {(1 - delta)} * Execution Time')
    plt.title('Performance')
    plt.grid(True)

    # 添加美化效果
    plt.rcParams['font.size'] = 12
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5

    plt.savefig(f"./pic/{delta}_Frequency.png")
    # 显示图形
    plt.show()
