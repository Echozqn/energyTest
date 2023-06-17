# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 假设你的数据保存在"data.csv"这个文件中
# df = pd.read_csv('./csv/model_energy.csv')
#
# # 按照GPU和模型进行排序
# df.sort_values(['GPU', 'model'], inplace=True)
#
# # 设置图像大小
# plt.figure(figsize=(10,6))
#
# # 对每个模型，绘制一条曲线
# for model in ['bert', 'resnet50', 'vgg19']:
#     model_df = df[df['model'] == model]
#     plt.plot(model_df['GPU'], model_df['energy'], marker='o', label=model)
#
# # 设置图像的标签
# plt.xlabel('GPU')
# plt.ylabel('Energy')
# plt.title('Energy consumption by GPU and model')
# plt.legend()
#
# plt.savefig("heter.pdf")
# # 展示图像
# plt.show()
#

import pandas as pd
import matplotlib.pyplot as plt

# 假设我们已经将上述数据保存到一个名为"data.csv"的文件中
df = pd.read_csv('./csv/model_energy.csv')

# 将GPU列按照[T4,V100,3090,A100]的顺序进行排序
gpu_order = ['T4', 'V100', '3090', 'A100']
df['GPU'] = df['GPU'].astype('category')
df['GPU'].cat.set_categories(gpu_order, inplace=True)
df = df.sort_values(['GPU'])

models = df['model'].unique()

plt.figure(figsize=(10, 6))

for model in models:
    model_data = df[df['model'] == model]
    plt.plot(model_data['GPU'], model_data['energy'], marker='o', label=model)

plt.xlabel('GPU')
plt.ylabel('Energy')
plt.title('Energy consumption for different models on different GPUs')
plt.legend()

plt.savefig("heter.pdf")
plt.show()

