import logging
import sys
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from Common import Constant
from Common import publicFunction

# 假设您已经下载并解压了MovieLens数据集，并将其放在名为"ml-1m"的文件夹中
# 数据集下载链接：https://grouplens.org/datasets/movielens/1m/
# wget https://files.grouplens.org/datasets/movielens/ml-1m.zip

batch_size = int(sys.argv[1])
device_batch_size = int(sys.argv[2])
file_name = f"{Constant.Log_DIR_NAME}/{sys.argv[3]}"
num_epochs = int(sys.argv[4])
GPU = sys.argv[5]
publicFunction.remove(file_name)
# 配置日志记录器
logging.basicConfig(filename=file_name, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')



# 加载数据
data = pd.read_csv('ml-1m/ratings.dat', sep='::', names=['user', 'item', 'rating', 'timestamp'], engine='python')
data['rating'] = data['rating'].astype(float)

# 构建用户和物品的映射
user_mapping = {j: i for i, j in enumerate(data['user'].unique())}
item_mapping = {j: i for i, j in enumerate(data['item'].unique())}
data['user'] = data['user'].map(user_mapping)
data['item'] = data['item'].map(item_mapping)

# 分割训练集和测试集
logging.info(f"数据集的大小为 {len(data)}")
train_data = data


# 定义数据集
class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.users = data['user'].values
        self.items = data['item'].values
        self.ratings = data['rating'].values

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return torch.tensor(self.users[idx], dtype=torch.long), torch.tensor(self.items[idx],
                                                                             dtype=torch.long), torch.tensor(
            self.ratings[idx], dtype=torch.float)


# NeuMF模型
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim, layers=[64, 32, 16, 8]):
        super(NeuMF, self).__init__()
        self.MF_Embedding_User = nn.Embedding(num_users, mf_dim)
        self.MF_Embedding_Item = nn.Embedding(num_items, mf_dim)
        self.layers = layers
        self.MLP_Embedding_User = nn.Embedding(num_users, layers[0] // 2)
        self.MLP_Embedding_Item = nn.Embedding(num_items, layers[0] // 2)
        self.mlp = nn.Sequential()
        for i in range(len(layers) - 1):
            self.mlp.add_module("linear%d" % i, nn.Linear(layers[i], layers[i + 1]))
            self.mlp.add_module("relu%d" % i, nn.ReLU())
        self.predict_layer = nn.Linear(mf_dim + layers[-1], 1)

    def forward(self, user, item):
        MF_user_embedding = self.MF_Embedding_User(user)
        MF_item_embedding = self.MF_Embedding_Item(item)
        MF_vector = MF_user_embedding * MF_item_embedding

        MLP_user_embedding = self.MLP_Embedding_User(user)
        MLP_item_embedding = self.MLP_Embedding_Item(item)
        MLP_vector = torch.cat((MLP_user_embedding, MLP_item_embedding), dim=-1)
        MLP_vector = self.mlp(MLP_vector)

        vector = torch.cat((MF_vector, MLP_vector), dim=-1)
        prediction = self.predict_layer(vector)

        return prediction

import pynvml

pynvml.nvmlInit()

handle = pynvml.nvmlDeviceGetHandleByIndex(0)
start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
before_time = time.time()
# 训练模型
def train(model, train_data, num_epochs, batch_size):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    train_dataloader = DataLoader(MovieLensDataset(train_data), batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

        for user, item, rating in train_dataloader:
            user = user.to(device)
            item = item.to(device)
            rating = rating.to(device)
            optimizer.zero_grad()
            outputs = model(user, item).squeeze()
            loss = criterion(outputs, rating)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_dataloader)

        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)

        energy_info = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - energy_before
        energy_usage = energy_info / 1000
        logging.info(f"Epoch {epoch + 1} elapsed time: {elapsed_time} ms,Train Loss: {train_loss:.4f}, energy Usage: {energy_usage} J")


num_users = data['user'].nunique()
num_items = data['item'].nunique()
mf_dim = 8
layers = [64, 32, 16, 8]

model = NeuMF(num_users, num_items, mf_dim, layers)
train(model, train_data, num_epochs, batch_size)


after_time = time.time()
exec_time = after_time - before_time
end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
energy = (end_energy - start_energy) / 1000
logging.info(f"Total energy Usage: {energy} J")
publicFunction.writeCSV(Constant.CSV_FILE_NAME,[GPU,"NeuMF",'MovieLens-1M',batch_size,device_batch_size,format(energy,'.2f'),format(exec_time,".2f")])

pynvml.nvmlShutdown()
