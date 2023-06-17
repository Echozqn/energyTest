import logging
import sys
import pynvml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim):
        super(NeuMF, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        mf_input = user_embed * item_embed

        mlp_input = torch.cat((user_embed, item_embed), dim=1)
        prediction = self.mlp(mlp_input)

        output = torch.sigmoid(prediction + mf_input)
        return output.squeeze()

class MovieLensDataset(Dataset):
    def __init__(self, ratings):
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user = self.ratings[idx][0]
        item = self.ratings[idx][1]
        rating = self.ratings[idx][2]
        return user, item, rating

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0

    for user, item, rating in train_loader:
        user = user.to(device)
        item = item.to(device)
        rating = rating.float().to(device)

        optimizer.zero_grad()
        output = model(user, item)
        loss = criterion(output, rating)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)


from Common import Constant
from Common import publicFunction


batch_size = int(sys.argv[1])
device_batch_size = int(sys.argv[2])
file_name = f"{Constant.Log_DIR_NAME}/{sys.argv[3]}"
num_epochs = int(sys.argv[4])
GPU = sys.argv[5]
publicFunction.remove(file_name)
# 配置日志记录器
logging.basicConfig(filename=file_name, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


# 设置超参数
learning_rate = 0.001
embedding_dim = 32
hidden_dim = 64

# 加载数据
# 假设您已经下载并解压了MovieLens数据集，并将其放在名为"ml-1m"的文件夹中
# 数据集下载链接：https://grouplens.org/datasets/movielens/1m/
ratings_file = "ml-1m/ratings.dat"

ratings = []
with open(ratings_file, 'r') as f:
    for line in f:
        user, item, rating, _ = line.strip().split("::")
        ratings.append((int(user), int(item), float(rating)))

# 将数据集分为训练集和测试集
logging.info(f"数据集大小为：{len(ratings)}")
train_size = int(0.8 * len(ratings))
train_ratings = ratings[:train_size]

train_dataset = MovieLensDataset(train_ratings)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_users = max([rating[0] for rating in ratings]) + 1
num_items = max([rating[1] for rating in ratings]) + 1

# 创建模型并将其移动到设备上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuMF(num_users, num_items, embedding_dim, hidden_dim).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize NVML
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

for epoch in range(num_epochs):
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    train_loss = train(model, train_loader, criterion, optimizer, device)

    end_time.record()
    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time)

    energy_info = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - energy_before
    energy_usage = energy_info / 1000
    logging.info(f"Epoch {epoch + 1} elapsed time: {elapsed_time} ms, energy Usage: {energy_usage} J")
    logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
energy = (end_energy - start_energy) / 1000
logging.info(f"Total energy Usage: {energy} J")
publicFunction.writeCSV(Constant.CSV_FILE_NAME,[GPU,"NeuMF",'MovieLens-1M',batch_size,device_batch_size,format(energy/num_epochs,'.2f')])
pynvml.nvmlShutdown()

print("Training and evaluation finished.")
