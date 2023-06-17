import logging
import sys

import torch
import pynvml
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizerFast, AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from Common import Constant
from Common import publicFunction


batch_size = int(sys.argv[1])
device_batch_size = int(sys.argv[2])
file_name = f"{Constant.Log_DIR_NAME}/{sys.argv[3]}"
num_epochs = int(sys.argv[4])
publicFunction.remove(file_name)
# 配置日志记录器
logging.basicConfig(filename=file_name, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin1', header=None)[:50000]
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

df['sentiment'] = df['sentiment'].replace(4, 1)

train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['sentiment'], test_size=.2)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


train_dataset = SentimentDataset(train_texts.to_list(), train_labels.to_list(), tokenizer)
test_dataset = SentimentDataset(test_texts.to_list(), test_labels.to_list(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

optimizer = AdamW(model.parameters(), lr=1e-5)

# Initialize pynvml and create device handle
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
logging.info("开始训练")
for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs[0]
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

    end_time.record()
    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time)
    energy_info = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - energy_before
    energy_usage = energy_info / 1000
    logging.info(f"Epoch {epoch + 1} elapsed time: {elapsed_time} ms, energy Usage: {energy_usage} J")

end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
energy = (end_energy - start_energy) / 1000
print(f"Total energy Usage: {energy} J")
publicFunction.writeCSV(Constant.CSV_FILE_NAME,["bert",'sentiment140',batch_size,device_batch_size,format(energy/num_epochs,'.2f')])
pynvml.nvmlShutdown()

