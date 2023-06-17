
import argparse
import logging
import os
import sys
import time
from transformers import AutoTokenizer, AutoModelForMaskedLM
import random
import numpy as np
import torch
from tqdm import tqdm
from Common import Constant
from Common import publicFunction
from albert_obqa_data_process import GetDataset
from albert_obqa_model_ import Moudle
import pynvml



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



batch_size = int(sys.argv[1])
device_batch_size = int(sys.argv[2])
file_name = f"{Constant.Log_DIR_NAME}/{sys.argv[3]}"
num_epochs = int(sys.argv[4])
GPU = sys.argv[5]

config = {
    "batch_size": batch_size,
    "PLM": "albert-base-v2",
    "device": torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    "USE_GPU": 1 if torch.cuda.is_available() else 0,
    "input_max_len": 128,
    "seed": 23,
    "train_size": 32,
    "num_epochs":num_epochs,
    "lr":1e-5
}

publicFunction.remove(file_name)
# 配置日志记录器
logging.basicConfig(filename=file_name, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


set_seed(config["seed"])
tokenizer = AutoTokenizer.from_pretrained(config["PLM"], is_fast=True)
trainDataset = GetDataset().getDataLoader("obqa/train", config["batch_size"], tokenizer,
                                                                 config["input_max_len"], config)

args_dict = dict(
    input_max_len=config["input_max_len"],
    choice_num=4,
    PLM=config["PLM"],
    device=config["device"],
    dropout=0.3,
    learning_rate=config["lr"],
    weight_decay=0.0,
    adam_epsilon=1e-08,
    gradient_accumulation_steps=1,
    len_train_loader=len(trainDataset),
    num_epochs=config["num_epochs"],
    tokenizer=tokenizer,
    n_gpu=1 if config["USE_GPU"] else 0,
    early_stop_callback=False,
    fp_16=False,
    opt_level='O1',
    max_grad_norm=1.0,
)

hparams = argparse.Namespace(**args_dict)
model = Moudle(hparams)
model = model.to(config["device"])
optimizer = model.configure_optimizers()



pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

mx = -99999999
# dev_acc, dec_sum = eval(model,devDataset)
# print("zero_acc",dev_acc, dec_sum)
# test_acc, test_sum = eval(model,testDataset)
model.train()
for epoch in range(hparams.num_epochs):
    energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    loss_sum = 0
    tot = 0
    for i, data in enumerate(tqdm(trainDataset)):
        tot += 1
        model.train()
        pre = time.time()
        loss = model.training_step(data)
        loss.backward()
        loss_sum += loss.item()
        model.optimizer_step(optimizer)

    avg_loss = loss_sum / tot
    energy_info = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - energy_before
    energy_usage = energy_info / 1000
    logging.info(f"Epoch {epoch + 1} energy Usage: {energy_usage} J avg_loss: {avg_loss}")

end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
energy = (end_energy - start_energy) / 1000
logging.info(f"Total energy Usage: {energy} J")
publicFunction.writeCSV(Constant.CSV_FILE_NAME,[GPU,"albert",'obqa',batch_size,device_batch_size,format(energy/num_epochs,'.2f')])

pynvml.nvmlShutdown()

