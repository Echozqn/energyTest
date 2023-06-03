import argparse
import logging
import os
import time
from datetime import datetime
import csv
import torch
import torchvision
import pynvml
from torch import distributed as dist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import DataLoader
import torch
from deepspeech_pytorch import DeepSpeech
from torch.utils.data import DataLoader
from torchaudio.datasets import LibriSpeech

def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)  # 与其他worker进行同步
        if rank == 0:
            tensor /= world_size


def train(epochs, local_rank, pre_name = "model_dataset",batch_size=128, job_id="Job0"):
    lr = 0.001
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    net = DeepSpeech()
    data_root = 'dataset'
    trainset = LibriSpeech(root=data_root, split="train-clean-100", download=True)

    sampler = DistributedSampler(trainset)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              sampler=sampler)

    file_name = f"{pre_name}_{job_id}_{batch_size}_{global_rank}.log"
    # 配置日志记录器
    logging.basicConfig(filename=file_name, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    criterion = torch.nn.CTCLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    net.cuda()
    net.train()

    print("Start")
    global_step = 0

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)

    csv_file = open(f"{pre_name}_{job_id}_{batch_size}_{global_rank}.csv", 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "Time", "Energy Consumption"])

    print(len(train_loader))
    for epoch in range(epochs):
        energy_consumption_begin = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        epoch_begin = time.time()

        sampler.set_epoch(epoch)
        epoch_step = 0
        for idx, (waveform, target, target_lengths) in enumerate(train_loader):

            epoch_step += 1
            global_step += 1
            waveform = waveform.cuda()  # loading
            target = target.cuda()  # loading
            output, output_sizes = net(waveform)  # running
            loss = criterion(output, target, output_sizes, target_lengths)
            opt.zero_grad()
            loss.backward()
            opt.step()
            reduce_loss(loss, global_rank, world_size)

            if idx % 10 == 0 and global_rank == 0:
                print('Epoch: {} step: {} loss: {}'.format(epoch, idx, loss.item()))

        print(f"epoch_step = {epoch_step}")
        epoch_time = (time.time() - epoch_begin) * 1000
        energy_consumption_end = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        epoch_energy_consumption = (energy_consumption_end - energy_consumption_begin) / 1000
        csv_writer.writerow([epoch, epoch_time, epoch_energy_consumption])

    pynvml.nvmlShutdown()
    csv_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, help="local gpu id")
    parser.add_argument('--batch_size', default=192, type=int, help="batch size")
    parser.add_argument('--JobID', default="Job0", type=str, help="JOB ID")
    parser.add_argument('--epochs', default=2, type=int, help="epoch num")

    args = parser.parse_args()
    pre_name = os.path.basename(__file__)
    pre_name = os.path.splitext(pre_name)[0]
    train(epochs=args.epochs, local_rank=args.local_rank, pre_name = pre_name,batch_size=args.batch_size, job_id=args.JobID)


"""

python3 -m torch.distributed.launch --nproc_per_node=1 --nnode=2 --node_rank=0 --master_addr=44.211.214.203 --master_port=5556 demo.py --batch_size=256 --JobID Job1Double --epochs 2 
python3 -m torch.distributed.launch --nproc_per_node=1 --nnode=2 --node_rank=1 --master_addr=44.211.214.203 --master_port=5556 demo.py --batch_size=256 --JobID Job2Double --epochs 2

python3 -m torch.distributed.launch --nproc_per_node=1 --nnode=1 --node_rank=0 --master_addr=44.204.86.82
 --master_port=5556 demo.py --batch_size=192 --JobID Job1Single --epochs 2 

OMP_NUM_THREADS=32 python3 -m torch.distributed.launch --nproc_per_node=2 --nnode=1 --node_rank=0 --master_addr=172.31.84.208 --master_port=5556 demo.py --batch_size=256 --JobID JobSM --epochs 3 

"""
