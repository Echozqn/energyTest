import argparse
import logging
import time
from datetime import datetime
import csv
import torch
import torchvision
import pynvml
from torch import distributed as dist
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)  # 与其他worker进行同步
        if rank == 0:
            tensor /= world_size


def train(epochs, local_rank, batch_size=128, job_id="Job0"):
    lr = 0.001
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    net = resnet50()
    data_root = 'dataset'
    trainset = CIFAR10(root=data_root,
                       download=True,
                       train=True,
                       transform=ToTensor())

    sampler = DistributedSampler(trainset)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              sampler=sampler)

    file_name = f"{job_id}_{batch_size}_{global_rank}.log"
    # 配置日志记录器
    logging.basicConfig(filename=file_name, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.cuda()
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)
    net.train()

    print("Start")
    global_step = 0

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)

    csv_file = open(f"epoch_stats_{job_id}_{batch_size}_{global_rank}.csv", 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "Time", "Energy Consumption"])

    for epoch in range(epochs):
        energy_consumption_begin = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        epoch_begin = time.time()

        sampler.set_epoch(epoch)
        for idx, (imgs, labels) in enumerate(train_loader):
            start = time.time()

            global_step += 1
            imgs = imgs.cuda()  # loading
            labels = labels.cuda()  # loading
            output = net(imgs)  # running
            loss = criterion(output, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            reduce_loss(loss, global_rank, world_size)

            if global_rank == 0 and global_step % 5 == 0:
                duration = time.time() - start
                examples_per_sec = batch_size / duration
                val = f"{datetime.now()}\t{global_step * world_size}\t{global_step * world_size * batch_size}\t{loss.item()}\t{examples_per_sec}\n"

            if idx % 10 == 0 and global_rank == 0:
                print('Epoch: {} step: {} loss: {}'.format(epoch, idx, loss.item()))

        epoch_time = (time.time() - epoch_begin) * 1000
        energy_consumption_end = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        epoch_energy_consumption = (energy_consumption_end - energy_consumption_begin) / 1000
        csv_writer.writerow([epoch, epoch_time, epoch_energy_consumption])

    pynvml.nvmlShutdown()
    csv_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, help="local gpu id")
    parser.add_argument('--batch_size', default=128, type=int, help="batch size")
    parser.add_argument('--JobID', default="Job0", type=str, help="JOB ID")
    parser.add_argument('--epochs', default=2, type=int, help="epoch num")

    args = parser.parse_args()
    train(epochs=args.epochs, local_rank=args.local_rank, batch_size=args.batch_size, job_id=args.JobID)

"""

python3 -m torch.distributed.launch --nproc_per_node=1 --nnode=2 --node_rank=0 --master_addr=172.31.92.152 --master_port=5556 demo.py --batch_size=256 --JobID Job1 --epochs 5 
python3 -m torch.distributed.launch --nproc_per_node=1 --nnode=2 --node_rank=1 --master_addr=172.31.92.152 --master_port=5556 demo.py --batch_size=256 --JobID Job2 --epochs 5

"""
