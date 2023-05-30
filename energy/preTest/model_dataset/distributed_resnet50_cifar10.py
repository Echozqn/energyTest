import argparse
import logging
import time
from datetime import datetime

import torch
import pynvml
import torchvision
from torch import distributed as dist
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from energy.preTest import Constant, publicFunction


def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)  # 与其他worker进行同步
        if rank == 0:
            tensor /= world_size


parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, help="local gpu id")
parser.add_argument('--batch_size', default=128, type=int, help="batch size")
parser.add_argument('--JobID', default="Job0", type=str, help="JOB ID")
parser.add_argument('--epoch', default=5, type=str, help="epoch num")
parser.add_argument('--file_name', default="distributed_resnet50_cifar10.log", type=str, help="log file name")

args = parser.parse_args()
batch_size = args.batch_size
jobId = args.JobID
epochs = args.epoch
file_name = f"{Constant.Log_DIR_NAME}/{args.file_name}"
publicFunction.remove(file_name)
lr = 0.001
# 配置日志记录器
logging.basicConfig(filename=file_name, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


dist.init_process_group(backend='nccl', init_method='env://')
torch.cuda.set_device(args.local_rank)
global_rank = dist.get_rank()
world_size = dist.get_world_size()


net = resnet50()

data_root = 'dataset'
trainset = CIFAR10(root=data_root,
                   download=True,
                   train=True,
                   transform=ToTensor())

valset = CIFAR10(root=data_root,
                 download=True,
                 train=False,
                 transform=ToTensor())


sampler = DistributedSampler(trainset)
train_loader = DataLoader(trainset,
                          batch_size=batch_size,
                          shuffle=False,
                          pin_memory=True,
                          sampler=sampler)

val_loader = DataLoader(valset,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=True)

file_name = f"{jobId}_{batch_size}_{global_rank}.log"
# data_file = open(file_name, "w")
# data_file.write("datetime\tg_step\tg_img\tloss_value\texamples_per_sec\n")
import torch.profiler
# 初始化NVML
pynvml.nvmlInit()

# 获取当前训练的GPU卡的handle
handle = pynvml.nvmlDeviceGetHandleByIndex(global_rank)
with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=f'./log_{jobId}'),
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True
) as p:
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.cuda()
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank)
    net.train()
    print("Start")
    global_step = 0
    train_begin = time.time()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        # 开始计时
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        for idx, (imgs, labels) in enumerate(train_loader):
            # start = time.time()
            global_step += 1
            imgs = imgs.cuda()  # loading
            labels = labels.cuda()  # loading
            output = net(imgs)  # running
            loss = criterion(output, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            reduce_loss(loss, global_rank, world_size)

            # if global_rank == 0 and global_step % 5 == 0:
            #     duration = time.time() - start
            #     examples_per_sec = batch_size / duration
            #     val = f"{datetime.now()}\t{global_step * world_size}\t{global_step * world_size * batch_size}\t{loss.item()}\t{examples_per_sec}\n"
            #     data_file.write(val)

            # if idx % 10 == 0 and global_rank == 0:
            #     print('Epoch: {} step: {} loss: {}'.format(e, idx, loss.item()))
            p.step()

        # 结束计时
        end_time.record()
        # 计算时间差（以毫秒为单位）
        elapsed_time = start_time.elapsed_time(end_time)
        # 计算功耗
        energy_info = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - energy_before
        energy_usage = energy_info / 1000  # 转换为瓦特
        if global_rank == 0:
            logging.info(f"Epoch {epoch + 1} elapsed time: {elapsed_time} ms, energy Usage: {energy_usage} J")

    # data_file.write("TrainTime\t%f\n" % (time.time() - train_begin))

# 最后，关闭NVML
pynvml.nvmlShutdown()
