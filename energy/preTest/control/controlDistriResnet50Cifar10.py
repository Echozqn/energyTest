
cmd = 'python -m torch.distributed.launch --nproc_per_node=4 --nnode=2 --node_rank=0 --master_addr=A_ip_address master_port=29500 main.py'
"""
python3 -m torch.distributed.launch --nproc_per_node=1 --nnode=1 --node_rank=0 --master_addr=34.229.205.76 master_port=29500 main.py
python3 -m torch.distributed.launch --nproc_per_node=1 --nnode=1 --node_rank=0 --master_addr="34.229.205.76" --master_port=5555 distributed_resnet50_cifar10.py --batch_size=256 --JobID Job0


python3 -m torch.distributed.launch --nproc_per_node=1 --nnode=1 --node_rank=0 --master_addr='34.229.205.76' master_port=29500 main.py
"""

