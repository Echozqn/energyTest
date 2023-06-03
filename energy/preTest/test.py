import numpy as np
import subprocess
# from profile_info import instanceinfo


def SSHCommand(key, ip, cmd, wait):
    print(cmd)
    # command = f'ssh -o StrictHostKeyChecking=no -i {key}.pem ubuntu@{ip} "{cmd}"'
    # if not wait: command += " &"
    # print(f"SSHCommand key = {key} ip = {ip} cmd = {cmd} \ncommand = {command}")
    # subprocess.run(command, shell=True)


def run(hosts, master_port, batch, jobId, openMPS):
    master_ip = hosts[0]
    key = "instanceinfo.key"
    # for ip in hosts:
    #     # pull
    #     pullcmd = "cd PyTorch-Distributed-Training && git pull"
    #     SSHCommand(key, ip, pullcmd, True)


    for i in range(len(hosts)):
        # if openMPS:
        #     # 启动MPS
        #     openMPSCmd = "nvidia-cuda-mps-control -d"
        #     SSHCommand(key, hosts[i], openMPSCmd, False)
        #     # 获得MPSID
        #     getMPSID = "source start.sh"
        #     SSHCommand(key, hosts[i], getMPSID, False)
        #     # 设置资源
        #     cmd = "python3 gpuAllocation.py --gpu-resource 50"
        #     SSHCommand(key, hosts[i], cmd, False)
        #
        # # 启动nvidia-smi监控
        # nvidia_capture_cmd = "nvidia-smi dmon -o T -f caputure.log &"
        # SSHCommand(key, hosts[i], nvidia_capture_cmd ,False)

        cmd = f"source activate pytorch && python -m torch.distributed.launch --nproc_per_node=1 --nnode=2" \
              f" --node_rank={i} --master_addr={master_ip} --master_port={master_port} /home/ubuntu/PyTorch-Distributed-Training/main.py " \
              f"--batch_size={batch[i]} --JobID Job{jobId}"
        SSHCommand(key, hosts[i], cmd, False)

    # if openMPS:
    #     subprocess.run("echo quit | nvidia-cuda-mps-control", shell=True)

run(['3.95.26.137','18.205.103.124'],5555,[256,256],0,False)
"""
python -m torch.distributed.launch --nproc_per_node=1 --nnode=1 --node_rank=0 --master_addr=44.201.175.219 --master_port=5556 /home/ubuntu/PyTorch-Distributed-Training/main.py --batch_size=256 --JobID Job1 

source activate pytorch && 
python -m torch.distributed.launch --nproc_per_node=1 --nnode=2 --node_rank=0 --master_addr=3.95.26.137 --master_port=5555 /home/ubuntu/PyTorch-Distributed-Training/main.py --batch_size=256 
source activate pytorch && 
python -m torch.distributed.launch --nproc_per_node=1 --nnode=2 --node_rank=1 --master_addr=3.95.26.137 --master_port=5555 /home/ubuntu/PyTorch-Distributed-Training/main.py --batch_size=256 

"""