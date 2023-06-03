import numpy as np


# Part 1 - get from AWS websites
instance_type=['p2.xlarge','g4dn.4xlarge','g3.8xlarge','g3.16xlarge','p2.8xlarge']
instance_gpus=[1,1,2,4,8]
instance_quota = np.array([2, 5, 5, 8, 5])
instance_spot_price = np.array([0.918, 0.3612, 0.6840, 1.5676, 2.1600])
bps = 1200
instance_type_ps=['m5.xlarge']

# Part 2 - hp
# aws
key = "tf-faye"
image_id = "ami-0f31a41ee53fb0d9b"
    # "ami-0921a2d983425b50e"
    # "ami-0615e79764448550f"
subnet_id = "subnet-b6716799"
SecurityGroupIds = "sg-0d9c3d7232d3f62f8"
# axure
resource_group = "mySpotGroup"
image = "UbuntuLTS"
admin_username = "azureuser"
ssh_key_name = "Azure-Faye"
vnet_name = "myVMVNET"
subnet = "myVMSubnet"
# gcp
zone = "us-west4-b"
network = "default"
subnet_gcp = "default"

# Part 3 - profile using profiler
instance_speed = np.array([1885, 750, 1018, 1995, 2115])
instance_batch = np.array([256, 128, 512, 512, 1024])
instance_time = instance_batch * instance_gpus / instance_speed

param = 11.5
instance_bandwith = 142
instance_comp = instance_time - (2 * param/instance_bandwith - 2 * np.array(instance_gpus) * param/ 10000)

a = 0.8
b = 1.92

r1 = 876.5
r2 = -0.2
r3 = 2507.6
r4 = 0.5

starttimestamp=0
objtimestamp=0