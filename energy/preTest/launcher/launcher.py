# coding:utf-8
import json
import subprocess
import time
import random
import string
# aws
from energy.preTest.launcher import instanceinfo

image_id = instanceinfo.image_id
subnet_id = instanceinfo.subnet_id
SecurityGroupIds = instanceinfo.SecurityGroupIds
key = instanceinfo.key

# azure
resource_group = instanceinfo.resource_group
image = instanceinfo.image
admin_username = instanceinfo.admin_username
ssh_key_name = instanceinfo.ssh_key_name
vnet_name = instanceinfo.vnet_name
subnet = instanceinfo.subnet

# gcp
zone = instanceinfo.zone
network = instanceinfo.network
subnet_gcp = instanceinfo.subnet_gcp


def getSpotInstance(instance_type, count):
    instance_id = ""
    hosts = []
    public_ips = []
    private_ips = []
    batchs = []
    instance_type_index = []

    print("\n****************** apply for spot instances ********************")

    for i in range(len(instance_type)):
        with open("launcher/specification.json.template", 'r') as f1, open("launcher/instanceSpec.json", 'w') as f2:
            template = f1.read()
            specification_file = template % (image_id, key, SecurityGroupIds, instance_type[i], subnet_id)
            f2.write(specification_file)

        if count[i] == 0:
            continue
        print("\n %d %s instances ... " % (count[i], instance_type[i]))
        command = """aws ec2 request-spot-instances --instance-count %d \
                                                    --type one-time \
                                                    --launch-specification file://launcher/instanceSpec.json""" % (
        count[i])

        spot_instance_request_id = ""
        try:
            output = subprocess.check_output(command, shell=True).decode()
            return_obj = json.loads(output)
            for j in range(count[i]):
                rid = return_obj["SpotInstanceRequests"][j]["SpotInstanceRequestId"]
                spot_instance_request_id = spot_instance_request_id + ' ' + rid
            # print(" spot_instance_request_id: ",spot_instance_request_id)
            command = """aws ec2 describe-spot-instance-requests --spot-instance-request-id %s""" % (
                spot_instance_request_id)
            time.sleep(10)

            try:
                output = subprocess.check_output(command, shell=True).decode()
                return_obj = json.loads(output)
                instanceid_one_type = ""
                for j in range(count[i]):
                    id = return_obj["SpotInstanceRequests"][j]["InstanceId"]
                    instance_id = instance_id + ' ' + id
                    instanceid_one_type = instanceid_one_type + ' ' + id

            except Exception as e:
                print("\n instance id not enough !")

            if instanceid_one_type is not None:
                command = """aws ec2 describe-instances --instance-ids %s \
                                                    --query "Reservations[*].Instances[*].{hosts:PublicIpAddress,hosts_private:PrivateIpAddress}" \
                                                    --output json""" % (instanceid_one_type)

                try:
                    output = subprocess.check_output(command, shell=True).decode()
                    return_obj = json.loads(output)
                    for j in range(count[i]):
                        public_ips.append(return_obj[0][j]["hosts"])
                        private_ips.append(return_obj[0][j]["hosts_private"])
                        batchs.append(instanceinfo.instance_batch[i])
                        instance_type_index.append(i)

                except Exception as e:
                    print("\n Failed to get ip address!")

        except Exception as e:
            print("\n Failed to get request id !")

        if spot_instance_request_id is not None:
            command = """aws ec2 cancel-spot-instance-requests --spot-instance-request-ids %s""" % (
                spot_instance_request_id)
            subprocess.check_output(command, shell=True)

    hosts.append(public_ips)
    hosts.append(private_ips)
    hosts.append(batchs)
    hosts.append(instance_type_index)

    print("\n----------------- result -----------------")
    print("\npublic_ips:", hosts[0])
    print("\nprivate_ips:", hosts[1])
    print("\ninstance_ids:", instance_id)

    print("\n******************** WRITE INSTANCES INFO INTO FILE !!*********************\n")

    with open("launcher/instancesInfo.txt", "w") as f:
        for item in hosts[0]:
            f.write(item + ' ')
        f.write('\n')
        for item in hosts[1]:
            f.write(item + ' ')
        f.write('\n')
        for item in hosts[2]:
            f.write(str(item) + ' ')
        f.write('\n')
        for item in hosts[3]:
            f.write(str(item) + ' ')
        f.write('\n')
        f.write(instance_id)

    return hosts


def getDemandInstance(instance_type, count, saveFile):
    instance_id = ""
    hosts = []
    public_ips = []
    private_ips = []
    batchs = []
    instance_type_index = []

    print("\n******************** apply for on-demand instance **********************")

    for i in range(len(instance_type)):
        print("\n %d 个 %s instances ... " % (count[i], instance_type[i]))
        command = """aws ec2 run-instances --instance-type %s \
                                               --count %d \
                                               --image-id %s \
                                               --key-name tf-faye \
                                               --security-group-ids %s \
                                               --subnet-id %s""" % (
            instance_type[i], count[i], image_id, SecurityGroupIds, subnet_id)

        try:
            output = subprocess.check_output(command, shell=True).decode()
            return_obj = json.loads(output)
            instanceid_one_type = ""
            for j in range(count[i]):
                id = return_obj["Instances"][j]["InstanceId"]
                instance_id = instance_id + ' ' + id
                instanceid_one_type = instanceid_one_type + ' ' + id

            command = """aws ec2 describe-instances --instance-ids %s \
                                                        --query "Reservations[*].Instances[*].{hosts:PublicIpAddress,hosts_private:PrivateIpAddress}" \
                                                        --output json""" % (instanceid_one_type)

            try:
                print('test!')
                output = subprocess.check_output(command, shell=True).decode()
                return_obj = json.loads(output)
                for j in range(count[i]):
                    public_ips.append(return_obj[0][j]["hosts"])
                    private_ips.append(return_obj[0][j]["hosts_private"])
                    batchs.append(instanceinfo.instance_batch[i])
                    instance_type_index.append(i)

            except Exception as e:
                print("\n 获取 instance ip 失败 !")

        except Exception as e:
            print("\n instanceid 数量不足!")

    hosts.append(public_ips)
    hosts.append(private_ips)
    hosts.append(batchs)
    hosts.append(instance_type_index)

    print("\n----------------- result -----------------")
    print("\npublic_ips:", hosts[0])
    print("\nprivate_ips:", hosts[1])
    print("\ninstance_ids:", instance_id)

    print("\n******************** WRITE INSTANCES INFO INTO FILE !!*********************\n")

    with open(saveFile, "w") as f:
        for item in hosts[0]:
            f.write(str(item) + ' ')
        f.write('\n')
        for item in hosts[1]:
            f.write(str(item) + ' ')
        f.write('\n')
        for item in hosts[2]:
            f.write(str(item) + ' ')
        f.write('\n')
        for item in hosts[3]:
            f.write(str(item) + ' ')
        f.write('\n')
        f.write(instance_id)
        f.write('\n')
        for item in instance_type:
            f.write(item + ' ')

    return hosts


def getSpotVM_Azure(vm_size, count):
    # requst one
    privateIpAddress = []
    publicIpAddress = []
    for i in range(len(vm_size)):
        if count[i] == 0:
            continue
        elif count[i] == 1:
            vm_name = ''.join(random.sample(string.ascii_letters + string.digits, 10))
            command = """az vm create \
                            --resource-group %s \
                            --name %s \
                            --image %s \
                            --admin-username %s \
                            --ssh-key-name %s \
                            --priority Spot \
                            --max-price -1 \
                            --eviction-policy Deallocate \
                            --size %s \
                            --vnet-name %s \
                            --subnet %s
                      """ % (
            resource_group, vm_name, image, admin_username, ssh_key_name, vm_size[i], vnet_name, subnet)

            output = subprocess.check_output(command, shell=True).decode()
            return_obj = json.loads(output)

            privateIpAddress.append(return_obj["privateIpAddress"])
            publicIpAddress.append(return_obj["publicIpAddress"])


        else:
            vm_name = ''.join(random.sample(string.ascii_letters + string.digits, 10))
            command = """az vm create \
                            --resource-group %s \
                            --name %s \
                            --image %s \
                            --admin-username %s \
                            --ssh-key-name %s \
                            --priority Spot \
                            --max-price -1 \
                            --eviction-policy Deallocate \
                            --size %s \
                            --vnet-name %s \
                            --subnet %s \
                            --count %d
                      """ % (
            resource_group, vm_name, image, admin_username, ssh_key_name, vm_size[i], vnet_name, subnet, count[i])

            output = subprocess.check_output(command, shell=True).decode()
            return_obj = json.loads(output)

            for j in range(count[i]):
                privateIpAddress.append(return_obj[j]["privateIps"])
                publicIpAddress.append(return_obj[j]["publicIps"])

    print("priveta IP address: ", privateIpAddress)
    print("public IP address: ", publicIpAddress)

    print("\n******************** WRITE INSTANCES INFO INTO FILE !!*********************\n")

    with open("launcher/instancesInfo.txt", "w") as f:
        for item in publicIpAddress:
            f.write(item + ' ')
        f.write('\n')
        for item in privateIpAddress:
            f.write(item + ' ')
        f.write('\n')
    return 0


def getSpotInstance_GCP(machine_type, count):
    index = 0
    privateIpAddress = []
    publicIpAddress = []
    for i in range(len(machine_type)):
        if count[i] == 0:
            continue
        else:
            instance_names = "spotvm" + str(index)
            index += 1
            for j in range(count[i] - 1):
                instance_names = instance_names + " " + "spotvm" + str(index)
                index += 1
            command = """gcloud compute instances create %s \
                                --zone=%s \
                                --provisioning-model=SPOT \
                                --machine-type=%s \
                                --network=%s \
                                --subnet=%s
                        """ % (instance_names, zone, machine_type[i], network, subnet_gcp)
            output = subprocess.check_output(command, shell=True)

            instance_names_list = instance_names.split(" ")
            print(instance_names_list)
            for instance in instance_names_list:
                command2 = """gcloud compute instances describe %s --format="json(networkInterfaces)"
                            """ % (instance)
                output2 = subprocess.check_output(command2, shell=True).decode()
                return_obj = json.loads(output2)
                privateIpAddress.append(return_obj["networkInterfaces"][0]["networkIP"])
                publicIpAddress.append(return_obj["networkInterfaces"][0]["accessConfigs"][0]["natIP"])
    print("priveta IP address: ", privateIpAddress)
    print("public IP address: ", publicIpAddress)

    print("\n******************** WRITE INSTANCES INFO INTO FILE !!*********************\n")

    with open("launcher/instancesInfo.txt", "w") as f:
        for item in publicIpAddress:
            f.write(item + ' ')
        f.write('\n')
        for item in privateIpAddress:
            f.write(item + ' ')
        f.write('\n')
    return 0


