import time

import pynvml


GPU_index = 4
handle = pynvml.nvmlDeviceGetHandleByIndex(int(GPU_index))  # 通过GPU索引获取handle
start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

time.sleep(10)

end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
energy = (end_energy - start_energy) / 1000
print(f"Total energy Usage: {energy} J")