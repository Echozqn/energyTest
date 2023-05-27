import csv
import os
def writeCSV(file_path,data):
    # 将数据写入CSV文件
    data = [format(float(x),'.2f') for x in data]
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def remove(file_path):
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 删除文件
        os.remove(file_path)
        print("文件删除成功！")
    else:
        print("文件不存在！")