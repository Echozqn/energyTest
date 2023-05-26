import csv
def writeCSV(file_path,data):
    # 将数据写入CSV文件
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)