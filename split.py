
import os
import random
import shutil

# 设置原始数据集路径和划分后数据集的保存路径
dataset_root = r'E:\Workspace\Datasets\NCT-CRC-HE-100K'
train_dir = r'E:\Workspace\Datasets\NCT-CRC-HE\train'
val_dir = r'E:\Workspace\Datasets\NCT-CRC-HE-100K\val'
test_dir = r'E:\Workspace\Datasets\NCT-CRC-HE-100K\test'

# 设置训练集、验证集和测试集的比例
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 创建保存划分后数据集的目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 遍历每个类别的文件夹
for class_name in os.listdir(dataset_root):
    class_dir = os.path.join(dataset_root, class_name)
    if not os.path.isdir(class_dir):
        continue

    # 获取每个类别下的文件列表
    file_list = os.listdir(class_dir)
    random.shuffle(file_list)

    # 计算划分后的数据集大小
    num_files = len(file_list)
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)
    num_test = num_files - num_train - num_val

    # 划分数据集并将文件复制到对应的目录中
    train_files = file_list[:num_train]
    val_files = file_list[num_train:num_train + num_val]
    test_files = file_list[num_train + num_val:]

    for filename in train_files:
        src_path = os.path.join(class_dir, filename)
        dst_path = os.path.join(train_dir, class_name, filename)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)

    for filename in val_files:
        src_path = os.path.join(class_dir, filename)
        dst_path = os.path.join(val_dir, class_name, filename)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)

    for filename in test_files:
        src_path = os.path.join(class_dir, filename)
        dst_path = os.path.join(test_dir, class_name, filename)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)
