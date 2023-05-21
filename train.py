import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import timm

from prepare_dataset import ColorectalCancerDataset
from PIL import Image

# ----> hyperparametes
batch_size = 32
epoches = 1
num_classes = 9
batch_size = 32
num_epochs = 10
learning_rate = 0.001


# 设置数据集路径
dataset_root = r'E:\Workspace\Datasets\NCT-CRC-HE-100K'

# 创建自定义数据集实例
dataset = ColorectalCancerDataset(dataset_root)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# image = transforms.ToPILImage(dataset[0][0])


# model_crossvit = timm.list_models('*crossvit*',pretrained=True)
# print(model_crossvit)
model = timm.create_model('crossvit_base_240',pretrained=True,num_classes=9,
    drop_rate=0.1,
    attn_drop_rate=0.0,
    drop_path_rate=0.1)


# 设置训练和验证数据集路径
train_dataset_root = 'path/to/your/train/dataset'
val_dataset_root = 'path/to/your/validation/dataset'

# 设置模型保存路径和一些训练参数
model_save_path = 'path/to/save/your/model.pth'


# 创建训练和验证数据集实例
train_dataset = ColorectalCancerDataset(train_dataset_root)
val_dataset = ColorectalCancerDataset(val_dataset_root)

# 创建数据加载器
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 将模型移至GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计训练损失和准确率
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_dataset)
    train_accuracy = train_correct / len(train_dataset)

    # 在验证集上进行评估
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item
