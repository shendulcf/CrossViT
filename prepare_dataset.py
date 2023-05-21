import glob
import os
import sys

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from PIL import Image
import openslide

import timm


class ColorectalCancerDataset(Dataset):
    
    def __init__(self, root_dir, **kwargs):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_images()
        print(self.class_to_idx)
    
    def _load_images(self):
        images = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for file_name in os.listdir(cls_dir):
                if file_name.endswith('tif'):
                    image_path = os.path.join(cls_dir, file_name)
                    images.append((image_path, self.class_to_idx[cls]))
        return images
    
    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        image = self._load_image(image_path)
        return image, label
    
    def __len__(self):
        return len(self.images)
    
    def _load_image(self, image_path):
        # load the image and preprocess it
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        # image = openslide.open_slide(image_path)
        image = Image.open(image_path)
        image = transform(image)
        return image




