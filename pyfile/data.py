import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms

from tqdm.notebook import tqdm   
import random
import numpy as np

random.seed(42)  # 랜덤 시드 고정
np.random.seed(42)  # numpy 랜덤 시드 고정  

class CityscapeDataset(Dataset):

  def __init__(self, image_dir, label_dir, label_model):
    self.image_dir = image_dir
    self.image_fns = os.listdir(image_dir)
    self.label_dir = label_dir
    self.label_fns = os.listdir(label_dir)
    self.label_model = label_model
    
  def __len__(self) :
    return len(self.image_fns)
    
  def __getitem__(self, index) :
    image_fn = self.image_fns[index]
    image_fp = os.path.join(self.image_dir, image_fn)
    label_fn = self.label_fns[index]
    label_fp = os.path.join(self.label_dir, label_fn)
    image = Image.open(image_fp)
    image = np.array(image)
    label = Image.open(label_fp)
    label = np.array(label)
    cityscape, label = self.split_image(image, label)
    label_class = self.label_model.predict(label.reshape(-1, 3)).reshape(540, 960)
    label_class = torch.Tensor(label_class).long()
    cityscape = self.transform(cityscape)
    return cityscape, label_class
    
  def split_image(self, image, label) :
    cityscape, label = np.array(image), np.array(label)
    return cityscape, label
    
  def transform(self, image) :
    transform_ops = transforms.Compose([
      			transforms.ToTensor(),
                        transforms.Normalize(mean = (0.485, 0.56, 0.406), std = (0.229, 0.224, 0.225))
    ])
    return transform_ops(image) 
  

class CityscapeDataset_predict(Dataset):
    def __init__(self, image_dir, label_model):
        self.image_dir = image_dir
        self.image_fns = os.listdir(image_dir)
        self.label_model = label_model

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp)
        image = np.array(image)
        cityscape = self.transform(image)
        return cityscape

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)

def label_model():
      num_items = 1000

      # 0~255 사이의 숫자를 3*num_items번 랜덤하게 뽑기
      color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)

      num_classes = 13

      # K-means clustering 알고리즘을 사용하여 label_model에 저장합니다.
      label_model = KMeans(n_clusters = num_classes)
      label_model.fit(color_array)
      return label_model
    
    
# class CityscapeDataset_image(Dataset):
#     def __init__(self, image, label_model):
#         self.image = image
#         self.label_model = label_model

#     def __len__(self):
#         return 1

#     def __getitem__(self, index):
#         image = self.image
#         cityscape = self.transform(image)
#         return cityscape

#     def transform(self, image):
#         transform_ops = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
#         ])
#         return transform_ops(image)