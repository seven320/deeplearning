
# coding: utf-8

# cnn model

# train
# data loader 作成

# In[1]:


import os
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

path_train = "/Users/kenkato/python/kaggle/06.dogs_vs_cats/train_train/"
path_test = "/Users/kenkato/python/kaggle/06.dogs_vs_cats/train_test/"

class DogCatDataset(Dataset):
    def __init__(self,path,transform=None):
        self.path = path
        self.transform = transform
        
    def __getitem__(self,index):
        img = Image.open(self.path+str(index)+".jpg")
        if index < 10000:
            label = 0
        else:
            label = 1
            
        if self.transform:
            img = self.transform(img)
            
        return [img,label]
    
    def __len__(self):
        dir = self.path
        files = os.listdir(dir)
        count = 0
        for file in files:
            index = re.search(".jpg",file)
            if index:
                count += 1
        return count

#transform part
# class
    
train_data_set = DogCatDataset(
    path = path_train,
    transform=transforms.Compose([
                        transforms.Resize((296,296)),
                        transforms.ToTensor()
                        ])
    )

train_dataloader = DataLoader(train_data_set,
                       batch_size=1,
                       shuffle=True)

for i,(image,label) in enumerate(train_dataloader):
    print(label)
    if i > 4:
        break


# test
# data loader 作成

# In[2]:




class DogCatDatasetTest(Dataset):
    def __init__(self,path,transform=None):
        self.path = path
        self.transform = transform
        
    def __getitem__(self,index):
        img = Image.open(self.path+str(index)+".jpg")
        if index < 2500:
            label = 0
        else:
            label = 1
            
        if self.transform:
            img = self.transform(img)
            
        return [img,label]
    
    def __len__(self):
        dir = self.path
        files = os.listdir(dir)
        count = 0
        for file in files:
            index = re.search(".jpg",file)
            if index:
                count += 1
        return count


test_data_set = DogCatDatasetTest(
    path = path_test,
    transform=transforms.Compose([
        transforms.Resize((296,296)),
        transforms.ToTensor()
        ])
    )

test_dataloader = DataLoader(test_data_set, 
                        batch_size=1,
                        shuffle=True)
                        

for i,(img,label) in enumerate(test_dataloader):
    print(label)
    print(img.size())
    if i > 4:
        break
# image,label = dogcatdata_loader[1]
# print(image.size)


# Network作成 3*92*92

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 6)
        self.pool = nn.MaxPool2d(4,4)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*17*17, 120)
        self.fc2 = nn.Linear(120, 10)
        self.fc3 = nn.Linear(10, 2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)      
        x = x.view(-1, 16*17*17)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x    


# model 01 画像処理サイズ224*224
# model 02 バッチサイズ1 92*92 70%
# model 03 

# In[13]:


model = Net()
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epocs in range(2):
    for i,(image,label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        output = model(image)
        
        loss = criterion(output,label)
        loss_show = loss.data.numpy()
        
        loss.backward()
        optimizer.step()
        torch.save(model.state_dict(),"model03.pth")

        if i % 100 == 0:
            print("epocs:",epocs,"number:",i,"loss:",loss_show)


# 精度

# In[14]:


model = Net()
# model.eval()

param = torch.load("model03.pth")
model.load_state_dict(param)


count = 0
for i,(image,label) in enumerate(test_dataloader):
    output = model(image)
    output = output.data.numpy().argmax() 
    label = label.data.numpy()
    
    if output == label:
        count += 1
    if i % 1000 == 0 and i != 0:
        print(count/i*100,"%")

