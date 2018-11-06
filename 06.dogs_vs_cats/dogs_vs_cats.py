
# coding: utf-8

# 猫犬の写真判定

# リサイズ
# 
# 調べた結果
# x = 42
# y = 33
# が最小なのでそれに合わせてリサイズを行う
# 

# In[33]:


#encoding utf-8

from PIL import Image
path = "/Users/kenkato/python/kaggle/06.dogs_vs_cats/train/"
min_size_x = 10**5
min_size_y = 10**5

# cats part
path = "/Users/kenkato/python/kaggle/06.dogs_vs_cats/train/"

min_size_x = 10**5
min_size_y = 10**5
animals = ["cat","dog"]
for animal in animals:
    for i in range(12500):
        img = Image.open(path+animal+"."+str(i)+".jpg")
        size = img.size
        if size[0] < min_size_x:
            min_size_x = size[0]
        if size[1] < min_size_y:
            min_size_y = size[1]
    
print(min_size_x,min_size_y)


# リサイズと名前変更
# 0~12499までは猫
# 12500~25000までは犬のファイルを作る

# In[38]:


path_save = "/Users/kenkato/python/kaggle/06.dogs_vs_cats/resize/"
for animal in animals:
    for i in range(12500):
        img = Image.open(path+animal+"."+str(i)+".jpg")
        img_resize = img.resize((42,33))
        if animal == "dog":
            i += 12500
        img_resize.save(path_save+str(i)+".jpg")


# テスト部分についてもリサイズ

# In[12]:


path = "/Users/kenkato/python/kaggle/06.dogs_vs_cats/test/"
path_save = "/Users/kenkato/python/kaggle/06.dogs_vs_cats/resize_test/"
for i in range(1,12501):
        img = Image.open(path+str(i)+".jpg")
        img_resize = img.resize((42,33))
#         if animal == "dog":
#             i += 12500
        img_resize.save(path_save+str(i)+".jpg")


# DataSet作成
# 
# label 0:cat, 1:dog
# image Img

# In[4]:


import os
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

class DogCatDataset(Dataset):
    def __init__(self,path,transform=None):
        self.path = path
        self.transform = transform
        
    def __getitem__(self,index):
        img = Image.open(self.path+str(index)+".jpg")
        
        if index < 12500:
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
    
data_set = DogCatDataset(
    path = "/Users/kenkato/python/kaggle/06.dogs_vs_cats/resize/",
    transform=transforms.Compose([
                        transforms.ToTensor()
                        ])
    )
dataloader = DataLoader(data_set, 
                        batch_size=1,
                        shuffle=True)
                        

for i,data in enumerate(dataloader):
    print(data[0])
    if i > 4:
        break
# image,label = dogcatdata_loader[1]
# print(image.size)


# Network作成
# 1*3*33*42

# In[5]:


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
        self.conv1 = nn.Conv2d(3, 6, (6,15))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)
        self.fc3 = nn.Linear(10, 2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x    


# In[8]:


model = Net()
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epocs in range(2):
    for i,(image,label) in enumerate(dataloader):
        optimizer.zero_grad()
        
        output = model(image)
        
        loss = criterion(output,label)
        loss_show = loss.data.numpy()
        
        loss.backward()
        optimizer.step()
#         torch.sabe(model.state_dict(),"dogcatmodel.pth")
        if i % 100 == 0:
            print("epocs:",epocs,"number:",i,"loss:",loss_show)


# 精度確認

# In[17]:


model = Net()

param = torch.load("dogcatmodel.pth")
model.load_state_dict(param)

count = 0
for i,(image,label) in enumerate(dataloader):
    output = model(image)
    output = output.data.numpy().argmax() 
    label = label.data.numpy()
    
    if output == label:
        count += 1
    if i % 1000 == 0 and i != 0:
        print(count/i*100,"%")


# In[ ]:


model = Net()

param = torch.load("dogcatmodel.pth")
model.load_state_dict(param)

test_num = 25000

for play in range(test_num):
    

