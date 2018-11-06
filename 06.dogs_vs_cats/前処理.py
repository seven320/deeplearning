
# coding: utf-8

# まずトレインデータをtrain_test,train_trainデータに分割する。フォルダごとに分割する。

# In[2]:


#encoding utf-8

from PIL import Image

path = "/Users/kenkato/python/kaggle/06.dogs_vs_cats/train/"
path_save = "/Users/kenkato/python/kaggle/06.dogs_vs_cats/train_train/"
path_test = "/Users/kenkato/python/kaggle/06.dogs_vs_cats/train_test/"
animals = ["cat","dog"]
for animal in animals:
    for i in range(12500):
        img = Image.open(path+animal+"."+str(i)+".jpg")
        if i < 10000:
            if animal == "dog":
                i += 10000
            img.save(path_save+str(i)+".jpg")
        else:
            i -= 10000
            if animal == "dog":
                i += 2500
            img.save(path_test+str(i)+".jpg")

