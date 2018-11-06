
# coding: utf-8

# In[1]:


import pandas as pd

train = pd.read_csv("/Users/kenkato/python/kaggle/kaggle_data/01.titanic/test.csv")
train.info()
train.head()

test = pd.read_csv("/Users/kenkato/python/kaggle/kaggle_data/01.titanic/test.csv")


# In[2]:


def deficit_table(data):
    null_val = data.isnull().sum()
    percent = 100*null_val/len(data)
    kesson_table = pd.concat([null_val,percent],axis=1)
    kesson_table_ren_columns = kesson_table.rename(
    columns = {0:"欠損数",1:"%"})
    return kesson_table_ren_columns

print("train")
print(deficit_table(train))

print("test")
print(deficit_table(test))

