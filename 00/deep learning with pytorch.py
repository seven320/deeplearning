
# coding: utf-8

# import torch
# attribute? 属性
# implementation 実装
# acyclic graph 非循環グラフ
# derivative 派生

# In[2]:


import torch


# Create a tensor and set requires_grad=True to track computation with it

# In[25]:


x = torch.ones(2, 2, requires_grad=True)
print(x)


# Do an operation of tensor

# In[26]:


y = x + 2
print(y)
middle = y.mean()
middle.backward()
print(x.grad)


# In[19]:


print(y.grad_fn)


# Do more operations on y

# In[32]:


x = torch.ones(2,2,requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

print(z, out)

print(z.grad_fn)
#out のバックプロパゲーションを計算
out.backward()
#xに関しての微分を表示
print(x.grad)


# In[12]:


a = torch.randn(2,2)
a = ((a*3) / (a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)


# In[46]:


x = torch.randn(3, requires_grad=True)

y = x*2
# y.data.norm()はベクトルの長さを表す。
while y.data.norm() < 1000:
    print(y)
    y = y*2
print(y)


# In[47]:


gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)
print(x.grad)


# In[48]:


print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad():
    print((x**2).requires_grad)


# In[45]:




