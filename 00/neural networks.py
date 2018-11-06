
# coding: utf-8

# In[30]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        #1input image channel, 6 output channels, 5*5 square convolution
        #画像のサイズによらずチャンネルはデータの厚みである、
        self.conv1 = nn.Conv2d(1, 6, (5, 5))
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        #an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
net = Net()
print(net)

#convlusion層とpooling層が前処理の様な形になっている。
# NN自体は最後のlinearとRELU関数でできる


# In[31]:


params = list(net.parameters())
print(len(params))
for i in range(len(params)):
    print(params[i].size())


# In[32]:


input = torch.randn(1,1,32,32)
out = net(input)
print(out)


# In[16]:


net.zero_grad()
out.backward(torch.randn(1,10))


# In[33]:


output = net(input)
target = torch.randn(10)
# print(target.size())
target = target.view(1, -1)
# print(target.size())
#MSE Mean Squared Error 平均２乗誤差
criterion = nn.MSELoss()
# MAE Mean Absolute Error
# RMSE Root Mean Squared Error 平均平方２乗誤差
#criterion: 標準
loss = criterion(output, target)
print(output)
print(target)
print(loss)


# In[34]:


print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


# In[35]:


net.zero_grad()

print("conv1.bias.grad before backward")
print(net.conv1.bias.grad)

loss.backward()

print("conv1.bias.grad after backward")
print(net.conv1.bias.grad)


# In[37]:


learning_rate = 0.01


print(net.parameters())
for f in net.parameters():
    f.data.sub_(f.grad.data*learning_rate)


# In[38]:


import torch.optim as optim

#create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()



