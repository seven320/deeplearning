#encoding utf-8

#参考文献 http://caffe.classcat.com/2018/05/12/pytorch-tutorial-intermediate-char-rnn-classification/

import copy
import random
import time
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from data import *
from model import *

N_HIDDEN = 128
EPOCHS = 10**5*3
LOG_FREQ = 500
PLOT_NUM = 1000

current_loss = 0
all_losses = []

start = time.time()
answer_rate = 0

#tensorのアウトプットデータからカテゴリーに変換
def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1)
    category_i = top_i[0][0]
    return all_categories[category_i],category_i

def randomChoice(l):
    return l[random.randint(0,len(l)-1)]

def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.LongTensor([all_categories.index(category)])
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def randomChoice2(l):
    number = random.randint(0,len(l)-1)
    return int(number),l[random.randint(0,len(l)-1)]


copy_all_categories = copy.deepcopy(all_categories)
copy_category_lines = copy.deepcopy(category_lines)

def allTrainingPair():
    done = False
    category_num, category = randomChoice2(copy_all_categories)
    line_num, line = randomChoice2(copy_category_lines[category])
    category_tensor = torch.LongTensor([copy_all_categories.index(category)])
    line_tensor = lineToTensor(line)
    del copy_category_lines[category][line_num]

    if not category in copy_category_lines:
        copy_category_lines.pop(category_num)
    if len(copy_category_lines) == 0:
        done = True

    return category, line, category_tensor, line_tensor, done


def train_rnn(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i],hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    rnn.optimizer.step()

    return output, loss.item()

def train_lstm(category_tensor, line_tensor):
    lstm.optimizer.zero_grad()

    output = lstm(line_tensor,hidden = None)

    loss = criterion(output, category_tensor)
    loss.backward()
    lstm.optimizer.step()

    return output, loss.item()

#train

correct = 0
all_corrects = []
current_loss = 0
all_losses = []


start = time.time

print("-"*30)
print("epoch total_step loss correct_rate ")

hidden = rnn.initHidden()
rnn.optimizer.zero_grad()

for epoch in range(1,EPOCHS+1):
    #random train
    category, line, category_tensor, line_tensor= randomTrainingPair()

    output_tensor,loss = train_rnn(category_tensor,line_tensor)
    # output_tensor,loss = train_lstm(category_tensor,line_tensor)

    current_loss += loss

    pre_category, pre_category_i = categoryFromOutput(output_tensor)
    if category == pre_category:
        correct += 1

    if epoch % LOG_FREQ == 0:
        print(epoch, current_loss/LOG_FREQ, correct/LOG_FREQ*100,"%")

        all_losses.append(current_loss/LOG_FREQ)
        all_corrects.append(correct/LOG_FREQ)
        correct = 0
        current_loss = 0

# torch.save(rnn, "char-lstm-classification.pt")


x = list(i*LOG_FREQ for i in range(int(EPOCHS/LOG_FREQ)))
plt.plot(x,all_corrects)
plt.show()
