#encoding utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import *

criterion = nn.NLLLoss()

LEARNING_RATE = 0.005


##########
#  RNN  #
##########
class RNN(nn.Module):
    def __init__(self,data_size,hidden_size,output_size):
        super(RNN,self).__init__()

        self.hidden_size = hidden_size
        input_size = data_size + hidden_size

        self.i2h = nn.Linear(input_size,hidden_size)
        self.h2o = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.00015, alpha=0.95, eps=0.01)

    def __call__(self,data,last_hidden):
        input = torch.cat((data,last_hidden),1)
        hidden = F.relu(self.i2h(input))
        output = self.h2o(hidden)
        output = self.softmax(output)

        return output,hidden

    def initHidden(self):
        return torch.zeros(1,self.hidden_size)



#RNN model
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)


##########
#  LSTM  #
##########
class LSTM(nn.Module):
    def __init__(self,data_size,hidden_size,output_size):
        super(LSTM,self).__init__()
        self.output_size = output_size
        #
        self.lstm = nn.LSTM(data_size,hidden_size,batch_first=True)
        self.hidden2output = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.00015, alpha=0.95, eps=0.01)

    def __call__(self, inputs, hidden = None):
        #lstmではtorchのモデルを使えるので入力に全入力情報を一度に渡せる
        output, (hn,cn) = self.lstm(inputs,hidden)
        #今回のlstmは名前１セットに対して最後のoutputしか使わないので[-1(次元残せる):,-1,:]
        output = self.hidden2output(output[-1:, -1, :])
        output = self.softmax(output)
        return output
#LSTM model
# lstm = torch.nn.LSTM(n_letters, n_hidden, n_categories)
lstm = LSTM(n_letters, n_hidden, n_categories)

hidden = torch.zeros(1,n_hidden)
cell = torch.zeros(1,n_hidden)

# output = lstm()








# class LSTM(nn.Module):
#     def __init__(self,data_size,hidden_size,output_size):
#         super(RNN,self).__init__()
#
#         self.hidden_size = hidden_size
#         self.input_size = data_size + hidden_size
#
#         self.input2f_t= nn.Linear(input_size,hidden_size)
#         self.input2i_t = nn.Linear(input_size,hidden_size)
#         self.input2z_t = nn.Linear(input_size,hidden_size)
#         self.input2o_t = nn.Linear(input_size,hidden_size)
#
#         self.softmax = nn.LogSoftmax(dim = 1)
#         self.optimizer = optim.RMSprop(self.parameters(), lr=0.00015, alpha=0.95, eps=0.01)
#
#     def __call__(self,hidden,input):
#         combined = torch.cat((data,hidden),1)
#         f_t = F.sigmoid(self.input2f_t(combined))
#         i_t = F.sigmoid(self.input2i_t(combined))
#         z_t = F.Tanh(self.input2z_t(combined))
#         o_t = F.sigmoid(self.input2o_t(combined))
#
#         output = self.softmax(hidden)
#         return hidden,output
