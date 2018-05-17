# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class CNN1D_LSTM(nn.Module):
    def __init__(self,in_channels,conv_out,kernel_size,hidden_size,out_channels):
        super(CNN1D_LSTM,self).__init__()
        # 1D CNN layers
        self.conv1 = nn.Conv1d(in_channels,32,kernel_size)
        self.conv2 = nn.Conv1d(32,conv_out,kernel_size)
        # paralell LSTM layers
        self.lstm1 = nn.LSTM(input_size = conv_out, hidden_size=hidden_size,num_layers=3, batch_first=True)
        self.lstm2 = nn.LSTM(input_size = conv_out, hidden_size=hidden_size,num_layers=3, batch_first=True)
        # fully connection and softmax layers
        self.fc1 = nn.Linear(2*hidden_size,16)
        self.fc2 = nn.Linear(16,out_channels)
        self.softmax = nn.Softmax()
        
    def reset(self, batch_size):
        self.h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim), requires_grad=False).cuda()
        self.c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim), requires_grad=False).cuda()
        self.h1 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim), requires_grad=False).cuda()
        self.c1 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim), requires_grad=False).cuda()
        
    def forward(self,x):
        #  feed through 1D-CNNs
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(F.relu(self.conv2(x)),2)
        # change the arangement of the tensor
        x = x.permute(0,2,1)
        # feed through LSTMs
        x1,_ = self.lstm1(x,None)
        x2,_ = self.lstm2(x,None)
        # combine the outputs from paralell LSTMs
        x = torch.cat([x1,x2],dim=2)
        # two more FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # retrieve results from the last time step
        x = x[:,-1,:]
        # classification depends on the last output
        x = self.softmax(x)
        return x
        
