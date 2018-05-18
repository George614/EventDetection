# -*- coding: utf-8 -*-
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matlab.engine
import numpy as np
torch.manual_seed(1) # reproducible
from cnn1d_lstm import CNN1D_LSTM

# Paths to the dataset
DATA_PATH = "F:\Event detection\George_ED_Mark1\\"
#DATA_PATH = "D:\Dataset\Event detection\George_ED_Mark1\\"

DATA_0by0 = "Data_0_0.mat"
DATA_3by3 = "Data_3_3.mat"
DATA_6by6 = "Data_6_6.mat"
DATA_9by9 = "Data_9_9.mat"
DATA_12by12 = "Data_12_12.mat"
DATA_15by15 = "Data_15_15.mat"
DATA_18by18 = "Data_18_18.mat"

# start a matlab engine to laod data from .mat file
mateng = matlab.engine.start_matlab()
dataset = mateng.load(DATA_PATH + DATA_3by3)
Weights = dataset['Weights']
TrainData = dataset['TrainData']
ID = dataset['ID']
Targets = dataset['Targets']

# convert matlab array to numpy arrays
def mlarray2nparray(mlarray):
    nparray = []
    for element in mlarray:
        element = np.array(element)
        nparray.append(element)
    return nparray

Weights = mlarray2nparray(Weights)
TrainData = mlarray2nparray(TrainData)
ID = mlarray2nparray(ID)
Targets = mlarray2nparray(Targets)

# Hyper-parameters 
BATCH_SIZE = 24
learning_rate = 0.005
num_classes = len(np.unique(Targets[0]))-1
hidden_size = 20
kernel_size = 3
num_cnn_filters = 128
EPOCHS = 10
num_features = np.shape(TrainData[0])[1]

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# inherit from the torch.utils.data, used for create a dataset class
class EDDataset(Data.Dataset):
    def __init__(self,trainData,trainLabels,trainWeights):
        self.trainData = trainData
        self.trainLabels = trainLabels
        self.trainWeights = trainWeights
    def __len__(self):
        return len(self.trainLabels)
    def __getitem__(self,idx):
        trainSample = self.trainData[idx,:,:]
        trainLabel  = self.trainLabels[idx]
        trainWeight = self.trainWeights[idx]
        sample = {'data':trainSample,'label':trainLabel,'Weight':trainWeight}
        return sample

#for i in len(TrainData):
i = 0 # which trial to use as the testing data
testData = TrainData[i]
trainData = TrainData[:i]+TrainData[i+1:]
trainData = np.vstack(trainData)
testWeights = Weights[i]
testWeights = np.squeeze(testWeights)
trainWeights = Weights[:i]+Weights[i+1:]
trainWeights = np.vstack(trainWeights)
trainWeights = np.squeeze(trainWeights)
testLabels = Targets[i]
testLabels = np.squeeze(testLabels)
trainLabels = Targets[:i]+Targets[i+1:]
trainLabels = np.vstack(trainLabels)
trainLabels = np.squeeze(trainLabels)
# change label==5 into label==1
indices = np.argwhere(testLabels==5)
testLabels[indices] = 1
indices2 = np.argwhere(trainLabels==5)
trainLabels[indices2] = 1

# calculate the number in each class
label_types,num_per_class = np.unique(trainLabels,return_counts=True)
# weights for each class during training 
W = Variable(torch.Tensor(1-num_per_class/np.sum(num_per_class)),requires_grad=False)

# create the CNN_LSTM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = CNN1D_LSTM(num_features,num_cnn_filters,kernel_size,hidden_size,num_classes).to(device)
#net = torch.nn.DataParallel(net,device_ids=range(torch.cuda.device_count()))
#cudnn.benchmark = True

# loss function 
loss_func = nn.CrossEntropyLoss(weight=W).to(device)
# optimizer
optimizer = optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-2)

# Data Loader for easy mini-batch return in training
trainData = torch.from_numpy(trainData).float().to(device)
trainLabels = torch.from_numpy(trainLabels).long().to(device)
trainWeights = torch.from_numpy(trainWeights).float().to(device)
torch_dataset = EDDataset(trainData,trainLabels,trainWeights)
train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)
#trainSet = Data.ConcatDataset(TrainData)
#trainW = Data.ConcatDataset(trainWeights)
#trainTarget = Data.ConcatDataset(trainLabels)

# convert testing data into tensor
testData = torch.from_numpy(testData).float().to(device)
testLabels = torch.from_numpy(testLabels).long().to(device)

# training loop
for epoch in range(EPOCHS):
    for step, train_dict in enumerate(train_loader):
        net.train()
#        net.reset(BATCH_SIZE)
        net.zero_grad()
        data_batch = torch.tensor(train_dict['data'],requires_grad=True).to(device)
        label_batch = torch.tensor(train_dict['label']).to(device)
        output = net(data_batch)
        loss = loss_func(output,label_batch)
#        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step%1000==0:
            net.eval()
            with torch.no_grad():
                test_output = net(testData).to(device)
                pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
                accuracy = torch.sum(pred_y == testLabels).float().item() / float(testLabels.size(0))
                print('Epoch: ', epoch, 'step: ',step,'| train loss: %.4f' % loss.item(), '| test accuracy: %.5f' % accuracy)
        
