import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=64,kernel_size=5,padding=2)
        self.conv2 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=5,padding=2)
        self.fc1 = nn.Linear(1152,100)
        self.fc2 = nn.Linear(100,2)
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.softmax(x)
        return x

class CustomDataset(Dataset):
    def __init__(self):
        self.lc_file = "Lightcurves.csv"
        self.label_path = "Labels.csv"
        self.lightcurves = pd.read_csv(self.lc_file,header=None)
        self.labels = pd.read_csv(self.label_path,header=None)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        lightcurvetemp = self.lightcurves.iloc[idx,:].values
        lightcurvetemp = torch.from_numpy(lightcurvetemp)
        lightcurve = torch.zeros(len(lightcurvetemp))
        count = 0
        for value in lightcurvetemp:
            lightcurve[count] = float(value)
            count = count + 1
        lightcurve = lightcurve.unsqueeze(0)
        labeltemp = self.labels.iloc[idx,:].values
        label = torch.zeros(len(labeltemp))
        count = 0
        for value in labeltemp:
            label[count] = float(value)
            count = count + 1
        #label = label.squeeze(0).squeeze(0)
        return lightcurve, label

trainingset = CustomDataset()
print(trainingset)
trainingloader = DataLoader(trainingset,batch_size=1,shuffle=True)

lightcurve, label = next(iter(trainingloader))
net = Net()
output = net(lightcurve)
print(output)
loss = nn.CrossEntropyLoss()
lossval = loss(output,label.squeeze(0))
print(lossval)



