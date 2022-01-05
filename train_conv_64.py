import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms
#from torchsummary import summary

import matplotlib.pyplot as plt
import random
import time
import os
import numpy as np

from font_dataset import MyDataset
import pdb
import argparse

# Device Configuration
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper - parameters
num_classes = 52
in_channel = 1
num_epochs = 10
learning_rate = 0.0005
batch_size = 50
max_pool_kernel = 2

# Fix Seed
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# size option
parser = argparse.ArgumentParser()
parser.add_argument\
('-s', '--size', type = int, default = 90, help = "image size, ex) 128 x 128")
parser.add_argument\
('-v', '--valid', type = bool, default = True, help = "use valid_data for selecting best_model")

args = parser.parse_args()

####################################################################
#                   LOAD DATA & DEFINE DATALOADER
####################################################################

# Load Data
train_data = MyDataset("./Font_npy_{}_train".format(str(args.size)))
valid_data = MyDataset("./Font_npy_{}_val".format(str(args.size)))
test_data = MyDataset("./Font_npy_{}_test".format(str(args.size)))
pdb.set_trace()
# split train,test,valid
# test_percent = 0.2
# valid_percent = 0.1

# pdb.set_trace()
# train_data, test_data = torch.utils.data.random_split\
# (dataset, [len(dataset)-int(len(dataset)*(test_percent)),int(len(dataset)*(test_percent))])

# train_data, valid_data = torch.utils.data.random_split\
# (train_data, [len(train_data)-int(len(train_data)*(test_percent)),int(len(train_data)*(test_percent))])

# Define Dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=False)

pdb.set_trace()


 
####################################################################
#                         DEFINE MODEL
####################################################################
# class convnet_64(nn.Module):
#     def __init__(self, num_classes=50):
#         super(convnet_64, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channel, out_channels=16, kernel_size=7, padding=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, out_channels=32, kernel_size=7, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(32, out_channels=64, kernel_size=5, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(64, out_channels=16, kernel_size=5, padding=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.fc1 = nn.Linear(in_features=144 ,out_features=120)
#         self.fc2 = nn.Linear(in_features=120, out_features=50)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = x.reshape(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return x
max_pool_kernel = 2
class convnet_64(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(convnet_64, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channel, 24, 5, stride=1, padding=2),
                nn.BatchNorm2d(24),
                nn.LeakyReLU(),
                nn.MaxPool2d(max_pool_kernel))
        self.layer2 = nn.Sequential(
                nn.Conv2d(24, 48, 5, stride=1, padding=2),
                nn.BatchNorm2d(48),
                nn.LeakyReLU(),
                nn.MaxPool2d(max_pool_kernel))
        self.layer3 = nn.Sequential(
                nn.Conv2d(48, 64, 5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.MaxPool2d(max_pool_kernel))
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



max_pool_kernel = 2
class convnet_50(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(convnet_50, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channel, 24, 5, stride=1, padding=2),
                nn.BatchNorm2d(24),
                nn.LeakyReLU(),
                nn.MaxPool2d(max_pool_kernel))
        self.layer2 = nn.Sequential(
                nn.Conv2d(24, 48, 5, stride=1, padding=2),
                nn.BatchNorm2d(48),
                nn.LeakyReLU(),
                nn.MaxPool2d(max_pool_kernel))
        self.layer3 = nn.Sequential(
                nn.Conv2d(48, 64, 5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.MaxPool2d(max_pool_kernel))
        self.fc1 = nn.Linear(2304, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class convnet_90(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(convnet_90, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channel, 24, 7, stride=1, padding=2),
                nn.BatchNorm2d(24),
                nn.LeakyReLU(),
                nn.MaxPool2d(max_pool_kernel))
        self.layer2 = nn.Sequential(
                nn.Conv2d(24, 48, 5, stride=1, padding=0),
                nn.BatchNorm2d(48),
                nn.LeakyReLU(),
                nn.MaxPool2d(max_pool_kernel))
        self.layer3 = nn.Sequential(
                nn.Conv2d(48, 64, 5, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.MaxPool2d(max_pool_kernel))
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
if args.size == 50:
    model = convnet_50().to(device)
if args.size == 64:
    model = convnet_64().to(device)   
if args.size == 90:
    model = convnet_90().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

####################################################################
#                       check receptive field
####################################################################
# from receptivefield import receptivefield
# print(receptivefield(model, (1, args.size,)))
####################################################################
#                             TRAIN
####################################################################
model.train()
total_step = len(train_loader)
loss_list = []

def evaluate(model, data_loader, criterion):
  epoch_loss, acc = 0, 0
  model.eval()

  for i, (images, labels) in enumerate(data_loader):
    images = images.to(device)
    labels = labels.to(device).long()
    output = model(images)
    loss = criterion(output, labels)

    epoch_loss += loss.item()
    _, pred = torch.max(output.data, 1)
    acc += (pred == labels).sum().item()
      
  return round(epoch_loss/(i+1),4) ,\
         round(acc / len(data_loader.dataset) * 100,2)

start = time.time()
best_acc = 0
best_epoch = 0
best_loss =  10
pdb.set_trace()
for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):
        # Assign Tensors to Configured Device
        images = images.to(device)
        labels = labels.to(device).long()
        
        # Forward Propagation
        outputs = model(images)

        # Get Loss, Compute Gradient, Update Parameters
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item()) # .item() : Tensor -> number

        if (i+1) % 300 == 0:
          print("Epoch [{}/{}], Step[{}/{}], Train Loss:{:.4f}".\
                format(epoch+1, num_epochs, i+1, total_step, loss.item()))
          

          if args.valid :
            valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
            print("Epoch [{}/{}], Step[{}/{}], Valid Loss:{:.4f}".\
                  format(epoch+1, num_epochs, i+1, total_step, valid_loss))
            # check if this one is best model
            if valid_loss <= best_loss:
                torch.save(model.state_dict(), 'con_best_model_{}.pth'.format(args.size))
                best_loss = valid_loss
                best_epoch = epoch
          

# Print training time
if epoch+1 == num_epochs:
  end = time.time()
  print("Train takes {:.2f} minutes".format((end-start)/60))
  torch.save(model.state_dict(), "con_last_model_{}.pth".format(args.size))



####################################################################
#                             TEST
####################################################################
pdb.set_trace()
if args.size == 50:
    model_test = convnet_50().to(device)
if args.size == 64:
    model_test = convnet_64().to(device)   
if args.size == 90:
    model_test = convnet_90().to(device)
if args.size == 96:
    model_test = convnet_90().to(device)
# modify structure
# model_test.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
#                         bias=False) # for channel 1 data
# model_test.fc = nn.Linear(512, num_classes)
# model_test = model_test.to(device)

# load pre-trained model
if args.valid:
  print("best epoch :{}".format(best_epoch))
  model_test.load_state_dict(torch.load('con_best_model_{}.pth'.format(args.size)))
else: 
  model_test.load_state_dict(torch.load('con_last_model_{}.pth'.format(args.size))) 

# test  
model_test.eval()
with torch.no_grad():
    correct = 0
    pdb.set_trace()
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_test(images)
        _,predicted = torch.max(outputs.data,1)
        correct += (predicted==labels).sum()
        
    print('Accuracy of the last_model network on the {} test images: {} %'.\
          format(len(test_loader)*batch_size, 100 * correct / (len(test_loader)*batch_size)))