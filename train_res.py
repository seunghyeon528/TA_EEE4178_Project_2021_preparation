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
learning_rate = 0.001
batch_size = 256
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
parser.add_argument\
('-p', '--pretrain', type = bool, default = False, help = "use pretrained resnet or not")

args = parser.parse_args()

####################################################################
#                   LOAD DATA & DEFINE DATALOADER
####################################################################

# Load Data
train_data = MyDataset("./Font_npy_{}_train".format(str(args.size)))
valid_data = MyDataset("./Font_npy_{}_val".format(str(args.size)))
test_data = MyDataset("./Font_npy_{}_test".format(str(args.size)))

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
from torchvision import models, transforms
model = models.resnet18(pretrained=args.pretrain)
# if args.pretrain :
#   for param in model.parameters():
#     param.requires_grad = False
    
# modify structure
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                        bias=False) # for channel 1 data
model.fc = nn.Linear(512, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



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

        if (i+1) % 100 == 0:
          print("Epoch [{}/{}], Step[{}/{}], Train Loss:{:.4f}".\
                format(epoch+1, num_epochs, i+1, total_step, loss.item()))
          

          if args.valid :
            valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
            print("Epoch [{}/{}], Step[{}/{}], Valid Loss:{:.4f}".\
                  format(epoch+1, num_epochs, i+1, total_step, valid_loss))
            # check if this one is best model
            if valid_acc >= best_acc:
                torch.save(model.state_dict(), 'best_model_{}.pth'.format(args.size))
                best_epoch = epoch
          

# Print training time
if epoch+1 == num_epochs:
  end = time.time()
  print("Train takes {:.2f} minutes".format((end-start)/60))
  torch.save(model.state_dict(), "last_model_{}.pth".format(args.size))



####################################################################
#                             TEST
####################################################################
model_test = models.resnet18(pretrained=False)

# modify structure
model_test.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                        bias=False) # for channel 1 data
model_test.fc = nn.Linear(512, num_classes)
model_test = model_test.to(device)
if args.valid:
  print("best epoch :{}".format(best_epoch))
  model_test.load_state_dict(torch.load('best_model_{}.pth'.format(args.size)))
else: 
  model_test.load_state_dict(torch.load('last_model_{}.pth'.format(args.size))) 

model_test.eval()
with torch.no_grad():
    correct = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_test(images)
        _,predicted = torch.max(outputs.data,1)
        correct += (predicted==labels).sum()
        
    print('Accuracy of the last_model network on the {} test images: {} %'.\
          format(len(test_data), 100 * correct / len(test_data)))