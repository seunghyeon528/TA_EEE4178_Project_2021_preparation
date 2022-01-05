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
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper - parameters
num_classes = 52
in_channel = 1
num_epochs = 10
learning_rate = 0.0005
batch_size = 128
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
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=num_classes, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
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
            torch.save(model.state_dict(), 'scratch_best_model_{}.pth'.format(args.size))
            best_epoch = epoch
          

# Print training time
if epoch+1 == num_epochs:
  end = time.time()
  print("Train takes {:.2f} minutes".format((end-start)/60))
  torch.save(model.state_dict(), "scratch_last_model_{}.pth".format(args.size))



####################################################################
#                             TEST
####################################################################
model_test = ResNet(BasicBlock, [2,2,2,2]).to(device)


if args.valid:
  print("best epoch :{}".format(best_epoch))
  model_test.load_state_dict(torch.load('scratch_best_model_{}.pth'.format(args.size)))
else: 
  model_test.load_state_dict(torch.load('scratch_last_model_{}.pth'.format(args.size))) 

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