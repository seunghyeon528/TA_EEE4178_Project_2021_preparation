import numpy as np
import torch
import os
import glob

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms


os.environ["CUDA_VISIBLE_DEVICES"]="1" # 특정 번호 GPU 만 사용하고 싶을 때
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## -- hyperparameters 모델 isntance 생성에 필요한 argument 들 적어주기
num_classes = 52 
batch_size = 780
in_channel = 1
max_pool_kernel = 2

## -- dataset & dataloader (test -> valid 로 바꿔서 돌아가는 것 확인하고 제출)
class MyDataset(Dataset):
    def __init__(self, npy_dir):
        self.dir_path = npy_dir
        self.to_tensor = transforms.ToTensor()

        # all npy path
        self.npy_path = glob.glob(os.path.join(npy_dir, '*','*.npy')) 

    def __getitem__(self, index):
        # load data
        single_data_path = self.npy_path[index]
        data = np.load(single_data_path, allow_pickle=True)
        
        image = data[0]
        image = self.to_tensor(image)
        label = data[1]
       
        return (image, label)

    def __len__(self):
        return len(self.npy_path)

test_data = MyDataset("./Font_npy_90_test")
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                           batch_size=batch_size,
                                           shuffle=False)



## -- model class
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



## -- load model
model_test = convnet_90().to(device)
model_test.load_state_dict(torch.load("con_best_model_90.pth"))



## -- measure testset accuracy 
model_test.eval()
with torch.no_grad():
    correct = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_test(images)
        _,predicted = torch.max(outputs.data,1)
        correct += (predicted==labels).sum()
        
    # batch size 가 7800의 양수라면 아래 방식으로 해도 정확
#     print('Accuracy of the last_model network on the {} test images: {} %'.\
#           format(len(test_loader)*batch_size, 100 * correct / (len(test_loader)*batch_size)))
            
    # batch size 가 7800 의 양수가 아니라면 아래 방식으로 해야만 정확한 accuracy 얻을 수 있음. 
    print('Accuracy of the last_model network on the {} test images: {} %'.\
          format(len(test_data), 100 * correct / len(test_data)))