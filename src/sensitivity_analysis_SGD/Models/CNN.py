'''
Created on Jan 8, 2020


'''
import torch
import sys, os

from torch import nn, optim

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
import torchvision
import time

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch.nn.functional as F

from collections import OrderedDict
# from sensitivity_analysis.DNN.utils import compute_model_para_diff2
try:
    from data_IO.Load_data import *
#     from MyDataset import MyDataset
except ImportError:
    from Load_data import *
    
    
class CNN0(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super().__init__()
    
        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5).double()
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5).double()
    
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120).double()
        self.fc2 = nn.Linear(in_features=120, out_features=60).double()
        self.out = nn.Linear(in_features=60, out_features=10).double()

  # define forward function
    def forward(self, t):
        # conv 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
    
        # conv 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
    
        # fc1
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)
    
        # fc2
        t = self.fc2(t)
        t = F.relu(t)
    
        # output
        t = self.out(t)
    # don't need softmax here since we'll use cross-entropy as activation.

        return t
    
    
    def get_all_parameters(self):
    
        para_list = []
        
        for param in self.parameters():
            para_list.append(param.data.clone())
            print(param.data.shape)
            
            
        return para_list



    def get_all_gradient(self):
        
        para_list = []
        
        for param in self.parameters():
            para_list.append(param.grad.clone())
            
            
        return para_list    
    
    
    class MyDataset(Dataset):
        def __init__(self, samples):
            
            self.data = samples
            
        def __getitem__(self, index):
            data, target = self.data[index]
            
    #         data = data.view(-1)
            
            # Your transformations here (or set it in CIFAR10)
            
            return data.type(torch.DoubleTensor), target, index
    
        def __len__(self):
            return len(self.data)

        