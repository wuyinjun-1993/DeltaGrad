'''
Created on Jan 8, 2020


'''
import sys

import numpy as np
import torch
from torch import nn
import os
from collections import deque 


from torch import nn, optim

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


try:
    from data_IO.Load_data import *
#     from MyDataset import MyDataset
except ImportError:
    from Load_data import *
#     from MyDataset import MyDataset





class Pretrained_resnet18(nn.Module):
    
    
    
    def __init__(self):
        super(Pretrained_resnet18, self).__init__()
        
        model_ft = models.resnet18(pretrained=True).double()
        
        for param in model_ft.parameters():
            param.requires_grad = False
        
        num_ftrs = model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Linear(num_ftrs, 10).double()
        
        self.model_ft = model_ft
        

    def forward(self, x):
         
        out = self.model_ft(x)

        return out
#     
    
    
    def get_all_parameters(self):
    
        para_list = []
        
        for param in self.parameters():
            para_list.append(param.data.clone())
            
            
        return para_list



    def get_all_gradient(self):
        
        para_list = []
        
        for param in self.parameters():
            para_list.append(param.grad.clone())
            
            
        return para_list    
    
    
    
    def get_output_each_layer(self, x):
        
        
        output_list = [None]*(len(self.linear_layers) + 3)
        
        non_linear_input_list = [None]*(len(self.linear_layers) + 3)
        
        k = 0
        
        output_list[k] = torch.cat((x, torch.ones([x.shape[0], 1], dtype = torch.double)), 1)
        
        non_linear_input_list[k]= x.clone()
        
        k = k + 1
        
        
        out = self.fc1(x)
        # Non-linearity 1
        
        non_linear_input_list[k]= out.clone()
        
        out = self.relu1(out)
        
        
        output_list[k] = torch.cat((out, torch.ones([out.shape[0], 1], dtype = torch.double)), 1)
        
        k = k + 1
        
        for i in range(len(self.linear_layers)):
            out = self.linear_layers[i](out)
            non_linear_input_list[k] = out.clone()
            out = self.activation_layers[i](out)
            output_list[k] = torch.cat((out, torch.ones([out.shape[0], 1], dtype = torch.double)), 1)
            
            k = k + 1
        
        
        # Linear function 2
        out = self.fc2(out)
        
        non_linear_input_list[k] = out.clone()
        
        out2 = self.fc3(out)
        
        output_list[k] = out2
        
        
        return output_list, non_linear_input_list
    
    class MyDataset(Dataset):
        def __init__(self, samples):
            
            self.data = samples
            
        def __getitem__(self, index):
            data, target = self.data[index]
            
#             data = data.view(-1)
            
            
#             y_onehot = torch.DoubleTensor(1, 10)
#     
#             target = torch.tensor([target])
#             target = target.type(torch.LongTensor)
#             
#         # In your for loop
#             y_onehot.zero_()
#             y_onehot.scatter_(1, target.view(-1, 1), 1)
            
            
            # Your transformations here (or set it in CIFAR10)
            
            return data.type(torch.DoubleTensor), target, index
    
        def __len__(self):
            return len(self.data)