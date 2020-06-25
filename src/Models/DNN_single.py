'''
Created on Jan 3, 2020


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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


try:
    from data_IO.Load_data import *
#     from MyDataset import MyDataset
except ImportError:
    from Load_data import *
#     from MyDataset import MyDataset

class Logistic_regression(nn.Module):
    
    
    
    def __init__(self, input_dim, output_dim):
        super(Logistic_regression, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, output_dim).double()
        
        
        
        
        self.fc2 = nn.LogSoftmax().double()
        
#         self.fc3 = nn.Softmax()
#         self.fc2 = nn.Sigmoid()


    def forward(self, x):

        out1 = self.fc1(x)
        
        out = self.fc2(out1)

#         out2 = self.fc3(out1)
#         
#         out2 = torch.log(out2)
# 
# #         out = self.fc2(out)
#         print(out2 - out)
        return out


    def forward_with_provenance(self, x, X_theta_prod_seq, X_theta_prod_softmax_seq):

        out1 = self.fc1(x)
        
        X_theta_prod_seq.append(out1.cpu())
        
        
        out = self.fc2(out1)
        
        X_theta_prod_softmax_seq.append(torch.exp(out).cpu())

#         out2 = self.fc3(out1)
#         
#         out2 = torch.log(out2)
# 
# #         out = self.fc2(out)
#         print(out2 - out)
        return out

    

    def forward_no_softmax(self, x):
        return self.fc1(x)
    
    
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
        def __init__(self, features, labels):
            
            self.data = features
            
            self.labels = labels.view(-1)
            
        def __getitem__(self, index):
            data, target = self.data[index], self.labels[index]
            
            data = data.view(-1)
            
            # Your transformations here (or set it in CIFAR10)
            
            return data.type(torch.DoubleTensor), target, index
    
        def __len__(self):
            return len(self.data)
    




class DNNModel_single(nn.Module):
    
    
    
    def __init__(self):
        super(DNNModel_single, self).__init__()
        
        self.fc1 = nn.Linear(32*32, 10).double()
        
        self.fc2 = nn.Sigmoid()


    def forward(self, x):

        out = self.fc1(x)

        out = self.fc2(out)
 
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
            
            data = data.view(-1)
            
            # Your transformations here (or set it in CIFAR10)
            
            return data.type(torch.DoubleTensor), target, index
    
        def __len__(self):
            return len(self.data)
    
    
def train(epoch, net, data_train_loader, optimizer, criterion, criterion2):
#     global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    
    
    for i, items in enumerate(data_train_loader):
        
        images, labels, ids =  items[0], items[1], items[2]
        
        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        output2 = net.forward_no_softmax(images)
        
        loss2 = criterion2(output2, labels)


        print(loss -loss2)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        # Update Visualization
#         if viz.check_connection():
#             cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
#                                      win=cur_batch_win, name='current_batch_loss',
#                                      update=(None if cur_batch_win is None else 'replace'),
#                                      opts=cur_batch_win_opts)
        loss.backward()
        
        optimizer.step()
        
        
    
    
    
if __name__ == '__main__':
    
#     configs = load_config_data(config_file)
#     
#     git_ignore_folder = configs['git_ignore_folder']
# 
#     file_name = '../../../data/minist.csv'
#     
#     [X, Y, test_X, test_Y] = load_data_multi_classes(True, file_name)

    
    
    
    data_train = DNNModel_single.MyDataset(MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
#                        transforms.Resize((32, 32)),
                       transforms.ToTensor()])))
    data_test = DNNModel_single.MyDataset(MNIST('./data/mnist',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
#                           transforms.Resize((32, 32)),
                          transforms.ToTensor()])))



    input_dim = 32*32
#     data_train = MyDataset(MNIST('./data/mnist',
#                    download=True,
#                    transform=transforms.Compose([
#                        transforms.Resize((-input_dim)),
#                        transforms.ToTensor()])))
#     data_test = MyDataset(MNIST('./data/mnist',
#                       train=False,
#                       download=True,
#                       transform=transforms.Compose([
#                           transforms.Resize((input_dim)),
#                           transforms.ToTensor()])))
    data_train_loader = DataLoader(data_train, batch_size=16, shuffle=True, num_workers=0)
    data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)
    
    
    
    hidden_dim= [300]
    
    output_dim = 10
    
    
    net = Logistic_regression(28*28, 10)
    
    net.get_all_parameters()

    
    criterion = nn.NLLLoss()
    
    criterion2 = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(net.parameters(), lr=2e-2)
    
    train(1, net, data_train_loader, optimizer, criterion, criterion2)
    
    net.get_all_parameters()
