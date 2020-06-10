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





class DNNModel(nn.Module):
    
    def __init__(self):
        super(DNNModel, self).__init__()
        
        self.fc1 = nn.Linear(28*28, 300).double()
        
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        
        self.relu1 = nn.ReLU()
        
#         self.fc.add_module("relu1", nn.Sigmoid())
        
        
#         self.linear_layers = nn.ModuleList([])
#          
#          
#         self.activation_layers = nn.ModuleList([])
#         
#         for i in range(len(hidden_dims) - 1):
# #             self.fc.add_module("fc" + str(i+2), nn.Linear(hidden_dims[i], hidden_dims[i + 1]).double())
#             
#             self.linear_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]).double())
#              
# #             self.linear_layers[i].weight.data.fill_(start_value)
# #              
# #             self.linear_layers[i].bias.data.fill_(start_value)
#             
# #             self.fc.add_module("relu" + str(i+2), nn.Sigmoid())
#             self.activation_layers.append(nn.ReLU())
        
        # Linear function 2: 100 --> 100
#         print(hidden_dims[len(hidden_dims) - 1])

#         self.fc.add_module("fc_final", nn.Linear(hidden_dims[len(hidden_dims) - 1], output_dim).double())

        self.fc2 = nn.Linear(300, 10).double()
        
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        
        
#         self.fc2.weight.data.fill_(start_value)
#         self.fc2.bias.data.fill_(start_value)
        
#         self.fc2.weight.fill_(0)
#         self.fc2.bias.fill_(0)
#         nn.init.constant_(self.fc2.weight, float(0.5))
#         
#         
#         nn.init.constant_(self.fc2.bias, float(0.5))
        
#         self.fc3 = nn.Softmax(dim=1)


#         self.fc.add_module("relu_final", nn.Sigmoid())
#         self.fc3 = nn.Softmax(dim=1)
#         self.fc3 = nn.Sigmoid()
        
        
#         # Non-linearity 2
#         self.tanh2 = nn.Tanh()
#         
#         # Linear function 3: 100 --> 100
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim)
#         # Non-linearity 3
#         self.elu3 = nn.ELU()
#         
#         # Linear function 4 (readout): 100 --> 10
#         self.fc4 = nn.Linear(hidden_dim, output_dim)  


#     def forward(self, x):
#         return self.fc.forward(x)


    def forward(self, x):
        # Linear function 1
         
#         para_square = 0
         
        out = self.fc1(x)
             
#         para_square += torch.sum(torch.pow(self.fc1.weight, 2))
#         
#         para_square += torch.sum(torch.pow(self.fc1.bias, 2))
         
         
        # Non-linearity 1
        out = self.relu1(out)
         
         
#         for i in range(len(self.linear_layers)):
#             out1 = self.linear_layers[i](out)
#             out = self.activation_layers[i](out1)
#             para_square += torch.sum(torch.pow(self.linear_layers[i].weight, 2))
#         
#             para_square += torch.sum(torch.pow(self.linear_layers[i].bias, 2))
         
         
#         print("dim::", out.shape)
#         
#         print(len(list(self.parameters())))
#         
#         print(self.fc2)
        # Linear function 2
        out = self.fc2(out)
         
#         para_square += torch.sum(torch.pow(self.fc2.weight,2))
#         
#         para_square += torch.sum(torch.pow(self.fc2.bias,2))
         
         
#         out = self.fc3(out)
        
#         out = torch.max(out, dim = 1)[0]
#         # Non-linearity 2
#         out = self.tanh2(out)
#         
#         # Linear function 2
#         out = self.fc3(out)
#         # Non-linearity 2
#         out = self.elu3(out)
#         
#         # Linear function 4 (readout)
#         out = self.fc4(out)
 
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
        def __init__(self, train_X, train_Y):
            
            self.data = train_X.type(torch.DoubleTensor).view(train_X.shape[0], -1)
            
            self.labels = train_Y.view(-1)
            
#             self.transformed_X = self.data.data
#             
#             for i in range(len(self.data.transforms.transform.transforms)):
#                 self.transformed_X = self.data.transforms.transform.transforms[i](self.transformed_X.numpy())
#             
#             self.transformed_X = self.transformed_X.transpose(0,1).transpose(1,2)
            
        def __getitem__(self, index):
            data, target = self.data[index],self.labels[index]
            
#             data = data.reshape(-1)
            
            # Your transformations here (or set it in CIFAR10)
            
            return data, target, index
        
#         def transform_data(self, ids):
# #             self.transformed_X = self.transformed_X.transpose(0,1)
#             
#             return self.transformed_X[ids].reshape(ids.shape[0], -1).type(torch.DoubleTensor), self.data.targets[ids]
        
        def __len__(self):
            return len(self.data)
    








def train(epoch, net, data_train_loader, optimizer, criterion, num_class):
#     global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    
    
    for i, items in enumerate(data_train_loader):
        
        images, labels, ids =  items[0], items[1], items[2]
        
        optimizer.zero_grad()

        output = net(images)

        y_onehot = torch.DoubleTensor(ids.shape[0], num_class)
    
        labels = labels.type(torch.LongTensor)
    
    # In your for loop
        y_onehot.zero_()
        y_onehot.scatter_(1, labels.view(-1, 1), 1)


        loss = criterion(output, y_onehot)


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

    
    
    
    data_train = DNNModel.MyDataset(MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()])))
    data_test = DNNModel.MyDataset(MNIST('./data/mnist',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
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
    
    
    net = DNNModel()
    
    net.get_all_parameters()

    
    criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=2e-2)
    
    train(1, net, data_train_loader, optimizer, criterion, 10)
    
    net.get_all_parameters()
