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
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


try:
    from data_IO.Load_data import *
    from Models.Data_preparer import *
#     from MyDataset import MyDataset
except ImportError:
    from Load_data import *
    from Models.Data_preparer import *
#     from MyDataset import MyDataset


class Transfer_learning():
    def prepare_resnet18(self):
    
    
        resnet = models.resnet18(pretrained=True)
            # freeze all model parameters
        for param in resnet.parameters():
            param.requires_grad = False    
            
        
        in_feature_num = resnet.fc.in_features
            
        return resnet, in_feature_num

    def compute_before_last_layer_resnet18(self, model, x):
        x = model.conv1.double()(x)
        x = model.bn1.double()(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1.double()(x)
        x = model.layer2.double()(x)
        x = model.layer3.double()(x)
        x = model.layer4.double()(x)

        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

    def get_last_layer_resnet18(self, model):
        
        
        last_layer = list(model.children())[-1].double()
        
        return last_layer
    
    
    def prepare_resnet50(self):
    
    
        resnet = models.resnet50(pretrained=True)
            # freeze all model parameters
        for param in resnet.parameters():
            param.requires_grad = False    
            
        
        in_feature_num = resnet.fc.in_features
            
        return resnet, in_feature_num

    def compute_before_last_layer_resnet50(self, model, x):
        x = model.conv1.double()(x)
        x = model.bn1.double()(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1.double()(x)
        x = model.layer2.double()(x)
        x = model.layer3.double()(x)
        x = model.layer4.double()(x)

        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

    def get_last_layer_resnet50(self, model):
        
        
        last_layer = list(model.children())[-1].double()
        
        return last_layer
    
    def prepare_resnet152(self):
    
    
        resnet = models.resnet152(pretrained=True)
            # freeze all model parameters
        for param in resnet.parameters():
            param.requires_grad = False    
            
        
        in_feature_num = resnet.fc.in_features
            
        return resnet, in_feature_num


    def compute_before_last_layer_resnet152(self, model, x):
        x = model.conv1.double()(x)
        x = model.bn1.double()(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1.double()(x)
        x = model.layer2.double()(x)
        x = model.layer3.double()(x)
        x = model.layer4.double()(x)

        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


    def get_last_layer_resnet152(self, model):
        
        
        last_layer = list(model.children())[-1].double()
        
        return last_layer

    def prepare_densenet121(self):
    
    
        densenet = models.densenet121(pretrained=True)
            # freeze all model parameters
        for param in densenet.parameters():
            param.requires_grad = False    
            
        
        in_feature_num = densenet.classifier.in_features
            
        return densenet, in_feature_num


    def compute_before_last_layer_densenet121(self, model, x):
        
        features = model.features.double()(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out
    
    
    
    def get_last_layer_densenet121(self, model):
        
        
        last_layer = list(model.children())[-1].double()
        
        return last_layer


    def prepare_alexnet(self):
    
    
#         resnet = models.resnet50(pretrained=True)
        alexnet = models.alexnet(pretrained=True)

            # freeze all model parameters
        for param in alexnet.parameters():
            param.requires_grad = False    
            
        
        in_feature_num = alexnet.classifier[6].in_features
        
        return alexnet, in_feature_num
    
    
    
    def compute_before_last_layer_alexnet(self, model, x):
        
        x = model.features.double()(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


    def get_last_layer_alexnet(self, model):
        
#         x = model.features.double()(x)
#         x = model.avgpool(x)
#         x = torch.flatten(x, 1)
        
        last_layer = list(model.children())[-1].double()
        
        
        return last_layer


    def prepare_vgg19(self):
    
    
#         resnet = models.resnet50(pretrained=True)
        alexnet = models.vgg19(pretrained=True)

            # freeze all model parameters
        for param in alexnet.parameters():
            param.requires_grad = False    
            
        
        in_feature_num = list(alexnet.classifier)[-1].in_features
        
        return alexnet, in_feature_num
    
    
    
    def compute_before_last_layer_vgg19(self, model, x):
        
        x = model.features.double()(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        
        modules = list(model.classifier)
        
        for i in range(len(modules)-1):
            x = modules[i].double()(x)
        
#         x = model.classifier.
        
        return x


    def get_last_layer_vgg19(self, model):
        
#         x = model.features.double()(x)
#         x = model.avgpool(x)
#         x = torch.flatten(x, 1)
        
        last_layer = list(model.classifier)[-1]
        
#         for i in range(len(modules)-1):
#             x = modules[i].double()(x)
        
#         x = model.classifier.
        
        return last_layer
    
#     def prepare_vgg19(self):
#     
#     
# #         resnet = models.resnet50(pretrained=True)
#         vgg19 = models.vgg19(pretrained=True)
# 
#             # freeze all model parameters
#         for param in vgg19.parameters():
#             param.requires_grad = False    
#             
#         
#         in_feature_num = vgg19.classifier[6].in_features
#         
#         return alexnet, in_feature_num


class ResNet_transfer(nn.Module):
    
    def __init__(self):
        super(ResNet_transfer, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        
        # new final layer with 16 classes
        num_ftrs = resnet.fc.in_features
        self.layer = torch.nn.Linear(num_ftrs, 10)
#         self.model = resnet

    def forward(self, x):
         
        out = self.layer(x)
             
        return out
#     
    
    
    def get_all_parameters(self):
    
        para_list = []
        
        for param in self.model.parameters():
            para_list.append(param.data.clone())
            
            
        return para_list



    def get_all_gradient(self):
        
        para_list = []
        
        for param in self.model.parameters():
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
            
        def __getitem__(self, index):
            data, target = self.data[index],self.labels[index]
            
            return data, target, index
        
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


#     data_train = ResNet_transfer.MyDataset(torchvision.datasets.CIFAR10(git_ignore_folder + '/cifar10',
#                    download=True,
#                    transform=transforms.Compose([
#                        transforms.RandomCrop(32, padding=4),
#                        transforms.RandomHorizontalFlip(),
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                        ])))
#     data_test = ResNet_transfer.MyDataset(torchvision.datasets.CIFAR10(git_ignore_folder+ '/cifar10',
#                       train=False,
#                       download=True,
#                       transform=transforms.Compose([
#                           transforms.ToTensor(),
#                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                           ])))


    data_preparer = Data_preparer()

#     function=getattr(Data_preparer, "prepare_MNIST2")
    
    function=getattr(Data_preparer, "prepare_cifar10_2")
    
    trans = transforms.Compose([transforms.Scale(224),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize])
    
    train_X, train_Y, test_X, test_Y = function(data_preparer)
    
    
    
    data_train = ResNet_transfer.MyDataset(train_X, train_Y)
    
    data_test = ResNet_transfer.MyDataset(test_X, test_Y)
    
#     data_train = ResNet_transfer.MyDataset(MNIST('./data/mnist',
#                    download=True,
#                    transform=transforms.Compose([
#                        transforms.Resize((32, 32)),
#                        transforms.ToTensor()])))
#     data_test = ResNet_transfer.MyDataset(MNIST('./data/mnist',
#                       train=False,
#                       download=True,
#                       transform=transforms.Compose([
#                           transforms.Resize((32, 32)),
#                           transforms.ToTensor()])))



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
    
    
    net = ResNet_transfer()
    
    net.get_all_parameters()

    
    criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=2e-2)
    
    train(10, net, data_train_loader, optimizer, criterion, 10)
    
    net.get_all_parameters()
