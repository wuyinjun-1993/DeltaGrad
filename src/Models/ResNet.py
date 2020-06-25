'''
Created on Jan 6, 2020


'''
import torch
import torch.nn as nn
import torch.nn.functional as F


import sys, os

from torch import optim

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import time

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from collections import OrderedDict
# from sensitivity_analysis.DNN.utils import compute_model_para_diff2
try:
    from data_IO.Load_data import *
#     from MyDataset import MyDataset
except ImportError:
    from Load_data import *

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False).double()
        self.bn1 = nn.BatchNorm2d(planes).double()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False).double()
        self.bn2 = nn.BatchNorm2d(planes).double()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False).double(),
                nn.BatchNorm2d(self.expansion*planes).double()
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False).double()
        self.bn1 = nn.BatchNorm2d(planes).double()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False).double()
        self.bn2 = nn.BatchNorm2d(planes).double()
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False).double()
        self.bn3 = nn.BatchNorm2d(self.expansion*planes).double()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False).double(),
                nn.BatchNorm2d(self.expansion*planes).double()
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).double()
        self.bn1 = nn.BatchNorm2d(64).double()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes).double()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

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
    
    def get_all_parameters(self):
    
        para_list = []
        
        for param in self.parameters():
            para_list.append(param.data.clone())
            print(param.data.shape)
            
            
        return para_list
    

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def get_expected_parameters(model, learning_rate, weight_decay):
    
    para_list = []
    
    
    for i in range(len(list(model.parameters()))):
        
        param = list(model.parameters())[i].data
        
        grad = list(model.parameters())[i].grad
        
        update_params = (1-learning_rate*weight_decay)*param - learning_rate*grad
        
        para_list.append(update_params)
    
    return para_list

        
    
    
def train(epoch, net, data_train_loader, optimizer, criterion, lr_scheduler):
#     global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    
    time1 = 0
    
    time2 = 0
    
    
    
    for i, items in enumerate(data_train_loader):
        
        images, labels, ids =  items[0], items[1], items[2]
        
        print("batch_size::", ids.shape)
        
        optimizer.zero_grad()

        output = net(images)

        t1 = time.time()

        loss = criterion(output, labels)

        t2 = time.time()
        
        time1 += (t2 - t1)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        if (i+1) % 10 == 0:
            lr_scheduler.step() 
            
        
        # Update Visualization
#         if viz.check_connection():
#             cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
#                                      win=cur_batch_win, name='current_batch_loss',
#                                      update=(None if cur_batch_win is None else 'replace'),
#                                      opts=cur_batch_win_opts)
        t1  =time.time()
        loss.backward()
        t2 = time.time()
        
#         print("learning_rate:", list(optimizer.param_groups)[0]['lr'])
        
#         expect_model_para = get_expected_parameters(net, list(optimizer.param_groups)[0]['lr'])
        
        expect_model_para = get_expected_parameters(net, list(optimizer.param_groups)[0]['lr'], list(optimizer.param_groups)[0]['weight_decay'])
        
        optimizer.step()
#         next_items = next(iter(data_train_loader))
        
        
#         compute_model_para_diff2(expect_model_para, list(net.parameters()))
        
        
        
        print((t2 - t1))
        
        time2 += (t2 - t1)
    
    
    print("forward time::", time1)
    
    print("backward time::", time2)
    
    

if __name__ == '__main__':
    
#     configs = load_config_data(config_file)
#     
#     git_ignore_folder = configs['git_ignore_folder']
# 
#     file_name = '../../../data/minist.csv'
#     
#     [X, Y, test_X, test_Y] = load_data_multi_classes(True, file_name)

    
    
    net = ResNet18()
    
    data_train = net.MyDataset(torchvision.datasets.CIFAR10(git_ignore_folder + '/cifar10',
                   download=True,
                   transform=transforms.Compose([
                       transforms.RandomCrop(32, padding=4),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                       ])))
    data_test = net.MyDataset(torchvision.datasets.CIFAR10(git_ignore_folder+ '/cifar10',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                          ])))
    data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=0)
    data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=0)
    
    net.get_all_parameters()

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=2e-3, weight_decay = 5e-4)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)
    
    
    train(1, net, data_train_loader, optimizer, criterion, lr_scheduler)
    
    net.get_all_parameters()

