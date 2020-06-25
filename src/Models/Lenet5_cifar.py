'''
Created on Jan 3, 2020


'''


import torch
import sys, os

from torch import nn, optim

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
import time

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from collections import OrderedDict
# from sensitivity_analysis.DNN.utils import compute_model_para_diff2
try:
    from data_IO.Load_data import *
#     from MyDataset import MyDataset
except ImportError:
    from Load_data import *
#     from MyDataset import MyDataset




class LeNet5_cifar(nn.Module):
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
        super(LeNet5_cifar, self).__init__()
        
        
        
#             model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape=(32,32,3)))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#     model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal'))
#     model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal'))
#     model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal'))

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 6, kernel_size=(5, 5)).double()),
            ('relu1', nn.ReLU().double()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2).double()),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5)).double()),
            ('relu3', nn.ReLU().double()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2).double()),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5)).double()),
            ('relu5', nn.ReLU().double())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84).double()),
            ('relu6', nn.ReLU().double()),
            ('f7', nn.Linear(84, 10).double()),
            ('sig7', nn.LogSoftmax(dim=-1).double())
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output
    
    
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
        
        print("learning_rate:", list(optimizer.param_groups)[0]['lr'])
        
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

    
    
    net = LeNet5_cifar()
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    
    data_train = LeNet5_cifar.MyDataset(torchvision.datasets.CIFAR10(git_ignore_folder + '/cifar10',
                   download=True,
                   transform=transform))
    data_test = LeNet5_cifar.MyDataset(torchvision.datasets.CIFAR10(git_ignore_folder + '/cifar10',
                      train=False,
                      download=True,
                      transform=transform))
    data_train_loader = DataLoader(data_train, batch_size=4096, shuffle=True, num_workers=0)
    data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=0)
    
    net.get_all_parameters()

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=2e-3, weight_decay = 0.001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)
    
    
    train(1, net, data_train_loader, optimizer, criterion, lr_scheduler)
    
    net.get_all_parameters()
    
    
#     cur_batch_win = None
#     cur_batch_win_opts = {
#         'title': 'Epoch Loss Trace',
#         'xlabel': 'Batch Number',
#         'ylabel': 'Loss',
#         'width': 1200,
#         'height': 600,
#     }
    
    

#     def get_output_each_layer(self, x):
#         
#         
#         output_list = [None]*(len(self.linear_layers) + 3)
#         
#         non_linear_input_list = [None]*(len(self.linear_layers) + 3)
#         
#         k = 0
#         
#         output_list[k] = torch.cat((x, torch.ones([x.shape[0], 1], dtype = torch.double)), 1)
#         
#         non_linear_input_list[k]= x.clone()
#         
#         k = k + 1
#         
#         
#         out = self.fc1(x)
#         # Non-linearity 1
#         
#         non_linear_input_list[k]= out.clone()
#         
#         out = self.relu1(out)
#         
#         
#         output_list[k] = torch.cat((out, torch.ones([out.shape[0], 1], dtype = torch.double)), 1)
#         
#         k = k + 1
#         
#         for i in range(len(self.linear_layers)):
#             out = self.linear_layers[i](out)
#             non_linear_input_list[k] = out.clone()
#             out = self.activation_layers[i](out)
#             output_list[k] = torch.cat((out, torch.ones([out.shape[0], 1], dtype = torch.double)), 1)
#             
#             k = k + 1
#         
#         
#         # Linear function 2
#         out = self.fc2(out)
#         
#         non_linear_input_list[k] = out.clone()
#         
#         out2 = self.fc3(out)
#         
#         output_list[k] = out2
#         
#         
#         return output_list, non_linear_input_list
#     
    
    
    
    
    
    
    
    
    
    