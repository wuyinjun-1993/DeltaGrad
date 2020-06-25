'''
Created on Jan 8, 2020


'''
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.mnist import MNIST

import torchvision.models as models


try:
    from data_IO.Load_data import *

except ImportError:
    from Load_data import *



class MyDataset(Dataset):
    def __init__(self, samples):
        
        self.data = samples
        
#             self.transformed_X = self.data.data
#             
#             for i in range(len(self.data.transforms.transform.transforms)):
#                 self.transformed_X = self.data.transforms.transform.transforms[i](self.transformed_X.numpy())
#             
#             self.transformed_X = self.transformed_X.transpose(0,1).transpose(1,2)
        
    def __getitem__(self, index):
        data, target = self.data[index]
        
        # Your transformations here (or set it in CIFAR10)
        
        return data, target, index
    
#         def transform_data(self, ids):
# #             self.transformed_X = self.transformed_X.transpose(0,1)
#             
#             return self.transformed_X[ids].reshape(ids.shape[0], -1).type(torch.DoubleTensor), self.data.targets[ids]
    
    def __len__(self):
        return len(self.data)

def train_model(model, criterion, optimizer, scheduler, num_epochs, data_train_loader, dataset_train_len):
    since = time.time()
    model.train()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for i, items in enumerate(data_train_loader):
            inputs = items[0]
            labels = items[1]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
#             with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
#                 if phase == 'train':
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
#         if phase == 'train':
            scheduler.step()
            
            if i % 5 == 0:
                print('Train - Epoch %d, Batch: %d, Loss: %f, acc::%f' % (epoch, i, loss.item(), torch.sum(preds == labels.data).item()/(inputs.shape[0])))
            

        epoch_loss = running_loss / dataset_train_len
        epoch_acc = running_corrects.double() / dataset_train_len

        print('iteration {} Loss: {:.4f} Acc: {:.4f}'.format(
            i, epoch_loss, epoch_acc))

        # deep copy the model
#         if phase == 'val' and epoch_acc > best_acc:
#             best_acc = epoch_acc
#             best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

data_dir = 'data/hymenoptera_data'


configs = load_config_data(config_file)
    
#     print(configs)
git_ignore_folder = configs['git_ignore_folder']
        
# train_data = MNIST(git_ignore_folder + '/mnist',

data_train = MyDataset(MNIST(git_ignore_folder + '/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.RandomResizedCrop(224),
                        transforms.RandomCrop(32),
                       transforms.RandomHorizontalFlip(),
                       transforms.Grayscale(3),
                       transforms.ToTensor(),
                       transforms.Normalize([0.5], [0.5])
                       ])))
data_test = MyDataset(MNIST('./data/mnist',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
                       transforms.RandomResizedCrop(224),
                       transforms.RandomCrop(32),
                       transforms.RandomHorizontalFlip(),
                       transforms.Grayscale(3),
                      transforms.ToTensor(),
                       transforms.Normalize([0.5], [0.5])
                        ])))


# data_train = MyDataset(torchvision.datasets.CIFAR10(git_ignore_folder + '/cifar10',
#                download=True,
#                transform=transforms.Compose([
#                    transforms.RandomCrop(32, padding=4),
#                    transforms.RandomHorizontalFlip(),
#                    transforms.ToTensor(),
#                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                    ])))
# data_test = MyDataset(torchvision.datasets.CIFAR10(git_ignore_folder+ '/cifar10',
#                   train=False,
#                   download=True,
#                   transform=transforms.Compose([
#                       transforms.ToTensor(),
#                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                       ])))
data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=0)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=0)


input_dim = 32*32




# model = torch.hub.load('pytorch/vision:v0.5.0', 'squeezenet1_0', pretrained=True)
# model = models.resnet18(pretrained=True)
model = models.resnet18(pretrained=True)

data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=0)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=0)

# model.num_classes = 10

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model.parameters(), lr=0.02, weight_decay = 0.05)


exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)




# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
# for x in ['train', 'val']}
# dataloaders = {x: torch.utils.data.DataLoader(data_train, batch_size=128,
#                                              shuffle=True, num_workers=0) for x in ['train', 'val']}
# # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# # class_names = image_datasets['train'].classes
# 
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 
# 
# model_ft = models.resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# # Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Linear(num_ftrs, 10)
# 
# model_ft = model_ft.to(device)
# 
# criterion = nn.CrossEntropyLoss()
# 
# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# 
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)




model_ft = train_model(model, criterion, optimizer_conv, exp_lr_scheduler,
                       2, data_train_loader, len(data_train))




