'''
Created on Jan 7, 2020

'''

import os, sys

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import time
from torchvision import datasets, transforms
from torch import nn, optim
import os
from collections import deque 
import random


from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import resnet18



sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Models')


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


try:
    from data_IO.Load_data import *
    from utils import *
    from Interpolation.piecewise_linear_interpolation_2D import *
    from Models.DNN import DNNModel
    from Models.Lenet5 import LeNet5
    from Models.Data_preparer import *
    from Models.DNN_single import DNNModel_single
    from Models.ResNet import *
    from Batch_samplers import Batch_sampler
    from benchmark_exp import *

except ImportError:
    from Load_data import *
    from utils import *
    from piecewise_linear_interpolation_2D import *
    from Models.DNN import DNNModel
    from Models.Lenet5 import LeNet5
    from Models.Data_preparer import *
    from Models.DNN_single import DNNModel_single
    from Models.ResNet import *
    from Batch_samplers import Batch_sampler
    from benchmark_exp import *
    



def model_training(epoch, net, data_train_loader, data_test_loader, data_train_size, data_test_size, optimizer, criterion, lr_scheduler, batch_size, is_GPU, device):
#     global cur_batch_win
    net.train()
    
    gradient_list_all_epochs = []
    
    para_list_all_epochs = []
    
#     output_list_all_epochs = []
    
    learning_rate_all_epochs = []
    
    
    
    loss_list, batch_list = [], []
    
    t1 = time.time()
    
    for j in range(epoch):
        
        random_ids = torch.zeros([data_train_size], dtype = torch.long)
    
        k = 0
        
        lr_scheduler.step()
        
        for i, items in enumerate(data_train_loader):
            
            if not is_GPU:
                images, labels, ids =  items[0], items[1], items[2]
            else:
                images, labels, ids =  items[0].to(device), items[1].to(device), items[2]
            
            end_id = k + batch_size
            
            if end_id > data_train_size:
                end_id = data_train_size
            
            random_ids[k:end_id] = ids
            
            
            k = k + batch_size
            
            optimizer.zero_grad()
    
            output = net(images)
    
            loss = criterion(output, labels)
    
    
            loss_list.append(loss.detach().cpu().item())
            batch_list.append(i+1)
    
            if i % 10 == 0:
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (j, i, loss.detach().cpu().item()))
            
#             if i % 20 == 0:
                
                 
    
            loss.backward()
    
            append_gradient_list(gradient_list_all_epochs, None, para_list_all_epochs, net, None, is_GPU, device)
    
            learning_rate = list(optimizer.param_groups)[0]['lr']
            
#             regularization_rate = list(optimizer.param_groups)[0]['weight_decay']
            
#             exp_model_param = update_model(net, learning_rate, regularization_rate)
            
            
            optimizer.step()
            
#             print('parameter comparison::')
#             
#             compute_model_para_diff(list(net.parameters()), exp_model_param)
            
            
            learning_rate_all_epochs.append(learning_rate)
        
        
        
#         item1 = data_train_loader.dataset.data[100]
#         print(torch.norm(item0[0] - item1[0]))
        
        random_ids_multi_super_iterations.append(random_ids)
        
        
        
    
    t2 = time.time()
    
    print("training_time::", (t2 - t1))
    
    return net, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs

    
if __name__ == '__main__':

    print(sys.version)
    
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']

    sys_argv = sys.argv
    
    noise_rate = float(sys_argv[1])
    
    batch_size = int(sys_argv[2])

    num_epochs = int(sys_argv[3])
    
    input = sys_argv[4]
    
    model_name = sys_argv[5]

    dataset_name = sys_argv[6]
    
    learning_rate = float(sys_argv[7])
    
    regularization_coeff = float(sys_argv[8])
    
    is_GPU = bool(int(sys_argv[9]))
    
    if not is_GPU:
        device = torch.device("cpu")
    else:    
        GPU_ID = int(sys_argv[10])
        device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")


    print(device)

    model_class = getattr(sys.modules[__name__], model_name)
    
    
    data_preparer = Data_preparer()
    
    
    dataset_train, dataset_test, data_train_loader, data_test_loader = get_train_test_data_loader_by_name(data_preparer, model_class, dataset_name, batch_size)
    
    hyper_para_function=getattr(Data_preparer, "get_hyperparameters_" + dataset_name)
    
#     model = model_class()
    
    model = ResNet18()
    
    if is_GPU:
        model.to(device)
    
    init_model_params = list(model.parameters())
    
    
    criterion, optimizer, lr_scheduler = hyper_para_function(data_preparer, model.parameters(), learning_rate, regularization_coeff)
    
    hyper_params = [criterion, optimizer, lr_scheduler]
    
    model, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs = model_training(num_epochs, model, data_train_loader, data_test_loader, len(dataset_train), len(dataset_test), optimizer, criterion, lr_scheduler, batch_size, is_GPU, device)

    
    
    torch.save(dataset_train, git_ignore_folder + "dataset_train")
    
    torch.save(dataset_test, git_ignore_folder + "test_data")
    
    torch.save(gradient_list_all_epochs, git_ignore_folder + 'gradient_list_all_epochs')
    
    torch.save(para_list_all_epochs, git_ignore_folder + 'para_list_all_epochs')
    
    torch.save(learning_rate_all_epochs, git_ignore_folder + 'learning_rate_all_epochs')


    torch.save(random_ids_multi_super_iterations, git_ignore_folder + 'random_ids_multi_super_iterations')
                  
    torch.save(num_epochs, git_ignore_folder+'epoch')    
    
    torch.save(hyper_params, git_ignore_folder + 'hyper_params')
    
    save_random_id_orders(random_ids_multi_super_iterations)
    
    torch.save(para_list_all_epochs[0], git_ignore_folder + 'init_para')
    
    torch.save(model, git_ignore_folder + 'origin_model')
    
    torch.save(model_class, git_ignore_folder + 'model_class')
    
    torch.save(data_train_loader, git_ignore_folder + 'data_train_loader')
    
    torch.save(data_test_loader, git_ignore_folder + 'data_test_loader')
    
    torch.save(learning_rate, git_ignore_folder + 'alpha')

    torch.save(regularization_coeff, git_ignore_folder + 'beta')
    
    torch.save(dataset_name, git_ignore_folder + 'dataset_name')
    
    torch.save(batch_size, git_ignore_folder + 'batch_size')

    torch.save(device, git_ignore_folder + 'device')

    torch.save(is_GPU, git_ignore_folder + 'is_GPU')
    
    torch.save(noise_rate, git_ignore_folder + 'noise_rate')    
    
    del dataset_train
    
    
    del data_train_loader
    
    
    
    if is_GPU:
        torch.cuda.empty_cache()
        
    
    
    test(model, data_test_loader, criterion, len(dataset_test), is_GPU, device)
    
