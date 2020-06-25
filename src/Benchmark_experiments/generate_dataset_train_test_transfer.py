'''
Created on Jan 27, 2020

'''


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
import sys, ast

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader



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
    from Models.Lenet5_cifar import LeNet5_cifar
    from Models.Data_preparer import *
    from Models.DNN_single import *
    from Models.DNN2 import DNNModel2
    from Models.DNN3 import DNNModel3
    from Models.ResNet import *
    from Models.Skipnet import *
    from Models.CNN import *
    from Models.Pretrained_models import *
    from Batch_samplers import Batch_sampler
    from benchmark_exp import *
    from Models.DNN_transfer import *

except ImportError:
    from Load_data import *
    from utils import *
    from piecewise_linear_interpolation_2D import *
    from Models.DNN import DNNModel
    from Models.DNN2 import DNNModel2
    from Models.DNN3 import DNNModel3
    from Models.Lenet5 import LeNet5
    from Models.Lenet5_cifar import LeNet5_cifar
    from Models.Data_preparer import *
    from Models.DNN_single import *
    from Models.ResNet import *
    from Models.Skipnet import *
    from Models.CNN import *
    from Models.Pretrained_models import *
    from Batch_samplers import Batch_sampler
    from benchmark_exp import *
    from Models.DNN_transfer import *

def generate_random_ids_list(dataset_train_len, epochs, repetition):
    
    
    
    
    for j in range(repetition): 
        
        random_ids_all_epochs = []
        
        for i in range(epochs):
            random_ids = torch.randperm(dataset_train_len)
        
            random_ids_all_epochs.append(random_ids)
        sorted_random_ids_all_epochs = get_sorted_random_ids(random_ids_all_epochs)
        
        
        torch.save(random_ids_all_epochs, git_ignore_folder + 'random_ids_multi_super_iterations_' + str(j))
        
        torch.save(sorted_random_ids_all_epochs, git_ignore_folder + 'sorted_ids_multi_super_iterations_' + str(j))

if __name__ == '__main__':
    
    sys_argv = sys.argv
    
    configs = load_config_data(config_file)
    
#     print(configs)
    global git_ignore_folder
    git_ignore_folder = configs['git_ignore_folder']
    
    transfer_model_name = sys_argv[1]
    
    dataset_name = sys_argv[2]
    
    batch_size = int(sys_argv[3])
    
    epochs = int(sys_argv[4])
    
    repetition = int(sys_argv[5])
    
    is_GPU = bool(int(sys_argv[6]))
    
    if not is_GPU:
        device = torch.device("cpu")
    else:    
        GPU_ID = int(sys_argv[7])
        device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    
    
    
    get_transfer_model_func = getattr(Transfer_learning, "prepare_" + transfer_model_name)
    
    
    tl = Transfer_learning()
    
    transfer_model, in_feature_num = get_transfer_model_func(tl)
    
#     model_class = getattr(sys.modules[__name__], model_name)
    model_class = Logistic_regression
        
        
    data_preparer = Data_preparer()
    
    
    dataset_train, dataset_test, data_train_loader, data_test_loader = get_train_test_data_loader_by_name_lr_transfer(data_preparer, transfer_model, transfer_model_name, model_class, dataset_name, batch_size, is_GPU, device)

#     generate_random_ids_list(50000, epochs, repetition)
    
    generate_random_ids_list(len(dataset_train), epochs, repetition)

    torch.save(dataset_train, git_ignore_folder + "dataset_train")
     
    torch.save(data_train_loader, git_ignore_folder + "data_train_loader")
     
    torch.save(data_test_loader, git_ignore_folder + "data_test_loader")
     
    torch.save(dataset_test, git_ignore_folder + "dataset_test")






