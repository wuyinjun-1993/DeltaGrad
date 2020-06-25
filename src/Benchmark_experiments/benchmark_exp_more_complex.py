'''
Created on Jan 8, 2020

'''
import sys



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

import ast

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
    from Models.DNN_single import DNNModel_single
    from Models.DNN2 import DNNModel2
    from Models.DNN3 import DNNModel3
    from Models.ResNet import *
    from Models.Skipnet import *
    from Models.CNN import *
    from Models.Pretrained_models import *
    from Batch_samplers import Batch_sampler
    from benchmark_exp import *

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
    from Models.DNN_single import DNNModel_single
    from Models.ResNet import *
    from Models.Skipnet import *
    from Models.CNN import *
    from Models.Pretrained_models import *
    from Batch_samplers import Batch_sampler
    
    from benchmark_exp import *
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':

    print(sys.version)
    
    configs = load_config_data(config_file)
    
#     print(configs)
    
    git_ignore_folder = configs['git_ignore_folder']


    


#     file_name = '../../../data/heartbeat/mitbih_train.csv'

#     file_name = '../../../data/covtype'
#     
    
#     file_name = '../../../data/Sensorless.scale'
    

#     file_name = '../../../data/shuttle.scale.tr'
    
#     file_name = '../../../data/skin_nonskin'

#     file_name = '../../../data/minist.csv'

    sys_argv = sys.argv
    
    

    
#     random.seed(random_seed)
#     os.environ['PYTHONHASHSEED'] = str(random_seed)
#     np.random.seed(random_seed)
#     torch.manual_seed(random_seed)
#     torch.cuda.manual_seed(random_seed)
#     torch.cuda.manual_seed_all(random_seed) # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = False
    
    
    
    
#     start = bool(int(sys_argv[1]))
#     
#     quantized = bool(int(sys_argv[2]))
    
    noise_rate = float(sys_argv[1])
    
    batch_size = int(sys_argv[2])
    
#     file_name = sys_argv[6]
    
#     epsilon = torch.tensor(float(sys_argv[7]), dtype = torch.double)

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
    
    init_model_params = torch.load(git_ignore_folder + 'origin_model0')
    
    data_preparer = Data_preparer()
    
    
    dataset_train, dataset_test, data_train_loader, data_test_loader = get_train_test_data_loader_by_name_lr(data_preparer, model_class, dataset_name, batch_size)
    
    hyper_para_function=getattr(Data_preparer, "get_hyperparameters_" + dataset_name)
    
    model = model_class()
    
    if is_GPU:
        model.to(device)
        
        
    init_model(model, list(init_model_params.parameters()))
    
#     init_model_params = list(init_model)
    
    
    criterion, optimizer, lr_scheduler = hyper_para_function(data_preparer, model.parameters(), learning_rate, regularization_coeff)
    
    hyper_params = [criterion, optimizer, lr_scheduler]
    
    
    
    
    lrs = ast.literal_eval(input)#map(float, input.strip('[]').split(','))
#     [2.0, 3.0, 4.0, 5.0]
    
#     model, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs, all_ids_list_all_epochs = model_training_skipnet(num_epochs, model, data_train_loader, data_test_loader, len(dataset_train), len(dataset_test), optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs)
    t1 = time.time()
    
    model, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs = model_training(num_epochs, model, data_train_loader, data_test_loader, len(dataset_train), len(dataset_test), optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs)


    t2 = time.time()
    
    print("training time full::", t2 - t1)
    
    torch.save(dataset_train, git_ignore_folder + "dataset_train")
    
    torch.save(dataset_test, git_ignore_folder + "test_data")
    
#     torch.save(all_ids_list_all_epochs, git_ignore_folder + "all_ids_list_all_epochs")
    
#     torch.save(ids2_list_all_epochs, git_ignore_folder + "ids2_list_all_epochs")
    
#     torch.save(delta_data_ids, git_ignore_folder + "delta_data_ids")
    
    
    torch.save(gradient_list_all_epochs, git_ignore_folder + 'gradient_list_all_epochs')
    
    torch.save(para_list_all_epochs, git_ignore_folder + 'para_list_all_epochs')
    
    torch.save(learning_rate_all_epochs, git_ignore_folder + 'learning_rate_all_epochs')


    torch.save(random_ids_multi_super_iterations, git_ignore_folder + 'random_ids_multi_super_iterations')
                  
    torch.save(num_epochs, git_ignore_folder+'epoch')    
    
    torch.save(hyper_params, git_ignore_folder + 'hyper_params')
    
    save_random_id_orders(git_ignore_folder, random_ids_multi_super_iterations)
    
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
    
    test(model, data_test_loader, criterion, len(dataset_test), is_GPU, device)
    
    