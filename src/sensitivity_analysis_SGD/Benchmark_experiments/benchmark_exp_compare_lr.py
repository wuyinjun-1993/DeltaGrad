'''
Created on Jan 12, 2020

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
    
if __name__ == '__main__':

    print(sys.version)
    
    configs = load_config_data(config_file)
    
#     print(configs)
    global git_ignore_folder
    git_ignore_folder = configs['git_ignore_folder']
    
    
#     random.seed(random_seed)
#     os.environ['PYTHONHASHSEED'] = str(random_seed)
#     np.random.seed(random_seed)
#     torch.manual_seed(random_seed)
#     torch.cuda.manual_seed(random_seed)
#     torch.cuda.manual_seed_all(random_seed) # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = False
    
    print(git_ignore_folder)
    
    sys_argv = sys.argv
    
    
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
    
    
    data_preparer = Data_preparer()
    
    dataset_train = torch.load(git_ignore_folder + "dataset_train")
    
    dataset_test = torch.load(git_ignore_folder + "dataset_test")
    
    data_train_loader = torch.load(git_ignore_folder + "data_train_loader")
    
    data_test_loader = torch.load(git_ignore_folder + "data_test_loader")
    
#     dataset_train, dataset_test, data_train_loader, data_test_loader = get_train_test_data_loader_by_name_lr(data_preparer, model_class, dataset_name, batch_size)
    
    dim = [len(dataset_train), len(dataset_train[0][0])]
    
    num_class = get_data_class_num_by_name(data_preparer, dataset_name)
    
    hyper_para_function=getattr(Data_preparer, "get_hyperparameters_" + dataset_name)
    
    model = model_class(dim[1], num_class)
    
    if is_GPU:
        model.to(device)
    
    init_model_params = list(model.parameters())
    
    
    criterion, optimizer, lr_scheduler = hyper_para_function(data_preparer, model.parameters(), learning_rate, regularization_coeff)
    
    hyper_params = [criterion, optimizer, lr_scheduler]
    
    
    
    
    lrs = ast.literal_eval(input)#map(float, input.strip('[]').split(','))
#     [2.0, 3.0, 4.0, 5.0]
    
#     model, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs, all_ids_list_all_epochs = model_training_skipnet(num_epochs, model, data_train_loader, data_test_loader, len(dataset_train), len(dataset_test), optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs)

# net, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs, X_theta_prod_seq, X_theta_prod_softmax_seq, random_ids_multi_super_iterations

    random_ids_all_epochs = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations_' + str(repetition))


    t1 = time.time()
    
#     model, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs, X_theta_prod_seq, X_theta_prod_softmax_seq, random_ids_multi_super_iterations = model_training_lr(random_ids_all_epochs, num_epochs, model, dataset_train, data_test_loader, len(dataset_train), len(dataset_test), optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs)
    model, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs, X_theta_prod_seq, X_theta_prod_softmax_seq, random_ids_multi_super_iterations = model_training_lr_test(random_ids_all_epochs, num_epochs, model, dataset_train, dataset_test, len(dataset_train), len(dataset_test), optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs)


    
    t2 = time.time()


    data_train_loader = update_data_train_loader(len(dataset_train), dataset_train, random_ids_multi_super_iterations, batch_size)
    
    
    t3 = time.time()
    
    capture_provenance(git_ignore_folder, data_train_loader, len(dataset_train), dim, num_epochs, num_class, batch_size, int((dim[0] - 1)/batch_size) + 1, torch.stack(random_ids_multi_super_iterations), X_theta_prod_softmax_seq, X_theta_prod_seq)

    data_train_loader.batch_sampler.reset_ids()
    

    x_sum_by_class_by_batch = compute_x_sum_by_class_by_batch(data_train_loader, len(dataset_train), batch_size, num_class, random_ids_multi_super_iterations)
    
    
    data_train_loader.batch_sampler.reset_ids()
    
    t4 = time.time()
    
    
    print("training time full::", t2 - t1)
    
    print("provenance prepare time::", t4 - t3)
    
    
    
    torch.save(dataset_train, git_ignore_folder + "dataset_train")
    
    torch.save(dataset_test, git_ignore_folder + "test_data")
    
    
    torch.save(x_sum_by_class_by_batch, git_ignore_folder+'x_sum_by_class')
#     torch.save(all_ids_list_all_epochs, git_ignore_folder + "all_ids_list_all_epochs")
    
#     torch.save(ids2_list_all_epochs, git_ignore_folder + "ids2_list_all_epochs")
    
#     torch.save(delta_data_ids, git_ignore_folder + "delta_data_ids")
    
    
    torch.save(gradient_list_all_epochs, git_ignore_folder + 'gradient_list_all_epochs')
    
    torch.save(para_list_all_epochs, git_ignore_folder + 'para_list_all_epochs')
    
    torch.save(learning_rate_all_epochs, git_ignore_folder + 'learning_rate_all_epochs')


    torch.save(random_ids_multi_super_iterations, git_ignore_folder + 'random_ids_multi_super_iterations')
                  
    torch.save(num_epochs, git_ignore_folder+'epoch')    
    
    torch.save(hyper_params, git_ignore_folder + 'hyper_params')
    
#     save_random_id_orders(git_ignore_folder, random_ids_multi_super_iterations)
    
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
    
    
    
    
    
    
    