'''
Created on Jun 24, 2020

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


sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Models')


# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/multi_nomial_logistic_regression')


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))





try:
#     from data_IO.Load_data import *
#     from utils import *
#     from Interpolation.piecewise_linear_interpolation_2D import *
    from Models.DNN import DNNModel
    from Models.Data_preparer import *
    from Models.DNN_single import DNNModel_single
    from Models.DNN2 import DNNModel2
    from Models.DNN3 import DNNModel3
    from Models.Pretrained_models import *
#     from Batch_samplers import Batch_sampler
#     from multi_nomial_logistic_regression.Multi_logistic_regression import *
#     from multi_nomial_logistic_regression.incremental_updates_logistic_regression_multi_dim import *
except ImportError:
#     from Load_data import *
#     from utils import *
#     from piecewise_linear_interpolation_2D import *
    from Models.DNN import DNNModel
    from Models.DNN2 import DNNModel2
    from Models.DNN3 import DNNModel3
    from Models.Data_preparer import *
    from Models.DNN_single import DNNModel_single
    from Models.Pretrained_models import *
#     from Batch_samplers import Batch_sampler
#     from multi_nomial_logistic_regression.Multi_logistic_regression import *
#     from multi_nomial_logistic_regression.incremental_updates_logistic_regression_multi_dim import *


baseline_method = 'baseline'

deltagrad_method = 'deltagrad'


gitignore_repo = '../.gitignore/'

def get_sampling_each_iteration0(random_ids_multi_super_iterations, add_num, num_mini_batches, id):


    added_random_ids_multi_super_iteration = []

    for i in range(len(random_ids_multi_super_iterations)):
        
        mini_batch_ids = torch.from_numpy(np.random.choice(num_mini_batches, add_num, replace=True))
            
            
        curr_added_random_ids = []
        
        for j in range(num_mini_batches):
            added_ids = torch.nonzero(mini_batch_ids == j)

            if added_ids.shape[0]> 0:
                added_ids += id

            curr_added_random_ids.append(added_ids.view(-1))
        
        added_random_ids_multi_super_iteration.append(curr_added_random_ids)


    return added_random_ids_multi_super_iteration

def get_subset_training_data(num, delta_data_ids):
    selected_rows = torch.tensor(list(set(range(num)) - set(delta_data_ids.tolist())))
    
    return selected_rows

def get_data_class_num_by_name(data_preparer, name):
    
    
    function=getattr(Data_preparer, "get_num_class_" + name)
    
    num_class = function(data_preparer)
        
    return num_class

def get_train_test_data_loader_lr(Model, train_X, train_Y, test_X, test_Y):
    
    
    dataset_train = Model.MyDataset(train_X, train_Y)
    dataset_test = Model.MyDataset(test_X, test_Y)
    
#     data_train_loader = DataLoader(dataset_train, batch_size=specified_batch_size, shuffle=True, num_workers=0)
#     data_test_loader = DataLoader(dataset_test, batch_size=specified_batch_size, num_workers=0)

    return dataset_train, dataset_test


def random_shuffle_data(X, Y, dim, noise_data_ids):
         
    random_ids = torch.randperm(dim[0])
     
    X = X[random_ids]
     
     
    Y = Y[random_ids]
    
    
    shuffled_noise_data_ids = torch.zeros(noise_data_ids.shape)
    
    for i in range(noise_data_ids.shape[0]):
        
        shuffled_id = torch.nonzero(random_ids == noise_data_ids[i])
        
#             print(shuffled_id)
        
        shuffled_noise_data_ids[i] = shuffled_id 
        
        
    return X, Y, shuffled_noise_data_ids

def get_train_test_data_loader_by_name_lr(data_preparer, Model, name, git_ignore_folder):
    
    
    function=getattr(Data_preparer, "prepare_" + name)
    
    train_X, train_Y, test_X, test_Y = function(data_preparer, git_ignore_folder)
    
    
    dataset_train, dataset_test= get_train_test_data_loader_lr(Model, train_X, train_Y, test_X, test_Y)
    
    return dataset_train, dataset_test


def get_lr_list(lrs, lens):
    
    learning_rates = []    
    
    for i in range(len(lrs)):
        
        learning_rates.extend([lrs[i]]*lens[i])
        
    return learning_rates
        

def update_learning_rate(optim, learning_rate):
    for g in optim.param_groups:
        g['lr'] = learning_rate
    
    
        
def append_gradient_list(gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, model, X, is_GPU, device):
    
    
    gradient_list = []
    
    para_list = []
    
    
    for param in model.parameters():
        if not is_GPU:
            gradient_list.append(param.grad.clone())
            para_list.append(param.data.clone())
        else:
            gradient_list.append(param.grad.cpu().clone())
            para_list.append(param.data.cpu().clone())
        
    
    
    if output_list_all_epochs is not None:
        
        output_list,_ = model.get_output_each_layer(X)   
        output_list_all_epochs.append(output_list)
        
            
    gradient_list_all_epochs.append(gradient_list)
    
    
    
    para_list_all_epochs.append(para_list)
    
    
    
def test(net, dataset_test, batch_size, criterion, data_test_size, is_GPU, device):
    net.eval()
    total_correct = 0
    avg_loss = 0.0
#     for i, items in enumerate(data_test_loader):
    for i in range(0, data_test_size, batch_size):
        
        
        end_id = i + batch_size
            
        if end_id > data_test_size:
            end_id = data_test_size
        
        if not is_GPU:
            images, labels = dataset_test.data[i:end_id], dataset_test.labels[i:end_id]
        else:
            images, labels = dataset_test.data[i:end_id].to(device), dataset_test.labels[i:end_id].to(device)
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            labels = torch.nonzero(labels)[:,1]
        
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= data_test_size
    
    net.train()
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / data_test_size))


def save_random_id_orders(git_ignore_folder, random_ids_multi_super_iterations):
    sorted_ids_multi_super_iterations = []
    
    
    for i in range(len(random_ids_multi_super_iterations)):
        sorted_ids_multi_super_iterations.append(random_ids_multi_super_iterations[i].numpy().argsort())
        
        
    torch.save(sorted_ids_multi_super_iterations, git_ignore_folder + 'sorted_ids_multi_super_iterations')


def random_generate_subset_ids2(delta_size, all_ids_list):
    
    num = len(all_ids_list)
    
    delta_data_ids = set()
    
    while len(delta_data_ids) < delta_size:
        id = random.randint(0, num-1)
        delta_data_ids.add(all_ids_list[id])
    
    return torch.tensor(list(delta_data_ids))

def generate_delta_ids(start, git_ignore_folder, noise_rate):
    if start:
        
#         training_data, trainin_labels, test_data, test_labels = function(data_preparer)
        dataset_train = torch.load(git_ignore_folder + "dataset_train")
        
        print(dataset_train.data.shape)
        
        full_ids_list = list(range(len(dataset_train.data)))
        
        delta_data_ids = random_generate_subset_ids2(int(len(dataset_train.data)*noise_rate), full_ids_list)
            
        train_data_len = len(dataset_train.data)
        
        torch.save(train_data_len, git_ignore_folder + 'train_data_len')
        
    else:
        old_delta_ids = torch.load(git_ignore_folder + "delta_data_ids")
    
        train_data_len = torch.load(git_ignore_folder + 'train_data_len')
    
        full_ids_list = list(range(train_data_len))
    
        remaining_size = int(train_data_len*noise_rate) - old_delta_ids.shape[0]
        
        remaining_full_ids_list = list(set(full_ids_list).difference(set(old_delta_ids.tolist())))
        
        if remaining_size > 0:
            curr_delta_data_ids = random_generate_subset_ids2(remaining_size, remaining_full_ids_list)
        
            delta_data_ids = torch.tensor(list(set(old_delta_ids.tolist()).union(set(curr_delta_data_ids.tolist()))))
        else:
            delta_data_ids = old_delta_ids
        
        
    
#     delta_data_ids = random_deletion(len(dataset_train), 1)
    
    print(delta_data_ids)
    
    torch.save(delta_data_ids, git_ignore_folder + "delta_data_ids")


def get_sorted_random_ids(random_ids_multi_epochs):
    
    sorted_ids_multi_epochs = []
    for i in range(len(random_ids_multi_epochs)):
        sorted_ids_multi_epochs.append(random_ids_multi_epochs[i].numpy().argsort())
        
    return sorted_ids_multi_epochs

def generate_random_ids_list(dataset_train, epochs, git_ignore_folder):
    
    
    
    
#     for j in range(repetition): 
        
    random_ids_all_epochs = []
    
    for i in range(epochs):
        random_ids = torch.randperm(len(dataset_train))
    
        random_ids_all_epochs.append(random_ids)
    sorted_random_ids_all_epochs = get_sorted_random_ids(random_ids_all_epochs)
    
    
    torch.save(random_ids_all_epochs, git_ignore_folder + 'random_ids_multi_epochs')
    
    torch.save(sorted_random_ids_all_epochs, git_ignore_folder + 'sorted_ids_multi_epochs')


def get_all_vectorized_parameters1(para_list):
    
    res_list = []
    
    i = 0
    
    for param in para_list:
        
        res_list.append(param.data.view(-1))
        
        i += 1
        
    return torch.cat(res_list, 0).view(1,-1)

def clear_gradients(para_list):
    for param in para_list:
        param.grad.zero_()


def get_model_para_shape_list(para_list):
    
    shape_list = []
    
    full_shape_list = []
    
    total_shape_size = 0
    
    for para in list(para_list):
        
        all_shape_size = 1
        
        
        for i in range(len(para.shape)):
            all_shape_size *= para.shape[i]
        
        total_shape_size += all_shape_size
        shape_list.append(all_shape_size)
        full_shape_list.append(para.shape)
        
    return full_shape_list, shape_list, total_shape_size

def post_processing_gradien_para_list_all_epochs(para_list_all_epochs, grad_list_all_epochs):
    
#     num = 0
    
    _,_,total_shape_size = get_model_para_shape_list(para_list_all_epochs[0])
        
    
    
    para_list_all_epoch_tensor = torch.zeros([len(para_list_all_epochs), total_shape_size], dtype = torch.double)
    
    grad_list_all_epoch_tensor = torch.zeros([len(grad_list_all_epochs), total_shape_size], dtype = torch.double)
    
    for i in range(len(para_list_all_epochs)):
        
        para_list_all_epoch_tensor[i] = get_all_vectorized_parameters1(para_list_all_epochs[i])
        
        grad_list_all_epoch_tensor[i] = get_all_vectorized_parameters1(grad_list_all_epochs[i])
        
    
    
    
    return para_list_all_epoch_tensor, grad_list_all_epoch_tensor

'''pre-fetch parts of the history parameters and gradients into GPU to save the IO overhead'''
def cache_grad_para_history(git_ignore_folder, cached_size, is_GPU, device):
    para_list_all_epochs = torch.load(git_ignore_folder + 'para_list_all_epochs')
    
    gradient_list_all_epochs = torch.load(git_ignore_folder + 'gradient_list_all_epochs')
    
    para_list_all_epoch_tensor, grad_list_all_epoch_tensor = post_processing_gradien_para_list_all_epochs(para_list_all_epochs, gradient_list_all_epochs)

    end_cached_id = cached_size
    
    if end_cached_id > len(para_list_all_epochs):
        end_cached_id =  len(para_list_all_epochs)
    

    para_list_GPU_tensor = para_list_all_epoch_tensor[0:cached_size]
    
    grad_list_GPU_tensor = grad_list_all_epoch_tensor[0:cached_size]

    if is_GPU:
        para_list_GPU_tensor = para_list_GPU_tensor.to(device)
        
        grad_list_GPU_tensor = grad_list_GPU_tensor.to(device) 
        
    return grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor

def compute_model_para_diff(model1_para_list, model2_para_list):
    
    diff = 0
    
    norm1 = 0
    
    norm2 = 0
    
    all_dot = 0
    
    
    for i in range(len(model1_para_list)):
        
        param1 = model1_para_list[i].to('cpu')
        
        param2 = model2_para_list[i].to('cpu')
        
        curr_diff = torch.norm(param1 - param2, p='fro')
        
        norm1 += torch.pow(torch.norm(param1, p='fro'), 2)
        
        norm2 += torch.pow(torch.norm(param2, p='fro'), 2)
        
        
        all_dot += torch.sum(param1*param2)
        
#         print("curr_diff:", i, curr_diff)
        
        diff += curr_diff*curr_diff
        
    print('model difference (l2 norm):', torch.sqrt(diff))
    
#     print(all_dot/torch.sqrt(norm1*norm2))


def compute_derivative_one_more_step(model, batch_X, batch_Y, criterion, optimizer):
    
    
    optimizer.zero_grad()

    output = model(batch_X)

    loss = criterion(output, batch_Y)
    
    loss.backward()
    
    
    return loss

def init_model(model, para_list):
    
    i = 0
    
    for m in model.parameters():
        
        
        
        m.data.copy_(para_list[i])
        if m.grad is not None:
            m.grad.zero_()
        m.requires_grad= True
        i += 1
        
        
def get_devectorized_parameters(params, full_shape_list, shape_list):
    
    params = params.view(-1)
    
    para_list = []
    
    pos = 0
    
    for i in range(len(full_shape_list)):
        
        param = 0
        if len(full_shape_list[i]) >= 2:
            
            curr_shape_list = list(full_shape_list[i])
            
            param = params[pos: pos+shape_list[i]].view(curr_shape_list)
            
        else:
            param = params[pos: pos+shape_list[i]].view(full_shape_list[i])
        
        para_list.append(param)
    
        
        pos += shape_list[i]
    
    return para_list
