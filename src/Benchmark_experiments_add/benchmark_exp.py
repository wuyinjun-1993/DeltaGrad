'''
Created on Feb 4, 2019

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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/multi_nomial_logistic_regression')


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))





try:
    from data_IO.Load_data import *
    from utils import *
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
#     from multi_nomial_logistic_regression.Multi_logistic_regression import *
#     from multi_nomial_logistic_regression.incremental_updates_logistic_regression_multi_dim import *
except ImportError:
    from Load_data import *
    from utils import *
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
#     from multi_nomial_logistic_regression.Multi_logistic_regression import *
#     from multi_nomial_logistic_regression.incremental_updates_logistic_regression_multi_dim import *
loss_threshold = 0.1
random_ids_multi_super_iterations = []
    
    
random_seed=0
#         if type(m) == nn.Linear:
#             nn.init.constant_(m.weight, 0)
#             
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
            
softmax_func = nn.Softmax(dim = 1)

sigmoid_func = nn.Sigmoid()

cut_off_epoch = 100

max_para_num_opt = 10000

default_epoch_num = 1


default_batch_size = 10


theta_record_threshold=0.7

# def create_models(input_dim, hidden_dims, output_dim):
#     layers = []
#     layers.append(nn.Linear(input_dim, hidden_dims[0]))
#     layers.append(nn.Sigmoid())
#     
#     for i in range(len(hidden_dims) - 1):
#         layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
#         layers.append(nn.Sigmoid())
#     
#     
#     
#     layers.append(nn.Linear(hidden_dims[-1], output_dim))
#     layers.append(nn.Sigmoid())
#     
#     net = nn.Sequential(*layers)
#     
#     return net
# 
# def get_output_each_layer(model, x):
#         
#         
#         output_list = []
#         
#         output_list.append(torch.cat((x, torch.ones(x.shape[0], 1)), 1))
#         
#         
#         out = self.fc1(x)
#         # Non-linearity 1
#         out = self.relu1(out)
#         
#         
#         output_list.append(torch.cat((out, torch.ones(out.shape[0], 1)), 1))
#         
#         for i in range(len(self.linear_layers)):
#             out = self.linear_layers[i](out)
#             out = self.activation_layers[i](out)
#             output_list.append(torch.cat((out, torch.ones(out.shape[0], 1)), 1))
#         
#         
#         
#         
#         # Linear function 2
#         out = self.fc2(out)
#         
#         out2 = self.fc3(out)
#         
#         output_list.append(out2)
#         
#         
#         return output_list
    
    
    
        
        
        
    

def get_onehot_y(Y, dim, num_class):
    
#     x_sum_by_class = torch.zeros([num_class, dim[1]], dtype = torch.double)
    
    
    y_onehot = torch.DoubleTensor(dim[0], num_class)

    Y = Y.type(torch.LongTensor)

# In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, Y.view(-1, 1), 1)
    
    
    return y_onehot


def get_onehot_y_time_X(X, Y, dim, num_class):
    
#     x_sum_by_class = torch.zeros([num_class, dim[1]], dtype = torch.double)
    
    
    y_onehot = torch.DoubleTensor(dim[0], num_class)

    Y = Y.type(torch.LongTensor)

# In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, Y.view(-1, 1), 1)
    
    
    return y_onehot

# def model_update_standard_lib(num_epochs, X, Y, test_X, test_Y, learning_rate, error, model):
#     count = 0
# #     for epoch in range(num_epochs):
# 
#     loss = np.infty
# 
#     while loss > loss_threshold and count < num_epochs:
# #         for i, (images, labels) in enumerate(train_loader):
#             
# #         for i in range(X.shape[0]):
#     
#     
# #             train = Variable(images.view(-1, 28*28))
#         train = Variable(X)
#         labels = Variable(Y.view(-1))
#         
#         # Clear gradients
# #         optimizer.zero_grad()
#         
#         # Forward propagation
#         outputs = model(train)
#         
#         # Calculate softmax and ross entropy loss
#         
# #         print(outputs)
# #         
# #         print(labels)
#         
#         labels = labels.type(torch.LongTensor)
#         
#         loss = error(outputs, labels)
#         
#         # Calculating gradients
# #         loss.backward(retain_graph = True, create_graph=True)
#         
#         loss.backward()
#         
#         update_and_zero_model_gradient(model,learning_rate)
#         
#         print("loss:", loss)
#         
#         if count % 10 == 0:
#             # Calculate Accuracy         
#             correct = 0
#             total = 0
#             # Predict test dataset
# #                 for images, labels in test_loader:
# #             for j in range(test_X.shape[0]):
# 
# #                     test = Variable(images.view(-1, 28*28))
#             test = Variable(test_X)
#             
#             labels = test_Y.view(-1).type(torch.LongTensor)
#             
#             # Forward propagation
#             outputs = model(test)
#             
#             # Get predictions from the maximum value
#             predicted = torch.max(outputs.data, 1)[1]
#             
#             # Total number of labels
#             total += len(labels)
# 
#             # Total correct predictions
#             correct += (predicted == labels).sum()
#             
#             accuracy = 100 * correct / float(total)
# #             if count % 500 == 0:
#                 # Print Loss
#                 
#                 
#             print("accuracy:: {} %", format(accuracy))
#         
#         
# #         print("epoch::", epoch)
#         
# #         print_model_para(model)
#         
#         
#         
#         # Update parameters
# #         optimizer.step()
#         
#         count += 1
# #             print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data[0].item(), accuracy.item()))
#     return model



def construct_gradient_list(gradient_list, res_list, model):
    
    gradient_list.clear()
    del gradient_list[:]
    
    for param in model.parameters():
        gradient_list.append(param.grad.clone())
        res_list.append(param.data.clone())
        
        
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
    
def append_gradient_list2(gradient_list_all_epochs, para_list_all_epochs, para, gradient_full, full_shape_list, shape_list, is_GPU, device):
    
    
#     gradient_list = []
    
    para_list = []
    
    
    for param in list(para):
        if not is_GPU:
#             gradient_list.append(param.grad.clone())
            para_list.append(param.data.clone())
        else:
#             gradient_list.append(param.grad.cpu().clone())
            para_list.append(param.data.cpu().clone())
        
    
    
    gradient_list_all_epochs.append(get_devectorized_parameters(gradient_full, full_shape_list, shape_list))
    
    
    
    para_list_all_epochs.append(para_list)
        
def get_model_gradient(model):
    
    gradient_list = []
    
    for param in model.parameters():
        gradient_list.append(param.grad.clone())
        
    return gradient_list

def create_piecewise_linea_class(linearized_Function):
#     x_l = torch.tensor(-10, dtype=torch.double)
#     x_u = torch.tensor(10, dtype=torch.double)
    x_l = -20.0
    x_u =20.0
    num = 1000000
    Pi = piecewise_linear_interpolication(x_l, x_u, linearized_Function, num)
    
    return Pi


def loss_function2(output, Y, dim):
    
#     res = 0
    
    
#     sigmoid_res = torch.stack(list(map(bia_function, Y*torch.mm(X, theta))))

#     sigmoid_res = Y*torch.mm(X, theta)
#     data_trans = sigmoid_res.apply(lambda x :  ())

#     sigmoid_res = -log_sigmoid_layer(Y*torch.mm(X, theta))
#     theta = theta.view(dim[1], num_class)




    X_theta_prod = output
    
    
    X_theta_prod_softmax = softmax_func(X_theta_prod)
    
    res = -torch.sum(torch.log(torch.gather(X_theta_prod_softmax, 1, Y.view(-1,1))))/dim[0]
    
    return res
    
#     return res + beta/2*torch.pow(torch.norm(theta, p =2), 2)


def get_subset_data_per_epoch(curr_rand_ids, full_id_set):
    
    
#     ids = torch.nonzero(curr_rand_ids.view(-1,1) == full_id_set.view(1,-1))
#     
#     return curr_rand_ids[ids[:,0]]
    
    
    
    curr_rand_id_set = set(curr_rand_ids.tolist())
            
    curr_matched_ids = torch.tensor(list(curr_rand_id_set.intersection(full_id_set)))
    
    return curr_matched_ids



def get_subset_data_per_epoch_skipnet(curr_rand_ids, full_id_set, all_ids_list):
    
    
#     ids = torch.nonzero(curr_rand_ids.view(-1,1) == full_id_set.view(1,-1))
#     
#     return curr_rand_ids[ids[:,0]]
    
    
    
    curr_rand_id_set = set(curr_rand_ids.tolist())
            
    curr_matched_id_set = curr_rand_id_set.intersection(full_id_set)        
    
    curr_matched_ids = torch.sort(torch.tensor(list(curr_matched_id_set)))[0]
    
    curr_remaining_id_tensor = torch.sort(torch.tensor(list(curr_rand_id_set.difference(curr_matched_id_set))))[0]
    
    
    
#     ids_list = []
    
    
    
    
    if curr_remaining_id_tensor.shape[0] > 0:
            
        for j in range(len(all_ids_list)):    
            removed_ids_set = []
            
            
            
            
            max_id_list = []
            
            for i in range(len(curr_remaining_id_tensor)):
            
                curr_id = torch.nonzero(curr_remaining_id_tensor[i] == curr_rand_ids).view(-1)
            
                curr_removed_ids = torch.nonzero(all_ids_list[j][:,0] == curr_id.item()).view(-1).tolist()
            
                removed_ids_set.extend(curr_removed_ids)
            
                max_id_list.append(max(curr_removed_ids))
                
            for id in max_id_list:
                all_ids_list[j][id+1:,0] -= 1
            
            
            
            curr_all_ids_list_numpy = all_ids_list[j].numpy()
            
            all_ids_list[j] = torch.tensor(np.delete(curr_all_ids_list_numpy, removed_ids_set, 0))
            
#             remaining_ids = np.sort(list(set(list(range(all_ids_list[j].shape[0]))).difference(removed_ids_set)))
#             
#             
#             all_ids_list[j] = all_ids_list[j][remaining_ids]
        
#         del curr_ids_list2[curr_id]
        
#         ids_list.append(curr_id.item())
        
#         if curr_matched_ids[i] in curr_rand_id_set:
#             ids_list.append(i)
            
            
    
    
    return curr_matched_ids



def get_remaining_subset_data_per_epoch(curr_rand_ids, removed_rand_ids):
    
    
    curr_rand_id_set = set(curr_rand_ids.tolist())
    
    curr_remaining_ids_set = list(curr_rand_id_set.difference(set(removed_rand_ids.tolist())))
    
    res = torch.sort(torch.tensor(curr_remaining_ids_set))[0]
    
    return res

def get_remaining_subset_data_per_epoch_skipnet(curr_rand_ids, removed_rand_ids, all_ids_list):
                
    
#     curr_removed_ids_list = []
#     
#     curr_removed_ids_list2 = []
    
    all_curr_removed_ids_list = []
    
#     for j in range(len(all_ids_list)):
#         all_curr_removed_ids_list.append([])
    
    
    sorted_removed_rand_ids = torch.sort(removed_rand_ids)[0]
    
    
    if removed_rand_ids.shape[0] > 0:
    
        for j in range(len(all_ids_list)):
        
        
            removed_ids_set = []
                
                
                
                
            max_id_list = []
            
        
            
        
            curr_ids_list_to_removed = []
            
            for i in range(sorted_removed_rand_ids.shape[0]):
            
                curr_id = torch.nonzero(sorted_removed_rand_ids[i] == curr_rand_ids)
                
                curr_removed_ids = torch.nonzero(all_ids_list[j][:,0] == curr_id.view(-1).item()).view(-1)
                
                removed_ids_set.extend(curr_removed_ids.tolist())
                
                max_id_list.append(max(curr_removed_ids))
            
                ids_list_to_removed = all_ids_list[j][curr_removed_ids]
                
                ids_list_to_removed[:,0] = i
                
                curr_ids_list_to_removed.append(ids_list_to_removed)
            
            all_curr_removed_ids_list.append(torch.cat(curr_ids_list_to_removed,0))
            
            
            for id in max_id_list:
                all_ids_list[j][id+1:,0] -= 1
            
            
            curr_all_ids_list_numpy = all_ids_list[j].numpy()
            
            all_ids_list[j] = torch.tensor(np.delete(curr_all_ids_list_numpy, removed_ids_set, 0))
            
#             remaining_ids = list(set(list(range(all_ids_list[j].shape[0]))).difference(removed_ids_set))
#             
#             remaining_ids = np.sort(remaining_ids)
#             
#             all_ids_list[j] = all_ids_list[j][remaining_ids]
#             
#             print(torch.max(all_ids_list[j] - torch.tensor(exp_all_ids_list)))
#             
#             print(torch.min(all_ids_list[j] - torch.tensor(exp_all_ids_list)))
#             
#             print("here")
        
#             del all_ids_list[j][curr_id]
#         curr_removed_ids_list.append(curr_ids_list[curr_id].copy())
#         
#         curr_removed_ids_list2.append(curr_ids_list2[curr_id].copy())
        
        
        
#         del curr_ids_list2[curr_id]
    
    
    
    return all_curr_removed_ids_list
    


def get_sampling_each_iteration0(random_ids_multi_super_iterations, add_num, num_mini_batches, id):


    added_random_ids_multi_super_iteration = []


#     for j in range(len(add_num)):
        

    for i in range(len(random_ids_multi_super_iterations)):
        
#         for j in range(len(add_num)):

        mini_batch_ids = torch.from_numpy(np.random.choice(num_mini_batches, add_num, replace=True))
            
            
        curr_added_random_ids = []
        
        for j in range(num_mini_batches):
            added_ids = torch.nonzero(mini_batch_ids == j)
            
#             if added_ids.shape[0] > 0:
#                 print("here")

            if added_ids.shape[0]> 0:
                added_ids += id

            curr_added_random_ids.append(added_ids.view(-1))
#             else:
#                 curr_added_random_ids.append([])
        
#         random_ids = torch.randperm(add_num)
        
        added_random_ids_multi_super_iteration.append(curr_added_random_ids)


    return added_random_ids_multi_super_iteration

def get_sampling_each_iteration(random_ids_multi_super_iterations, add_num, num_mini_batches):


    added_random_ids_multi_super_iteration = []


#     for j in range(len(add_num)):
        

    for i in range(len(random_ids_multi_super_iterations)):
        
#         for j in range(len(add_num)):

        mini_batch_ids = torch.from_numpy(np.random.choice(num_mini_batches, add_num, replace=True))
            
            
        curr_added_random_ids = []
        
        for j in range(num_mini_batches):
            added_ids = torch.nonzero(mini_batch_ids == j)
            
#             if added_ids.shape[0] > 0:
#                 print("here")
            curr_added_random_ids.append(added_ids.view(-1))
#             else:
#                 curr_added_random_ids.append([])
        
#         random_ids = torch.randperm(add_num)
        
        added_random_ids_multi_super_iteration.append(curr_added_random_ids)


    return added_random_ids_multi_super_iteration

def get_sampling_each_iteration_union_prev(random_ids_multi_super_iterations, add_num, num_mini_batches, prev_added_random_ids_multi_super_iteartion, id):


    added_random_ids_multi_super_iteration = []


#     for j in range(len(add_num)):
        

    for i in range(len(random_ids_multi_super_iterations)):
        
#         for j in range(len(add_num)):

        mini_batch_ids = torch.from_numpy(np.random.choice(num_mini_batches, add_num, replace=True))
        
        
        prev_added_random_ids = prev_added_random_ids_multi_super_iteartion[i]
            
        curr_added_random_ids = []
        
        for j in range(num_mini_batches):
            added_ids = torch.nonzero(mini_batch_ids == j)
            
            prev_added_random_ids_this_iter = prev_added_random_ids[j]
            
#             if prev_added_random_ids_this_iter.shape[0] > 0:
#                 print("here")
            
            if added_ids.shape[0]> 0:
                added_ids += id
            
#             if added_ids.shape[0] > 0:
#                 print("here")

            unioned_random_ids = set(prev_added_random_ids_this_iter.view(-1).tolist()).union(set(added_ids.view(-1).tolist()))

            curr_added_random_ids.append(torch.tensor(list(unioned_random_ids)))
#             else:
#                 curr_added_random_ids.append([])
        
#         random_ids = torch.randperm(add_num)
        
        added_random_ids_multi_super_iteration.append(curr_added_random_ids)


    return added_random_ids_multi_super_iteration


def model_update_standard_lib(max_epoch, dataset_train, dim, model, random_ids_multi_super_iterations, batch_size, learning_rate_all_epochs, added_random_ids_multi_super_iteration, X_to_add, Y_to_add, criterion, optimizer, para_list_all_epochs, gradient_list_all_epochs, is_GPU, device, record_params, regularization_coeff):
    count = 0
#     for epoch in range(num_epochs):
    loss = np.infty

    elapse_time = 0

    overhead = 0
    
    overhead2 = 0

    t1 = time.time()
    
    para = list(model.parameters())
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    exp_gradient_list_all_epochs = []
    
    exp_para_list_all_epochs = []
    old_lr = -1
#     selected_rows_set = set(selected_rows.view(-1).tolist())
    
#     train = Variable(X)
#     labels = Variable(Y.view(-1))
#     labels = labels.type(torch.LongTensor)
#     for k in range(len(random_ids_multi_super_iterations)):
    for k in range(max_epoch):
        
        random_ids = random_ids_multi_super_iterations[k]
        
        added_random_ids = added_random_ids_multi_super_iteration[k]
        

        
#         for i in range(len(batch_X_list)):

#         sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
#         all_indexes = np.sort(np.searchsorted(random_ids.numpy(), delta_ids.numpy()))
        
#         all_indexes = np.sort(sort_idx[np.searchsorted(random_ids.numpy(),selected_rows.numpy(),sorter = sort_idx)])
        
#         all_indexes = np.sort(sort_idx[selected_rows])

        id_start = 0
        
        id_end = 0

#         print('epoch::', k)

        j = 0
        
        to_add = True

        for i in range(0, dim[0], batch_size):
            
            end_id = i + batch_size
            
#             added_end_id = j + added_batch_size
            
            curr_to_add_rand_ids = added_random_ids[j]
            
            
            if curr_to_add_rand_ids.shape[0] > 0:
                full_random_ids = torch.cat([curr_to_add_rand_ids, random_ids[i:end_id]], 0)
            else:
                full_random_ids = random_ids[i:end_id]
            
#             print(j, curr_to_add_rand_ids.shape[0])
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            
#             if added_end_id > X_to_add.shape[0]:
#                 added_end_id = X_to_add.shape[0]
            
            
#             if curr_to_add_rand_ids.shape[0] <= 0:
#                 to_add = False
#             
#             else:
#                 to_add = True
            
#             if curr_to_add_rand_ids.shape[0] > 0:
#                 print("here")
            
#             print(count)
            
            learning_rate = learning_rate_all_epochs[count]
            
            
            
            
#             regularization_rate = list(optimizer.param_groups)[0]['weight_decay']
            
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate
            
#             curr_rand_ids = random_ids[i:end_id]
            
#             curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)

#             if all_indexes[-1] < end_id:
#                 id_end = all_indexes.shape[0]
#             else:
#                 id_end = np.argmax(all_indexes >= end_id)
                
#             curr_rand_ids = random_ids[i:end_id]
            t5 = time.time()
            
            grad_dual = 0
            
            curr_to_add_size = 0
            
            
            init_model(model, para)
            
#             if to_add:
# #                 curr_to_add_rand_ids = added_random_ids[j:added_end_id]
#             
#                 curr_to_add_size = curr_to_add_rand_ids.shape[0]
#             
# #             curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
#             
# #             curr_matched_ids,_ = torch.sort(curr_matched_ids)
# #             while 1:
# #                 if id_end >= all_indexes.shape[0] or all_indexes[id_end] >= end_id:
# #                     break
# #                 
# #                 id_end = id_end + 1
#             
# #             curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
#             
# #             curr_matched_ids_size = curr_matched_ids.shape[0]
# 
# 
# 
# #             curr_matched_ids,_ = torch.sort(curr_matched_ids)
# 
# #             print(curr_matched_ids)
#             
#             
# #             if curr_matched_ids_size <= 0:
# #                 continue
#                 if is_GPU:
#                     compute_derivative_one_more_step(model, X_to_add[curr_to_add_rand_ids].to(device), Y_to_add[curr_to_add_rand_ids].to(device), criterion, optimizer)
#             
#                 else:
#                     compute_derivative_one_more_step(model, X_to_add[curr_to_add_rand_ids], Y_to_add[curr_to_add_rand_ids], criterion, optimizer)
#             
#                 grad_dual = get_all_vectorized_parameters1(model.get_all_gradient())
            
            
#                 batch_X = torch.cat([dataset_train.data[curr_rand_ids], ], dim = 0)
#                 
#                 batch_Y = torch.cat([dataset_train.labels[curr_rand_ids], ], dim = 0)
                
#         outputs = model(train)
        
#         loss = error(outputs, labels)
#             else:
                
            batch_X = dataset_train.data[full_random_ids]
            
            batch_Y = dataset_train.labels[full_random_ids]
            
            if is_GPU:
                batch_X = batch_X.to(device)
                
                batch_Y = batch_Y.to(device)
            
            compute_derivative_one_more_step(model, batch_X, batch_Y, criterion, optimizer)

            grad_full = get_all_vectorized_parameters1(model.get_all_gradient())
            
#             grad_full = (grad_curr*batch_X.shape[0] + grad_dual*curr_to_add_size)/(batch_X.shape[0] + curr_to_add_size)
            
            if record_params:
#                 append_gradient_list(exp_gradient_list_all_epochs, None, exp_para_list_all_epochs, model, batch_X, is_GPU, device)
                append_gradient_list2(exp_gradient_list_all_epochs, exp_para_list_all_epochs, para, grad_full, full_shape_list, shape_list, is_GPU, device)

            
            
            para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*get_all_vectorized_parameters1(para) - learning_rate*grad_full, full_shape_list, shape_list)
            
#             if count == 1:
#             
#                 torch.save(batch_X, git_ignore_folder + 'tmp_batch_x')
#                  
#                 torch.save(batch_Y, git_ignore_folder + 'tmp_batch_y')
#                  
#                 torch.save(X_to_add[curr_to_add_rand_ids], git_ignore_folder + 'tmp_added_x')
#                  
#                 torch.save(Y_to_add[curr_to_add_rand_ids], git_ignore_folder + 'tmp_added_y')
#                  
#                 torch.save(grad_curr, git_ignore_folder + 'tmp_grad_remaining')
#                  
#                 torch.save(grad_dual, git_ignore_folder + 'tmp_grad_dual')
#                  
#                 print("here")
            
            
            
#             t6 = time.time()
#             
#             overhead2 += (t6 - t5)
#             
#             t3 = time.time()
#             
#             optimizer.zero_grad()
# 
# #             batch_X = dataset_train.data.data[curr_matched_ids]
# #             
# #             batch_Y = dataset_train.data.targets[curr_matched_ids]
#             
#             output = model(batch_X)
#             
# #             print(output[0])
# #             
# #             print(torch.sort(items[2])[0])
#     
#             loss = criterion(output, batch_Y)
#             
#             loss.backward()
#             
#             t4 = time.time()
#             
#             overhead += (t4 - t3)
#             print('parameter difference::')
#                 
#             compute_model_para_diff(para_list_all_epochs[count], list(model.parameters()))
#                 
#             print('gradient difference::')
#                 
#             compute_model_para_diff(gradient_list_all_epochs[count], list(get_model_gradient(model)))
#             loss = compute_loss(model, error, batch_X, batch_Y, beta)
#         
#             loss.backward()
            
#             optimizer.step()
#             update_and_zero_model_gradient(model,learning_rate)
            
    #         print("iteration::", count)
    #         
    #         print("loss::", loss)
            
            count += 1
            
            j += 1
             
#             print("loss::", loss)
    init_model(model, para)
    
    
    t2 = time.time()
        
    elapse_time += (t2 - t1)  

    print("training time is", elapse_time)
    
    print("overhead::", overhead)
    
    print("overhead2::", overhead2)
    
    return model, count, exp_para_list_all_epochs, exp_gradient_list_all_epochs
    

# def model_update_standard_lib(num_epochs, dataset_train, model, random_ids_multi_super_iterations, selected_rows, batch_size, learning_rate_all_epochs, criterion, optimizer, para_list_all_epochs, gradient_list_all_epochs, is_GPU, device, record_params):
#     count = 0
# #     for epoch in range(num_epochs):
# #     loss = np.infty
# 
#     elapse_time = 0
# 
#     t1 = time.time()
#     
#     
#     exp_gradient_list_all_epochs = []
#       
#     exp_para_list_all_epochs = []
#     
#     selected_rows_set = set(selected_rows.view(-1).tolist())
#     
# #     train = Variable(X)
# #     labels = Variable(Y.view(-1))
# #     labels = labels.type(torch.LongTensor)
# 
#     old_lr = -1
#     
# #     data_train_loader.shuffle = False
# 
# 
#     random_list_all_epochs = []
#     
# 
# 
#     for k in range(len(random_ids_multi_super_iterations)):
#         
#         random_ids = random_ids_multi_super_iterations[k]
#         
# #         id_start = 0
# #         
# #         id_end = 0
#         
#         random_ids_list = []
#         
#         for i in range(0, len(dataset_train), batch_size):
#             
#             end_id = i + batch_size
#             
#             if end_id > len(dataset_train):
#                 end_id = len(dataset_train)
#         
#             
#             curr_rand_ids = random_ids[i:end_id]
#             
#             curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
#             
#             random_ids_list.append(list(curr_matched_ids.numpy()))
#         
#         random_list_all_epochs.append(random_ids_list)    
# #             curr_matched_ids_size = curr_matched_ids.shape[0]
# 
#             
#             
#     curr_batch_sampler = Batch_sampler(random_list_all_epochs)
# 
# #     data_train_loader.batch_sampler = curr_batch_sampler
# #     data_train_loader2 = DataLoader(dataset_train, shuffle = True)
#     
# #     data_train_loader = DataLoader(dataset_train, num_workers=0, shuffle=False, batch_sampler=curr_batch_sampler)
#     
#     for k in range(len(random_ids_multi_super_iterations)):   
#         
#         print("epoch::", k)
#         
#         random_ids_list = random_list_all_epochs[k]
#         
# #         for i, items in enumerate(data_train_loader):
# #         for i in range(0, )
# #         for j in range(0, dim[0], batch_size):
#         for j in range(len(random_ids_list)):
#         
#         
#             curr_random_ids = random_ids_list[j]
# #             end_id = j + batch_size
# #             
# #             if end_id > dim[0]:
# #                 end_id = dim[0]
#             
# #             random.seed(random_seed)
# #             os.environ['PYTHONHASHSEED'] = str(random_seed)
# #             np.random.seed(random_seed)
# #             torch.manual_seed(random_seed)
# #             torch.cuda.manual_seed(random_seed)
# #             torch.cuda.manual_seed_all(random_seed) # if you are using multi-GPU.
# #             torch.backends.cudnn.benchmark = False
# #             torch.backends.cudnn.deterministic = True
# #             torch.backends.cudnn.enabled = False
#             
# #             curr_matched_ids_size = items[2].shape[0]
#             
#             curr_matched_ids_size = len(curr_random_ids)
#             
#             if curr_matched_ids_size <= 0:
#                 
#                 count += 1
#                 
#                 continue
#             
# #             print(items[2])
# #             
# #             print(random_ids_list[i])
#             
# #             print(torch.max(items[2] - torch.tensor(random_ids_list[i])))
# #             
# #             print(torch.min(items[2] - torch.tensor(random_ids_list[i])))
#             if not is_GPU:
# #                 batch_X = items[0]
# #                 
# #                 batch_Y = items[1]
# 
#                 batch_X = dataset_train.data[curr_random_ids]
#                 
#                 batch_Y = dataset_train.labels[curr_random_ids]
#                 
#             else:
# #                 batch_X = items[0].to(device)
# #                 
# #                 batch_Y = items[1].to(device)
#                 batch_X = dataset_train.data[curr_random_ids].to(device)
#                 
#                 batch_Y = dataset_train.labels[curr_random_ids].to(device)
#                 
#                 
# #             print(items[2].shape)
#             
# #             exp_ids = torch.sort(torch.tensor(random_list_all_epochs[k][i]))[0]
# #              
# #             curr_ids = torch.sort(torch.tensor(items[2]))[0]
# #              
# #             print("compare_len::", len(exp_ids) - len(curr_ids))
# #              
# #             if len(exp_ids) - len(curr_ids) == 0:
# #                 print(torch.max(exp_ids - curr_ids))
# #                 print(torch.min(exp_ids - curr_ids))
#             
#             learning_rate = learning_rate_all_epochs[count]
#             
#             
#             
#             
# #             regularization_rate = list(optimizer.param_groups)[0]['weight_decay']
#             
#             if not learning_rate == old_lr:
#                 update_learning_rate(optimizer, learning_rate)
#             
#             old_lr = learning_rate
#                 
# #             torch.save(batch_X, git_ignore_folder + 'tmp_batch_X')
# #             
# #             torch.save(batch_Y, git_ignore_folder + 'tmp_batch_Y')
# #             
# #             torch.save(learning_rate, git_ignore_folder + 'tmp_learning_rate')
# #             
# #             torch.save(curr_random_ids, git_ignore_folder + 'tmp_rand_ids')
# #             
# #             torch.save(dataset_train.data, git_ignore_folder + 'tmp_X')
#             
#             
# #             x_sum_by_class = compute_single_x_sum_by_class(batch_X, batch_Y, num_class)
#             
# #             torch.save(x_sum_by_class, git_ignore_folder + 'tmp_x_sum_by_class')
#             
# #             batch_X = X[curr_matched_ids]
# #             
# #             batch_Y = Y[curr_matched_ids]
# 
#             optimizer.zero_grad()
# 
# #             batch_X = dataset_train.data.data[curr_matched_ids]
# #             
# #             batch_Y = dataset_train.data.targets[curr_matched_ids]
#             
#             output = model(batch_X)
#             
# #             print(output[0])
# #             
# #             print(torch.sort(items[2])[0])
#     
#             loss = criterion(output, batch_Y)
#             
#             loss.backward()
#             
# #             print('parameter difference::')
# #                
# #             compute_model_para_diff(para_list_all_epochs[count], list(model.parameters()))
# #                
# #             print('gradient difference::')
# #                
# #             compute_model_para_diff(gradient_list_all_epochs[count], list(get_model_gradient(model)))
#              
# #             exp_model_param = update_model(model, learning_rate, regularization_rate)
#             
#             if record_params:
#                 append_gradient_list(exp_gradient_list_all_epochs, None, exp_para_list_all_epochs, model, batch_X, is_GPU, device)
# 
#             optimizer.step()
#             
# #             print('parameter comparison::')
# #             
# #             compute_model_para_diff(list(model.parameters()), exp_model_param)
#             
# #             update_and_zero_model_gradient(model,learning_rate)
#             
#     #         print("iteration::", count)
#     #         
#     #         print("loss::", loss)
#             
#             count += 1
#         
# #         data_train_loader.batch_sampler.increm_ids()
#            
# #             curr_batch_sampler.increm_ids()
#              
# #             print("loss::", loss)
#         
#     t2 = time.time()
#         
#     elapse_time += (t2 - t1)  
# 
#     print("training time is", elapse_time)
#     
#     return model, count, exp_para_list_all_epochs, exp_gradient_list_all_epochs


def generate_added_random_ids_all_epochs(dataset_train_len, X_to_add, mini_batch_num, random_ids_multi_super_iterations):
    
    prev_added_random_ids_multi_super_iteartion = None
    
    
    all_added_random_ids_list_all_samples = []
    
    for r in range(X_to_add.shape[0]):
        
        if prev_added_random_ids_multi_super_iteartion is None:
 
            added_random_ids_multi_super_iteration = get_sampling_each_iteration0(random_ids_multi_super_iterations, 1, mini_batch_num, dataset_train_len)
     
        else:
            
            added_random_ids_multi_super_iteration = get_sampling_each_iteration_union_prev(random_ids_multi_super_iterations, 1, mini_batch_num, prev_added_random_ids_multi_super_iteartion, r + dataset_train_len)
        
            
            
        prev_added_random_ids_multi_super_iteartion = added_random_ids_multi_super_iteration
        
        all_added_random_ids_list_all_samples.append(added_random_ids_multi_super_iteration)
        
    return all_added_random_ids_list_all_samples
        
        


def model_update_standard_lib_multi(all_added_random_ids_list_all_samples, origin_model, max_epoch, dataset_train, dim, model, random_ids_multi_super_iterations, batch_size, learning_rate_all_epochs, added_batch_size, X_to_add, Y_to_add, criterion, optimizer, para_list_all_epochs, gradient_list_all_epochs, is_GPU, device, record_params, regularization_coeff, mini_batch_num):
    
#     for epoch in range(num_epochs):
    loss = np.infty

    elapse_time = 0

    overhead = 0
    
    overhead2 = 0

    t1 = time.time()
    
    para = list(model.parameters())
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    exp_gradient_list_all_epochs = []
    
    exp_para_list_all_epochs = []
    
#     selected_rows_set = set(selected_rows.view(-1).tolist())
    
#     train = Variable(X)
#     labels = Variable(Y.view(-1))
#     labels = labels.type(torch.LongTensor)

#     for r in range(len(delta_data_ids)):


    

    all_res = []


    prev_added_random_ids_multi_super_iteartion = None
        
    for r in range(X_to_add.shape[0]):
        
        t5 = time.time()
        
        init_model(model, para_list_all_epochs[0])
        
        para = list(model.parameters())
        
        count = 0
        
        old_lr = -1

#         curr_X_to_add = X_to_add[0:r+1]
#         
#         curr_Y_to_add = Y_to_add[0:r+1]
        
        curr_exp_para_list_all_epochs = []
        
        curr_exp_grad_list_all_epochs = []
        
        added_random_ids_multi_super_iteration = all_added_random_ids_list_all_samples[r]


        for k in range(len(random_ids_multi_super_iterations)):
            
#             print("epoch ", k)
            
            random_ids = random_ids_multi_super_iterations[k]
            
            added_random_ids = added_random_ids_multi_super_iteration[k]
            
    #         for i in range(len(batch_X_list)):
    
    #         sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
    #         all_indexes = np.sort(np.searchsorted(random_ids.numpy(), delta_ids.numpy()))
            
    #         all_indexes = np.sort(sort_idx[np.searchsorted(random_ids.numpy(),selected_rows.numpy(),sorter = sort_idx)])
            
    #         all_indexes = np.sort(sort_idx[selected_rows])
    
            id_start = 0
            
            id_end = 0
    
    #         print('epoch::', k)
    
            j = 0
            
            to_add = True
    
            for i in range(0, dim[0], batch_size):
                
                end_id = i + batch_size
                
#                 added_end_id = j + added_batch_size
                curr_added_random_ids = added_random_ids[j]
                
                
                
                
                
                if end_id > dim[0]:
                    end_id = dim[0]
                
                
#                 if added_end_id > curr_X_to_add.shape[0]:
#                     added_end_id = curr_X_to_add.shape[0]
                
                
                if curr_added_random_ids.shape[0] <= 0:
                    to_add = False
                else:
                    to_add = True
    #             print(count)
                
                learning_rate = learning_rate_all_epochs[count]
                
                
                
                
    #             regularization_rate = list(optimizer.param_groups)[0]['weight_decay']
                
                if not learning_rate == old_lr:
                    update_learning_rate(optimizer, learning_rate)
                
                old_lr = learning_rate
                
    #             curr_rand_ids = random_ids[i:end_id]
                
    #             curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
    
    #             if all_indexes[-1] < end_id:
    #                 id_end = all_indexes.shape[0]
    #             else:
    #                 id_end = np.argmax(all_indexes >= end_id)
                    
                curr_rand_ids = random_ids[i:end_id]
#                 t5 = time.time()
                
                grad_dual = 0
                
                curr_to_add_size = 0
                
                
                init_model(model, para)
                
                if to_add:
#                     curr_to_add_rand_ids = added_random_ids[j:added_end_id]
                
                    curr_to_add_size = curr_added_random_ids.shape[0]
                
    #             curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
                
    #             curr_matched_ids,_ = torch.sort(curr_matched_ids)
    #             while 1:
    #                 if id_end >= all_indexes.shape[0] or all_indexes[id_end] >= end_id:
    #                     break
    #                 
    #                 id_end = id_end + 1
                
    #             curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
                
    #             curr_matched_ids_size = curr_matched_ids.shape[0]
    
    
    
    #             curr_matched_ids,_ = torch.sort(curr_matched_ids)
    
    #             print(curr_matched_ids)
                
                
    #             if curr_matched_ids_size <= 0:
    #                 continue
                    if is_GPU:
                        compute_derivative_one_more_step(model, X_to_add[curr_added_random_ids].to(device), Y_to_add[curr_added_random_ids].to(device), criterion, optimizer)
                
                    else:
                        compute_derivative_one_more_step(model, X_to_add[curr_added_random_ids], Y_to_add[curr_added_random_ids], criterion, optimizer)
                
                    grad_dual = get_all_vectorized_parameters1(model.get_all_gradient())
                
                
    #                 batch_X = torch.cat([dataset_train.data[curr_rand_ids], ], dim = 0)
    #                 
    #                 batch_Y = torch.cat([dataset_train.labels[curr_rand_ids], ], dim = 0)
                    
    #         outputs = model(train)
            
    #         loss = error(outputs, labels)
    #             else:
                    
                batch_X = dataset_train.data[curr_rand_ids]
                
                batch_Y = dataset_train.labels[curr_rand_ids]
                
                if is_GPU:
                    batch_X = batch_X.to(device)
                    
                    batch_Y = batch_Y.to(device)
                
                compute_derivative_one_more_step(model, batch_X, batch_Y, criterion, optimizer)
    
                grad_curr = get_all_vectorized_parameters1(model.get_all_gradient())
                
                grad_full = (grad_curr*batch_X.shape[0] + grad_dual*curr_to_add_size)/(batch_X.shape[0] + curr_to_add_size)
                
                if record_params:
    #                 append_gradient_list(exp_gradient_list_all_epochs, None, exp_para_list_all_epochs, model, batch_X, is_GPU, device)
                    append_gradient_list2(curr_exp_grad_list_all_epochs, curr_exp_para_list_all_epochs, para, grad_full, full_shape_list, shape_list, is_GPU, device)
    
#                     curr_exp_para_list_all_epochs.append(para)
#                     
#                     curr_exp_grad_list_all_epochs.append(get_devectorized_parameters(grad_full, full_shape_list, shape_list))
                
                para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*get_all_vectorized_parameters1(para) - learning_rate*grad_full, full_shape_list, shape_list)
                
    #             if count == 1:
    #             
    #                 torch.save(batch_X, git_ignore_folder + 'tmp_batch_x')
    #                  
    #                 torch.save(batch_Y, git_ignore_folder + 'tmp_batch_y')
    #                  
    #                 torch.save(X_to_add[curr_to_add_rand_ids], git_ignore_folder + 'tmp_added_x')
    #                  
    #                 torch.save(Y_to_add[curr_to_add_rand_ids], git_ignore_folder + 'tmp_added_y')
    #                  
    #                 torch.save(grad_curr, git_ignore_folder + 'tmp_grad_remaining')
    #                  
    #                 torch.save(grad_dual, git_ignore_folder + 'tmp_grad_dual')
    #                  
    #                 print("here")
                
                
                
    #             t6 = time.time()
    #             
    #             overhead2 += (t6 - t5)
    #             
    #             t3 = time.time()
    #             
    #             optimizer.zero_grad()
    # 
    # #             batch_X = dataset_train.data.data[curr_matched_ids]
    # #             
    # #             batch_Y = dataset_train.data.targets[curr_matched_ids]
    #             
    #             output = model(batch_X)
    #             
    # #             print(output[0])
    # #             
    # #             print(torch.sort(items[2])[0])
    #     
    #             loss = criterion(output, batch_Y)
    #             
    #             loss.backward()
    #             
    #             t4 = time.time()
    #             
    #             overhead += (t4 - t3)
    #             print('parameter difference::')
    #                 
    #             compute_model_para_diff(para_list_all_epochs[count], list(model.parameters()))
    #                 
    #             print('gradient difference::')
    #                 
    #             compute_model_para_diff(gradient_list_all_epochs[count], list(get_model_gradient(model)))
    #             loss = compute_loss(model, error, batch_X, batch_Y, beta)
    #         
    #             loss.backward()
                
    #             optimizer.step()
    #             update_and_zero_model_gradient(model,learning_rate)
                
        #         print("iteration::", count)
        #         
        #         print("loss::", loss)
                
                count += 1
                
                j += 1
                
                
                del batch_X, batch_Y
        
        t6 = time.time()
        
        overhead2 += (t6 - t5)
        
        if r % 10 == 0:
            print("Num of deletion:: %d, running time baseline::%f" %(r, overhead2))
        
        
        if record_params:
            all_res.append(get_all_vectorized_parameters1(para).clone())
            exp_para_list_all_epochs.append(curr_exp_para_list_all_epochs)
        
            exp_gradient_list_all_epochs.append(curr_exp_grad_list_all_epochs)
            
#         compute_model_para_diff(para, list(origin_model.parameters()))
            
        
        
        
            
#             print("loss::", loss)
    init_model(model, para)
    
    
    t2 = time.time()
        
    elapse_time += (t2 - t1)  

    print("training time is", elapse_time)
    
    print("overhead::", overhead)
    
    print("overhead2::", overhead2)
    
    return model, count, exp_para_list_all_epochs, exp_gradient_list_all_epochs, all_res


def model_update_standard_lib_multi0(all_added_random_ids_list_all_samples, origin_model, max_epoch, dataset_train, dim, model, random_ids_multi_super_iterations, batch_size, learning_rate_all_epochs, added_batch_size, X_to_add, Y_to_add, criterion, optimizer, para_list_all_epochs, gradient_list_all_epochs, is_GPU, device, record_params, regularization_coeff, mini_batch_num, origin_train_data_size):
    
#     for epoch in range(num_epochs):
    loss = np.infty

    elapse_time = 0

    overhead = 0
    
    overhead2 = 0

    t1 = time.time()
    
    para = list(model.parameters())
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    exp_gradient_list_all_epochs = []
    
    exp_para_list_all_epochs = []
    
#     selected_rows_set = set(selected_rows.view(-1).tolist())
    
#     train = Variable(X)
#     labels = Variable(Y.view(-1))
#     labels = labels.type(torch.LongTensor)

#     for r in range(len(delta_data_ids)):


    

    all_res = []


    prev_added_random_ids_multi_super_iteartion = None
        
    for r in range(X_to_add.shape[0]):
        
        t5 = time.time()
        
        init_model(model, para_list_all_epochs[0])
        
        para = list(model.parameters())
        
        count = 0
        
        old_lr = -1

#         curr_X_to_add = X_to_add[0:r+1]
#         
#         curr_Y_to_add = Y_to_add[0:r+1]
        
        curr_exp_para_list_all_epochs = []
        
        curr_exp_grad_list_all_epochs = []
        
        added_random_ids_multi_super_iteration = all_added_random_ids_list_all_samples[r]


        for k in range(len(random_ids_multi_super_iterations)):
            
#             print("epoch ", k)
            
            random_ids = random_ids_multi_super_iterations[k]
            
            added_random_ids = added_random_ids_multi_super_iteration[k]
            
            
    #         for i in range(len(batch_X_list)):
    
    #         sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
    #         all_indexes = np.sort(np.searchsorted(random_ids.numpy(), delta_ids.numpy()))
            
    #         all_indexes = np.sort(sort_idx[np.searchsorted(random_ids.numpy(),selected_rows.numpy(),sorter = sort_idx)])
            
    #         all_indexes = np.sort(sort_idx[selected_rows])
    
            id_start = 0
            
            id_end = 0
    
    #         print('epoch::', k)
    
            j = 0
            
            to_add = True
    
            for i in range(0, dim[0], batch_size):
                
                end_id = i + batch_size
                
#                 added_end_id = j + added_batch_size
                curr_added_random_ids = added_random_ids[j]
                
                
                curr_rand_ids = random_ids[i:end_id]
                
                if curr_added_random_ids.shape[0] > 0:
                    full_random_ids = torch.cat([curr_rand_ids, curr_added_random_ids], 0)
                else:
                    full_random_ids = curr_rand_ids
                
                
                
                if end_id > dim[0]:
                    end_id = dim[0]
                
                
#                 if added_end_id > curr_X_to_add.shape[0]:
#                     added_end_id = curr_X_to_add.shape[0]
                
                
#                 if curr_added_random_ids.shape[0] <= 0:
#                     to_add = False
#                 else:
#                     to_add = True
    #             print(count)
                
                learning_rate = learning_rate_all_epochs[count]
                
                
                
                
    #             regularization_rate = list(optimizer.param_groups)[0]['weight_decay']
                
                if not learning_rate == old_lr:
                    update_learning_rate(optimizer, learning_rate)
                
                old_lr = learning_rate
                
    #             curr_rand_ids = random_ids[i:end_id]
                
    #             curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
    
    #             if all_indexes[-1] < end_id:
    #                 id_end = all_indexes.shape[0]
    #             else:
    #                 id_end = np.argmax(all_indexes >= end_id)
                    
                
#                 t5 = time.time()
                
                grad_dual = 0
                
                curr_to_add_size = 0
                
                
                init_model(model, para)
                
#                 if to_add:
# #                     curr_to_add_rand_ids = added_random_ids[j:added_end_id]
#                 
#                     curr_to_add_size = curr_added_random_ids.shape[0]
#                 
#     #             curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
#                 
#     #             curr_matched_ids,_ = torch.sort(curr_matched_ids)
#     #             while 1:
#     #                 if id_end >= all_indexes.shape[0] or all_indexes[id_end] >= end_id:
#     #                     break
#     #                 
#     #                 id_end = id_end + 1
#                 
#     #             curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
#                 
#     #             curr_matched_ids_size = curr_matched_ids.shape[0]
#     
#     
#     
#     #             curr_matched_ids,_ = torch.sort(curr_matched_ids)
#     
#     #             print(curr_matched_ids)
#                 
#                 
#     #             if curr_matched_ids_size <= 0:
#     #                 continue
#                     if is_GPU:
#                         compute_derivative_one_more_step(model, X_to_add[curr_added_random_ids].to(device), Y_to_add[curr_added_random_ids].to(device), criterion, optimizer)
#                 
#                     else:
#                         compute_derivative_one_more_step(model, X_to_add[curr_added_random_ids], Y_to_add[curr_added_random_ids], criterion, optimizer)
#                 
#                     grad_dual = get_all_vectorized_parameters1(model.get_all_gradient())
                
                
    #                 batch_X = torch.cat([dataset_train.data[curr_rand_ids], ], dim = 0)
    #                 
    #                 batch_Y = torch.cat([dataset_train.labels[curr_rand_ids], ], dim = 0)
                    
    #         outputs = model(train)
            
    #         loss = error(outputs, labels)
    #             else:
                    
                batch_X = dataset_train.data[full_random_ids]
                
                batch_Y = dataset_train.labels[full_random_ids]
                
                if is_GPU:
                    batch_X = batch_X.to(device)
                    
                    batch_Y = batch_Y.to(device)
                
                compute_derivative_one_more_step(model, batch_X, batch_Y, criterion, optimizer)
    
                grad_full = get_all_vectorized_parameters1(model.get_all_gradient())
                
#                 grad_full = (grad_curr*batch_X.shape[0] + grad_dual*curr_to_add_size)/(batch_X.shape[0] + curr_to_add_size)
                
                if record_params:
    #                 append_gradient_list(exp_gradient_list_all_epochs, None, exp_para_list_all_epochs, model, batch_X, is_GPU, device)
                    append_gradient_list2(curr_exp_grad_list_all_epochs, curr_exp_para_list_all_epochs, para, grad_full, full_shape_list, shape_list, is_GPU, device)
    
#                     curr_exp_para_list_all_epochs.append(para)
#                     
#                     curr_exp_grad_list_all_epochs.append(get_devectorized_parameters(grad_full, full_shape_list, shape_list))
                
                para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*get_all_vectorized_parameters1(para) - learning_rate*grad_full, full_shape_list, shape_list)
                
    #             if count == 1:
    #             
    #                 torch.save(batch_X, git_ignore_folder + 'tmp_batch_x')
    #                  
    #                 torch.save(batch_Y, git_ignore_folder + 'tmp_batch_y')
    #                  
    #                 torch.save(X_to_add[curr_to_add_rand_ids], git_ignore_folder + 'tmp_added_x')
    #                  
    #                 torch.save(Y_to_add[curr_to_add_rand_ids], git_ignore_folder + 'tmp_added_y')
    #                  
    #                 torch.save(grad_curr, git_ignore_folder + 'tmp_grad_remaining')
    #                  
    #                 torch.save(grad_dual, git_ignore_folder + 'tmp_grad_dual')
    #                  
    #                 print("here")
                
                
                
    #             t6 = time.time()
    #             
    #             overhead2 += (t6 - t5)
    #             
    #             t3 = time.time()
    #             
    #             optimizer.zero_grad()
    # 
    # #             batch_X = dataset_train.data.data[curr_matched_ids]
    # #             
    # #             batch_Y = dataset_train.data.targets[curr_matched_ids]
    #             
    #             output = model(batch_X)
    #             
    # #             print(output[0])
    # #             
    # #             print(torch.sort(items[2])[0])
    #     
    #             loss = criterion(output, batch_Y)
    #             
    #             loss.backward()
    #             
    #             t4 = time.time()
    #             
    #             overhead += (t4 - t3)
    #             print('parameter difference::')
    #                 
    #             compute_model_para_diff(para_list_all_epochs[count], list(model.parameters()))
    #                 
    #             print('gradient difference::')
    #                 
    #             compute_model_para_diff(gradient_list_all_epochs[count], list(get_model_gradient(model)))
    #             loss = compute_loss(model, error, batch_X, batch_Y, beta)
    #         
    #             loss.backward()
                
    #             optimizer.step()
    #             update_and_zero_model_gradient(model,learning_rate)
                
        #         print("iteration::", count)
        #         
        #         print("loss::", loss)
                
                count += 1
                
                j += 1
                
                
                del batch_X, batch_Y
        
        t6 = time.time()
        
        overhead2 += (t6 - t5)
        
        if r % 10 == 0:
            print("Num of deletion:: %d, running time baseline::%f" %(r, overhead2))
        
        
        if record_params:
            all_res.append(get_all_vectorized_parameters1(para).clone())
            exp_para_list_all_epochs.append(curr_exp_para_list_all_epochs)
        
            exp_gradient_list_all_epochs.append(curr_exp_grad_list_all_epochs)
            
#         compute_model_para_diff(para, list(origin_model.parameters()))
            
        
        
        
            
#             print("loss::", loss)
    init_model(model, para)
    
    
    t2 = time.time()
        
    elapse_time += (t2 - t1)  

    print("training time is", elapse_time)
    
    print("overhead::", overhead)
    
    print("overhead2::", overhead2)
    
    return model, count, exp_para_list_all_epochs, exp_gradient_list_all_epochs, all_res



def model_update_standard_lib_multi2(origin_model, max_epoch, dataset_train, dim, model, random_ids_multi_super_iterations, batch_size, learning_rate_all_epochs, added_random_ids_multi_super_iteration, added_batch_size, X_to_add, Y_to_add, criterion, optimizer, para_list_all_epochs, gradient_list_all_epochs, is_GPU, device, record_params, regularization_coeff, mini_batch_num, all_added_random_ids_list_all_samples):
    
#     for epoch in range(num_epochs):
    loss = np.infty

    elapse_time = 0

    overhead = 0
    
    overhead2 = 0

    t1 = time.time()
    
    para = list(model.parameters())
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    exp_gradient_list_all_epochs = []
    
    exp_para_list_all_epochs = []
    
#     selected_rows_set = set(selected_rows.view(-1).tolist())
    
#     train = Variable(X)
#     labels = Variable(Y.view(-1))
#     labels = labels.type(torch.LongTensor)

#     for r in range(len(delta_data_ids)):


#     all_added_random_ids_list_all_samples = []

    all_res = []

        
    for r in range(X_to_add.shape[0]):
        
        init_model(model, para_list_all_epochs[0])
        
        para = list(model.parameters())
        
        count = 0
        
        old_lr = -1

        curr_X_to_add = X_to_add[0:r+1]
        
        curr_Y_to_add = Y_to_add[0:r+1]
        
        curr_exp_para_list_all_epochs = []
        
        curr_exp_grad_list_all_epochs = []

#         added_random_ids_multi_super_iteration = get_sampling_each_iteration(random_ids_multi_super_iterations, r+1, mini_batch_num)
        
        
        added_random_ids_multi_super_iteration = all_added_random_ids_list_all_samples[r]
        
#         all_added_random_ids_list_all_samples.append(added_random_ids_multi_super_iteration)

        for k in range(len(random_ids_multi_super_iterations)):
            
            print("epoch ", k)
            
            random_ids = random_ids_multi_super_iterations[k]
            
            added_random_ids = added_random_ids_multi_super_iteration[k]
            
    #         for i in range(len(batch_X_list)):
    
    #         sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
    #         all_indexes = np.sort(np.searchsorted(random_ids.numpy(), delta_ids.numpy()))
            
    #         all_indexes = np.sort(sort_idx[np.searchsorted(random_ids.numpy(),selected_rows.numpy(),sorter = sort_idx)])
            
    #         all_indexes = np.sort(sort_idx[selected_rows])
    
            id_start = 0
            
            id_end = 0
    
    #         print('epoch::', k)
    
            j = 0
            
            to_add = True
    
            for i in range(0, dim[0], batch_size):
                
                end_id = i + batch_size
                
#                 added_end_id = j + added_batch_size
                curr_added_random_ids = added_random_ids[j]
                
                
                if r > 0:
                    if count == 15: 
                
                        print("here")
                
                
                if end_id > dim[0]:
                    end_id = dim[0]
                
                
#                 if added_end_id > curr_X_to_add.shape[0]:
#                     added_end_id = curr_X_to_add.shape[0]
                
                
                if curr_added_random_ids.shape[0] <= 0:
                    to_add = False
                else:
                    to_add = True
    #             print(count)
                
                learning_rate = learning_rate_all_epochs[count]
                
                
                
                
    #             regularization_rate = list(optimizer.param_groups)[0]['weight_decay']
                
                if not learning_rate == old_lr:
                    update_learning_rate(optimizer, learning_rate)
                
                old_lr = learning_rate
                
    #             curr_rand_ids = random_ids[i:end_id]
                
    #             curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
    
    #             if all_indexes[-1] < end_id:
    #                 id_end = all_indexes.shape[0]
    #             else:
    #                 id_end = np.argmax(all_indexes >= end_id)
                    
                curr_rand_ids = random_ids[i:end_id]
                t5 = time.time()
                
                grad_dual = 0
                
                curr_to_add_size = 0
                
                
                init_model(model, para)
                
                if to_add:
#                     curr_to_add_rand_ids = added_random_ids[j:added_end_id]
                
                    curr_to_add_size = curr_added_random_ids.shape[0]
                
    #             curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
                
    #             curr_matched_ids,_ = torch.sort(curr_matched_ids)
    #             while 1:
    #                 if id_end >= all_indexes.shape[0] or all_indexes[id_end] >= end_id:
    #                     break
    #                 
    #                 id_end = id_end + 1
                
    #             curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
                
    #             curr_matched_ids_size = curr_matched_ids.shape[0]
    
    
    
    #             curr_matched_ids,_ = torch.sort(curr_matched_ids)
    
    #             print(curr_matched_ids)
                
                
    #             if curr_matched_ids_size <= 0:
    #                 continue
                    if is_GPU:
                        compute_derivative_one_more_step(model, X_to_add[curr_added_random_ids].to(device), Y_to_add[curr_added_random_ids].to(device), criterion, optimizer)
                
                    else:
                        compute_derivative_one_more_step(model, X_to_add[curr_added_random_ids], Y_to_add[curr_added_random_ids], criterion, optimizer)
                
                    grad_dual = get_all_vectorized_parameters1(model.get_all_gradient())
                
                
    #                 batch_X = torch.cat([dataset_train.data[curr_rand_ids], ], dim = 0)
    #                 
    #                 batch_Y = torch.cat([dataset_train.labels[curr_rand_ids], ], dim = 0)
                    
    #         outputs = model(train)
            
    #         loss = error(outputs, labels)
    #             else:
                    
                batch_X = dataset_train.data[curr_rand_ids]
                
                batch_Y = dataset_train.labels[curr_rand_ids]
                
                if is_GPU:
                    batch_X = batch_X.to(device)
                    
                    batch_Y = batch_Y.to(device)
                
                compute_derivative_one_more_step(model, batch_X, batch_Y, criterion, optimizer)
    
                grad_curr = get_all_vectorized_parameters1(model.get_all_gradient())
                
                grad_full = (grad_curr*batch_X.shape[0] + grad_dual*curr_to_add_size)/(batch_X.shape[0] + curr_to_add_size)
                
                if record_params:
    #                 append_gradient_list(exp_gradient_list_all_epochs, None, exp_para_list_all_epochs, model, batch_X, is_GPU, device)
                    append_gradient_list2(curr_exp_grad_list_all_epochs, curr_exp_para_list_all_epochs, para, grad_full, full_shape_list, shape_list, is_GPU, device)
    
#                     curr_exp_para_list_all_epochs.append(para)
#                     
#                     curr_exp_grad_list_all_epochs.append(get_devectorized_parameters(grad_full, full_shape_list, shape_list))
                
                para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*get_all_vectorized_parameters1(para) - learning_rate*grad_full, full_shape_list, shape_list)
                
    #             if count == 1:
    #             
    #                 torch.save(batch_X, git_ignore_folder + 'tmp_batch_x')
    #                  
    #                 torch.save(batch_Y, git_ignore_folder + 'tmp_batch_y')
    #                  
    #                 torch.save(X_to_add[curr_to_add_rand_ids], git_ignore_folder + 'tmp_added_x')
    #                  
    #                 torch.save(Y_to_add[curr_to_add_rand_ids], git_ignore_folder + 'tmp_added_y')
    #                  
    #                 torch.save(grad_curr, git_ignore_folder + 'tmp_grad_remaining')
    #                  
    #                 torch.save(grad_dual, git_ignore_folder + 'tmp_grad_dual')
    #                  
    #                 print("here")
                
                
                
    #             t6 = time.time()
    #             
    #             overhead2 += (t6 - t5)
    #             
    #             t3 = time.time()
    #             
    #             optimizer.zero_grad()
    # 
    # #             batch_X = dataset_train.data.data[curr_matched_ids]
    # #             
    # #             batch_Y = dataset_train.data.targets[curr_matched_ids]
    #             
    #             output = model(batch_X)
    #             
    # #             print(output[0])
    # #             
    # #             print(torch.sort(items[2])[0])
    #     
    #             loss = criterion(output, batch_Y)
    #             
    #             loss.backward()
    #             
    #             t4 = time.time()
    #             
    #             overhead += (t4 - t3)
    #             print('parameter difference::')
    #                 
    #             compute_model_para_diff(para_list_all_epochs[count], list(model.parameters()))
    #                 
    #             print('gradient difference::')
    #                 
    #             compute_model_para_diff(gradient_list_all_epochs[count], list(get_model_gradient(model)))
    #             loss = compute_loss(model, error, batch_X, batch_Y, beta)
    #         
    #             loss.backward()
                
    #             optimizer.step()
    #             update_and_zero_model_gradient(model,learning_rate)
                
        #         print("iteration::", count)
        #         
        #         print("loss::", loss)
                
                count += 1
                
                j += 1
        
        
        all_res.append(get_all_vectorized_parameters1(para).clone())
            
        compute_model_para_diff(para, list(origin_model.parameters()))
            
        exp_para_list_all_epochs.append(curr_exp_para_list_all_epochs)
        
        exp_gradient_list_all_epochs.append(curr_exp_grad_list_all_epochs)
        
        
            
#             print("loss::", loss)
    init_model(model, para)
    
    
    t2 = time.time()
        
    elapse_time += (t2 - t1)  

    print("training time is", elapse_time)
    
    print("overhead::", overhead)
    
    print("overhead2::", overhead2)
    
    return model, count, exp_para_list_all_epochs, exp_gradient_list_all_epochs, all_added_random_ids_list_all_samples, all_res
 


def model_update_standard_lib_skipnet(num_epochs, dataset_train, model, random_ids_multi_super_iterations, selected_rows, batch_size, learning_rate_all_epochs, criterion, optimizer, para_list_all_epochs, gradient_list_all_epochs, is_GPU, device, record_params, all_ids_list_all_epochs):
    count = 0
#     for epoch in range(num_epochs):
#     loss = np.infty

    elapse_time = 0

    t1 = time.time()
    
    
    exp_gradient_list_all_epochs = []
      
    exp_para_list_all_epochs = []
    
    selected_rows_set = set(selected_rows.view(-1).tolist())
    
#     train = Variable(X)
#     labels = Variable(Y.view(-1))
#     labels = labels.type(torch.LongTensor)

    old_lr = -1
    
#     data_train_loader.shuffle = False


    random_list_all_epochs = []
    
#     dropout_ids_ids_list =[]

    for k in range(len(random_ids_multi_super_iterations)):
        
        random_ids = random_ids_multi_super_iterations[k]
        
#         id_start = 0
#         
#         id_end = 0
        
        random_ids_list = []
        
        for i in range(0, len(dataset_train), batch_size):
            
            end_id = i + batch_size
            
            if end_id > len(dataset_train):
                end_id = len(dataset_train)
        
            
            curr_rand_ids = random_ids[i:end_id]
            
            curr_matched_ids = get_subset_data_per_epoch_skipnet(curr_rand_ids, selected_rows_set, all_ids_list_all_epochs[count])
            
            random_ids_list.append(list(curr_matched_ids.numpy()))
            
#             dropout_ids_ids_list.append(ids_list)
            
            count += 1
        random_list_all_epochs.append(random_ids_list)    
#             curr_matched_ids_size = curr_matched_ids.shape[0]

            
            
    curr_batch_sampler = Batch_sampler(random_list_all_epochs)

#     data_train_loader.batch_sampler = curr_batch_sampler
#     data_train_loader2 = DataLoader(dataset_train, shuffle = True)
    
    data_train_loader = DataLoader(dataset_train, num_workers=0, shuffle=False, batch_sampler=curr_batch_sampler)
    
    
    count = 0
    
    for k in range(len(random_ids_multi_super_iterations)):   
        
        print("epoch::", k)
        
        for i, items in enumerate(data_train_loader):
            
#             random.seed(random_seed)
#             os.environ['PYTHONHASHSEED'] = str(random_seed)
#             np.random.seed(random_seed)
#             torch.manual_seed(random_seed)
#             torch.cuda.manual_seed(random_seed)
#             torch.cuda.manual_seed_all(random_seed) # if you are using multi-GPU.
#             torch.backends.cudnn.benchmark = False
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.enabled = False
            
            curr_matched_ids_size = items[2].shape[0]
            
#             print(items[2])
#             
#             print(random_ids_list[i])
            
#             print(torch.max(items[2] - torch.tensor(random_ids_list[i])))
#             
#             print(torch.min(items[2] - torch.tensor(random_ids_list[i])))
            if not is_GPU:
                batch_X = items[0]
                
                batch_Y = items[1]
            else:
                batch_X = items[0].to(device)
                
                batch_Y = items[1].to(device)
            
#             exp_ids = torch.sort(torch.tensor(random_list_all_epochs[k][i]))[0]
#              
#             curr_ids = torch.sort(torch.tensor(items[2]))[0]
#              
#             print("compare_len::", len(exp_ids) - len(curr_ids))
#              
#             if len(exp_ids) - len(curr_ids) == 0:
#                 print(torch.max(exp_ids - curr_ids))
#                 print(torch.min(exp_ids - curr_ids))
            
            learning_rate = learning_rate_all_epochs[count]
            
#             regularization_rate = list(optimizer.param_groups)[0]['weight_decay']
            
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate
                
            
            if curr_matched_ids_size <= 0:
                continue
            
            if curr_matched_ids_size < batch_size:
 
                print("here!!")
            
#             batch_X = X[curr_matched_ids]
#             
#             batch_Y = Y[curr_matched_ids]

            optimizer.zero_grad()

#             batch_X = dataset_train.data.data[curr_matched_ids]
#             
#             batch_Y = dataset_train.data.targets[curr_matched_ids]
            
#             ids_list = ids_list_all_epochs[count]
# 
#             ids_list2 = ids_list2_all_epochs[count]
            
            all_ids_list = all_ids_list_all_epochs[count]
            
#             curr_dropout_ids_ids_list = dropout_ids_ids_list[count]
            
            sorted_ids, sorted_id_ids = torch.sort(items[2])
            
#             print(sorted_ids)
            
#             print(torch.sort(items[2])[0])
#             print(count)
#             if count >= 31:
#                 print("here")
            
            need_load = False

            if k == 1 and i == 0:
                need_load = True
            
            output = model.forward_with_known_dropout(batch_X[sorted_id_ids], torch.tensor(list(range(batch_size))), all_ids_list, need_load)
            
#             print(output[0])
#             
#             print(torch.sort(items[2])[0])
    
            loss = criterion(output, batch_Y[sorted_id_ids])
            
            loss.backward()
            
            
            
#             exp_output = torch.load(git_ignore_folder + 'tmp_output')
#     
#             exp_batch_X = torch.load(git_ignore_folder + 'tmp_batch_X')
#     
#             exp_batch_Y = torch.load(git_ignore_folder + 'tmp_batch_Y')
#             
#             
#             exp_all_ids_list = torch.load(git_ignore_folder + 'tmp_ids_list')
            
            
            
            
            print('parameter difference::')
                  
            compute_model_para_diff(para_list_all_epochs[count], list(model.parameters()))
                  
            print('gradient difference::')
                  
            compute_model_para_diff(gradient_list_all_epochs[count], list(get_model_gradient(model)))
             
#             exp_model_param = update_model(model, learning_rate, regularization_rate)
            
            if record_params:
                append_gradient_list(exp_gradient_list_all_epochs, None, exp_para_list_all_epochs, model, batch_X, is_GPU, device)

            optimizer.step()
            
#             print('parameter comparison::')
#             
#             compute_model_para_diff(list(model.parameters()), exp_model_param)
            
#             update_and_zero_model_gradient(model,learning_rate)
            
    #         print("iteration::", count)
    #         
    #         print("loss::", loss)
            
            count += 1
        
        data_train_loader.batch_sampler.increm_ids()
           
#             curr_batch_sampler.increm_ids()
             
#             print("loss::", loss)
        
    t2 = time.time()
        
    elapse_time += (t2 - t1)  

    print("training time is", elapse_time)
    
    return model, count, exp_para_list_all_epochs, exp_gradient_list_all_epochs


def quantize_model_param(model, epsilon):
    
    for param in model.parameters():
        
        quantized_param = quantize_vectors_no_random(param, epsilon)
        
        param.data.copy_(quantized_param)


def quantize_model_param2(params, epsilon):
    
    para_list = []
    
    for param in params:
        
        quantized_param = quantize_vectors_no_random(param, epsilon)
        
        para_list.append(quantized_param)
        
    return para_list


def model_update_standard_lib_quantize(num_epochs, dataset_train, model, random_ids_multi_super_iterations, selected_rows, batch_size, learning_rate_all_epochs, criterion, optimizer, para_list_all_epochs, gradient_list_all_epochs, is_GPU, device, record_params, epsilon):
    count = 0
#     for epoch in range(num_epochs):
#     loss = np.infty

    elapse_time = 0

    t1 = time.time()
    
    
    exp_gradient_list_all_epochs = []
      
    exp_para_list_all_epochs = []
    
    selected_rows_set = set(selected_rows.view(-1).tolist())
    
#     train = Variable(X)
#     labels = Variable(Y.view(-1))
#     labels = labels.type(torch.LongTensor)

    old_lr = -1
    
#     data_train_loader.shuffle = False


    random_list_all_epochs = []

    for k in range(len(random_ids_multi_super_iterations)):
        
        random_ids = random_ids_multi_super_iterations[k]
        
#         id_start = 0
#         
#         id_end = 0
        
        random_ids_list = []
        
        for i in range(0, len(dataset_train), batch_size):
            
            end_id = i + batch_size
            
            if end_id > len(dataset_train):
                end_id = len(dataset_train)
        
            
            curr_rand_ids = random_ids[i:end_id]
            
            curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
            
            random_ids_list.append(list(curr_matched_ids.numpy()))
        
        random_list_all_epochs.append(random_ids_list)    
#             curr_matched_ids_size = curr_matched_ids.shape[0]

            
            
    curr_batch_sampler = Batch_sampler(random_list_all_epochs)

#     data_train_loader.batch_sampler = curr_batch_sampler
    
    
    data_train_loader = DataLoader(dataset_train, num_workers=0, shuffle=False, batch_sampler=curr_batch_sampler)
    
    for k in range(len(random_ids_multi_super_iterations)):   
        
        print("epoch::", k)
        
        for i, items in enumerate(data_train_loader):
            
            curr_matched_ids_size = items[2].shape[0]
            
#             print(items[2])
#             
#             print(random_ids_list[i])
            
#             print(torch.max(items[2] - torch.tensor(random_ids_list[i])))
#             
#             print(torch.min(items[2] - torch.tensor(random_ids_list[i])))
            if not is_GPU:
                batch_X = items[0]
                
                batch_Y = items[1]
            else:
                batch_X = items[0].to(device)
                
                batch_Y = items[1].to(device)
            
#             exp_ids = torch.sort(torch.tensor(random_list_all_epochs[k][i]))[0]
#              
#             curr_ids = torch.sort(torch.tensor(items[2]))[0]
#              
#             print("compare_len::", len(exp_ids) - len(curr_ids))
#              
#             if len(exp_ids) - len(curr_ids) == 0:
#                 print(torch.max(exp_ids - curr_ids))
#                 print(torch.min(exp_ids - curr_ids))
            
            learning_rate = learning_rate_all_epochs[count]
            
#             regularization_rate = list(optimizer.param_groups)[0]['weight_decay']
            
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate
                
            
            if curr_matched_ids_size <= 0:
                continue
            
#             batch_X = X[curr_matched_ids]
#             
#             batch_Y = Y[curr_matched_ids]

            optimizer.zero_grad()

#             batch_X = dataset_train.data.data[curr_matched_ids]
#             
#             batch_Y = dataset_train.data.targets[curr_matched_ids]
            
            output = model(batch_X)
    
            loss = criterion(output, batch_Y)
            
            loss.backward()
            
            print('parameter difference::')
               
            compute_model_para_diff(para_list_all_epochs[count], list(model.parameters()))
               
            print('gradient difference::')
               
            compute_model_para_diff(gradient_list_all_epochs[count], list(get_model_gradient(model)))
            
#             exp_model_param = update_model(model, learning_rate, regularization_rate)
            
            if record_params:
                append_gradient_list(exp_gradient_list_all_epochs, None, exp_para_list_all_epochs, model, batch_X, is_GPU, device)

            optimizer.step()
            
            quantize_model_param(model, epsilon)
            
#             print('parameter comparison::')
#             
#             compute_model_para_diff(list(model.parameters()), exp_model_param)
            
#             update_and_zero_model_gradient(model,learning_rate)
            
    #         print("iteration::", count)
    #         
    #         print("loss::", loss)
            
            count += 1
        
        data_train_loader.batch_sampler.increm_ids()
           
#             curr_batch_sampler.increm_ids()
             
#             print("loss::", loss)
        
    t2 = time.time()
        
    elapse_time += (t2 - t1)  

    print("training time is", elapse_time)
    
    return model, count, exp_para_list_all_epochs, exp_gradient_list_all_epochs


def model_update_standard_lib_stochastic(batch_size, num_epochs, X, Y, learning_rate, error, model):
    count = 0
#     for epoch in range(num_epochs):
    loss = np.infty

    elapse_time = 0

    t1 = time.time()

#     train = Variable(X)

    labels = Variable(Y.view(-1))
    
    labels = labels.type(torch.LongTensor)
    
    while count < num_epochs:
        
        
        
        random_ids = torch.randperm(X.shape[0])
        
        curr_X = Variable(X[random_ids])
        
        curr_Y = Variable(labels[random_ids])
        
        
        for i in range(0, X.shape[0], batch_size):
            
            
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
        
                
                
            batch_x = curr_X[i: end_id]
            
            batch_y = curr_Y[i: end_id]    
            
        
            outputs = model(batch_x)
            
            loss = error(outputs, batch_y)
            
            loss.backward()
            
            update_and_zero_model_gradient(model,learning_rate)
            
    #         print("iteration::", count)
    #         
    #         print("loss::", loss)
            
        count += 1
         
#         print("loss::", loss)
        
    t2 = time.time()
        
    elapse_time += (t2 - t1)  

    print("training time is", elapse_time)
    
    return model, count

def model_update_standard_lib_with_recording_parameters(num_epochs, X, Y, learning_rate, error, model):
    count = 0
#     for epoch in range(num_epochs):
    loss = np.infty

    elapse_time = 0

    t1 = time.time()
    
    para_list = []
    
    
    para_list.append(model.get_all_parameters())

    while count < num_epochs:
    
        train = Variable(X)
        labels = Variable(Y.view(-1))
        
        outputs = model(train)
        
        labels = labels.type(torch.LongTensor)
        
        loss = error(outputs, labels)
        
        loss.backward()
        
        update_and_zero_model_gradient(model,learning_rate)
        
        
        para_list.append(model.get_all_parameters)
        
        print("iteration::", count)
        
        print("loss::", loss)
        
        count += 1
         
#         print("loss::", loss)
        
    t2 = time.time()
        
    elapse_time += (t2 - t1)  

    print("training time is", elapse_time)
    
    return model, count, para_list


# def compute_hessian(model, error):


def model_compute_loss(model, X, Y, error):
    train = Variable(X)
    outputs = model(train)
    labels = Variable(Y.view(-1))
    
    loss = error(outputs, labels)
    
    return loss


def compute_test_acc(model, test_X, test_Y):
    correct = 0
    total = 0
    # Predict test dataset
#                 for images, labels in test_loader:
#             for j in range(test_X.shape[0]):

#                     test = Variable(images.view(-1, 28*28))
    test = Variable(test_X)
    
    labels = test_Y.view(-1).type(torch.LongTensor)
    
    # Forward propagation
    outputs = model(test)
    
    # Get predictions from the maximum value
    predicted = torch.max(outputs.data, 1)[1]
    
    # Total number of labels
    total += len(labels)

    # Total correct predictions
    correct += (predicted == labels).sum()
    
    accuracy = 100 * correct / float(total)
    
    # store loss and iteration
#     loss_list.append(loss.data)
#     
#     iteration_list.append(count)
#     accuracy_list.append(accuracy)
#             if count % 500 == 0:
        # Print Loss
        
        
    print("accuracy:: {} %", format(accuracy))



def update_learning_rate(optim, learning_rate):
    for g in optim.param_groups:
        g['lr'] = learning_rate


# def test(net, data_test_loader, criterion, data_test_size, is_GPU, device):
#     net.eval()
#     total_correct = 0
#     avg_loss = 0.0
#     for i, items in enumerate(data_test_loader):
#         
#         if not is_GPU:
#             images, labels = items[0], items[1]
#         else:
#             images, labels = items[0].to(device), items[1].to(device)
#         output = net(images)
#         avg_loss += criterion(output, labels).sum()
#         pred = output.detach().max(1)[1]
#         
#         if len(labels.shape) > 1 and labels.shape[1] > 1:
#             labels = torch.nonzero(labels)[:,1]
#         
#         total_correct += pred.eq(labels.view_as(pred)).sum()
# 
#     avg_loss /= data_test_size
#     
#     net.train()
#     print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / data_test_size))


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


def test_skipnet(net, data_test_loader, criterion, data_test_size, is_GPU, device):
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, items in enumerate(data_test_loader):
        
        if not is_GPU:
            images, labels = items[0], items[1]
        else:
            images, labels = items[0].to(device), items[1].to(device)
        output,_ = net.forward(images, False)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            labels = torch.nonzero(labels)[:,1]
        
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= data_test_size
    
    
    net.train()
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / data_test_size))


def post_processing_gradien_para_list_all_epochs(para_list_all_epochs, grad_list_all_epochs):
    
#     num = 0
    
    _,_,total_shape_size = get_model_para_shape_list(para_list_all_epochs[0])
        
    
    
    para_list_all_epoch_tensor = torch.zeros([len(para_list_all_epochs), total_shape_size], dtype = torch.double)
    
    grad_list_all_epoch_tensor = torch.zeros([len(grad_list_all_epochs), total_shape_size], dtype = torch.double)
    
    for i in range(len(para_list_all_epochs)):
        
        para_list_all_epoch_tensor[i] = get_all_vectorized_parameters1(para_list_all_epochs[i])
        
        grad_list_all_epoch_tensor[i] = get_all_vectorized_parameters1(grad_list_all_epochs[i])
        
    
    
    
    return para_list_all_epoch_tensor, grad_list_all_epoch_tensor



def model_training(epoch, net, data_train_loader, data_test_loader, data_train_size, data_test_size, optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs):
#     global cur_batch_win
    net.train()
    
    gradient_list_all_epochs = []
    
    para_list_all_epochs = []
    
#     output_list_all_epochs = []
    
    learning_rate_all_epochs = []
    
    
    
    loss_list, batch_list = [], []
    
    t1 = time.time()
    
    
#     test(net, data_test_loader, criterion, data_test_size, is_GPU, device)

    for j in range(epoch):
        
        random_ids = torch.zeros([data_train_size], dtype = torch.long)
    
        k = 0
        
        learning_rate = lrs[j]
        update_learning_rate(optimizer, learning_rate)
        
#         item0 = data_train_loader.dataset.data[100]
    
        for i, items in enumerate(data_train_loader):
            
            
#             random.seed(random_seed)
#             os.environ['PYTHONHASHSEED'] = str(random_seed)
#             np.random.seed(random_seed)
#             torch.manual_seed(random_seed)
#             torch.cuda.manual_seed(random_seed)
#             torch.cuda.manual_seed_all(random_seed) # if you are using multi-GPU.
#             torch.backends.cudnn.benchmark = False
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.enabled = False
            
            
            if not is_GPU:
                images, labels, ids =  items[0], items[1], items[2]
            else:
                images, labels, ids =  items[0].to(device), items[1].to(device), items[2]
            
            end_id = k + batch_size
            
            if end_id > data_train_size:
                end_id = data_train_size
            
#             print(k, end_id)
            random_ids[k:end_id] = ids
            
            
            k = k + batch_size
            
            optimizer.zero_grad()
    
            output = net(images)
            
#             print(output[0])
#              
#             print(torch.sort(ids)[0])
    
            loss = criterion(output, labels)
    
    
            loss_list.append(loss.detach().cpu().item())
            batch_list.append(i+1)
    
            if i % 10 == 0:
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (j, i, loss.detach().cpu().item()))
            
#             if i % 10 == 0:
#                 lr_scheduler.step()
                 
    
            loss.backward()
    
            append_gradient_list(gradient_list_all_epochs, None, para_list_all_epochs, net, None, is_GPU, device)
    
            # Update Visualization
    #         if viz.check_connection():
    #             cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
    #                                      win=cur_batch_win, name='current_batch_loss',
    #                                      update=(None if cur_batch_win is None else 'replace'),
    #                                      opts=cur_batch_win_opts)
#             learning_rate = list(optimizer.param_groups)[0]['lr']
            
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
        
        test(net, data_test_loader, criterion, data_test_size, is_GPU, device)
        
    
    t2 = time.time()
    
    print("training_time::", (t2 - t1))
    
    return net, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs

# def model_training(num_epochs, origin_X, origin_Y, test_X, test_Y, init_learning_rate, decay, regularization_coeff, error, model, is_tracking_paras, batch_size, dim):
#     count = 0
# #     loss_list = []
# #     iteration_list = []
# #     accuracy_list = []
# #     for epoch in range(num_epochs):
#     loss = np.infty
# 
#     gradient_list_all_epochs = []
#     
#     para_list_all_epochs = []
#     
#     output_list_all_epochs = []
#     
#     learning_rate_all_epochs = []
#     
# #     para_lists_all_epochs = []
#     
# #     gradient_lists_all_epochs = []
#     
# #     para_lists_all_epochs.append(model.get_all_parameters)
#     
# #     gradient_lists_all_epochs.append(model.get_all_gradient())
# 
# #     construct_gradient_list(gradient_list, model)
# 
# 
#     elapse_time = 0
#     
#     b_time = 0
#     
#     f_time = 0
# 
#     iter = 0
# 
#     while loss > loss_threshold and count < num_epochs:
# #         for i, (images, labels) in enumerate(train_loader):
#             
# #         for i in range(X.shape[0]):
# 
#         random_ids = torch.randperm(dim[0])
# #         random_ids = torch.tensor(list(range(dim[0])))
#         
# #         print('rand_ids::', random_ids)
#         
#         X = origin_X[random_ids]
#         
#         Y = origin_Y[random_ids]
#         
#         random_ids_multi_super_iterations.append(random_ids)
#         
#         print("iteration::", count)
# 
#         for i in range(0,X.shape[0], batch_size):
#             
#             
# #             optimizer.zero_grad()
#     
#             end_id = i + batch_size
#             
#             if end_id >= X.shape[0]:
#                 end_id = X.shape[0]
#     
#     
# #             indices = permutation[i:i+batch_size]
#             batch_x, batch_y = X[i:end_id], Y[i:end_id]
# 
# 
#             t1 = time.time()
#         
#     #             train = Variable(images.view(-1, 28*28))
#             train = Variable(batch_x)
#             labels = Variable(batch_y.view(-1))
#             
#             # Clear gradients
#     #         optimizer.zero_grad()
#             
#             # Forward propagation
#             
#             t3 = time.time()
#             
#             outputs = model(train)
#             
#             t4 = time.time()
#             
#             f_time += (t4 - t3)
#             
#             # Calculate softmax and ross entropy loss
#             
#     #         print(outputs)
#     #         
#     #         print(labels)
#             
#             labels = labels.type(torch.LongTensor)
#             
#             loss = error(outputs, labels) + regularization_coeff*get_regularization_term(model.parameters())
#             
#     #         print("loss0::", loss)
#     # 
#     #         loss = loss_function2(outputs, labels, X.shape)
#             
#             # Calculating gradients
#     #         loss.backward(retain_graph = True, create_graph=True)
#             t3 = time.time()
#             
#             loss.backward()
#             
#             t4 = time.time()
#             
#             b_time += (t4 - t3)
#             
#     #         construct_gradient_list(gradient_list_all_epochs, res_list, model)
#     
#             learning_rate = init_learning_rate/(1+decay*iter)
#             
#             learning_rate_all_epochs.append(learning_rate)
#     
#             if is_tracking_paras:
#                 append_gradient_list(gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, model,train)
#             
#     #         gradient_lists_all_epochs.append(model.get_all_gradient())
#             
#             
#             update_and_zero_model_gradient(model,learning_rate)
#             
#             
#             if len(para_list_all_epochs) > 1:
#                 print("para_changes_first_layer::", torch.norm(para_list_all_epochs[-1][0] - para_list_all_epochs[-2][0]))
#                 print("para_changes_last_layer::", torch.norm(para_list_all_epochs[-1][-1] - para_list_all_epochs[-2][-1]))
# #             decompose_model_paras(para_list_all_epochs[-1], list(model.parameters()), gradient_list_all_epochs[-1], learning_rate)
#             
#             
#             t2 = time.time()
#             
#             elapse_time += (t2 - t1)
#             
#             print("loss::", loss)
#             
#             
#             iter += 1
#     #         print("training time::", (t2 - t1))
#             
# #         print_model_para(model)
#         
# #         para_lists_all_epochs.append(model.get_all_parameters)
#         
# #         print("epoch::", epoch)
#         
# #         print_model_para(model)
#         
#         
#         
#         # Update parameters
# #         optimizer.step()
#         
#         count += 1
#         
#         
#         
#         if count % 10 == 0:
#             compute_test_acc(model, test_X, test_Y)
#             # Calculate Accuracy         
# 
# #             print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data[0].item(), accuracy.item()))
# 
# 
#     print("training time is ", elapse_time)
#     
#     print("forward time is ", f_time)
#     
#     print("backaward time is ", b_time)
#     
# #     para_list_all_epochs.append(model.get_all_parameters)
#     
#     return model, gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs, count
# 
# 
# # def compute_hessian(model, error):


# def postprocessing_ids_list(random_list_all_epochs, ids_list_all_epochs, ids_list2_all_epochs):
#     
#     for i in range(len(random_list_all_epochs)):
#         
#         random_list = random_list_all_epochs[i]
#         
#         
#         
#         
#         for j in range(len(random_list)):
#             
#             curr_rand_ids = random_list[j]
   
   
   



# def model_training_test(random_ids_multi_super_iterations, epoch, net, data_train_loader, data_test_loader, data_train_size, data_test_size, optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs):
# #     global cur_batch_win
#     net.train()
#     
#     gradient_list_all_epochs = []
#     
#     para_list_all_epochs = []
#     
# #     output_list_all_epochs = []
#     
#     learning_rate_all_epochs = []
#     
#     
#     
#     loss_list, batch_list = [], []
#     
#     t1 = time.time()
#     
#     
# #     test(net, data_test_loader, criterion, data_test_size, is_GPU, device)
# 
#     for j in range(epoch):
#         
# #         random_ids = torch.zeros([data_train_size], dtype = torch.long)
#         random_ids = random_ids_multi_super_iterations[j]
#     
#         k = 0
#         
#         learning_rate = lrs[j]
#         update_learning_rate(optimizer, learning_rate)
#         
# #         item0 = data_train_loader.dataset.data[100]
#     
#         for i, items in enumerate(data_train_loader):
#             
#             
# #             random.seed(random_seed)
# #             os.environ['PYTHONHASHSEED'] = str(random_seed)
# #             np.random.seed(random_seed)
# #             torch.manual_seed(random_seed)
# #             torch.cuda.manual_seed(random_seed)
# #             torch.cuda.manual_seed_all(random_seed) # if you are using multi-GPU.
# #             torch.backends.cudnn.benchmark = False
# #             torch.backends.cudnn.deterministic = True
# #             torch.backends.cudnn.enabled = False
#             
#             
#             if not is_GPU:
#                 images, labels, ids =  items[0], items[1], items[2]
#             else:
#                 images, labels, ids =  items[0].to(device), items[1].to(device), items[2]
#             
#             end_id = k + batch_size
#             
#             if end_id > data_train_size:
#                 end_id = data_train_size
#             
# #             print(k, end_id)
#             random_ids[k:end_id] = ids
#             
#             
#             k = k + batch_size
#             
#             optimizer.zero_grad()
#     
#             output = net(images)
#             
# #             print(output[0])
# #              
# #             print(torch.sort(ids)[0])
#     
#             loss = criterion(output, labels)
#     
#     
#             loss_list.append(loss.detach().cpu().item())
#             batch_list.append(i+1)
#     
#             if i % 10 == 0:
#                 print('Train - Epoch %d, Batch: %d, Loss: %f' % (j, i, loss.detach().cpu().item()))
#             
# #             if i % 10 == 0:
# #                 lr_scheduler.step()
#                  
#     
#             loss.backward()
#     
#             append_gradient_list(gradient_list_all_epochs, None, para_list_all_epochs, net, None, is_GPU, device)
#     
#             # Update Visualization
#     #         if viz.check_connection():
#     #             cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
#     #                                      win=cur_batch_win, name='current_batch_loss',
#     #                                      update=(None if cur_batch_win is None else 'replace'),
#     #                                      opts=cur_batch_win_opts)
# #             learning_rate = list(optimizer.param_groups)[0]['lr']
#             
# #             regularization_rate = list(optimizer.param_groups)[0]['weight_decay']
#             
# #             exp_model_param = update_model(net, learning_rate, regularization_rate)
#             
#             
#             optimizer.step()
#             
# #             print('parameter comparison::')
# #             
# #             compute_model_para_diff(list(net.parameters()), exp_model_param)
#             
#             
#             learning_rate_all_epochs.append(learning_rate)
#         
#         
#         
# #         item1 = data_train_loader.dataset.data[100]
# #         print(torch.norm(item0[0] - item1[0]))
#         
# #         random_ids_multi_super_iterations.append(random_ids)
#         
#         test(net, data_test_loader, criterion, data_test_size, is_GPU, device)
#         
#     
#     t2 = time.time()
#     
#     print("training_time::", (t2 - t1))
#     
#     return net, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs, random_ids_multi_super_iterations



def model_training_test(random_ids_multi_super_iterations, epoch, net, dataset_train, dataset_test, data_train_size, data_test_size, optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs):
#     global cur_batch_win
    net.train()
    
    gradient_list_all_epochs = []
    
    para_list_all_epochs = []
    
#     output_list_all_epochs = []
    
    learning_rate_all_epochs = []
    
    
    
    loss_list, batch_list = [], []
    
    t1 = time.time()
    
    
#     test(net, data_test_loader, criterion, data_test_size, is_GPU, device)

    for j in range(epoch):
        
#         random_ids = torch.zeros([data_train_size], dtype = torch.long)
    
    
#         random_ids = torch.randperm(data_train_size)
        random_ids = random_ids_multi_super_iterations[j]
    
#         k = 0
        
        learning_rate = lrs[j]
        update_learning_rate(optimizer, learning_rate)
        
#         item0 = data_train_loader.dataset.data[100]
    
#         for i, items in enumerate(data_train_loader):
        i = 0

        for k in range(0, data_train_size, batch_size):
            
            
            
            
#             random.seed(random_seed)
#             os.environ['PYTHONHASHSEED'] = str(random_seed)
#             np.random.seed(random_seed)
#             torch.manual_seed(random_seed)
#             torch.cuda.manual_seed(random_seed)
#             torch.cuda.manual_seed_all(random_seed) # if you are using multi-GPU.
#             torch.backends.cudnn.benchmark = False
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.enabled = False
            
            end_id = k + batch_size
            
            curr_rand_ids = random_ids[k:end_id]
            
            if end_id > data_train_size:
                end_id = data_train_size
            if not is_GPU:
                images, labels =  dataset_train.data[curr_rand_ids], dataset_train.labels[curr_rand_ids]
            else:
                images, labels =  dataset_train.data[curr_rand_ids].to(device), dataset_train.labels[curr_rand_ids].to(device)
            

            
#             print(k, end_id)
#             random_ids[k:end_id] = ids
            
            
#             k = k + batch_size
            
            optimizer.zero_grad()
    
            output = net(images)
            
#             print(output[0])
#              
#             print(torch.sort(ids)[0])
    
            loss = criterion(output, labels)
    
    
#             loss_list.append(loss.detach().cpu().item())
#             batch_list.append(i+1)
    
            if i % 10 == 0:
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (j, i, loss.detach().cpu().item()))
            
#             if i % 10 == 0:
#                 lr_scheduler.step()
                 
    
            loss.backward()
    
            append_gradient_list(gradient_list_all_epochs, None, para_list_all_epochs, net, None, is_GPU, device)
    
    
            i += 1
            # Update Visualization
    #         if viz.check_connection():
    #             cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
    #                                      win=cur_batch_win, name='current_batch_loss',
    #                                      update=(None if cur_batch_win is None else 'replace'),
    #                                      opts=cur_batch_win_opts)
#             learning_rate = list(optimizer.param_groups)[0]['lr']
            
#             regularization_rate = list(optimizer.param_groups)[0]['weight_decay']
            
#             exp_model_param = update_model(net, learning_rate, regularization_rate)
            
            
            optimizer.step()
            
#             print('parameter comparison::')
#             
#             compute_model_para_diff(list(net.parameters()), exp_model_param)
            
            
            learning_rate_all_epochs.append(learning_rate)
        
        
        
#         item1 = data_train_loader.dataset.data[100]
#         print(torch.norm(item0[0] - item1[0]))
        
#         random_ids_multi_super_iterations.append(random_ids)
    test(net, dataset_test, batch_size, criterion, data_test_size, is_GPU, device)
#     test(net, data_test_loader, criterion, data_test_size, is_GPU, device)
        
    
    t2 = time.time()
    
    print("training_time::", (t2 - t1))
    
    return net, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs





# def postprocessing_ids_list(random_list_all_epochs, ids_list_all_epochs, ids_list2_all_epochs):
#     
#     for i in range(len(random_list_all_epochs)):
#         
#         random_list = random_list_all_epochs[i]
#         
#         
#         
#         
#         for j in range(len(random_list)):
#             
#             curr_rand_ids = random_list[j]


         
def prepare_term_1_2_large_feature_space(git_ignore_folder, data_set_train_loader, dataset_train_len, weights, offsets, dim, max_epoch, num_class, cut_off_epoch, batch_size, curr_rand_ids_multi_super_iterations, mini_epochs_per_super_iteration):

    term2 = torch.zeros([cut_off_epoch, num_class*dim[1]], dtype = torch.double)
    
    epoch = 0
    
    end = False
    
    cut_off_super_iteration = (int((cut_off_epoch - 1)/mini_epochs_per_super_iteration) + 1)
    
    avg_term1  = 0
    
    directory = git_ignore_folder + 'svd_folder'
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for k in range(curr_rand_ids_multi_super_iterations.shape[0]):
    
        curr_rand_ids = curr_rand_ids_multi_super_iterations[k]
        
        
        weights_this_super_iter = weights[k*dataset_train_len:(k+1)*dataset_train_len]
        
        offsets_this_super_iter = offsets[k*dataset_train_len:(k+1)*dataset_train_len]
    
#         for i in range(0, X.shape[0], batch_size):
        i = 0


        for t, items in enumerate(data_set_train_loader):
            
            
#             if epoch >= 126:
#                 print("here")
            
            end_id = i + batch_size
            
            if end_id > dataset_train_len:
                end_id = dataset_train_len
    
            batch_X = items[0]
            
            batch_weights = weights_this_super_iter[curr_rand_ids[i:end_id]]
            
            batch_offsets = offsets_this_super_iter[curr_rand_ids[i:end_id]]
            
            batch_term1 = prepare_sub_term_1(batch_X, batch_weights, batch_X.shape, num_class)
            
            
            if epoch >= cut_off_epoch - mini_epochs_per_super_iteration:
                avg_term1 = avg_term1 + batch_term1
            
            
            sub_u, sub_v = compute_single_svd(epoch, batch_term1, batch_size)
            
#             if k == 1 and i == 0:
#                 torch.save(batch_X, git_ignore_folder + 'tmp_batch_X')
#                 
#                 torch.save(batch_term1, git_ignore_folder + 'tmp_batch_term1')
#                 
#                 torch.save(batch_weights, git_ignore_folder + 'tmp_batch_weights')
#                 
#                 torch.save(batch_offsets, git_ignore_folder + 'tmp_batch_offsets')
#                 
#                 torch.save(curr_rand_ids[i:end_id], git_ignore_folder + 'tmp_rand_ids')
#                 
#                 torch.save(sub_u, git_ignore_folder + 'tmp_sub_u')
#                 
#                 torch.save(sub_v, git_ignore_folder + 'tmp_sub_v')
            
            
            
            np.save(directory + '/u_' + str(epoch), sub_u)
    
            np.save(directory + '/v_' + str(epoch), sub_v)
    
            del sub_u, sub_v
    
            batch_term2 = prepare_sub_term_2(batch_X, batch_offsets, batch_X.shape, num_class)
            
            term2[epoch] = batch_term2
            
            i += batch_size
            epoch = epoch + 1
            
            if epoch >= cut_off_epoch:
                end = True
                break
        
        
        data_set_train_loader.batch_sampler.increm_ids()
        
        
        if end == True:
            break
        
        
    
    torch.save(torch.tensor([cut_off_epoch]), directory + '/len')
    
    return avg_term1, term2            
            
def model_training_lr(epoch, net, dataset_train, data_test_loader, data_train_size, data_test_size, optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs):
#     global cur_batch_win
    net.train()
    
    gradient_list_all_epochs = []
    
    para_list_all_epochs = []
    
#     output_list_all_epochs = []
    
    learning_rate_all_epochs = []
    
    
    X_theta_prod_seq, X_theta_prod_softmax_seq = [], []
#     loss_list, batch_list = [], []
    
#     random_ids_all_iterations = []
    
    t1 = time.time()
    
    
#     test(net, data_test_loader, criterion, data_test_size, is_GPU, device)

    for j in range(epoch):
        
#         random_ids = torch.zeros([data_train_size], dtype = torch.long)
        random_ids = torch.randperm(data_train_size)
    
#         k = 0

        i = 0
        
        learning_rate = lrs[j]
        update_learning_rate(optimizer, learning_rate)
        
#         item0 = data_train_loader.dataset.data[100]
    
        for k in range(0, data_train_size, batch_size):

            end_id = k + batch_size
            
            if end_id > data_train_size:
                end_id = data_train_size
            
            
            curr_rand_ids = random_ids[k: end_id]
            
#             if not is_GPU:
#                 images, labels, ids =  items[0], items[1], items[2]
#             else:
#                 images, labels, ids =  items[0].to(device), items[1].to(device), items[2]
            if not is_GPU:
                images, labels = dataset_train.data[curr_rand_ids], dataset_train.labels[curr_rand_ids]
            else:
                images, labels = dataset_train.data[curr_rand_ids].to(device), dataset_train.labels[curr_rand_ids].to(device)
            
#             print(k, end_id)
#             random_ids[k:end_id] = ids
            
            
#             k = k + batch_size
            
            optimizer.zero_grad()
    
            output = net.forward_with_provenance(images, X_theta_prod_seq, X_theta_prod_softmax_seq)
    
            loss = criterion(output, labels)
    
    
#             loss_list.append(loss.detach().cpu().item())
#             batch_list.append(i+1)
    
            if i % 10 == 0:
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (j, i, loss.detach().cpu().item()))
            
            i += 1
            
#             if i % 10 == 0:
#                 lr_scheduler.step()
                 
    
            loss.backward()
    
            append_gradient_list(gradient_list_all_epochs, None, para_list_all_epochs, net, None, is_GPU, device)
    
            # Update Visualization
    #         if viz.check_connection():
    #             cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
    #                                      win=cur_batch_win, name='current_batch_loss',
    #                                      update=(None if cur_batch_win is None else 'replace'),
    #                                      opts=cur_batch_win_opts)
#             learning_rate = list(optimizer.param_groups)[0]['lr']
            
#             regularization_rate = list(optimizer.param_groups)[0]['weight_decay']
            
#             exp_model_param = update_model(net, learning_rate, regularization_rate)
            
            
            optimizer.step()
            
#             print('parameter comparison::')
#             
#             compute_model_para_diff(list(net.parameters()), exp_model_param)
            
            
            learning_rate_all_epochs.append(learning_rate)
            
            del images, labels
        
        
#         item1 = data_train_loader.dataset.data[100]
#         print(torch.norm(item0[0] - item1[0]))
        
        random_ids_multi_super_iterations.append(random_ids)
        
#     test(net, data_test_loader, criterion, data_test_size, is_GPU, device)
        
    
    t2 = time.time()
    
    print("training_time::", (t2 - t1))
    
    return net, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs, X_theta_prod_seq, X_theta_prod_softmax_seq, random_ids_multi_super_iterations




def model_training_lr_test(random_ids_multi_super_iterations, epoch, net, dataset_train, data_test_loader, data_train_size, data_test_size, optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs):
#     global cur_batch_win
    net.train()
    
    gradient_list_all_epochs = []
    
    para_list_all_epochs = []
    
#     output_list_all_epochs = []
    
    learning_rate_all_epochs = []
    
    
    X_theta_prod_seq, X_theta_prod_softmax_seq = [], []
#     loss_list, batch_list = [], []
    
#     random_ids_all_iterations = []
    
    t1 = time.time()
    
    
#     test(net, data_test_loader, criterion, data_test_size, is_GPU, device)

    for j in range(epoch):
        
#         random_ids = torch.zeros([data_train_size], dtype = torch.long)
#         random_ids = torch.randperm(data_train_size)
        random_ids = random_ids_multi_super_iterations[j]
    
#         k = 0

        i = 0
        
        learning_rate = lrs[j]
        update_learning_rate(optimizer, learning_rate)
        
#         item0 = data_train_loader.dataset.data[100]
    
        for k in range(0, data_train_size, batch_size):

            end_id = k + batch_size
            
            if end_id > data_train_size:
                end_id = data_train_size
            
            
            curr_rand_ids = random_ids[k: end_id]
            
#             if not is_GPU:
#                 images, labels, ids =  items[0], items[1], items[2]
#             else:
#                 images, labels, ids =  items[0].to(device), items[1].to(device), items[2]
            if not is_GPU:
                images, labels = dataset_train.data[curr_rand_ids], dataset_train.labels[curr_rand_ids]
            else:
                images, labels = dataset_train.data[curr_rand_ids].to(device), dataset_train.labels[curr_rand_ids].to(device)
            
#             print(k, end_id)
#             random_ids[k:end_id] = ids
            
            
#             k = k + batch_size
            
            optimizer.zero_grad()
    
            output = net.forward_with_provenance(images, X_theta_prod_seq, X_theta_prod_softmax_seq)
    
            loss = criterion(output, labels)
    
    
#             loss_list.append(loss.detach().cpu().item())
#             batch_list.append(i+1)
    
            if i % 10 == 0:
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (j, i, loss.detach().cpu().item()))
            
            i += 1
            
#             if i % 10 == 0:
#                 lr_scheduler.step()
                 
    
            loss.backward()
    
            append_gradient_list(gradient_list_all_epochs, None, para_list_all_epochs, net, None, is_GPU, device)
    
            # Update Visualization
    #         if viz.check_connection():
    #             cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
    #                                      win=cur_batch_win, name='current_batch_loss',
    #                                      update=(None if cur_batch_win is None else 'replace'),
    #                                      opts=cur_batch_win_opts)
#             learning_rate = list(optimizer.param_groups)[0]['lr']
            
#             regularization_rate = list(optimizer.param_groups)[0]['weight_decay']
            
#             exp_model_param = update_model(net, learning_rate, regularization_rate)
            
            
            optimizer.step()
            
#             print('parameter comparison::')
#             
#             compute_model_para_diff(list(net.parameters()), exp_model_param)
            
            
            learning_rate_all_epochs.append(learning_rate)
        
        
        
#         item1 = data_train_loader.dataset.data[100]
#         print(torch.norm(item0[0] - item1[0]))
        
#         random_ids_multi_super_iterations.append(random_ids)
        
#     test(net, data_test_loader, criterion, data_test_size, is_GPU, device)
        
    
    t2 = time.time()
    
    print("training_time::", (t2 - t1))
    
    return net, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs, X_theta_prod_seq, X_theta_prod_softmax_seq, random_ids_multi_super_iterations


def compute_single_x_sum_by_class(batch_X, batch_Y, num_class):
    y_onehot = torch.DoubleTensor(batch_X.shape[0], num_class)


    batch_Y = batch_Y.type(torch.LongTensor)

# In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, batch_Y.view(-1, 1), 1)
    
    
    x_sum_by_class = torch.mm(torch.t(y_onehot), batch_X)
    
    return x_sum_by_class
        
def compute_x_sum_by_class_by_batch(dataset_train_loader, dataset_train_len, batch_size, num_class, random_ids_multi_super_iterations):
    
#     x_sum_by_class = torch.zeros([num_class, dim[1]], dtype = torch.double)
    
    
    x_sum_by_class_list = []
    
    
    for j in range(len(random_ids_multi_super_iterations)):
#         random_ids = random_ids_multi_super_iterations[j]
    
    
#         curr_X = X[random_ids]
#         
#         curr_Y = Y[random_ids]
        i = 0
    
        for t, items in enumerate(dataset_train_loader):
            end_id = i + batch_size
                
            if end_id > dataset_train_len:
                end_id = dataset_train_len
            
            
#             curr_rand_ids = random_ids[i:end_id]
            
            batch_X, batch_Y = items[0], items[1]
        
            x_sum_by_class = compute_single_x_sum_by_class(batch_X, batch_Y, num_class)
            
            x_sum_by_class_list.append(x_sum_by_class.view(-1,1))
    
    
        dataset_train_loader.batch_sampler.increm_ids()
    
#     for i in range(num_class):
#         Y_mask = (Y == i)
#         
#         Y_mask = Y_mask.type(torch.DoubleTensor)
#         
#         x_sum_by_class[i] = torch.mm(torch.t(Y_mask), X) 
        
    return torch.stack(x_sum_by_class_list, dim = 0)        


def update_data_train_loader(dataset_train_len, dataset_train, random_ids_multi_super_iterations, batch_size):
    
    random_list_all_epochs = []
    


    for k in range(len(random_ids_multi_super_iterations)):
        
        random_ids = random_ids_multi_super_iterations[k]
        
#         id_start = 0
#         
#         id_end = 0
        
        random_ids_list = []
        
        for i in range(0, dataset_train_len, batch_size):
            
            end_id = i + batch_size
            
            if end_id > dataset_train_len:
                end_id = dataset_train_len
        
            
            curr_rand_ids = random_ids[i:end_id]
            
#             curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
            
            random_ids_list.append(list(curr_rand_ids.numpy()))
        
        random_list_all_epochs.append(random_ids_list)    
#             curr_matched_ids_size = curr_matched_ids.shape[0]

            
            
    curr_batch_sampler = Batch_sampler(random_list_all_epochs)

#     data_train_loader.batch_sampler = curr_batch_sampler
#     data_train_loader2 = DataLoader(dataset_train, shuffle = True)
    
    data_train_loader = DataLoader(dataset_train, num_workers=0, shuffle=False, batch_sampler=curr_batch_sampler)
    
    return data_train_loader


def load_svd(directory):
    u_list = []
 
#     s_list = []
     
    v_s_list = []
    
    directory = directory + 'svd_folder/'
    
    term1_len = torch.load(directory + 'len')[0]
    
    for i in range(term1_len):
        u_list.append(torch.from_numpy(np.load(directory + '/u_' + str(i) + '.npy')))
        v_s_list.append(torch.from_numpy(np.load(directory + '/v_' + str(i) + '.npy')))
    
    
    
    return u_list, v_s_list

def compute_model_parameter_by_approx_incremental_1_2(cut_off_epoch, exp_para_list_all_epochs, exp_gradient_list_all_epochs, theta_list, grad_list, dataset_train, weights, offsets, delta_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, term2, dim, theta, max_epoch, learning_rate_all_epochs, beta, batch_size, num_class, x_sum_by_class_list, avg_term2, u_list, v_s_list, is_GPU, device):
    
    total_time = 0.0
    
#     min_batch_num_per_epoch = int((origin_X.shape[0] - 1)/batch_size) + 1
    cut_off_super_iteration = int((weights.shape[0]-1)/dim[0])
#     


    epoch = 0

    overhead = 0

    overhead2 = 0
    
    overhead3 = 0
#     
#     is_GPU = False
    
    if is_GPU:
        theta = theta.to(device)
    
    
    vectorized_theta = torch.reshape(torch.t(theta), [-1,1])
    
#     theta = Variable(theta)
    
    
    
#     sub_term_2_list = []
    
    avg_A = 0
    
    avg_B = 0
    
#     theta_list = []
#     
#     output_list = []
#     
#     sub_term2_list = []
#     
#     x_sum_by_list = []
#     
#     sub_term_1_theta_list = []
    
    end = False
    
    num = 0
    
#     for k in range(max_epoch):
    removed_batch_empty_list = []
    
    random_ids_list_all_epochs = []
    
    i = 0
    
    for k in range(len(random_ids_multi_super_iterations)):
        

    
        random_ids = random_ids_multi_super_iterations[k]
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
        if delta_ids.shape[0] > 1:
            all_indexes = np.sort(sort_idx[delta_ids])
        else:
            all_indexes = torch.tensor([sort_idx[delta_ids]])
                
        id_start = 0
    
        id_end = 0
        
        random_ids_list = []
        
        for j in range(0, dim[0], batch_size):
        
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]

            curr_matched_id_num = curr_matched_ids.shape[0]

            if curr_matched_id_num > 0:
                random_ids_list.append(curr_matched_ids)
                removed_batch_empty_list.append(False)
            else:
                random_ids_list.append(random_ids[0:1])
                removed_batch_empty_list.append(True)
            
            
            i += 1
            
            id_start = id_end
                
        random_ids_list_all_epochs.append(random_ids_list)        
    
    
#     curr_batch_sampler = Batch_sampler(random_ids_list_all_epochs)
    
        
#     data_train_loader = DataLoader(dataset_train, num_workers=0, shuffle=False, batch_sampler = curr_batch_sampler, pin_memory=True)

    
#     is_GPU = False

    for k in range(len(random_ids_multi_super_iterations)):
    
#     for k in range(5):
        
        
        random_ids = random_ids_multi_super_iterations[k]
        
        random_ids_list = random_ids_list_all_epochs[k]
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
        if delta_ids.shape[0] > 1:
            all_indexes = np.sort(sort_idx[delta_ids])
        else:
            all_indexes = torch.tensor([sort_idx[delta_ids]])
        
        
        end_id_super_iteration = (k + 1)*dim[0]
        
        id_start = 0
    
        id_end = 0
        
        if end_id_super_iteration > weights.shape[0]:
            end_id_super_iteration = weights.shape[0]
            
        
#         for i in range(0, dim[0], batch_size):
        i = 0
#         for t, items in enumerate(data_train_loader):


        for p in range(len(random_ids_list)):
            
#             curr_matched_ids = items[2]        
            curr_matched_ids = random_ids_list[p]
            end_id = i + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]

            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
            
#             curr_matched_ids = items[2]#random_ids[all_indexes[id_start:id_end]]
            
            curr_matched_ids_size = 0#curr_matched_ids.shape[0]
            
            if not removed_batch_empty_list[epoch]:
                curr_matched_ids_size = curr_matched_ids.shape[0]
            
            
            if (end_id - i - curr_matched_ids_size) <= 0:
                epoch = epoch + 1

                
                continue
            

            if curr_matched_ids_size > 0:
                
                
#                 batch_delta_X = items[0]#(X[curr_matched_ids])
#                 
#                 batch_delta_Y = items[1].type(torch.LongTensor)#(Y[curr_matched_ids])
                
                batch_delta_X = dataset_train.data[curr_matched_ids]
                
                batch_delta_Y = dataset_train.labels[curr_matched_ids].type(torch.LongTensor)
                
                if is_GPU:
                    batch_delta_X = batch_delta_X.to(device)
                    
                    batch_delta_Y = batch_delta_Y.type(torch.LongTensor).to(device)
                

#             print(curr_matched_ids_size)   

    
            sub_term2 = 0
            
            if curr_matched_ids_size > 0:

                coeff_rand_ids = curr_matched_ids + k*dim[0]
                
                batch_weights = weights[coeff_rand_ids]
            
                batch_offsets = offsets[coeff_rand_ids]
                
                if is_GPU:
                    batch_weights = batch_weights.to(device)
                    
                    batch_offsets = batch_offsets.to(device)

                
                batch_X_multi_theta = torch.mm(batch_delta_X, theta)
        

                sub_term2 = torch.mm(torch.t(batch_offsets.view(curr_matched_ids_size, num_class)), batch_delta_X).view(-1,1)#prepare_sub_term_2(batch_delta_X, batch_offsets, batch_delta_X.shape, num_class).view(-1,1)

                
                
#                 sub_term2_cp = torch.load(git_ignore_folder + "tmp_sub_term2")
#                 
#                 print(torch.norm(sub_term2 - sub_term2_cp))
            
            full_term2 = None
            
            if is_GPU:
                full_term2 = term2[epoch].clone().to(device)
            else:
                full_term2 = term2[epoch].clone()

            sub_term2 = full_term2.view(-1,1) - sub_term2
            
            del full_term2

            
            delta_x_sum_by_class = 0
            if curr_matched_ids_size > 0:
                delta_x_sum_by_class = compute_x_sum_by_class(batch_delta_X, batch_delta_Y, num_class, batch_delta_X.shape, is_GPU, device)
#                 exp_delta_x_sum_by_class = torch.load(git_ignore_folder + "tmp_delta_x_sum_by_class")
#                 
#                 print(torch.norm(delta_x_sum_by_class - exp_delta_x_sum_by_class))
            
#             exp_x_sum_by_class = torch.load(git_ignore_folder + 'tmp_x_sum_by_class')
#             
#             print(exp_x_sum_by_class - x_sum_by_class_list[epoch])
            
            
            full_x_sum_by_class = None
            
            if is_GPU:
                full_x_sum_by_class = x_sum_by_class_list[epoch].clone().to(device)
            else:
                full_x_sum_by_class = x_sum_by_class_list[epoch].clone()
            
            
            
            
            delta_x_sum_by_class = full_x_sum_by_class - delta_x_sum_by_class


            del full_x_sum_by_class

            vectorized_sub_term_1 = 0
            if curr_matched_ids_size > 0:
#                 t5 = time.time()
                
                res1 = torch.bmm(batch_X_multi_theta.view(curr_matched_ids_size, 1, num_class), batch_weights).view(curr_matched_ids_size, num_class)
    
                '''dim[1],num_class, num_class*num_class'''
                vectorized_sub_term_1 = torch.mm(torch.t(res1), batch_delta_X).view(num_class, dim[1]).view(-1,1)
                
#                 vectorized_sub_term_1 = vectorized_sub_term_1.cpu()
                
                del res1


            curr_u_list = None
            
            curr_v_s_list = None

#             if is_GPU:
#                 
# #                 print(u_list[epoch][0])
#                 
#                 curr_u_list = u_list[epoch].clone().to(device)
#                 curr_v_s_list = v_s_list[epoch].clone().to(device)
#                 
#             else:
            curr_u_list = u_list[epoch].clone()
            curr_v_s_list = v_s_list[epoch].clone()
            
            
            
#             if is_GPU:
#                 curr_u_list = curr_u_list.to(device)
#                 
#                 curr_v_s_list = curr_v_s_list.to(device)

            output = torch.mm(curr_u_list, torch.mm(curr_v_s_list, vectorized_theta.cpu()))
            
            
            if is_GPU:
                output = output.to(device)
            
            del curr_u_list
            
            del curr_v_s_list
            
            
#             if is_GPU:
#                 output = output.to(device)
            
            
            output -= vectorized_sub_term_1
            output += sub_term2
            output -= delta_x_sum_by_class
            output /= (end_id - i - curr_matched_ids_size)
            
#             exp_grad = exp_gradient_list_all_epochs[epoch][0]
#  
#             exp_para = exp_para_list_all_epochs[epoch][0]
#  
#             print("grad_diff::")
#              
# #             print(exp_grad.T)
# #              
# #             print(output.view(num_class, dim[1]).T)
#              
#             print(torch.norm(exp_grad.T - output.view(num_class, dim[1]).T))
#              
#             print("para diff::")
#              
#             print(torch.norm(exp_para.T - theta))
#             
#             print(epoch, curr_matched_ids_size)
            
            output += vectorized_theta*beta     

            
            del delta_x_sum_by_class
            
            del sub_term2
            
            
            
            if curr_matched_ids_size > 0:
                del batch_delta_X
                del batch_delta_Y
                del batch_weights
                del batch_offsets
                del batch_X_multi_theta
                del vectorized_sub_term_1
            
#             if is_GPU:
#                 vectorized_theta -= (output*learning_rate_all_epochs[num]).to(device)
#             else:
            vectorized_theta -= output*learning_rate_all_epochs[num]
            
            
            del theta
            
            theta = vectorized_theta.view(num_class, dim[1]).T
            

            del output
            torch.cuda.empty_cache()
#             print("epoch::", epoch)
            
            epoch = epoch + 1
            
            num += 1
            
            i += batch_size
            
            id_start = id_end
#         data_train_loader.batch_sampler.increm_ids()   
#             if epoch >= max_epoch:
#                 end = True
#                 
#                 break
#             
#                 
#         if end == True:
#             break
                

    return theta#, theta_list, output_list, sub_term2_list, x_sum_by_list, sub_term_1_theta_list
    
        

def capture_provenance(git_ignore_folder, dataset_train_loader, dataset_train_len, dim, epoch, num_class, batch_size, mini_epochs_per_super_iteration, random_ids_multi_super_iterations, X_theta_prod_softmax_seq, X_theta_prod_seq):
    
    save_random_id_orders(git_ignore_folder, random_ids_multi_super_iterations)
    
    super_iteration = (int((len(X_theta_prod_softmax_seq) - 1)/mini_epochs_per_super_iteration) + 1)
    
    cut_off_epoch= len(X_theta_prod_softmax_seq)
    
    
    print('super_iteration::', super_iteration)

    print('cut_off_epoch::', cut_off_epoch)
    
    weights, offsets = prepare_term_1_batch3_0(X_theta_prod_softmax_seq, X_theta_prod_seq, dim, epoch, num_class, cut_off_epoch, batch_size) 

    cut_off_super_iteration = (int((cut_off_epoch - 1)/mini_epochs_per_super_iteration) + 1)
    
    
    curr_rand_ids_multi_super_iterations = random_ids_multi_super_iterations[0:(int((cut_off_epoch - 1)/mini_epochs_per_super_iteration) + 1)*dataset_train_len]
    
    '''T*dim[0]'''
    
    curr_rand_ids_multi_super_iterations = curr_rand_ids_multi_super_iterations.view(-1, dataset_train_len)
    
    _, sorted_ids_multi_super_iterations = torch.sort(curr_rand_ids_multi_super_iterations)


    weights_copy = torch.zeros([dataset_train_len*cut_off_super_iteration, num_class, num_class], dtype = torch.double)
    
    offsets_copy = torch.zeros([dataset_train_len*cut_off_super_iteration, num_class], dtype = torch.double)
    
    weights_copy[0:weights.shape[0]] = weights
    
    offsets_copy[0:offsets.shape[0]] = offsets
    
    weights_copy = weights_copy.view(cut_off_super_iteration, dataset_train_len, num_class*num_class)
    
    offsets_copy = offsets_copy.view(cut_off_super_iteration, dataset_train_len, num_class)

    for i in range(cut_off_super_iteration):
        weights_copy[i, :, :] = weights_copy[i, sorted_ids_multi_super_iterations[i], :]
        offsets_copy[i, :, :] = offsets_copy[i, sorted_ids_multi_super_iterations[i], :]
    
    
    weights_copy = weights_copy.view(dim[0]*cut_off_super_iteration, num_class*num_class)
    
    offsets_copy = offsets_copy.view(dim[0]*cut_off_super_iteration, num_class)


    print('compute_weights_offsets_done!!')

    print(weights_copy.shape)

    avg_term1, term2 = prepare_term_1_2_large_feature_space(git_ignore_folder, dataset_train_loader,dataset_train_len, weights_copy, offsets_copy, dim, epoch, num_class, cut_off_epoch, batch_size, curr_rand_ids_multi_super_iterations, mini_epochs_per_super_iteration)
    if avg_term1.shape[1] < max_para_num_opt:
        eigen_decomposition3(git_ignore_folder, avg_term1)

    cut_off_super_iteration = int(super_iteration*theta_record_threshold)#(int((cut_off_epoch - 1)/mini_epochs_per_super_iteration) + 1)
    
    
    cut_off_epoch = cut_off_super_iteration*mini_epochs_per_super_iteration


    
    torch.save(weights_copy, git_ignore_folder+'weights')
    
    
    torch.save(cut_off_epoch, git_ignore_folder + 'cut_off_epoch')
    
    del weights
    
    print('save weights and term 1 done!!!')
    
     
    torch.save(offsets_copy, git_ignore_folder+'offsets')
    
    torch.save(term2, git_ignore_folder+'term2') 
    
    del term2
    
    del offsets
    
    print('save offsets and term 2 done!!!')    
    
    

def model_training_skipnet(epoch, net, data_train_loader, data_test_loader, data_train_size, data_test_size, optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs):
#     global cur_batch_win
    net.train()
    
    gradient_list_all_epochs = []
    
    para_list_all_epochs = []
    
#     output_list_all_epochs = []
    
    learning_rate_all_epochs = []
    
    
    
    loss_list, batch_list = [], []
    
    t1 = time.time()
    
    
    all_ids_list_all_epochs = []
    
#     ids_list_all_epochs = []
#     
#     ids2_list_all_epochs = []
    
#     test(net, data_test_loader, criterion, data_test_size, is_GPU, device)

    for j in range(epoch):
        
        random_ids = torch.zeros([data_train_size], dtype = torch.long)
    
        k = 0
        
        learning_rate = lrs[j]
        update_learning_rate(optimizer, learning_rate)
#         item0 = data_train_loader.dataset.data[100]
    
        for i, items in enumerate(data_train_loader):
            
            
#             random.seed(random_seed)
#             os.environ['PYTHONHASHSEED'] = str(random_seed)
#             np.random.seed(random_seed)
#             torch.manual_seed(random_seed)
#             torch.cuda.manual_seed(random_seed)
#             torch.cuda.manual_seed_all(random_seed) # if you are using multi-GPU.
#             torch.backends.cudnn.benchmark = False
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.enabled = False
            
            
            if not is_GPU:
                images, labels, ids =  items[0], items[1], items[2]
            else:
                images, labels, ids =  items[0].to(device), items[1].to(device), items[2]
            
            end_id = k + batch_size
            
            if end_id > data_train_size:
                end_id = data_train_size
            
#             print(k, end_id)
            
            
            
            
            
            optimizer.zero_grad()
    
            sorted_ids, sorted_id_ids = torch.sort(ids)
            random_ids[k:end_id] = sorted_ids
#             print(sorted_ids)
            
            need_record = False
            
            if j == 1 and i == 0:
                need_record = True
                
            output, all_ids_list = net(images[sorted_id_ids], need_record)
            
            k = k + batch_size
            
            all_ids_list_all_epochs.append(all_ids_list)
            
#             print(ids_list2)
            
            
            
#             ids_list_all_epochs.append(ids_list)
#             
#             ids2_list_all_epochs.append(ids_list2)
            
            
#             print(output[0])
#              
#             
    
            loss = criterion(output, labels[sorted_id_ids])
    
    
            loss_list.append(loss.detach().cpu().item())
            batch_list.append(i+1)
    
            if i % 10 == 0:
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (j, i, loss.detach().cpu().item()))
            
#             if i % 100 == 0:
#                 lr_scheduler.step()
                 
    
            loss.backward()
    
            append_gradient_list(gradient_list_all_epochs, None, para_list_all_epochs, net, None, is_GPU, device)
    
            # Update Visualization
    #         if viz.check_connection():
    #             cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
    #                                      win=cur_batch_win, name='current_batch_loss',
    #                                      update=(None if cur_batch_win is None else 'replace'),
    #                                      opts=cur_batch_win_opts)
#             learning_rate = list(optimizer.param_groups)[0]['lr']
            
#             regularization_rate = list(optimizer.param_groups)[0]['weight_decay']
            
#             exp_model_param = update_model(net, learning_rate, regularization_rate)
            
            
            optimizer.step()
            
#             print('parameter comparison::')
#             
#             compute_model_para_diff(list(net.parameters()), exp_model_param)
            
            
            learning_rate_all_epochs.append(learning_rate)
        
            del images, labels
        
#         item1 = data_train_loader.dataset.data[100]
#         print(torch.norm(item0[0] - item1[0]))
        
        random_ids_multi_super_iterations.append(random_ids)
        
        test_skipnet(net, data_test_loader, criterion, data_test_size, is_GPU, device)
        
    
    t2 = time.time()
    
    print("training_time::", (t2 - t1))
    
    return net, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs, all_ids_list_all_epochs



def model_training_quantized(random_ids_multi_super_iterations, epsilon, num_epochs, origin_X, origin_Y, test_X, test_Y, learning_rate, regularization_coeff, error, model, batch_size, dim):
    count = 0

    loss = np.infty

    elapse_time = 0
    
    b_time = 0
    
    f_time = 0
    
    iter = 0
    
    random_theta_list_all_epochs = []
    
    gradient_list_all_epochs = []
    
    para_list_all_epochs = []

    output_list_all_epochs = []

    while loss > loss_threshold and count < num_epochs:
        
#         random_ids = torch.randperm(dim[0])
        
        random_ids = random_ids_multi_super_iterations[count]
        
        X = origin_X[random_ids]
        
        Y = origin_Y[random_ids]
        
#         random_ids_multi_super_iterations.append(random_ids)
        
        print("iteration::", count)

        for i in range(0,X.shape[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id >= X.shape[0]:
                end_id = X.shape[0]

            batch_x, batch_y = X[i:end_id], Y[i:end_id]


            t1 = time.time()
        
            train = Variable(batch_x)
            labels = Variable(batch_y.view(-1))
            
            t3 = time.time()
            
            outputs = model(train)
            
            t4 = time.time()
            
            f_time += (t4 - t3)
            
            labels = labels.type(torch.LongTensor)
            
            loss = error(outputs, labels) + regularization_coeff*get_regularization_term(model.parameters())
            
            t3 = time.time()
            
            loss.backward()
            
            t4 = time.time()
            
            b_time += (t4 - t3)
            
            random_theta_list = update_and_zero_model_gradient_quantized(model,learning_rate, epsilon, gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, iter, train)
            
            random_theta_list_all_epochs.append(random_theta_list)
            
            if len(para_list_all_epochs) > 1:
                print("para_changes_first_layer::", torch.norm(para_list_all_epochs[-1][0] - para_list_all_epochs[-2][0]))
                print("para_changes_last_layer::", torch.norm(para_list_all_epochs[-1][-1] - para_list_all_epochs[-2][-1]))
            
            t2 = time.time()
            
            elapse_time += (t2 - t1)
            
            print("loss::", loss)
            
            
            iter += 1
        
        count += 1
        
        
        
        if count % 10 == 0:
            compute_test_acc(model, test_X, test_Y)

    print("training time is ", elapse_time)
    
    print("forward time is ", f_time)
    
    print("backaward time is ", b_time)
    
    return model, count, random_theta_list_all_epochs, para_list_all_epochs, gradient_list_all_epochs, output_list_all_epochs


def model_training_quantized_updates(random_ids_multi_super_iterations, epsilon, num_epochs, origin_X, origin_Y, test_X, test_Y, learning_rate, regularization_coeff, error, model, is_tracking_paras, batch_size, dim, gradient_list_all_epochs, para_list_all_epochs, random_theta_list_all_epochs):
    count = 0

    loss = np.infty

    elapse_time = 0
    
    b_time = 0
    
    f_time = 0
    
    iter = 0
    
    random_theta_list_all_epochs = []

    while loss > loss_threshold and count < num_epochs:
        
        random_ids = random_ids_multi_super_iterations[count]
        
        X = origin_X[random_ids]
        
        Y = origin_Y[random_ids]
        
        print("iteration::", count)

        for i in range(0,X.shape[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id >= X.shape[0]:
                end_id = X.shape[0]

            batch_x, batch_y = X[i:end_id], Y[i:end_id]


            t1 = time.time()
        
            train = Variable(batch_x)
            labels = Variable(batch_y.view(-1))
            
            t3 = time.time()
            
            outputs = model(train)
            
            t4 = time.time()
            
            f_time += (t4 - t3)
            
            labels = labels.type(torch.LongTensor)
            
            loss = error(outputs, labels) + regularization_coeff*get_regularization_term(model.parameters())
            
            t3 = time.time()
            
            loss.backward()
            
            t4 = time.time()
            
            b_time += (t4 - t3)
            
            random_theta_list = update_and_zero_model_gradient_quantized(model,learning_rate, epsilon, gradient_list_all_epochs, para_list_all_epochs, iter)
            
            
            random_theta_list_all_epochs.append(random_theta_list)
            
            if len(para_list_all_epochs) > 1:
                print("para_changes_first_layer::", torch.norm(para_list_all_epochs[-1][0] - para_list_all_epochs[-2][0]))
                print("para_changes_last_layer::", torch.norm(para_list_all_epochs[-1][-1] - para_list_all_epochs[-2][-1]))
            
            t2 = time.time()
            
            elapse_time += (t2 - t1)
            
            print("loss::", loss)
            
            
            iter += 1
        
        count += 1
        
        
        
        if count % 10 == 0:
            compute_test_acc(model, test_X, test_Y)

    print("training time is ", elapse_time)
    
    print("forward time is ", f_time)
    
    print("backaward time is ", b_time)
    
    return model, count, random_theta_list_all_epochs




def get_all_gradients(selected_rows, output_list_all_epochs, model, para_list_all_epochs, input_dim, hidden_dims, output_dim, num_class, X, Y):
    
    depth = len(hidden_dims) + 1
    

    delta_gradient_all_epochs = []
    
    delta_all_epochs = []
    
    for k in range(len(output_list_all_epochs)):

#     for k in range(1):
        
        output_list = output_list_all_epochs[k]
        pred = output_list[len(output_list) - 1][selected_rows]
            
        para_list = para_list_all_epochs[k]
        
        init_model(model, para_list)
    
    
    
#         A_list = [None]*depth
#     
#     
#         B_list = [None]*depth
#         
#         
#     #     A0_list = [None]*depth
#     #     
#     #     B0_list = [None]*depth
#     #         loss = error(pred, Y)
        
        delta = softmax_func(pred) - get_onehot_y(Y[selected_rows], X[selected_rows].shape, num_class)
        
#         delta = delta[selected_rows]
        
        delta_all_epochs.append(delta)
        delta_gradient = []
        
        for m in range(20):
            
            curr_gradient_list = []
            
            print("derivitive_times::", m)
            
            for l in range(delta.shape[1]):
                delta[m][l].backward(retain_graph=True)
                
                curr_gradient = model.get_all_gradient()
                
                curr_gradient_list.append(curr_gradient)
                
            delta_gradient.append(curr_gradient_list)
            
            
        delta_gradient_all_epochs.append(delta_gradient)
        
    return delta_gradient_all_epochs, delta_all_epochs   
            
            
            
            
            
    
        
        
    
    







def get_all_vectorized_gradients(para_list):
    
    res_list = []
    
    for param in para_list:
        res_list.append(param.grad.clone().view(-1))
        
        
#         para_list.append(param.grad.clone())
        
        
    return torch.cat(res_list, 0)    


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
# def get_model_para_shape_list(para_list):
#     
#     shape_list = []
#     
#     full_shape_list = []
#     
#     total_shape_size = 0
#     
#     for para in list(para_list):
#         
#         all_shape_size = 1
#         
#         
#         for i in range(len(para.shape)):
#             all_shape_size *= para.shape[i]
#         
#         total_shape_size += all_shape_size
#         shape_list.append(all_shape_size)
#         full_shape_list.append(para.shape)
#         
#     return full_shape_list, shape_list, total_shape_size
    
    
    

def get_all_vectorized_parameters_with_gradient(para_list):
    
    res_list = []
    
    for param in para_list:
        res_list.append(param.view(-1))
        
        
#         para_list.append(param.grad.clone())
        
        
    return torch.cat(res_list, 0).view(-1)



def get_all_vectorized_parameters(para_list):
    
    res_list = []
    
    i = 0
    
    for param in para_list:
        
#         print(param.data.view(-1).view(shape_list[i]) - param)
#         
#         print(torch.norm(param.data.view(-1).view(shape_list[i]) - param))
        
        res_list.append(param.data.to('cpu').view(-1))
        
        i += 1
#         para_list.append(param.grad.clone())
        
        
    return torch.cat(res_list, 0).view(1,-1)


def get_all_vectorized_parameters1(para_list):
    
    res_list = []
    
    i = 0
    
    for param in para_list:
        
#         print(param.data.view(-1).view(shape_list[i]) - param)
#         
#         print(torch.norm(param.data.view(-1).view(shape_list[i]) - param))
        
        res_list.append(param.data.view(-1))
        
        i += 1
#         para_list.append(param.grad.clone())
        
        
    return torch.cat(res_list, 0).view(1,-1)


def get_all_vectorized_parameters2(para_list, is_GPU, device):
    
    res_list = []
    
    i = 0
    
    for param in para_list:
        
#         print(param.data.view(-1).view(shape_list[i]) - param)
#         
#         print(torch.norm(param.data.view(-1).view(shape_list[i]) - param))

#         print(param.data.shape)

        if is_GPU:
            res_list.append(param.data.to(device).view(-1))
        else:
            res_list.append(param.data.view(-1))
        
        i += 1
#         para_list.append(param.grad.clone())
        
        
    return torch.cat(res_list, 0).view(1,-1)


def compute_diff_vectorized_parameters(para_list1, para_list2, vec_para_diff, shape_list):
#     res_list = []
    
    
    id_start = 0
    
    for i in range(len(para_list1)):
        
        vec_para_diff[id_start:shape_list[i] + id_start].copy_((para_list1[i] - para_list2[i]).view(-1, 1)) 

        id_start = shape_list[i] + id_start


def compute_diff_vectorized_parameters2(para_list1, para_list2, vec_para_diff, shape_list, is_GPU, device):
#     res_list = []
    
    
    id_start = 0
    
    for i in range(len(para_list1)):
        
        if is_GPU:
            vec_para_diff[id_start:shape_list[i] + id_start].copy_((para_list1[i] - para_list2[i].to(device)).view(-1, 1))
        else:
            vec_para_diff[id_start:shape_list[i] + id_start].copy_((para_list1[i] - para_list2[i]).view(-1, 1))

        id_start = shape_list[i] + id_start
        
#         para_list.append(param.grad.clone())
        
        
#     return torch.cat(res_list, 0).view(1,-1)
    
    

def get_all_vectorized_parameters_by_layers(para_list, layer_num):
    
    res_list = []
    
    
    for i in range(len(para_list) - 2*layer_num):
#     for param in para_list:
        param = para_list[i + 2*layer_num]
        res_list.append(param.data.contiguous().view(-1))
        
        
#         para_list.append(param.grad.clone())
        
        
    return torch.cat(res_list, 0).view(1,-1)

def compute_model_para_diff2(para_list1, para_list2):
    para_res = []
    
    for i in range(len(para_list1)):
        para_res.append(para_list1[i] - para_list2[i])
        
        
    return para_res
        




def get_regularization_term(para_list):
    
    regularization_term = 0
    
    for param in para_list:
        regularization_term += param.pow(2).sum()
        
        
    return regularization_term
        
        
        
        
        


        
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

def get_devectorized_parameters_by_layers(origin_para, params, input_dim, hidden_dims, output_dim, first_few_layers):
    
    params = params.view(-1)
    
    para_list = []
    
    pos = 0
    
    for i in range(first_few_layers):
        
#         if i <= 1:
#             pos += (input_dim+1)*hidden_dims[0]
#         else:
#             pos = pos+hidden_dims[i]*hidden_dims[i+1]
#             pos = pos+hidden_dims[i+1]
        
        para_list.append(origin_para[2*i])
        
        para_list.append(origin_para[2*i + 1])
    
#     para_list.append(params[0: input_dim*hidden_dims[0]].view(hidden_dims[0], input_dim))
#     
#     para_list.append(params[input_dim*hidden_dims[0]: (input_dim+1)*hidden_dims[0]].view(hidden_dims[0]))
#     
#     pos = (input_dim+1)*hidden_dims[0]
    
    for i in range(len(hidden_dims) - first_few_layers):
        
        j = i + first_few_layers - 1
        
        para_list.append(params[pos: pos+hidden_dims[j]*hidden_dims[j+1]].view(hidden_dims[j+1], hidden_dims[j]))
        pos = pos+hidden_dims[j]*hidden_dims[j+1]
        para_list.append(params[pos: pos+hidden_dims[j+1]].view(hidden_dims[j+1]))
        pos = pos+hidden_dims[j+1]
        
    
    
    para_list.append(params[pos: pos+ hidden_dims[-1]*output_dim].view(output_dim, hidden_dims[-1]))
    
    pos = pos+hidden_dims[-1]*output_dim
    
    para_list.append(params[pos: pos + output_dim].view(output_dim))
    
    return para_list
    

def update_and_zero_model_gradient(model, alpha):
    
#     all_parameters = list(model.parameters())
    
#     for i in range(len(all_parameters)):

    for param in model.parameters():
        param.data = param.data - alpha*param.grad
        
#         print(param.grad)
        
        param.grad.zero_()
        
        
def update_model(model, alpha, regularization_rate):
    
#     all_parameters = list(model.parameters())
    
#     for i in range(len(all_parameters)):

    para_list = []

    for param in model.parameters():
        res = param.data - alpha*param.grad - alpha*regularization_rate*param.data
        
        para_list.append(res)
        
    return para_list
        
#         print(param.grad)
        
#         param.grad.zero_()
        
#         all_parameters[i].data = all_parameters[i] - alpha*all_parameters[i].grad
        
#         all_parameters[i].grad.zero_()


# def quantize_vectors(data, epsilon):
#     
#     theta = torch.rand(data.shape, dtype = torch.double) - 0.5
#     
#     res_id = ((data - theta*epsilon)/epsilon + 0.5).type(torch.IntTensor)
#     
#     res = (res_id.type(torch.DoubleTensor) + theta)*epsilon
#     
#     print(torch.max(torch.abs(res - data)))
#     
#     return res
    
def quantize_vectors(data, epsilon):
    
    theta = (torch.rand(data.shape, dtype = torch.double) - 0.5)
    
    
#     print((data - theta*epsilon)/epsilon)
    
    ids = (data - theta*epsilon)/epsilon
    
    
    discretized_ids = ids.type(torch.IntTensor)
    
    signs = (((discretized_ids > 0).type(torch.DoubleTensor) - 0.5)*2).type(torch.IntTensor)
    
     
    res_id = (torch.abs(ids - discretized_ids.type(torch.DoubleTensor)) + 0.5).type(torch.IntTensor)*signs
    
    res_id += discretized_ids
    
#     res_id = ((data - theta*epsilon)/epsilon + 0.5).type(torch.IntTensor)
    
    res = (res_id.type(torch.DoubleTensor) + theta)*epsilon
    
#     print(res_id)
#     
#     print(torch.max(torch.abs(res - data)))
    
    return res, theta


def quantize_vectors_no_random(data, epsilon):
    
    theta = 0#(torch.rand(data.shape, dtype = torch.double) - 0.5)
    
    
#     print((data - theta*epsilon)/epsilon)
    
    ids = (data - theta*epsilon)/epsilon
    
    
    discretized_ids = ids.type(torch.IntTensor)
    
    signs = (((discretized_ids > 0).type(torch.DoubleTensor) - 0.5)*2).type(torch.IntTensor)
    
     
    res_id = (torch.abs(ids - discretized_ids.type(torch.DoubleTensor)) + 0.5).type(torch.IntTensor)*signs
    
    res_id += discretized_ids
    
#     res_id = ((data - theta*epsilon)/epsilon + 0.5).type(torch.IntTensor)
    
    res = (res_id.type(torch.DoubleTensor) + theta)*epsilon
    
#     print(res_id)
#     
#     print(torch.max(torch.abs(res - data)))
    
    return res


def quantize_vectors_incremental(data, epsilon, theta):
    
    
#     theta = (torch.rand(data.shape, dtype = torch.double) - 0.5)
    
    ids = (data - theta*epsilon)/epsilon
    
    
    discretized_ids = ids.type(torch.IntTensor)
    
    signs = (((discretized_ids > 0).type(torch.DoubleTensor) - 0.5)*2).type(torch.IntTensor)
    
     
    res_id = (torch.abs(ids - discretized_ids.type(torch.DoubleTensor)) + 0.5).type(torch.IntTensor)*signs
    
    res_id += discretized_ids
    
#     res_id = ((data - theta*epsilon)/epsilon + 0.5).type(torch.IntTensor)
    
    res = (res_id.type(torch.DoubleTensor) + theta)*epsilon
    
#     print(res_id)
#     
#     print(torch.max(torch.abs(res - data)))
    
    return res


def quantize_model_parameters(paras, epsilon):
    
    update_para_list = []
    
    
    for para in paras:
        para = quantize_vectors(para, epsilon)
    
        update_para_list.append(para)
        
    return update_para_list


    





def update_and_zero_model_gradient_quantized(model, alpha, epsilon, gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, count, train):
    
#     all_parameters = list(model.parameters())
    
#     for i in range(len(all_parameters)):

    
    random_theta_list= [] 
    
    
    para_list = []
    
    grad_list = []

    for i in range(len(list(model.parameters()))):
        
        param = list(model.parameters())[i]
        
        
#         old_grad = old_grad_list[i]
#         
#         
#         old_para = old_para_list[i]
        
        
#         print("para_diff:", i, torch.max(torch.abs(old_para - param.data)))
#         
#         print("grad_diff::", i, torch.max(torch.abs(old_grad - param.grad)))
        para_list.append(param.data.clone())
        
        param.data, theta = quantize_vectors(param.data - alpha*param.grad, epsilon)
        
        
        
        
        grad_list.append(param.grad.clone())
        
        random_theta_list.append(theta)
#         print(param.grad)
        
        param.grad.zero_()


    gradient_list_all_epochs.append(grad_list)
    
    para_list_all_epochs.append(para_list)

#     append_gradient_list(gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, model,train)

    return random_theta_list

def update_and_zero_model_gradient_quantized_incremental(model, alpha, epsilon, gradient_list_all_epochs, para_list_all_epochs, count, random_theta_list, origin_batch_size, delta_batch_size):
    
    old_grad_list = gradient_list_all_epochs[count]
    
    
    old_para_list = para_list_all_epochs[count]

    for i in range(len(list(model.parameters()))):
        
        param = list(model.parameters())[i]
        
        
        old_grad = old_grad_list[i]

        old_para = old_para_list[i]
        
        
        para_diff = torch.max(torch.abs(old_para - param.data))
        
#         print(i, torch.max(torch.abs(para_diff)))

#         if para_diff >= epsilon:
#             return False
        
#         if not old_para == param.data:
#         
#             return False
        para_before_quantization = param.data - alpha*(old_grad*origin_batch_size - param.grad*delta_batch_size)/(origin_batch_size - delta_batch_size)
        
        para_after_quantization = quantize_vectors_incremental(para_before_quantization, epsilon, random_theta_list[i])
        
        
        
        
        param.data = para_after_quantization
        
        param.grad.zero_()


    return True

def update_and_zero_model_gradient_quantized_baseline(model, alpha, epsilon, gradient_list_all_epochs, para_list_all_epochs, count, random_theta_list, origin_batch_size, delta_batch_size):
    
    old_grad_list = gradient_list_all_epochs[count]
    
    
    old_para_list = para_list_all_epochs[count]

    for i in range(len(list(model.parameters()))):
        
        param = list(model.parameters())[i]
        
        
        old_grad = old_grad_list[i]

        old_para = old_para_list[i]
        
        
        para_diff = old_para - param.data
        
        print(i, torch.max(torch.abs(para_diff)))
#         if not old_para == param.data:
#         
#             return False
        para_before_quantization = param.data - alpha*param.grad
        
        para_after_quantization = quantize_vectors_incremental(para_before_quantization, epsilon, random_theta_list[i])
        
        
        
        
        param.data = para_after_quantization
        
        param.grad.zero_()


    return True


def update_and_zero_model_gradient_quantized_incremental_no_delta(model, alpha, epsilon, gradient_list_all_epochs, para_list_all_epochs, count, random_theta_list, origin_batch_size, delta_batch_size):
    
    old_grad_list = gradient_list_all_epochs[count]
    
    
    old_para_list = para_list_all_epochs[count]

    for i in range(len(list(model.parameters()))):
        
        param = list(model.parameters())[i]
        
        
        old_grad = old_grad_list[i]

        old_para = old_para_list[i]
        
        
        para_diff = torch.max(torch.abs(old_para - param.data))
        
#         print(i, torch.max(torch.abs(para_diff)))

#         if para_diff >= epsilon:
#             return False
#         if not old_para == param.data:
#         
#             return False
        para_before_quantization = param.data - alpha*old_grad
        
        para_after_quantization = quantize_vectors_incremental(para_before_quantization, epsilon, random_theta_list[i])
        
        
        
        
        param.data = para_after_quantization
        
#         param.grad.zero_()


    return True



def zero_model_gradient(model):
    for param in model.parameters():
        param.grad.zero_()

def init_model(model, para_list):
    
    i = 0
    
    for m in model.parameters():
        
        
        
        m.data.copy_(para_list[i])
        if m.grad is not None:
            m.grad.zero_()
        m.requires_grad= True
        i += 1
        
#     model.zero_grad()


def get_model_para(model):
    
    para_list = []
    
    for param in model.parameters():
        print(param.data.shape)
        para_list.append(param.data.clone())
        
    return para_list

def print_model_para(model):
    
    for param in model.parameters():
#         print(param.data.shape)
        print(param.data)



def sigmoid_diff_function(x):
    return np.exp(-x)/(np.power(1+np.exp(-x), 2))


def model_update_provenance_by_dual(alpha, dim, dual_para_list, hessian_matrix, origin_gradient_list, vectorized_orign_params, epoch, input_dim, hidden_dims, output_dim, delta_ids, expected_gradient_list_all_epochs, epxected_para_list_all_epochs_all_epochs):
    
    
    vectorized_origin_gradient = get_all_vectorized_parameters(origin_gradient_list)
    
    res_vectorized_paras = vectorized_orign_params.clone()
    
    n = dim[0]
    
    
    k = delta_ids.shape[0]
    
    
    
    for i in range(epoch):
        
#         expected_para = get_all_vectorized_parameters(epxected_para_list_all_epochs_all_epochs[i])
        
        dual_vectorized_paras = get_all_vectorized_parameters(dual_para_list[i])
        
        dual_vectorized_paras_next_epoch = get_all_vectorized_parameters(dual_para_list[i+1])
        
        res_vectorized_paras = (0 - (n-k)*res_vectorized_paras - k *dual_vectorized_paras - alpha*(torch.mm((vectorized_orign_params - res_vectorized_paras), hessian_matrix)) + k*dual_vectorized_paras_next_epoch)/(k-n)
    
#         expected_grad = get_all_vectorized_parameters(expected_gradient_list_all_epochs[i])
    
    return get_devectorized_parameters(res_vectorized_paras, input_dim, hidden_dims, output_dim)


def update_hessian(hessian_matrix, old_para, new_para, old_gradient, new_gradient):
    
    y = get_all_vectorized_parameters(new_gradient) - get_all_vectorized_parameters(old_gradient)
    
    s = get_all_vectorized_parameters(new_para) - get_all_vectorized_parameters(old_para)
    
    
    
    updated_hessian_matrix = hessian_matrix + torch.mm(torch.t(y), y)/(torch.mm(y, torch.t(s))) - (torch.mm(torch.mm(hessian_matrix, torch.t(s)), torch.mm(s, torch.t(hessian_matrix)))/torch.mm(torch.mm(s, (hessian_matrix)), torch.t(s)))

    
    return updated_hessian_matrix
    
def cal_approx_hessian_vec_prod3(i, m, v_vec, para_list_all_epochs, gradient_list_all_epoch):
    
    curr_S_k = torch.zeros([v_vec.shape[0],m], dtype = torch.double)#S_k_list[:,i-m:i]
        
    curr_Y_k = torch.zeros([v_vec.shape[0],m], dtype = torch.double)
    
    for k in range(m):
        curr_S_k[:,k] = get_all_vectorized_parameters(para_list_all_epochs[i-(k+1)]) - get_all_vectorized_parameters(para_list_all_epochs[i])
    
        curr_Y_k[:,k] = get_all_vectorized_parameters(gradient_list_all_epoch[i-(k+1)]) - get_all_vectorized_parameters(gradient_list_all_epoch[i])
    
    
    res = torch.mm(torch.inverse(torch.mm(torch.t(curr_S_k), curr_S_k)), torch.mm(torch.t(curr_S_k), v_vec))
    
    return torch.mm(curr_Y_k, res.view(-1,1))





def cal_approx_hessian_vec_prod4(truncted_s, extended_Y_k_list, i, m, v_vec):
    
#     curr_S_k = torch.zeros([v_vec.shape[0],m], dtype = torch.double)#S_k_list[:,i-m:i]
#         
#     curr_Y_k = torch.zeros([v_vec.shape[0],m], dtype = torch.double)
#     
#     for k in range(m):
#         curr_S_k[:,k] = get_all_vectorized_parameters(para_list_all_epochs[i-(k+1)]) - get_all_vectorized_parameters(para_list_all_epochs[i])
#     
#         curr_Y_k[:,k] = get_all_vectorized_parameters(gradient_list_all_epoch[i-(k+1)]) - get_all_vectorized_parameters(gradient_list_all_epoch[i])
    
    
    res = torch.mm(torch.inverse(torch.mm(torch.t(truncted_s), truncted_s)), torch.mm(torch.t(truncted_s), v_vec))
    
    print(res)
    
    print(torch.norm(torch.mm(truncted_s, res) - v_vec))
    
    return torch.mm(extended_Y_k_list, res.view(-1,1)), torch.mm(truncted_s, res)


    
#     results = np.linalg.solve(curr_S_k.numpy(), v_vec.numpy())
#     
#     results_tensor = torch.from_numpy(results)
#     
#     
#     return torch.mm(curr_Y_k, results_tensor.view(-1,1))
    
    
    
    
    


def cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec):
    
    curr_S_k = S_k_list[:,i-m:i]
        
    curr_Y_k = Y_k_list[:,i-m:i]
    
    S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
    
    
    S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)
    
    
    R_k = torch.triu(S_k_time_Y_k)
    
    L_k = S_k_time_Y_k - R_k
    
    D_k_diag = torch.diag(S_k_time_Y_k)
    
    
    sigma_k = torch.dot(Y_k_list[:,i-1],S_k_list[:,i-1])/(torch.dot(S_k_list[:,i-1], S_k_list[:,i-1]))
    
    
    interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
    
    J_k = torch.from_numpy(np.linalg.cholesky(interm.detach().numpy())).type(torch.DoubleTensor)
    
    
#     v_vec = S_k_list[:,i-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
    
#         v_vec = torch.rand([para_num, 1], dtype = torch.double)
    
    
    p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
    
    
    D_k_sqr_root = torch.pow(D_k_diag, 0.5)
    
    D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
    
    upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
    
    lower_mat_1 = torch.cat([torch.zeros([m, m], dtype = torch.double), torch.t(J_k)], dim = 1)
    
    
    mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
    
    
    upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([m, m], dtype = torch.double)], dim = 1)
    
    lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
    
    mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
    
    
    p_mat = torch.mm(torch.inverse(mat_1), torch.mm(torch.inverse(mat_2), p_mat))
    
    
    approx_prod = sigma_k*v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    
    return approx_prod






def prepare_hessian_vec_prod0(S_k_list, Y_k_list, i, m, k, period):
 
 
#     t3  = time.time()
    
#     period_num = int(i/period)
#     
#     
#     ids = torch.tensor(range(m)).view(-1)
#     
#     if period_num > 0:
#         ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
# #     else:
# #         ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
#     ids = ids - 1
#     
#     ids = ids[ids >= 0]
#     
#     if ids.shape[0] > k:
#         ids = ids[-k:]
    
#     if i-k >= 1:
#         lb = i-k
#         
#         zero_mat_dim = ids.shape[0] + k
#         
#     else:
#         lb = 1
#         
#         zero_mat_dim = ids.shape[0] + i-1
    zero_mat_dim = k#ids.shape[0]
    
    
    
#     curr_S_k = torch.cat([S_k_list[:, ids],S_k_list[:,lb:i]], dim=1) 
#           
#     curr_Y_k = torch.cat([Y_k_list[:, ids],Y_k_list[:,lb:i]], dim=1)

#     print(ids)

#     curr_S_k = torch.cat([S_k_list[:, ids]], dim=1) 
#           
#     curr_Y_k = torch.cat([Y_k_list[:, ids]], dim=1)

    curr_S_k = torch.stack(list(S_k_list), dim = 0)
          
    curr_Y_k = torch.stack(list(Y_k_list), dim = 0)
#     curr_S_k = S_k_list[:,k:m] 
#          
#     curr_Y_k = Y_k_list[:,k:m] 
    
    S_k_time_Y_k = torch.mm(curr_S_k, torch.t(curr_Y_k))
    
    
    S_k_time_S_k = torch.mm(curr_S_k, torch.t(curr_S_k))
    
    
    R_k = np.triu(S_k_time_Y_k.numpy())
    
    L_k = S_k_time_Y_k - torch.from_numpy(R_k)
    
    D_k_diag = torch.diag(S_k_time_Y_k)
    
    
    sigma_k = torch.dot(Y_k_list[-1],S_k_list[-1])/(torch.dot(S_k_list[-1], S_k_list[-1]))
    
    
    interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
    
    J_k = torch.from_numpy(np.linalg.cholesky(interm.detach().numpy())).type(torch.DoubleTensor)
    
    
#     v_vec = S_k_list[:,i-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
    
#         v_vec = torch.rand([para_num, 1], dtype = torch.double)
#     t1 = time.time()
    
#     p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double)
#     
# #     tmp = 
#     
#     p_mat[0:zero_mat_dim] = torch.mm(torch.t(curr_Y_k), v_vec)
#     
#     p_mat[zero_mat_dim:zero_mat_dim*2] = torch.mm(torch.t(curr_S_k), v_vec)*sigma_k
    
#     t2 = time.time()
    
#     p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
    
    
    D_k_sqr_root = torch.pow(D_k_diag, 0.5)
    
    D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
    
    upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
    
    lower_mat_1 = torch.cat([torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double), torch.t(J_k)], dim = 1)
    
    
    mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
    
    
    upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double)], dim = 1)
    
    lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
    
    mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
    
    
#     p_mat = torch.mm(torch.inverse(mat_1), torch.mm(torch.inverse(mat_2), p_mat))
#     
#     
#     approx_prod = sigma_k*v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    
#     t4  = time.time()
    
    
#     print('time1::', t4 - t3)
#     
#     print('key time::', t2 - t1)
    
    
    return zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2

# def prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, para_vec):
#     
#     sigma_k = torch.dot(Y_k_list[0],S_k_list[0])/(torch.dot(S_k_list[0], S_k_list[0]))
#     
#     B_k_vec_prod = sigma_k*para_vec
#     
#     B_k_s_k_prod = sigma_k*S_k_list[0]
#     
#     for i in range(len(S_k_list)):
#         B_k_vec_prod = B_k_vec_prod - torch.mm(B_k_s_k_prod, 
#     
    

def prepare_hessian_vec_prod0_2(S_k_list, Y_k_list, i, m, k):
 
 
#     t3  = time.time()
    
#     period_num = int(i/period)
#     
#     
#     ids = torch.tensor(range(m)).view(-1)
#     
#     if period_num > 0:
#         ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
# #     else:
# #         ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
#     ids = ids - 1
#     
#     ids = ids[ids >= 0]
#     
#     if ids.shape[0] > k:
#         ids = ids[-k:]
    
#     if i-k >= 1:
#         lb = i-k
#         
#         zero_mat_dim = ids.shape[0] + k
#         
#     else:
#         lb = 1
#         
#         zero_mat_dim = ids.shape[0] + i-1
    zero_mat_dim = k#ids.shape[0]
    
    
    
#     curr_S_k = torch.cat([S_k_list[:, ids],S_k_list[:,lb:i]], dim=1) 
#           
#     curr_Y_k = torch.cat([Y_k_list[:, ids],Y_k_list[:,lb:i]], dim=1)

#     print(ids)

#     curr_S_k = torch.cat([S_k_list[:, ids]], dim=1) 
#           
#     curr_Y_k = torch.cat([Y_k_list[:, ids]], dim=1)

    curr_S_k = torch.stack(list(S_k_list), dim = 0)
          
    curr_Y_k = torch.stack(list(Y_k_list), dim = 0)
#     curr_S_k = S_k_list[:,k:m] 
#          
#     curr_Y_k = Y_k_list[:,k:m] 
    
    S_k_time_Y_k = torch.mm(curr_S_k, torch.t(curr_Y_k))
    
    
    S_k_time_S_k = torch.mm(curr_S_k, torch.t(curr_S_k))
    
    
    R_k = np.triu(S_k_time_Y_k.numpy())
    
    L_k = S_k_time_Y_k - torch.from_numpy(R_k)
    
    D_k_diag = torch.diag(S_k_time_Y_k)
    
    
    sigma_k = torch.dot(Y_k_list[-1],S_k_list[-1])/(torch.dot(S_k_list[-1], S_k_list[-1]))
    
    
#     interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
    
#     J_k = torch.from_numpy(np.linalg.cholesky(interm.detach().numpy())).type(torch.DoubleTensor)
    
    
#     v_vec = S_k_list[:,i-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
    
#         v_vec = torch.rand([para_num, 1], dtype = torch.double)
#     t1 = time.time()
    
#     p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double)
#     
# #     tmp = 
#     
#     p_mat[0:zero_mat_dim] = torch.mm(torch.t(curr_Y_k), v_vec)
#     
#     p_mat[zero_mat_dim:zero_mat_dim*2] = torch.mm(torch.t(curr_S_k), v_vec)*sigma_k
    
#     t2 = time.time()
    
#     p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
    
    
#     D_k_sqr_root = torch.pow(D_k_diag, 0.5)
#     
#     D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
    
    upper_mat = torch.cat([-torch.diag(D_k_diag), torch.t(L_k)], dim = 1)
    
    lower_mat = torch.cat([L_k, sigma_k*S_k_time_S_k], dim = 1)
    
    mat = torch.cat([upper_mat, lower_mat], dim = 0)
    
#     upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
#     
#     lower_mat_1 = torch.cat([torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double), torch.t(J_k)], dim = 1)
#     
#     
#     mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
#     
#     
#     upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double)], dim = 1)
#     
#     lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
#     
#     mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
    
    
#     p_mat = torch.mm(torch.inverse(mat_1), torch.mm(torch.inverse(mat_2), p_mat))
#     
#     
#     approx_prod = sigma_k*v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    
#     t4  = time.time()
    
    
#     print('time1::', t4 - t3)
#     
#     print('key time::', t2 - t1)
    
    
    return zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat


# def prepare_hessian_vec_prod0_2(S_k_list, Y_k_list, i, m, k, period):
#  
#  
# #     t3  = time.time()
# 
# 
#     period_num = int(i/period)
#     
#     
#     ids = torch.tensor(range(m)).view(-1)
#     
#     if period_num > 0:
#         ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
# #     else:
# #         ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
#     ids = ids - 1
#     
#     ids = ids[ids >= 0]
#     
#     if ids.shape[0] > k:
#         ids = ids[-k:]
#     
# #     if i-k >= 1:
# #         lb = i-k
# #         
# #         zero_mat_dim = ids.shape[0] + k
# #         
# #     else:
# #         lb = 1
# #         
# #         zero_mat_dim = ids.shape[0] + i-1
#     zero_mat_dim = ids.shape[0]
# 
#     curr_S_k_list = []
#     
#     curr_Y_k_list = []
#     
#     sigma_k = 0
#     
#     mat_1 = 0
#     
#     mat_2 = 0
#     
#     
#     S_k_time_Y_k = 0
#     
#     
#     S_k_time_S_k = 0
#     
# #     for id in ids:
# #         
# #     
# #         this_S_k_list = []
# #         
# #         this_Y_k_list = []
#     
#     last_S_k_time_S_k = 0
#     
#     last_S_k_time_Y_k = 0
#     
#     
#     for r in range(len(S_k_list)):
#     
#         curr_S_k = S_k_list[r][:, ids]
#         
#         
# #         this_S_k_list.append(curr_S_k)
#         
#             
#             
#         curr_Y_k = Y_k_list[r][:, ids]
#         
#         
#         S_k_time_S_k += torch.mm(torch.t(curr_S_k), curr_S_k)
#         
#         S_k_time_Y_k += torch.mm(torch.t(curr_S_k), curr_Y_k)
#         
#         
#         
#         last_S_k_time_Y_k += torch.dot(Y_k_list[r][:,ids[-1]],S_k_list[r][:,ids[-1]])
#         
#         last_S_k_time_S_k += torch.dot(S_k_list[r][:,ids[-1]], S_k_list[r][:,ids[-1]])
# #         this_Y_k_list.append(curr_Y_k)
# # 
# #         curr_S_k_list.append(torch.cat(this_S_k_list, dim = 1))
# #         
# #         curr_Y_k_list.append(torch.cat(this_Y_k_list, dim = 1))
#     
#     
# #     curr_S_k = torch.cat(curr_S_k_list, dim = 1)
# #     
# #     curr_Y_k = torch.cat(curr_Y_k_list, dim = 1)
#     
#     
# #     curr_S_k = torch.cat([S_k_list[:, ids],S_k_list[:,lb:i]], dim=1) 
# #           
# #     curr_Y_k = torch.cat([Y_k_list[:, ids],Y_k_list[:,lb:i]], dim=1)
# 
# #     print(ids)
# 
# #     curr_S_k = torch.cat([S_k_list[:, ids]], dim=1) 
# #           
# #     curr_Y_k = torch.cat([Y_k_list[:, ids]], dim=1)
# 
# #         curr_S_k = S_k_list[r][:, ids]
# #               
# #         curr_Y_k = Y_k_list[r][:, ids]
# #     curr_S_k = S_k_list[:,k:m] 
# #          
# #     curr_Y_k = Y_k_list[:,k:m] 
#     
# #     S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
# # 
# # 
# #     S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)
# 
# 
#     R_k = np.triu(S_k_time_Y_k.numpy())
# 
#     L_k = S_k_time_Y_k - torch.from_numpy(R_k)
# 
#     D_k_diag = torch.diag(S_k_time_Y_k)
# 
# 
# #     sigma_k = torch.dot(Y_k_list[:,ids[-1]],S_k_list[:,ids[-1]])/(torch.dot(S_k_list[:,ids[-1]], S_k_list[:,ids[-1]]))
# 
#     sigma_k = last_S_k_time_Y_k/last_S_k_time_S_k
# 
# 
#     interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
# 
#     J_k = torch.from_numpy(np.linalg.cholesky(interm.detach().numpy())).type(torch.DoubleTensor)
# 
# 
# #     v_vec = S_k_list[:,i-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
# 
# #         v_vec = torch.rand([para_num, 1], dtype = torch.double)
# #     t1 = time.time()
# 
# #     p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double)
# #     
# # #     tmp = 
# #     
# #     p_mat[0:zero_mat_dim] = torch.mm(torch.t(curr_Y_k), v_vec)
# #     
# #     p_mat[zero_mat_dim:zero_mat_dim*2] = torch.mm(torch.t(curr_S_k), v_vec)*sigma_k
# 
# #     t2 = time.time()
# 
# #     p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
# 
# 
#     D_k_sqr_root = torch.pow(D_k_diag, 0.5)
#     
#     D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
#     
#     upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
#     
#     lower_mat_1 = torch.cat([torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double), torch.t(J_k)], dim = 1)
#     
#     
#     mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
#     
#     
#     upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double)], dim = 1)
#     
#     lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
#     
#     mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
#     
#     
# #     p_mat = torch.mm(torch.inverse(mat_1), torch.mm(torch.inverse(mat_2), p_mat))
# #     
# #     
# #     approx_prod = sigma_k*v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
#     
#     
# #     t4  = time.time()
#     
#     
# #     print('time1::', t4 - t3)
# #     
# #     print('key time::', t2 - t1)
#     
#     
#     return zero_mat_dim, sigma_k, mat_1, mat_2, ids


# def prepare_hessian_vec_prod0_2(S_k_list, Y_k_list, i, m, k, period):
#  
#  
# #     t3  = time.time()
# 
# 
#     period_num = int(i/period)
#     
#     
#     ids = torch.tensor(range(m)).view(-1)
#     
#     if period_num > 0:
#         ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
# #     else:
# #         ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
#     ids = ids - 1
#     
#     ids = ids[ids >= 0]
#     
#     if ids.shape[0] > k:
#         ids = ids[-k:]
#     
# #     if i-k >= 1:
# #         lb = i-k
# #         
# #         zero_mat_dim = ids.shape[0] + k
# #         
# #     else:
# #         lb = 1
# #         
# #         zero_mat_dim = ids.shape[0] + i-1
#     zero_mat_dim = ids.shape[0]
# 
#     curr_S_k_list = []
#     
#     curr_Y_k_list = []
#     
#     sigma_k = 0
#     
#     mat_1 = 0
#     
#     mat_2 = 0
#     
#     
#     S_k_time_Y_k = 0
#     
#     
#     S_k_time_S_k = 0
#     
# #     for id in ids:
# #         
# #     
# #         this_S_k_list = []
# #         
# #         this_Y_k_list = []
#     
#     last_S_k_time_S_k = 0
#     
#     last_S_k_time_Y_k = 0
#     
#     
#     for r in range(len(S_k_list)):
#     
#         curr_S_k = S_k_list[r][:, ids]
#         
#         
# #         this_S_k_list.append(curr_S_k)
#         
#             
#             
#         curr_Y_k = Y_k_list[r][:, ids]
#         
#         
#         S_k_time_S_k += torch.mm(torch.t(curr_S_k), curr_S_k)
#         
#         S_k_time_Y_k += torch.mm(torch.t(curr_S_k), curr_Y_k)
#         
#         
#         
#         last_S_k_time_Y_k += torch.dot(Y_k_list[r][:,ids[-1]],S_k_list[r][:,ids[-1]])
#         
#         last_S_k_time_S_k += torch.dot(S_k_list[r][:,ids[-1]], S_k_list[r][:,ids[-1]])
# #         this_Y_k_list.append(curr_Y_k)
# # 
# #         curr_S_k_list.append(torch.cat(this_S_k_list, dim = 1))
# #         
# #         curr_Y_k_list.append(torch.cat(this_Y_k_list, dim = 1))
#     
#     
# #     curr_S_k = torch.cat(curr_S_k_list, dim = 1)
# #     
# #     curr_Y_k = torch.cat(curr_Y_k_list, dim = 1)
#     
#     
# #     curr_S_k = torch.cat([S_k_list[:, ids],S_k_list[:,lb:i]], dim=1) 
# #           
# #     curr_Y_k = torch.cat([Y_k_list[:, ids],Y_k_list[:,lb:i]], dim=1)
# 
# #     print(ids)
# 
# #     curr_S_k = torch.cat([S_k_list[:, ids]], dim=1) 
# #           
# #     curr_Y_k = torch.cat([Y_k_list[:, ids]], dim=1)
# 
# #         curr_S_k = S_k_list[r][:, ids]
# #               
# #         curr_Y_k = Y_k_list[r][:, ids]
# #     curr_S_k = S_k_list[:,k:m] 
# #          
# #     curr_Y_k = Y_k_list[:,k:m] 
#     
# #     S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
# # 
# # 
# #     S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)
# 
# 
#     R_k = np.triu(S_k_time_Y_k.numpy())
# 
#     L_k = S_k_time_Y_k - torch.from_numpy(R_k)
# 
#     D_k_diag = torch.diag(S_k_time_Y_k)
# 
# 
# #     sigma_k = torch.dot(Y_k_list[:,ids[-1]],S_k_list[:,ids[-1]])/(torch.dot(S_k_list[:,ids[-1]], S_k_list[:,ids[-1]]))
# 
#     sigma_k = last_S_k_time_Y_k/last_S_k_time_S_k
# 
# 
#     interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
# 
#     J_k = torch.from_numpy(np.linalg.cholesky(interm.detach().numpy())).type(torch.DoubleTensor)
# 
# 
# #     v_vec = S_k_list[:,i-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
# 
# #         v_vec = torch.rand([para_num, 1], dtype = torch.double)
# #     t1 = time.time()
# 
# #     p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double)
# #     
# # #     tmp = 
# #     
# #     p_mat[0:zero_mat_dim] = torch.mm(torch.t(curr_Y_k), v_vec)
# #     
# #     p_mat[zero_mat_dim:zero_mat_dim*2] = torch.mm(torch.t(curr_S_k), v_vec)*sigma_k
# 
# #     t2 = time.time()
# 
# #     p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
# 
# 
#     D_k_sqr_root = torch.pow(D_k_diag, 0.5)
#     
#     D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
#     
#     upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
#     
#     lower_mat_1 = torch.cat([torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double), torch.t(J_k)], dim = 1)
#     
#     
#     mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
#     
#     
#     upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double)], dim = 1)
#     
#     lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
#     
#     mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
#     
#     
# #     p_mat = torch.mm(torch.inverse(mat_1), torch.mm(torch.inverse(mat_2), p_mat))
# #     
# #     
# #     approx_prod = sigma_k*v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
#     
#     
# #     t4  = time.time()
#     
#     
# #     print('time1::', t4 - t3)
# #     
# #     print('key time::', t2 - t1)
#     
#     
#     return zero_mat_dim, sigma_k, mat_1, mat_2, ids



def prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, m, k, is_GPU, device):
 
 
    zero_mat_dim = k#ids.shape[0]
    

    curr_S_k = torch.cat(list(S_k_list), dim = 0)
          
    curr_Y_k = torch.cat(list(Y_k_list), dim = 0)
    
    S_k_time_Y_k = torch.mm(curr_S_k, torch.t(curr_Y_k))
    
    
    S_k_time_S_k = torch.mm(curr_S_k, torch.t(curr_S_k))
    
    
    if is_GPU:
        R_k = np.triu(S_k_time_Y_k.to('cpu').numpy())
        L_k = S_k_time_Y_k - torch.from_numpy(R_k).to(device)
    else:
        R_k = np.triu(S_k_time_Y_k.numpy())
        L_k = S_k_time_Y_k - torch.from_numpy(R_k)
    
    
    
    
    
    
    D_k_diag = torch.diag(S_k_time_Y_k)
    
    
    sigma_k = torch.mm(Y_k_list[-1],torch.t(S_k_list[-1]))/(torch.mm(S_k_list[-1], torch.t(S_k_list[-1])))
    
    upper_mat = torch.cat([-torch.diag(D_k_diag), torch.t(L_k)], dim = 1)
    
    lower_mat = torch.cat([L_k, sigma_k*S_k_time_S_k], dim = 1)
    
    mat = torch.cat([upper_mat, lower_mat], dim = 0)
    
    
    return zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat



def compute_approx_hessian_vector_prod_with_prepared_terms(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, v_vec):
    
    
    p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double)
     
#     tmp = 
     
#     curr_Y_k_transpose = torch.t(curr_Y_k)
#     
#     curr_S_k_transpose = torch.t(curr_S_k)
     
#     p_mat[0:zero_mat_dim] = curr_Y_k_transpose@v_vec.view(-1, 1)
    
#     torch.sum((curr_Y_k_transpose*v_vec), dim = 1, out = p_mat[0:zero_mat_dim]) 
    
    torch.mm(curr_Y_k, v_vec, out = p_mat[0:zero_mat_dim])
    
#     torch.sum(curr_S_k_transpose*v_vec*sigma_k, dim = 1, out = p_mat[zero_mat_dim:zero_mat_dim*2])
    
    
    torch.mm(curr_S_k, v_vec*sigma_k, out = p_mat[zero_mat_dim:zero_mat_dim*2])

    p_mat = torch.mm(mat, p_mat)
    
#     print(curr_Y_k_transpose.shape, curr_S_k_transpose.shape, v_vec.shape)
    
    approx_prod = sigma_k*v_vec
    
#     approx_prod -= torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
#     print((torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1)).shape)
#     
#     print((torch.sum((torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1))*p_mat.view(1, -1), dim = 1)).shape)
#     
#     print(approx_prod.shape)
    
    
    
#     approx_prod -= torch.sum((torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1))*p_mat.view(1, -1), dim = 1).view(-1,1)
    
    approx_prod -= (torch.mm(torch.t(curr_Y_k), p_mat[0:zero_mat_dim]) + torch.mm(sigma_k*torch.t(curr_S_k), p_mat[zero_mat_dim:zero_mat_dim*2]))
    
    
#     approx_prod -= torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    
    return approx_prod

def compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, v_vec, is_GPU, device):
    
    
    if is_GPU:
        p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double, device =device)
    else:
        p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double)
     
    
    torch.mm(curr_Y_k, v_vec, out = p_mat[0:zero_mat_dim])
    
#     torch.sum(curr_S_k_transpose*v_vec*sigma_k, dim = 1, out = p_mat[zero_mat_dim:zero_mat_dim*2])
    
    
    torch.mm(curr_S_k, v_vec*sigma_k, out = p_mat[zero_mat_dim:zero_mat_dim*2])

#     torch.mm(mat, p_mat, out = p_mat)

    p_mat = torch.mm(mat, p_mat)
    
#     print(curr_Y_k_transpose.shape, curr_S_k_transpose.shape, v_vec.shape)
    
    approx_prod = sigma_k*v_vec
    
#     approx_prod -= torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
#     print((torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1)).shape)
#     
#     print((torch.sum((torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1))*p_mat.view(1, -1), dim = 1)).shape)
#     
#     print(approx_prod.shape)
    
    
    
#     approx_prod -= torch.sum((torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1))*p_mat.view(1, -1), dim = 1).view(-1,1)
    
    approx_prod -= (torch.mm(torch.t(curr_Y_k), p_mat[0:zero_mat_dim]) + torch.mm(sigma_k*torch.t(curr_S_k), p_mat[zero_mat_dim:zero_mat_dim*2]))
    
    
#     approx_prod -= torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    
    return approx_prod


def compute_approx_hessian_vector_prod_with_prepared_terms2(ids, zero_mat_dim, Y_k_list, S_k_list, sigma_k, mat_1, mat_2, v_vec):
    
    
    p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double)
     
#     tmp = 
    approx_prod = []
     
    for k in range(len(S_k_list)): 
     
     
        v_vec_para = v_vec[k]
     
    
        curr_Y_k_transpose = torch.t(Y_k_list[k][:,ids])
        
        curr_S_k_transpose = torch.t(S_k_list[k][:,ids])
     
#     p_mat[0:zero_mat_dim] = curr_Y_k_transpose@v_vec.view(-1, 1)
    
    
        p_mat[0:zero_mat_dim] += torch.sum(curr_Y_k_transpose*v_vec_para.view(1,-1), dim = 1).view(-1,1)
            
        p_mat[zero_mat_dim:zero_mat_dim*2] += torch.sum(curr_S_k_transpose*v_vec_para.view(1,-1)*sigma_k, dim = 1).view(-1,1)

    
#         id_start = 0
        
    #     torch.sum((curr_Y_k_transpose*v_vec.view(-1)), dim = 1, out = p_mat[0:zero_mat_dim]) 
        
    #     torch.mm(curr_Y_k_transpose, v_vec, out = p_mat[0:zero_mat_dim])
        
    #     torch.sum(curr_S_k_transpose*v_vec.view(-1)*sigma_k, dim = 1, out = p_mat[zero_mat_dim:zero_mat_dim*2])
        
        
    #     torch.mm(curr_S_k_transpose, v_vec*sigma_k, out = p_mat[zero_mat_dim:zero_mat_dim*2])
    
    p_mat = torch.mm(torch.inverse(mat_1), torch.mm(torch.inverse(mat_2), p_mat))
        
    #     print(curr_Y_k_transpose.shape, curr_S_k_transpose.shape, v_vec.shape)
        
        
        
#         curr_approx_prod = torch.zeros([curr_Y_k.shape[0], 1], dtype = torch.double)
        
#         for i in range(len(v_vec)):
            
#             v_vec_para = v_vec[i]
    for k in range(len(S_k_list)): 
        
        v_vec_para = v_vec[k]
            
        curr_approx_prod = sigma_k*v_vec_para.view(-1,1)
            
    #     approx_prod -= torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
        
    #     print((torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1)).shape)
    #     
    #     print((torch.sum((torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1))*p_mat.view(1, -1), dim = 1)).shape)
    #     
    #     print(approx_prod.shape)
        
        
        
    #     approx_prod -= torch.sum((torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1))*p_mat.view(1, -1), dim = 1).view(-1,1)
        
        curr_approx_prod -= (torch.mm(Y_k_list[k][:,ids],p_mat[0:zero_mat_dim]) + (torch.mm(sigma_k*S_k_list[k][:,ids], p_mat[zero_mat_dim:zero_mat_dim*2]))) 
        
        approx_prod.append(curr_approx_prod)
        
#         curr_approx_prod -= torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
        
#         approx_prod -= torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    
    return approx_prod


def cal_approx_hessian_vec_prod0_2(S_k_list, Y_k_list, i, m, k, v_vec, period):
 
 
#     t3  = time.time()
    
    period_num = int(i/period)
    
    
    ids = torch.tensor(range(m)).view(-1)
    
    if period_num > 0:
        ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
#     else:
#         ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
    ids = ids - 1
    
    ids = ids[ids >= 0]
    
    if ids.shape[0] > k:
        ids = ids[-k:]
    
#     if i-k >= 1:
#         lb = i-k
#         
#         zero_mat_dim = ids.shape[0] + k
#         
#     else:
#         lb = 1
#         
#         zero_mat_dim = ids.shape[0] + i-1
    zero_mat_dim = k#ids.shape[0]
    
    
    
#     curr_S_k = torch.cat([S_k_list[:, ids],S_k_list[:,lb:i]], dim=1) 
#           
#     curr_Y_k = torch.cat([Y_k_list[:, ids],Y_k_list[:,lb:i]], dim=1)

#     print(ids)

#     curr_S_k = torch.cat([S_k_list[:, ids]], dim=1) 
#           
#     curr_Y_k = torch.cat([Y_k_list[:, ids]], dim=1)

#     curr_S_k = S_k_list[:, ids]
#           
#     curr_Y_k = Y_k_list[:, ids]
    
    curr_S_k = torch.t(torch.stack(list(S_k_list), dim = 0))
          
    curr_Y_k = torch.t(torch.stack(list(Y_k_list), dim = 0))
    
#     curr_S_k = S_k_list[:,k:m] 
#          
#     curr_Y_k = Y_k_list[:,k:m] 
    
    S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
    
    
    S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)
    
    
    R_k = np.triu(S_k_time_Y_k.numpy())
    
    L_k = S_k_time_Y_k - torch.from_numpy(R_k)
    
    D_k_diag = torch.diag(S_k_time_Y_k)
    
    
    sigma_k = torch.dot(Y_k_list[-1],S_k_list[-1])/(torch.dot(S_k_list[-1], S_k_list[-1]))
    
    
#     interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
#     
#     J_k = torch.from_numpy(np.linalg.cholesky(interm.detach().numpy())).type(torch.DoubleTensor)
    
    
#     v_vec = S_k_list[:,i-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
    
#         v_vec = torch.rand([para_num, 1], dtype = torch.double)
#     t1 = time.time()
    
    p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double)
    
    tmp = torch.mm(torch.t(curr_Y_k), v_vec)
    
    p_mat[0:zero_mat_dim] = tmp
    
    p_mat[zero_mat_dim:zero_mat_dim*2] = torch.mm(torch.t(curr_S_k), v_vec)*sigma_k
    
#     t2 = time.time()
    
#     p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
    
    
#     D_k_sqr_root = torch.pow(D_k_diag, 0.5)
#     
#     D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
#     
#     upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
#     
#     lower_mat_1 = torch.cat([torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double), torch.t(J_k)], dim = 1)
#     
#     
#     mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
#     
#     
#     upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double)], dim = 1)
#     
#     lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
#     
#     mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
    
    
    
    
    
    upper_mat = torch.cat([-torch.diag(D_k_diag), torch.t(L_k)], dim = 1)
    
    lower_mat = torch.cat([L_k, sigma_k*S_k_time_S_k], dim = 1)
    
    mat = torch.cat([upper_mat, lower_mat], dim = 0)
    
    
    
    p_mat = torch.mm(torch.inverse(mat), p_mat)
    
    
    approx_prod = sigma_k*v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    
#     t4  = time.time()
    
    
#     print('time1::', t4 - t3)
#     
#     print('key time::', t2 - t1)
    
    
    return approx_prod,zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat


def cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, i, m, k, v_vec, period, is_GPU, device):
 
 
#     t3  = time.time()
    
    period_num = int(i/period)
    
    
    ids = torch.tensor(range(m)).view(-1)
    
    if period_num > 0:
        ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
#     else:
#         ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
    ids = ids - 1
    
    ids = ids[ids >= 0]
    
    if ids.shape[0] > k:
        ids = ids[-k:]
    
#     if i-k >= 1:
#         lb = i-k
#         
#         zero_mat_dim = ids.shape[0] + k
#         
#     else:
#         lb = 1
#         
#         zero_mat_dim = ids.shape[0] + i-1
    zero_mat_dim = k#ids.shape[0]
    
    
    
#     curr_S_k = torch.cat([S_k_list[:, ids],S_k_list[:,lb:i]], dim=1) 
#           
#     curr_Y_k = torch.cat([Y_k_list[:, ids],Y_k_list[:,lb:i]], dim=1)

#     print(ids)

#     curr_S_k = torch.cat([S_k_list[:, ids]], dim=1) 
#           
#     curr_Y_k = torch.cat([Y_k_list[:, ids]], dim=1)

#     curr_S_k = S_k_list[:, ids]
#           
#     curr_Y_k = Y_k_list[:, ids]
    
    curr_S_k = torch.t(torch.cat(list(S_k_list), dim = 0))
          
    curr_Y_k = torch.t(torch.cat(list(Y_k_list), dim = 0))
    
#     curr_S_k = S_k_list[:,k:m] 
#          
#     curr_Y_k = Y_k_list[:,k:m] 
    
    S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
    
    
    S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)
    
    
    if is_GPU:
        
        R_k = np.triu(S_k_time_Y_k.to('cpu').numpy())
        
        L_k = S_k_time_Y_k - torch.from_numpy(R_k).to(device)
        
    else:
        R_k = np.triu(S_k_time_Y_k.numpy())
    
        L_k = S_k_time_Y_k - torch.from_numpy(R_k)
    
    D_k_diag = torch.diag(S_k_time_Y_k)
    
    
    sigma_k = torch.mm(Y_k_list[-1],torch.t(S_k_list[-1]))/(torch.mm(S_k_list[-1], torch.t(S_k_list[-1])))
    
    
#     interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
#     
#     J_k = torch.from_numpy(np.linalg.cholesky(interm.detach().numpy())).type(torch.DoubleTensor)
    
    
#     v_vec = S_k_list[:,i-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
    
#         v_vec = torch.rand([para_num, 1], dtype = torch.double)
#     t1 = time.time()
    if is_GPU:
        p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double, device = device)
    else:
        p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double)
    
    tmp = torch.mm(torch.t(curr_Y_k), v_vec)
    
    p_mat[0:zero_mat_dim] = tmp
    
    p_mat[zero_mat_dim:zero_mat_dim*2] = torch.mm(torch.t(curr_S_k), v_vec)*sigma_k
    
#     t2 = time.time()
    
#     p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
    
    
#     D_k_sqr_root = torch.pow(D_k_diag, 0.5)
#     
#     D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
#     
#     upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
#     
#     lower_mat_1 = torch.cat([torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double), torch.t(J_k)], dim = 1)
#     
#     
#     mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
#     
#     
#     upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double)], dim = 1)
#     
#     lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
#     
#     mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
    
    
    
    
    
    upper_mat = torch.cat([-torch.diag(D_k_diag), torch.t(L_k)], dim = 1)
    
    lower_mat = torch.cat([L_k, sigma_k*S_k_time_S_k], dim = 1)
    
    mat = torch.cat([upper_mat, lower_mat], dim = 0)
    

    mat = np.linalg.inv(mat.cpu().numpy())
        
    inv_mat = torch.from_numpy(mat)
    
    if is_GPU:
        
        inv_mat = inv_mat.to(device)
        
        
    
    p_mat = torch.mm(inv_mat, p_mat)
    
    
    approx_prod = sigma_k*v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    
#     t4  = time.time()
    
    
#     print('time1::', t4 - t3)
#     
#     print('key time::', t2 - t1)
    
    
    return approx_prod,zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat


def cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, m, k, v_vec, period):
 
 
#     t3  = time.time()
    
    period_num = int(i/period)
    
    
    ids = torch.tensor(range(m)).view(-1)
    
    if period_num > 0:
        ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
#     else:
#         ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
    ids = ids - 1
    
    ids = ids[ids >= 0]
    
    if ids.shape[0] > k:
        ids = ids[-k:]
    
#     if i-k >= 1:
#         lb = i-k
#         
#         zero_mat_dim = ids.shape[0] + k
#         
#     else:
#         lb = 1
#         
#         zero_mat_dim = ids.shape[0] + i-1
    zero_mat_dim = k#ids.shape[0]
    
    
    
#     curr_S_k = torch.cat([S_k_list[:, ids],S_k_list[:,lb:i]], dim=1) 
#           
#     curr_Y_k = torch.cat([Y_k_list[:, ids],Y_k_list[:,lb:i]], dim=1)

#     print(ids)

#     curr_S_k = torch.cat([S_k_list[:, ids]], dim=1) 
#           
#     curr_Y_k = torch.cat([Y_k_list[:, ids]], dim=1)

#     curr_S_k = S_k_list[:, ids]
#           
#     curr_Y_k = Y_k_list[:, ids]
    
    curr_S_k = torch.t(torch.stack(list(S_k_list), dim = 0))
          
    curr_Y_k = torch.t(torch.stack(list(Y_k_list), dim = 0))
    
#     curr_S_k = S_k_list[:,k:m] 
#          
#     curr_Y_k = Y_k_list[:,k:m] 
    
    S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
    
    
    S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)
    
    
    R_k = np.triu(S_k_time_Y_k.numpy())
    
    L_k = S_k_time_Y_k - torch.from_numpy(R_k)
    
    D_k_diag = torch.diag(S_k_time_Y_k)
    
    
    sigma_k = torch.dot(Y_k_list[-1],S_k_list[-1])/(torch.dot(S_k_list[-1], S_k_list[-1]))
    
    
    interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
    
    J_k = torch.from_numpy(np.linalg.cholesky(interm.detach().numpy())).type(torch.DoubleTensor)
    
    
#     v_vec = S_k_list[:,i-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
    
#         v_vec = torch.rand([para_num, 1], dtype = torch.double)
#     t1 = time.time()
    
    p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double)
    
    tmp = torch.mm(torch.t(curr_Y_k), v_vec)
    
    p_mat[0:zero_mat_dim] = tmp
    
    p_mat[zero_mat_dim:zero_mat_dim*2] = torch.mm(torch.t(curr_S_k), v_vec)*sigma_k
    
#     t2 = time.time()
    
#     p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
    
    
    D_k_sqr_root = torch.pow(D_k_diag, 0.5)
    
    D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
    
    upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
    
    lower_mat_1 = torch.cat([torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double), torch.t(J_k)], dim = 1)
    
    
    mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
    
    
    upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double)], dim = 1)
    
    lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
    
    mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
    
    
    p_mat = torch.mm(torch.inverse(mat_1), torch.mm(torch.inverse(mat_2), p_mat))
    
    
    approx_prod = sigma_k*v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    
#     t4  = time.time()
    
    
#     print('time1::', t4 - t3)
#     
#     print('key time::', t2 - t1)
    
    
    return approx_prod

def cal_approx_hessian_vec_prod2(S_k_list, Y_k_list, i, m, v_vec, delta_para, delta_grad):
    
    curr_S_k = torch.cat([S_k_list[:,i-m:i], delta_para], dim = 1)
                         
        
    curr_Y_k = torch.cat([Y_k_list[:,i-m:i], delta_grad], dim = 1)
    
    
    
    
    S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
    
    
    S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)
    
    
    R_k = torch.triu(S_k_time_Y_k)
    
    L_k = S_k_time_Y_k - R_k
    
    D_k_diag = torch.diag(S_k_time_Y_k)
    
    
    sigma_k = torch.dot(Y_k_list[:,i-1],S_k_list[:,i-1])/(torch.dot(S_k_list[:,i-1], S_k_list[:,i-1]))
    
    
    interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
    
    J_k = torch.from_numpy(np.linalg.cholesky(interm.detach().numpy())).type(torch.DoubleTensor)
    
    
#     v_vec = S_k_list[:,i-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
    
#         v_vec = torch.rand([para_num, 1], dtype = torch.double)
    
    
    p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
    
    
    D_k_sqr_root = torch.pow(D_k_diag, 0.5)
    
    D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
    
    upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
    
    lower_mat_1 = torch.cat([torch.zeros([m+1, m+1], dtype = torch.double), torch.t(J_k)], dim = 1)
    
    
    mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
    
    
    upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([m+1, m+1], dtype = torch.double)], dim = 1)
    
    lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
    
    mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
    
    
    p_mat = torch.mm(torch.inverse(mat_1), torch.mm(torch.inverse(mat_2), p_mat))
    
    
    approx_prod = sigma_k*v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    
    return approx_prod


def model_update_provenance_test(truncted_s, extended_Y_k_list, exp_gradient_list_all_epochs, exp_para_list_all_epochs, X, Y, model, S_k_list, Y_k_list, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, input_dim, hidden_dims, output_dim, m, alpha, beta, selected_rows, error):
    
    
    para_num = S_k_list.shape[0]
    
    
    model_dual = DNNModel(input_dim, hidden_dims, output_dim)
    
    
    para = list(model.parameters())
    
    expected_para = list(model.parameters())
    
    last_gradient_full = None

    last_para = None
    
    for i in range(epoch):
        
        print('epoch::', i)
        
        init_model(model, para)
        
        compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
        
        expect_gradients = get_all_vectorized_parameters(model.get_all_gradient())
        
        
        init_model(model, expected_para)
        
        compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
        
#         compute_model_para_diff(expected_para, exp_para_list_all_epochs[i])

        
        expected_para = get_devectorized_parameters(get_all_vectorized_parameters(expected_para) - alpha*get_all_vectorized_parameters(model.get_all_gradient()), input_dim, hidden_dims, output_dim)
        
        
        
        if i >= 50:
        
            init_model(model_dual, para)
            
            compute_derivative_one_more_step(model_dual, error, X[delta_ids], Y[delta_ids], beta)
            
            
            gradient_dual = get_all_vectorized_parameters(model_dual.get_all_gradient())
            
            
            v_vec = get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])
            
            print('para_diff::', torch.norm(v_vec))
            
            print('para_angle::', torch.dot(get_all_vectorized_parameters(para).view(-1), get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))/(torch.norm(get_all_vectorized_parameters(para).view(-1))*torch.norm(get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))))
            
            
            
            init_model(model, para_list_all_epochs[i])
             
            curr_vec_gradient,_ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, gradient_list_all_epochs[i], beta)
             
            hessian_matrix1 = compute_hessian_matrix(model, curr_vec_gradient, input_dim, hidden_dims, output_dim)
            
            
#             hessian_para_prod = cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, 50, m, v_vec.view(-1,1))
            
            hessian_para_prod = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1))
            
#             hessian_para_prod = cal_approx_hessian_vec_prod2(S_k_list, Y_k_list, i, m-1, v_vec.view(-1,1), last_v_vec.view(-1,1), last_gradient_full.view(-1, 1) - get_all_vectorized_parameters(gradient_list_all_epochs[i-1]).view(-1,1))
            
#             hessian_para_prod = cal_approx_hessian_vec_prod3(i, m, v_vec.view(-1,1), para_list_all_epochs, gradient_list_all_epochs)
            
#             hessian_para_prod, tmp_res = cal_approx_hessian_vec_prod4(truncted_s, extended_Y_k_list[i], i, m, v_vec.view(-1,1))
            
            
            delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
            
#             para_diff = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
            
            gradient_full = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
            
            delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1), error, model, beta)
            gradient_full2 = torch.mm(v_vec.view(1,-1), hessian_matrix1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
#             gradient_full2 = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) - delta_const*v_vec.view(1,-1)
            
            gradients = (gradient_full*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
            
            gradients2 = (gradient_full2*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
            
            print('hessian_vector_prod_diff::', torch.norm(torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1) - hessian_para_prod))
            
            print('gradient_diff::', torch.norm(gradients - expect_gradients))
            
            print('gradient2_diff::', torch.norm(gradients2 - expect_gradients))
            
            compute_model_para_diff(exp_para_list_all_epochs[i], para)
            
            S_k_list[:,i-1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
            
            para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradients2, input_dim, hidden_dims, output_dim)
        
            Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
            
            last_gradient_full = gradient_full
            
            last_v_vec = v_vec.clone()
        
        
        else:
            if i >= 1:
                S_k_list[:,i - 1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
            last_para = para
            
            para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*expect_gradients, input_dim, hidden_dims, output_dim)
        
            if i == m-1:
                init_model(model, para)
                
                compute_derivative_one_more_step(model, error, X, Y, beta)
            
            
                last_gradient_full = get_all_vectorized_parameters(model.get_all_gradient())
                
                last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
                
            if i >= 1:
                
                init_model(model, para)
                
                compute_derivative_one_more_step(model, error, X, Y, beta)
            
            
                gradient_full = get_all_vectorized_parameters(model.get_all_gradient())
                
#                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
                
                Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
        
#         last_gradient = expect_gradients
#             init_model(model, get_devectorized_parameters(updated_para, input_dim, hidden_dims, output_dim))
            
            
            
            
            
            
            
    return para
            


# def update_para_final(para, hessian_para_prod, gradient_dual, grad_list, size1, size2, alpha):
#     
#     if gradient_dual is not None:
#     
#         for i in range(len(para)):
#             
#         
#             hessian_para_prod[i] += grad_list[i]
#         
#             gradients = (hessian_para_prod[i]*size1 + gradient_dual[i]*size2)/(size1 + size2)
#         
#             para[i] -= alpha*gradients
#             
#     else:
#         for i in range(len(para)):
#             
#         
#             hessian_para_prod[i] += grad_list[i]
#         
# #             gradients = (gradient_full*size1)/(size1 - size2)
#         
#             para[i] -= alpha*hessian_para_prod[i]

def compute_grad_final(para, hessian_para_prod, gradient_dual, grad_list, para_list, size1, size2, alpha, beta):
    
    gradient_list = []
    
    exp_grad_list = []
    
    gradients = None
    
    
#     old_para = []
    
    if gradient_dual is not None:
    
        for i in range(len(para)):
            
        
            hessian_para_prod[i] += (grad_list[i] + beta*para_list[i])
            
            
#             old_para.append(para[i].clone())
#             if is_GPU:
#                 gradients = (hessian_para_prod[i].to(device)*size1 - gradient_dual[i]*size2)/(size1 - size2)
#             else:
#             print(gradient_dual[i].shape)
#             print(para_vec.shape)
#             print(hessian_para_prod.shape)
            gradients = (hessian_para_prod[i]*size1 + (gradient_dual[i].to('cpu') + beta*para[i])*size2)/(size1 + size2)
        
#             para[i] *= (1-alpha*beta)
        
#             para[i] -=  alpha*gradients
            
            gradient_list.append(gradients)
            
#             if exp_gradient is not None:
#                 
#                 exp_grad_list.append(exp_gradient[i] + beta*exp_para[i])
            
    else:
        for i in range(len(para)):
            
        
            hessian_para_prod[i] += (grad_list[i] + beta*para_list[i])
        
#             gradients = (gradient_full*size1)/(size1 - size2)
#             para[i] *= (1-alpha*beta)
            
#             if is_GPU:
#                 para[i] -= alpha*hessian_para_prod[i].to(device)
#             else:

#             old_para.append(para[i].clone())
            gradients = hessian_para_prod[i]
#             para[i] -= alpha*gradients
            
            gradient_list.append(gradients)
            
#             if exp_gradient is not None:
# #                 gradient_list.append(hessian_para_prod[i])
#                 exp_grad_list.append(exp_gradient[i] + beta*exp_para[i])

#     tmp_res = torch.dot(, )

    delta_para = get_all_vectorized_parameters(para).view(-1) - get_all_vectorized_parameters(para_list).view(-1)
    
    delta_grad = get_all_vectorized_parameters(gradient_list).view(-1) - (get_all_vectorized_parameters(grad_list).view(-1) + beta*get_all_vectorized_parameters(para_list).view(-1))
    
    tmp_res = 0
    
#     print(torch.norm(delta_grad)/torch.norm(delta_para))
    
#     print("delta param::")
#     
#     print(torch.norm(delta_para))
#     
#     print("delta grad::")
#     
#     print(torch.norm(delta_grad))
    
    if torch.norm(delta_para) > torch.norm(delta_grad):
        return True, gradient_list
#     if exp_gradient is not None:
#         print("grad_diff::")
#          
#         compute_model_para_diff(gradient_list, exp_grad_list)
#         
#         tmp_res = torch.dot(get_all_vectorized_parameters(old_para).view(-1) - get_all_vectorized_parameters(para_list).view(-1), get_all_vectorized_parameters(gradient_list).view(-1) - get_all_vectorized_parameters(grad_list).view(-1))
#         
#         print(tmp_res)
#           
#         print("here!!")

#     if tmp_res < 0:
#         return False, gradient_list
    else:
        return False, gradient_list
    
def compute_grad_final2(para, hessian_para_prod, gradient_dual, grad_list, para_list, size1, size2, alpha, beta, is_GPU, device):
    
    gradient_list = []
    
    exp_grad_list = []
    
    gradients = None
    
    if gradient_dual is not None:
    
        for i in range(len(para)):
            
            if is_GPU:
                hessian_para_prod[i] += (grad_list[i] + beta*para_list[i]).to(device)
            else:
                hessian_para_prod[i] += (grad_list[i] + beta*para_list[i])
            
            
            curr_hessian_para_prod = hessian_para_prod[i]
            
            
            gradients = (curr_hessian_para_prod*size1 + (gradient_dual[i] + beta*para[i])*size2)/(size1 + size2)
        
            
            gradient_list.append(gradients)
            
    else:
        for i in range(len(para)):
            

            if is_GPU:        
                hessian_para_prod[i] += (grad_list[i] + beta*para_list[i]).to(device)
            else:
                hessian_para_prod[i] += (grad_list[i] + beta*para_list[i])
            
            gradients = hessian_para_prod[i]
            
            
            gradient_list.append(gradients)
            

    delta_para = get_all_vectorized_parameters1(para).view(-1) - get_all_vectorized_parameters1(para_list).view(-1).to(device)
    
    delta_grad = get_all_vectorized_parameters1(gradient_list).view(-1) - (get_all_vectorized_parameters1(grad_list).view(-1).to(device) + beta*get_all_vectorized_parameters1(para_list).view(-1).to(device))
    
    tmp_res = 0
    
    
    if torch.norm(delta_para) > torch.norm(delta_grad):
        return True, gradient_list

    else:
        return False, gradient_list



def compute_grad_final3(para, hessian_para_prod, gradient_dual, grad_list_tensor, para_list_tensor, size1, size2, alpha, beta, is_GPU, device):
    
#     gradient_list = []
#     
#     exp_grad_list = []
    
    gradients = None
    
    if gradient_dual is not None:
        
        hessian_para_prod += grad_list_tensor 
        
        hessian_para_prod += beta*para_list_tensor 
        
        
#         tmp = (grad_list_tensor + beta*para_list_tensor)
        
#         hessian_para_prod += tmp.view(-1,1)
        gradients = hessian_para_prod*size1
        
        gradients += (gradient_dual + beta*para_list_tensor)*size2
        
        gradients /= (size1 + size2)
        
        
#         for i in range(len(para)):
#             
#             if is_GPU:
#                 hessian_para_prod[i] += (grad_list[i] + beta*para_list[i]).to(device)
#             else:
#                 hessian_para_prod[i] += (grad_list[i] + beta*para_list[i])
#             
#             
#             curr_hessian_para_prod = hessian_para_prod[i]
#             
#             
#             gradients = (curr_hessian_para_prod*size1 + (gradient_dual[i] + beta*para[i])*size2)/(size1 + size2)
#         
#             
#             gradient_list.append(gradients)
            
    else:
        
        hessian_para_prod += (grad_list_tensor + beta*para_list_tensor)
        
        gradients = hessian_para_prod
        
#         for i in range(len(para)):
#             
# 
#             if is_GPU:        
#                 hessian_para_prod[i] += (grad_list[i] + beta*para_list[i]).to(device)
#             else:
#                 hessian_para_prod[i] += (grad_list[i] + beta*para_list[i])
#             
#             gradients = hessian_para_prod[i]
#             
#             
#             gradient_list.append(gradients)
            
#     print(gradients.shape)
#     delta_para = get_all_vectorized_parameters1(para).view(-1) - para_list_tensor.view(-1)
#      
#     delta_grad = gradients.view(-1) - (grad_list_tensor.view(-1) + beta*para_list_tensor.view(-1))
#      
#     tmp_res = 0
#      
#      
#     if torch.norm(delta_para) > torch.norm(delta_grad):
#         return True, gradients
#  
#     else:
#         return False, gradients
    return True, gradients


    
def compute_grad_final4(para, hessian_para_prod, gradient_dual, grad_list_tensor, para_list_tensor, size1, size2, alpha, beta, is_GPU, device):
    
#     gradient_list = []
#     
#     exp_grad_list = []
    
    gradients = None
    
    if gradient_dual is not None:
        
        hessian_para_prod += grad_list_tensor 
        
        hessian_para_prod += beta*para_list_tensor 
        
        
#         tmp = (grad_list_tensor + beta*para_list_tensor)
        
#         hessian_para_prod += tmp.view(-1,1)
        gradients = hessian_para_prod*size1
        
        gradients += (gradient_dual + beta*para_list_tensor)*size2
        
        gradients /= (size1 + size2)
        
        
#         for i in range(len(para)):
#             
#             if is_GPU:
#                 hessian_para_prod[i] += (grad_list[i] + beta*para_list[i]).to(device)
#             else:
#                 hessian_para_prod[i] += (grad_list[i] + beta*para_list[i])
#             
#             
#             curr_hessian_para_prod = hessian_para_prod[i]
#             
#             
#             gradients = (curr_hessian_para_prod*size1 + (gradient_dual[i] + beta*para[i])*size2)/(size1 + size2)
#         
#             
#             gradient_list.append(gradients)
            
    else:
        
        hessian_para_prod += (grad_list_tensor + beta*para_list_tensor)
        
        gradients = hessian_para_prod
        
#         for i in range(len(para)):
#             
# 
#             if is_GPU:        
#                 hessian_para_prod[i] += (grad_list[i] + beta*para_list[i]).to(device)
#             else:
#                 hessian_para_prod[i] += (grad_list[i] + beta*para_list[i])
#             
#             gradients = hessian_para_prod[i]
#             
#             
#             gradient_list.append(gradients)
            
#     print(gradients.shape)
    delta_para = para - para_list_tensor
     
    delta_grad = hessian_para_prod - (grad_list_tensor + beta*para_list_tensor)
     
    tmp_res = 0
     
     
    if torch.norm(delta_para) > torch.norm(delta_grad):
        return True, gradients
 
    else:
        return False, gradients
#     return True, gradients    



def compute_grad_final5(para, hessian_para_prod, gradient_dual, grad_list_tensor, para_list_tensor, size1, size2, alpha, beta, is_GPU, device):
    
#     gradient_list = []
#     
#     exp_grad_list = []
    
    gradients = None
    
    exp_grad_list_full = None
    
    
    delta_grad_norm = torch.norm(hessian_para_prod)
    
    if gradient_dual is not None:
        
        hessian_para_prod += grad_list_tensor 
        
        hessian_para_prod += beta*para_list_tensor 
        
        
#         tmp = (grad_list_tensor + beta*para_list_tensor)
        
#         hessian_para_prod += tmp.view(-1,1)
        gradients = hessian_para_prod*size1
        
        gradients += (gradient_dual + beta*para_list_tensor)*size2
        
        gradients /= (size1 + size2)
        
        
#         for i in range(len(para)):
#             
#             if is_GPU:
#                 hessian_para_prod[i] += (grad_list[i] + beta*para_list[i]).to(device)
#             else:
#                 hessian_para_prod[i] += (grad_list[i] + beta*para_list[i])
#             
#             
#             curr_hessian_para_prod = hessian_para_prod[i]
#             
#             
#             gradients = (curr_hessian_para_prod*size1 + (gradient_dual[i] + beta*para[i])*size2)/(size1 + size2)
#         
#             
#             gradient_list.append(gradients)
        
    else:
        
        hessian_para_prod += (grad_list_tensor + beta*para_list_tensor)
        
        gradients = hessian_para_prod
        
#         for i in range(len(para)):
#             
# 
#             if is_GPU:        
#                 hessian_para_prod[i] += (grad_list[i] + beta*para_list[i]).to(device)
#             else:
#                 hessian_para_prod[i] += (grad_list[i] + beta*para_list[i])
#             
#             gradients = hessian_para_prod[i]
#             
#             
#             gradient_list.append(gradients)
            
#     print(gradients.shape)

    exp_grad_list_full = gradients - beta*para
    delta_para = para - para_list_tensor
     
    
     
    tmp_res = 0
     
     
    if torch.norm(delta_para) > delta_grad_norm:
        return True, gradients, exp_grad_list_full
 
    else:
        return False, gradients, exp_grad_list_full
   

def update_para_final(para, gradient_list, alpha, beta, exp_gradient, exp_para):
    
#     gradient_list = []
    
    exp_grad_list = []
    
    gradients = None
    
    
    old_para = []
    
    for i in range(len(para)):
        para[i] -=  alpha*gradient_list[i]
    
#     if gradient_dual is not None:
#     
#         for i in range(len(para)):
#             
#         
#             hessian_para_prod[i] += (grad_list[i] + beta*para_list[i])
#             
#             
#             old_para.append(para[i].clone())
# #             if is_GPU:
# #                 gradients = (hessian_para_prod[i].to(device)*size1 - gradient_dual[i]*size2)/(size1 - size2)
# #             else:
# #             print(gradient_dual[i].shape)
# #             print(para_vec.shape)
# #             print(hessian_para_prod.shape)
#             gradients = (hessian_para_prod[i]*size1 - (gradient_dual[i].to('cpu') + beta*para[i])*size2)/(size1 - size2)
#         
# #             para[i] *= (1-alpha*beta)
#         
#             para[i] -=  alpha*gradients
#             
#             gradient_list.append(gradients)
#             
#             if exp_gradient is not None:
#                 
#                 exp_grad_list.append(exp_gradient[i] + beta*exp_para[i])
#             
#     else:
#         for i in range(len(para)):
#             
#         
#             hessian_para_prod[i] += (grad_list[i] + beta*para_list[i])
#         
# #             gradients = (gradient_full*size1)/(size1 - size2)
# #             para[i] *= (1-alpha*beta)
#             
# #             if is_GPU:
# #                 para[i] -= alpha*hessian_para_prod[i].to(device)
# #             else:
# 
#             old_para.append(para[i].clone())
#             gradients = hessian_para_prod[i]
#             para[i] -= alpha*gradients
#             
#             gradient_list.append(gradients)
            
        if exp_gradient is not None:
#                 gradient_list.append(hessian_para_prod[i])
            exp_grad_list.append(exp_gradient[i] + beta*exp_para[i])

#     tmp_res = torch.dot(get_all_vectorized_parameters(old_para).view(-1) - get_all_vectorized_parameters(para_list).view(-1), get_all_vectorized_parameters(gradient_list).view(-1) - get_all_vectorized_parameters(grad_list).view(-1))

    if exp_gradient is not None:
        print("grad_diff::")
         
        compute_model_para_diff(gradient_list, exp_grad_list)
        
#         tmp_res = torch.dot(get_all_vectorized_parameters(old_para).view(-1) - get_all_vectorized_parameters(para_list).view(-1), get_all_vectorized_parameters(gradient_list).view(-1) - get_all_vectorized_parameters(grad_list).view(-1))
#         
#         print(tmp_res)
          
        print("here!!")



def update_para_final2(vec_para, gradient_list, alpha, beta, exp_gradient, exp_para):
    
#     gradient_list = []
    
#     exp_grad_list = []
    
#     gradients = None
#     
#     
#     old_para = []
#     vec_para = get_all_vectorized_parameters1(para)
    
    vec_para -= alpha*gradient_list
    
#     for i in range(len(para)):
#         para[i] -=  alpha*gradient_list[i]
    
#     if gradient_dual is not None:
#     
#         for i in range(len(para)):
#             
#         
#             hessian_para_prod[i] += (grad_list[i] + beta*para_list[i])
#             
#             
#             old_para.append(para[i].clone())
# #             if is_GPU:
# #                 gradients = (hessian_para_prod[i].to(device)*size1 - gradient_dual[i]*size2)/(size1 - size2)
# #             else:
# #             print(gradient_dual[i].shape)
# #             print(para_vec.shape)
# #             print(hessian_para_prod.shape)
#             gradients = (hessian_para_prod[i]*size1 - (gradient_dual[i].to('cpu') + beta*para[i])*size2)/(size1 - size2)
#         
# #             para[i] *= (1-alpha*beta)
#         
#             para[i] -=  alpha*gradients
#             
#             gradient_list.append(gradients)
#             
#             if exp_gradient is not None:
#                 
#                 exp_grad_list.append(exp_gradient[i] + beta*exp_para[i])
#             
#     else:
#         for i in range(len(para)):
#             
#         
#             hessian_para_prod[i] += (grad_list[i] + beta*para_list[i])
#         
# #             gradients = (gradient_full*size1)/(size1 - size2)
# #             para[i] *= (1-alpha*beta)
#             
# #             if is_GPU:
# #                 para[i] -= alpha*hessian_para_prod[i].to(device)
# #             else:
# 
#             old_para.append(para[i].clone())
#             gradients = hessian_para_prod[i]
#             para[i] -= alpha*gradients
#             
#             gradient_list.append(gradients)
            
#         if exp_gradient is not None:
# #                 gradient_list.append(hessian_para_prod[i])
#             exp_grad_list.append(exp_gradient[i] + beta*exp_para[i])

#     tmp_res = torch.dot(get_all_vectorized_parameters(old_para).view(-1) - get_all_vectorized_parameters(para_list).view(-1), get_all_vectorized_parameters(gradient_list).view(-1) - get_all_vectorized_parameters(grad_list).view(-1))

    if exp_gradient is not None:
        print("grad_diff::")
         
        compute_model_para_diff(gradient_list, exp_grad_list)
        
#         tmp_res = torch.dot(get_all_vectorized_parameters(old_para).view(-1) - get_all_vectorized_parameters(para_list).view(-1), get_all_vectorized_parameters(gradient_list).view(-1) - get_all_vectorized_parameters(grad_list).view(-1))
#         
#         print(tmp_res)
          
        print("here!!")
        
        
    return vec_para


def get_model_parameter_shape_list(model):
    
    shape_list = []
    
    for param in model.parameters():
        shape_list.append(param.shape)
        
    return shape_list
    
    

# init_epochs, 1, init_epochs, None, None, None, None, dataset_train, model, gradient_list_all_epochs, para_list_all_epochs, max_epoch, period, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, added_random_ids_multi_super_iteration, added_batch_size, X_to_add, Y_to_add, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device


def model_update_provenance_test2(period, length, init_epochs, truncted_s, extended_Y_k_list, exp_gradient_list_all_epochs, exp_para_list_all_epochs, dataset_train, model, gradient_list_all_epochs, para_list_all_epochs, epoch, m, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, added_random_ids_multi_super_iteration, added_batch_size, X_to_add, Y_to_add, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device):
    
    
#     para_num = S_k_list.shape[0]
    
    
#     model_dual = DNNModel(input_dim, hidden_dims, output_dim)
#     selected_rows_set = set(selected_rows.view(-1).tolist())
    
    para = list(model.parameters())
    
#     expected_para = list(model.parameters())
#     
#     last_gradient_full = None
# 
#     last_para = None
    
    use_standard_way = False
    
    recorded = 0
    
    overhead = 0
    
#     vecorized_paras = get_all_vectorized_parameters_with_gradient(model.parameters())
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
#     shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    
#     remaining_shape_num = 0
#     
#     for i in range(len(shape_list) - first_few_layer_num):
#         remaining_shape_num += shape_list[i+first_few_layer_num]
#         
#     S_k_list = torch.zeros(remaining_shape_num, len(random_ids_multi_super_iterations))
#     
#     
#     Y_k_list = torch.zeros(remaining_shape_num, len(random_ids_multi_super_iterations))
    
    i = 0
    
    S_k_list = deque()
    
    Y_k_list = deque()
    
    overhead2 = 0
    
    overhead3 = 0
    
    overhead4 = 0
    
    overhead5 = 0
    
    old_lr = 0
    
    for k in range(len(random_ids_multi_super_iterations)):
    
        random_ids = random_ids_multi_super_iterations[k]
        
        added_to_random_ids = added_random_ids_multi_super_iteration[k]
        
#         sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
#         all_indexes = np.sort(sort_idx[delta_ids])
                
        id_start = 0
    
        id_end = 0
        
        jj = 0
        
        to_add = True
        
        for j in range(0, dim[0], batch_size):
        
            end_id = j + batch_size
            
            added_end_id = jj + added_batch_size
            
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            if added_end_id >= X_to_add.shape[0]:
                added_end_id = X_to_add.shape[0]
            
            
            if jj >= X_to_add.shape[0]:
                to_add = False
            

            curr_added_size = 0

            

            if to_add:
                
                curr_added_random_ids = added_to_random_ids[jj:added_end_id]
                
                batch_delta_X = X_to_add[curr_added_random_ids]
                
                batch_delta_Y = Y_to_add[curr_added_random_ids]
            
                curr_added_size = curr_added_random_ids.shape[0]
                
                
                if is_GPU:
                    batch_delta_X = batch_delta_X.to(device)
                    
                    batch_delta_Y = batch_delta_Y.to(device)
                
            
            
            learning_rate = learning_rate_all_epochs[i]
            
            
#             if end_id - j - curr_matched_ids_size <= 0:
#                 
#                 i += 1
#                 
#                 continue
            
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate    
                
                      
            if (i-init_epochs)%period == 0:
                
                recorded = 0
                
                use_standard_way = True
                
                
            if i< init_epochs or use_standard_way == True:
                t7 = time.time()
                
                curr_rand_ids = random_ids[j:end_id]
            
            
                batch_remaining_X = dataset_train.data[curr_rand_ids]
                
                batch_remaining_Y = dataset_train.labels[curr_rand_ids]
                
#                 if is_GPU:
#                     batch_remaining_X = batch_remaining_X.to(device)
#                     
#                     batch_remaining_Y = batch_remaining_Y.to(device)
#                 
#                 
#                 
#                 t8 = time.time()
#             
#                 overhead4 += (t8 - t7)
#                 
#                 
#                 t5 = time.time()
#                 
#                 init_model(model, para)
#                 
#                 compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
#                 
#                 
# 
#                 
#                 
#                 expect_gradients = get_all_vectorized_parameters1(model.get_all_gradient())
#                 
#                 t6 = time.time()
# 
#                 overhead3 += (t6 - t5)
#                 
#                 gradient_remaining = 0
# #                 if curr_matched_ids_size > 0:
#                 if to_add:
#                     
#                     t3 = time.time()
#                     
#                     clear_gradients(model.parameters())
#                         
#                     compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
#                 
#                 
#                     gradient_remaining = get_all_vectorized_parameters1(model.get_all_gradient())     
#                     
#                     
#                     t4 = time.time()
#                 
#                 
#                     overhead2 += (t4  -t3)
#                 
#                 with torch.no_grad():
#                                
#                 
#                     curr_para = get_all_vectorized_parameters1(para)
#                 
#                     if i>0:
#                         
#                         
#                         
#                         
#                         
# 
#                         
#                         if i <= 10:
#                             t9 = time.time()
#                         
#                         prev_para = get_all_vectorized_parameters2(para_list_all_epochs[i], is_GPU, device)
#                         
#                         
#                         if i <= 10:
#                             t10 = time.time()
#                             print(len(para_list_all_epochs))
#                             overhead5 += (t10 - t9)
#                         
#                         curr_s_list = (curr_para - prev_para).view(-1)
#                         
#                         
# #                         if is_GPU:
# #                             curr_s_list = curr_s_list.to(device)
#                         S_k_list.append(curr_s_list)
#                         if len(S_k_list) > m:
#                             removed_s_k = S_k_list.popleft()
#                             
#                             del removed_s_k
#                         
#                         
#                             
#     #                     print(i-1)
#     #                     
#     #                     print(S_k_list[:,i - 1])
#                     
#         #             init_model(model, para)
#                     
# #                     gradient_full = (expect_gradients*curr_rand_ids.shape[0] + gradient_remaining*curr_added_random_id_size)/(curr_added_random_id_size + curr_rand_ids.shape[0])
# 
#                     gradient_full = (expect_gradients*curr_rand_ids.shape[0] + gradient_remaining*curr_added_size)/(curr_rand_ids.shape[0] + curr_added_size)
#                     
#                     if i>0:
#                         
#                         
#                         Y_k_list.append((expect_gradients - get_all_vectorized_parameters1(gradient_list_all_epochs[i]).to(device)).view(-1))
#                         if len(Y_k_list) > m:
#                             removed_y_k = Y_k_list.popleft()
#                             
#                             del removed_y_k
#                     
#                     
#                     
#     #                 batch_X = X[curr_rand_ids]
#     #                 
#     #                 batch_Y = Y[curr_rand_ids]
#     #                 clear_gradients(model.parameters())
#     #                     
#     #                 compute_derivative_one_more_step(model, error, batch_X, batch_Y, beta)
#     #                 
#     #                 expect_gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
#                     
#     #                 decompose_model_paras2(para, para_list_all_epochs[i], get_devectorized_parameters(gradient_full, input_dim, hidden_dims, output_dim), gradient_list_all_epochs[i])
#                     
#     #                     print(torch.dot(Y_k_list[:,i-1], S_k_list[:,i-1]))
#     #                     y=0
#     #                     y+=1
#                     alpha = learning_rate_all_epochs[i]
#                     
#                     para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_para - learning_rate*gradient_full, full_shape_list, shape_list)
# #                     para = get_devectorized_parameters(params, full_shape_list, shape_list)
# #                     
# #                     para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradient_full, input_dim, hidden_dims, output_dim)
#         #                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
#                     
#     
#                     
#                     
#                     recorded += 1
#                     
#                     
#                     del gradient_full
#                     
#                     del gradient_remaining
#                     
#                     del expect_gradients
#                     
#                     del batch_remaining_X
#                     
#                     del batch_remaining_Y
#                     
#                     if to_add:
#                         
#                         del batch_delta_X
#                         
#                         del batch_delta_Y
#                     
#                     if i > 0:
#                         del prev_para
#                     
#                         del curr_para
#                     
#                     if recorded >= length:
#                         use_standard_way = False
                if is_GPU:
                    batch_remaining_X = batch_remaining_X.to(device)
                    
                    batch_remaining_Y = batch_remaining_Y.to(device)
                
#                 if not is_GPU:
#                 
#                     batch_remaining_X = next_items[0]
#                     
#                     batch_remaining_Y = next_items[1]
#                     
#                 else:
#                     batch_remaining_X = next_items[0].to(device)
#                     
#                     batch_remaining_Y = next_items[1].to(device)
                
                init_model(model, para)
                
                
#                 t7 = time.time()
                                
                compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
                
                
                t3 = time.time()
                expect_gradients = get_all_vectorized_parameters1(model.get_all_gradient()).cpu()

                t4 = time.time()
                        
                overhead2 += t4 - t3
#                 t8 = time.time()
                
#                 count3 += 1
#                 
#                 overhead3 += (t8 - t7)

                gradient_remaining = 0
                if to_add:
                    clear_gradients(model.parameters())
                        
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                
                    gradient_remaining = get_all_vectorized_parameters(model.get_all_gradient())     
                    
                with torch.no_grad():
                               
                
                    if i>0:
                        
#                         torch.cuda.synchronize()
                        
                        
    
                        S_k_list.append((get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1))
                        
                        
#                         torch.cuda.synchronize()
                        
                        
                        
                        if len(S_k_list) > m:
                            S_k_list.popleft()
                    
#                     gradient_full = (expect_gradients*next_items[2].shape[0] + gradient_remaining*curr_matched_ids_size)/(next_items[2].shape[0] + curr_matched_ids_size)
                        gradient_full = (expect_gradients*curr_rand_ids.shape[0]+ gradient_remaining*curr_added_size)/(curr_added_size + curr_rand_ids.shape[0])
                            

                    
                    if i>0:
                        
                        
                        Y_k_list.append((gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i]) + regularization_coeff*S_k_list[-1]).view(-1))
                        
#                         print("period::", i)
#                         
#                         print("secont condition::", torch.dot(Y_k_list[-1].view(-1), S_k_list[-1].view(-1)))
#                         
#                         print("batch size check::", curr_matched_ids_size + next_items[2].shape[0])
                        
                        
                        if len(Y_k_list) > m:
                            Y_k_list.popleft()
                    
                    exp_gradient = None
                    
                    para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*get_all_vectorized_parameters(para) - learning_rate*expect_gradients, full_shape_list, shape_list)
                    
                    recorded += 1
                    
                    
                    if recorded >= length:
                        use_standard_way = False
                
            else:
                
#                 print('epoch::', i)
                
                
    #             delta_X = X[delta_ids]
    #             
    #             delta_Y = Y[delta_ids]
#                 t1 = time.time()
                gradient_dual = None
    
#                 if curr_matched_ids_size > 0:
                if to_add:
                
#                     t3 = time.time()
                    init_model(model, para)
                    
                    
                    
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                    
#                     t4 = time.time()
#                     
#                     overhead2 += (t4 - t3)
                    
                    gradient_dual = model.get_all_gradient()
                
                with torch.no_grad():
                
                
#                 t5 = time.time()
#                     v_vec = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
                    
                    compute_diff_vectorized_parameters(para, para_list_all_epochs[i], vec_para_diff, shape_list)
    #                 v_vec2 = compute_model_para_diff2(para, para_list_all_epochs[i])
                    
#                     if i/period >= 1:
#                         if i % period == 1:
                    if (i-init_epochs)/period >= 1:
                        if (i-init_epochs) % period == 1:
    #                         print(i)
    #                         
    #                         if i >= 370:
    #                             y = 0
    #                             y+=1
                            zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, is_GPU, device)
#                             zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2 = prepare_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, period)
                            
                            mat = torch.inverse(mat_prime)
                    
                        hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff, is_GPU, device)
                        
                    else:
                        
                        hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period, is_GPU, device)
#                         hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
    #                     hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms2(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2, v_vec2, shape_list)
                        
#                     else:
#                         hessian_para_prod = cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
                    
                    
        #             print('para_diff::', torch.norm(v_vec))
        #             
        #             print('para_angle::', torch.dot(get_all_vectorized_parameters(para).view(-1), get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))/(torch.norm(get_all_vectorized_parameters(para).view(-1))*torch.norm(get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))))
                    
                    
                    
        #             init_model(model, para_list_all_epochs[i])
                     
        #             curr_vec_gradient,_ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, gradient_list_all_epochs[i], beta)
                     
        #             hessian_matrix1 = compute_hessian_matrix(model, curr_vec_gradient, input_dim, hidden_dims, output_dim)
                    
                    
                    
    #                 cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, v_vec.view(-1,1), period)
                    
                    
                    
                    
        #             hessian_para_prod = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1))
                    
        #             hessian_para_prod = cal_approx_hessian_vec_prod2(S_k_list, Y_k_list, i, m-1, v_vec.view(-1,1), last_v_vec.view(-1,1), last_gradient_full.view(-1, 1) - get_all_vectorized_parameters(gradient_list_all_epochs[i-1]).view(-1,1))
                    
        #             hessian_para_prod = cal_approx_hessian_vec_prod3(i, m, v_vec.view(-1,1), para_list_all_epochs, gradient_list_all_epochs)
                    
        #             hessian_para_prod, tmp_res = cal_approx_hessian_vec_prod4(truncted_s, extended_Y_k_list[i], i, m, v_vec.view(-1,1))
                    exp_gradient, exp_param = None, None
                    
                    delta_const = 0#compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
                    
        #             para_diff = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
                    
                    alpha = learning_rate_all_epochs[i]
                    
                    
                    
                    is_positive, final_gradient_list = compute_grad_final(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_added_size, learning_rate, regularization_coeff, is_GPU, device)
                    
#                     update_para_final(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, exp_gradient, exp_param)
                    update_para_final(para, final_gradient_list, learning_rate, regularization_coeff, exp_gradient, exp_param)

                    
#                     update_para_final(para, get_devectorized_parameters(hessian_para_prod, input_dim, hidden_dims, output_dim), gradient_dual, gradient_list_all_epochs[i], end_id - j, curr_added_size, alpha)
#                     gradient_full = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i])
                    
        #             delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1), error, model, beta)
        #             gradient_full2 = torch.mm(v_vec.view(1,-1), hessian_matrix1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
        #             gradient_full2 = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) - delta_const*v_vec.view(1,-1)
                    
#                     gradients = (gradient_full*(end_id - j) - gradient_dual*curr_matched_ids_size)/(end_id - j - curr_matched_ids_size)
                    
        #             gradients2 = (gradient_full2*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
                    
        #             print('hessian_vector_prod_diff::', torch.norm(torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1) - hessian_para_prod))
                    
        #             print('gradient_diff::', torch.norm(gradients - expect_gradients))
                    
        #             print('gradient2_diff::', torch.norm(gradients2 - expect_gradients))
                    
        #             compute_model_para_diff(exp_para_list_all_epochs[i], para)
                    
    #                 S_k_list[:,i-1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
                    
#                     para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradients, input_dim, hidden_dims, output_dim)
            
#                 t6 = time.time()
#                     
#                 overhead3 += (t6 - t5)
#                 Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
#                 print(torch.norm(get_all_vectorized_parameters(para)))
                
#                 print(Y_k_list[:,i-1])
#                 t2 = time.time()
                    
#                 overhead += (t2 - t1)
                 
                
            i = i + 1
            
            id_start = id_end
            
            jj += added_batch_size
            
#             last_gradient_full = gradient_full
#             
#             last_v_vec = v_vec.clone()
        
        
#         else:
#             if i >= 1:
#                 S_k_list[:,i - 1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
#             last_para = para
#             
#             para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*expect_gradients, input_dim, hidden_dims, output_dim)
#         
#             if i == m-1:
#                 init_model(model, para)
#                 
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             
#                 last_gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
#                 
#                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
#                 
#             if i >= 1:
#                 
#                 init_model(model, para)
#                 
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             
#                 gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
#                 
# #                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
#                 
#                 Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
        
#         last_gradient = expect_gradients
#             init_model(model, get_devectorized_parameters(updated_para, input_dim, hidden_dims, output_dim))
            
            
            
            
    print('overhead::', overhead)
    
    print('overhead2::', overhead2)
    
    print('overhead3::', overhead3)
    
    print('overhead4::', overhead4)
    
    print('overhead5::', overhead5)
    
            
    return para






def model_update_provenance_test1(period, length, init_epochs, truncted_s, extended_Y_k_list, exp_gradient_list_all_epochs, exp_para_list_all_epochs, dataset_train, model, gradient_list_all_epochs, para_list_all_epochs, epoch, m, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, added_random_ids_multi_super_iteration, added_batch_size, X_to_add, Y_to_add, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device):
    
    
#     para_num = S_k_list.shape[0]
    
    
#     model_dual = DNNModel(input_dim, hidden_dims, output_dim)
#     selected_rows_set = set(selected_rows.view(-1).tolist())
    
    para = list(model.parameters())
    
#     expected_para = list(model.parameters())
#     
#     last_gradient_full = None
# 
#     last_para = None
    
    use_standard_way = False
    
    recorded = 0
    
    overhead = 0
    
#     vecorized_paras = get_all_vectorized_parameters_with_gradient(model.parameters())
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
#     shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    if not is_GPU:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    else:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double, device=device)
    
#     remaining_shape_num = 0
#     
#     for i in range(len(shape_list) - first_few_layer_num):
#         remaining_shape_num += shape_list[i+first_few_layer_num]
#         
#     S_k_list = torch.zeros(remaining_shape_num, len(random_ids_multi_super_iterations))
#     
#     
#     Y_k_list = torch.zeros(remaining_shape_num, len(random_ids_multi_super_iterations))
    
    i = 0
    
    S_k_list = deque()
    
    Y_k_list = deque()
    
    overhead2 = 0
    
    overhead3 = 0
    
    overhead4 = 0
    
    overhead5 = 0
    
    old_lr = 0
    
    for k in range(len(random_ids_multi_super_iterations)):
    
        random_ids = random_ids_multi_super_iterations[k]
        
        added_to_random_ids = added_random_ids_multi_super_iteration[k]
        
#         sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
#         all_indexes = np.sort(sort_idx[delta_ids])
                
        id_start = 0
    
        id_end = 0
        
        jj = 0
        
        to_add = True
        
        for j in range(0, dim[0], batch_size):
        
            end_id = j + batch_size
            
            added_end_id = jj + added_batch_size
            
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            if added_end_id >= X_to_add.shape[0]:
                added_end_id = X_to_add.shape[0]
            
            
            if jj >= X_to_add.shape[0]:
                to_add = False
            

            curr_added_size = 0

            

            if to_add:
                
                curr_added_random_ids = added_to_random_ids[jj:added_end_id]
                
                batch_delta_X = X_to_add[curr_added_random_ids]
                
                batch_delta_Y = Y_to_add[curr_added_random_ids]
            
                curr_added_size = curr_added_random_ids.shape[0]
                
                
                if is_GPU:
                    batch_delta_X = batch_delta_X.to(device)
                    
                    batch_delta_Y = batch_delta_Y.to(device)
                
            
            
            learning_rate = learning_rate_all_epochs[i]
            
            
#             if end_id - j - curr_matched_ids_size <= 0:
#                 
#                 i += 1
#                 
#                 continue
            
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate    
                
                      
            if (i-init_epochs)%period == 0:
                
                recorded = 0
                
                use_standard_way = True
                
                
            if i< init_epochs or use_standard_way == True:
                t7 = time.time()
                
                curr_rand_ids = random_ids[j:end_id]
            
            
                batch_remaining_X = dataset_train.data[curr_rand_ids]
                
                batch_remaining_Y = dataset_train.labels[curr_rand_ids]
                
                if is_GPU:
                    batch_remaining_X = batch_remaining_X.to(device)
                    
                    batch_remaining_Y = batch_remaining_Y.to(device)
                
                
                
                t8 = time.time()
            
                overhead4 += (t8 - t7)
                
                
                t5 = time.time()
                
                init_model(model, para)
                
                compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
                
                

                
                
                expect_gradients = get_all_vectorized_parameters1(model.get_all_gradient())
                
                t6 = time.time()

                overhead3 += (t6 - t5)
                
                gradient_remaining = 0
#                 if curr_matched_ids_size > 0:
                if to_add:
                    
                    t3 = time.time()
                    
                    clear_gradients(model.parameters())
                        
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                
                
                    gradient_remaining = get_all_vectorized_parameters1(model.get_all_gradient())     
                    
                    
                    t4 = time.time()
                
                
                    overhead2 += (t4  -t3)
                
                with torch.no_grad():
                               
                
                    curr_para = get_all_vectorized_parameters1(para)
                
                    if i>0:
                        
                        
                        
                        
                        

                        torch.cuda.synchronize()
                        
                        
                        prev_para = get_all_vectorized_parameters1(para_list_all_epochs[i])
                        
                        t9 = time.time()
                        
                        if is_GPU:
                            prev_para = prev_para.to(device)
                        
                        torch.cuda.synchronize()
                        t10 = time.time()
                        overhead5 += (t10 - t9)
                        
                        curr_s_list = (curr_para - prev_para).view(-1)
                        
                        
#                         if is_GPU:
#                             curr_s_list = curr_s_list.to(device)
                        S_k_list.append(curr_s_list)
                        if len(S_k_list) > m:
                            removed_s_k = S_k_list.popleft()
                            
                            del removed_s_k
                        
                        
                            
    #                     print(i-1)
    #                     
    #                     print(S_k_list[:,i - 1])
                    
        #             init_model(model, para)
                    
#                     gradient_full = (expect_gradients*curr_rand_ids.shape[0] + gradient_remaining*curr_added_random_id_size)/(curr_added_random_id_size + curr_rand_ids.shape[0])

                    gradient_full = (expect_gradients*curr_rand_ids.shape[0] + gradient_remaining*curr_added_size)/(curr_rand_ids.shape[0] + curr_added_size)
                    
                    if i>0:
                        
                        
                        Y_k_list.append((expect_gradients - get_all_vectorized_parameters1(gradient_list_all_epochs[i]).to(device)).view(-1))
                        if len(Y_k_list) > m:
                            removed_y_k = Y_k_list.popleft()
                            
                            del removed_y_k
                    
                    
                    
    #                 batch_X = X[curr_rand_ids]
    #                 
    #                 batch_Y = Y[curr_rand_ids]
    #                 clear_gradients(model.parameters())
    #                     
    #                 compute_derivative_one_more_step(model, error, batch_X, batch_Y, beta)
    #                 
    #                 expect_gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
                    
    #                 decompose_model_paras2(para, para_list_all_epochs[i], get_devectorized_parameters(gradient_full, input_dim, hidden_dims, output_dim), gradient_list_all_epochs[i])
                    
    #                     print(torch.dot(Y_k_list[:,i-1], S_k_list[:,i-1]))
    #                     y=0
    #                     y+=1
                    alpha = learning_rate_all_epochs[i]
                    
                    para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_para - learning_rate*gradient_full, full_shape_list, shape_list)
#                     para = get_devectorized_parameters(params, full_shape_list, shape_list)
#                     
#                     para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradient_full, input_dim, hidden_dims, output_dim)
        #                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
                    
    
                    
                    
                    recorded += 1
                    
                    
                    del gradient_full
                    
                    del gradient_remaining
                    
                    del expect_gradients
                    
                    del batch_remaining_X
                    
                    del batch_remaining_Y
                    
                    if to_add:
                        
                        del batch_delta_X
                        
                        del batch_delta_Y
                    
                    if i > 0:
                        del prev_para
                    
                        del curr_para
                    
                    if recorded >= length:
                        use_standard_way = False
                
                
            else:
                
#                 print('epoch::', i)
                
                
    #             delta_X = X[delta_ids]
    #             
    #             delta_Y = Y[delta_ids]
#                 t1 = time.time()
                gradient_dual = None
    
#                 if curr_matched_ids_size > 0:
                if to_add:
                
#                     t3 = time.time()
                    init_model(model, para)
                    
                    
                    
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                    
#                     t4 = time.time()
#                     
#                     overhead2 += (t4 - t3)
                    
                    gradient_dual = model.get_all_gradient()
                
                with torch.no_grad():
                
                
#                 t5 = time.time()
#                     v_vec = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
                    
                    compute_diff_vectorized_parameters2(para, para_list_all_epochs[i], vec_para_diff, shape_list, is_GPU, device)
    #                 v_vec2 = compute_model_para_diff2(para, para_list_all_epochs[i])
                    
#                     if i/period >= 1:
#                         if i % period == 1:
                    if (i-init_epochs)/period >= 1:
                        if (i-init_epochs) % period == 1:
    #                         print(i)
    #                         
    #                         if i >= 370:
    #                             y = 0
    #                             y+=1
                            zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, is_GPU, device)
#                             zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2 = prepare_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, period)
                            
                            mat = torch.inverse(mat_prime)
                    
                        hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff, is_GPU, device)
                        
                    else:
                        
                        hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period, is_GPU, device)
#                         hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
    #                     hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms2(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2, v_vec2, shape_list)
                        
#                     else:
#                         hessian_para_prod = cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
                    
                    
        #             print('para_diff::', torch.norm(v_vec))
        #             
        #             print('para_angle::', torch.dot(get_all_vectorized_parameters(para).view(-1), get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))/(torch.norm(get_all_vectorized_parameters(para).view(-1))*torch.norm(get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))))
                    
                    
                    
        #             init_model(model, para_list_all_epochs[i])
                     
        #             curr_vec_gradient,_ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, gradient_list_all_epochs[i], beta)
                     
        #             hessian_matrix1 = compute_hessian_matrix(model, curr_vec_gradient, input_dim, hidden_dims, output_dim)
                    
                    
                    
    #                 cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, v_vec.view(-1,1), period)
                    
                    
                    
                    
        #             hessian_para_prod = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1))
                    
        #             hessian_para_prod = cal_approx_hessian_vec_prod2(S_k_list, Y_k_list, i, m-1, v_vec.view(-1,1), last_v_vec.view(-1,1), last_gradient_full.view(-1, 1) - get_all_vectorized_parameters(gradient_list_all_epochs[i-1]).view(-1,1))
                    
        #             hessian_para_prod = cal_approx_hessian_vec_prod3(i, m, v_vec.view(-1,1), para_list_all_epochs, gradient_list_all_epochs)
                    
        #             hessian_para_prod, tmp_res = cal_approx_hessian_vec_prod4(truncted_s, extended_Y_k_list[i], i, m, v_vec.view(-1,1))
                    exp_gradient, exp_param = None, None
                    
                    delta_const = 0#compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
                    
        #             para_diff = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
                    
                    alpha = learning_rate_all_epochs[i]
                    
                    
                    
                    is_positive, final_gradient_list = compute_grad_final2(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_added_size, learning_rate, regularization_coeff, is_GPU, device)
                    
#                     update_para_final(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, exp_gradient, exp_param)
                    update_para_final(para, final_gradient_list, learning_rate, regularization_coeff, exp_gradient, exp_param)

                    
#                     update_para_final(para, get_devectorized_parameters(hessian_para_prod, input_dim, hidden_dims, output_dim), gradient_dual, gradient_list_all_epochs[i], end_id - j, curr_added_size, alpha)
#                     gradient_full = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i])
                    
        #             delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1), error, model, beta)
        #             gradient_full2 = torch.mm(v_vec.view(1,-1), hessian_matrix1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
        #             gradient_full2 = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) - delta_const*v_vec.view(1,-1)
                    
#                     gradients = (gradient_full*(end_id - j) - gradient_dual*curr_matched_ids_size)/(end_id - j - curr_matched_ids_size)
                    
        #             gradients2 = (gradient_full2*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
                    
        #             print('hessian_vector_prod_diff::', torch.norm(torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1) - hessian_para_prod))
                    
        #             print('gradient_diff::', torch.norm(gradients - expect_gradients))
                    
        #             print('gradient2_diff::', torch.norm(gradients2 - expect_gradients))
                    
        #             compute_model_para_diff(exp_para_list_all_epochs[i], para)
                    
    #                 S_k_list[:,i-1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
                    
#                     para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradients, input_dim, hidden_dims, output_dim)
            
#                 t6 = time.time()
#                     
#                 overhead3 += (t6 - t5)
#                 Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
#                 print(torch.norm(get_all_vectorized_parameters(para)))
                
#                 print(Y_k_list[:,i-1])
#                 t2 = time.time()
                    
#                 overhead += (t2 - t1)
                 
                
            i = i + 1
            
            id_start = id_end
            
            jj += added_batch_size
            
#             last_gradient_full = gradient_full
#             
#             last_v_vec = v_vec.clone()
        
        
#         else:
#             if i >= 1:
#                 S_k_list[:,i - 1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
#             last_para = para
#             
#             para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*expect_gradients, input_dim, hidden_dims, output_dim)
#         
#             if i == m-1:
#                 init_model(model, para)
#                 
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             
#                 last_gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
#                 
#                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
#                 
#             if i >= 1:
#                 
#                 init_model(model, para)
#                 
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             
#                 gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
#                 
# #                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
#                 
#                 Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
        
#         last_gradient = expect_gradients
#             init_model(model, get_devectorized_parameters(updated_para, input_dim, hidden_dims, output_dim))
            
            
            
            
    print('overhead::', overhead)
    
    print('overhead2::', overhead2)
    
    print('overhead3::', overhead3)
    
    print('overhead4::', overhead4)
    
    print('overhead5::', overhead5)
    
            
    return para


def model_update_provenance_test3(period, length, init_epochs, truncted_s, extended_Y_k_list, exp_gradient_list_all_epochs, exp_para_list_all_epochs, dataset_train, model, gradient_list_all_epochs_tensor, para_list_all_epochs_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, epoch, m, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, added_random_ids_multi_super_iteration, X_to_add, Y_to_add, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device):
    
    
#     para_num = S_k_list.shape[0]
    
    
#     model_dual = DNNModel(input_dim, hidden_dims, output_dim)
#     selected_rows_set = set(selected_rows.view(-1).tolist())
    
    para = list(model.parameters())
    
#     expected_para = list(model.parameters())
#     
#     last_gradient_full = None
# 
#     last_para = None
    
    use_standard_way = False
    
    recorded = 0
    
    overhead = 0
    
#     vecorized_paras = get_all_vectorized_parameters_with_gradient(model.parameters())
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
#     shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    if not is_GPU:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    else:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double, device=device)
    
#     remaining_shape_num = 0
#     
#     for i in range(len(shape_list) - first_few_layer_num):
#         remaining_shape_num += shape_list[i+first_few_layer_num]
#         
#     S_k_list = torch.zeros(remaining_shape_num, len(random_ids_multi_super_iterations))
#     
#     
#     Y_k_list = torch.zeros(remaining_shape_num, len(random_ids_multi_super_iterations))
    
    i = 0
    
    S_k_list = deque()
    
    Y_k_list = deque()
    
    overhead2 = 0
    
    overhead3 = 0
    
    overhead4 = 0
    
    overhead5 = 0
    
    old_lr = 0
    
    cached_id = 0
    
    batch_id = 1
    
    res_para = []
    
    res_grad = []
    
    for k in range(len(random_ids_multi_super_iterations)):
    
        random_ids = random_ids_multi_super_iterations[k]
        
        added_to_random_ids = added_random_ids_multi_super_iteration[k]
        
#         sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
#         all_indexes = np.sort(sort_idx[delta_ids])
                
        id_start = 0
    
        id_end = 0
        
        j = 0
        
        to_add = True
        
        curr_init_epochs = init_epochs
#         if k == 0:
#              
#             for p in range(len(added_to_random_ids)):
#                 if len(added_to_random_ids[p]) > 0:
#                     i = p
#                     break
#                 else:
#                     res_para.append(None)
#                     
#                     res_grad.append(None)
#                  
#             added_to_random_ids = added_to_random_ids[i:]
# #             i = torch.nonzero(torch.tensor(removed_batch_empty_list).view(-1) == False)[0].item()
#             init_model(model, get_devectorized_parameters(para_list_all_epochs_tensor[i], full_shape_list, shape_list))
#          
#             para = list(model.parameters())
#          
# #             random_ids_list = random_ids_list[i:]
#              
# #             remaining_ids_list = remaining_ids_list[i:]
#          
#             j = batch_size*i
#              
#             cached_id = i
#              
#             curr_init_epochs = init_epochs + i
#              
#              
#             if cached_id >= cached_size:
#                  
#                 batch_id = cached_id/cached_size
#                  
#                 GPU_tensor_end_id = (batch_id + 1)*cached_size
#                  
#                 if GPU_tensor_end_id > para_list_all_epochs_tensor.shape[0]:
#                     GPU_tensor_end_id = para_list_all_epochs_tensor.shape[0] 
#                 print("end_tensor_id::", GPU_tensor_end_id)
#                  
#                 para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(para_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
#                  
#                 grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(gradient_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
#                  
#                 batch_id += 1
#                  
#                 cached_id = 0
        
        
        
#         for j in range(0, dim[0], batch_size):
            
        for jj in range(len(added_to_random_ids)):
        
            end_id = j + batch_size
            
#             added_end_id = jj + added_batch_size
            curr_added_random_ids = added_to_random_ids[jj]
            
            if end_id > dim[0]:
                end_id = dim[0]
            
#             if added_end_id >= X_to_add.shape[0]:
#                 added_end_id = X_to_add.shape[0]
            
            
            if curr_added_random_ids.shape[0] <= 0:
                to_add = False
            else:
                to_add = True

            curr_added_size = 0

            

            if to_add:
                
#                 curr_added_random_ids = added_to_random_ids[jj:added_end_id]
                
                batch_delta_X = dataset_train.data[curr_added_random_ids]
                
                batch_delta_Y = dataset_train.labels[curr_added_random_ids]
            
                curr_added_size = curr_added_random_ids.shape[0]
                
                
                if is_GPU:
                    batch_delta_X = batch_delta_X.to(device)
                    
                    batch_delta_Y = batch_delta_Y.to(device)
                
            
            
            learning_rate = learning_rate_all_epochs[i]
            
            
#             if end_id - j - curr_matched_ids_size <= 0:
#                 
#                 i += 1
#                 
#                 continue
            
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate    
                
                      
            if (i-curr_init_epochs)%period == 0:
                
                recorded = 0
                
                use_standard_way = True
                
                
            if i< curr_init_epochs or use_standard_way == True:
                t7 = time.time()
                
                curr_rand_ids = random_ids[j:end_id]
            
            
                batch_remaining_X = dataset_train.data[curr_rand_ids]
                
                batch_remaining_Y = dataset_train.labels[curr_rand_ids]
                
                if is_GPU:
                    batch_remaining_X = batch_remaining_X.to(device)
                    
                    batch_remaining_Y = batch_remaining_Y.to(device)
                
                
                
                t8 = time.time()
            
                overhead4 += (t8 - t7)
                
                
                t5 = time.time()
                
                init_model(model, para)
                
                compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
                
                

                
                
                expect_gradients = get_all_vectorized_parameters1(model.get_all_gradient())
                
                t6 = time.time()

                overhead3 += (t6 - t5)
                
                gradient_remaining = 0
#                 if curr_matched_ids_size > 0:
                if to_add:
                    
                    t3 = time.time()
                    
                    clear_gradients(model.parameters())
                        
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                
                
                    gradient_remaining = get_all_vectorized_parameters1(model.get_all_gradient())     
                    
                    
                    t4 = time.time()
                
                
                    overhead2 += (t4  -t3)
                
                with torch.no_grad():
                               
                
                    curr_para = get_all_vectorized_parameters1(para)
                
                    if k > 0 or (k == 0 and jj > 0):
                        
                        
                        
                        
                        

#                         torch.cuda.synchronize()
                        
                        
#                         prev_para = get_all_vectorized_parameters1(para_list_all_epochs[i])
                        prev_para = para_list_GPU_tensor[cached_id]
                        
#                         t9 = time.time()
                        
#                         if is_GPU:
#                             prev_para = prev_para.to(device)
                        
#                         torch.cuda.synchronize()
#                         t10 = time.time()
#                         overhead5 += (t10 - t9)
                        
                        curr_s_list = (curr_para - prev_para)+ 1e-16
                        
                        
#                         if is_GPU:
#                             curr_s_list = curr_s_list.to(device)
                        S_k_list.append(curr_s_list)
                        if len(S_k_list) > m:
                            removed_s_k = S_k_list.popleft()
                            
                            del removed_s_k
                        
                        
                            
    #                     print(i-1)
    #                     
    #                     print(S_k_list[:,i - 1])
                    
        #             init_model(model, para)
                    
#                     gradient_full = (expect_gradients*curr_rand_ids.shape[0] + gradient_remaining*curr_added_random_id_size)/(curr_added_random_id_size + curr_rand_ids.shape[0])

                    gradient_full = (expect_gradients*curr_rand_ids.shape[0] + gradient_remaining*curr_added_size)/(curr_rand_ids.shape[0] + curr_added_size)
                    
                    if k > 0 or (k == 0 and jj > 0):
                        
                        
#                         Y_k_list.append((expect_gradients - get_all_vectorized_parameters1(gradient_list_all_epochs[i]).to(device)).view(-1))
                        
                        Y_k_list.append((expect_gradients - grad_list_GPU_tensor[cached_id] + regularization_coeff*curr_s_list)+ 1e-16)
                        
                        if len(Y_k_list) > m:
                            removed_y_k = Y_k_list.popleft()
                            
                            del removed_y_k
                    
                    
                    
    #                 batch_X = X[curr_rand_ids]
    #                 
    #                 batch_Y = Y[curr_rand_ids]
    #                 clear_gradients(model.parameters())
    #                     
    #                 compute_derivative_one_more_step(model, error, batch_X, batch_Y, beta)
    #                 
    #                 expect_gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
                    
    #                 decompose_model_paras2(para, para_list_all_epochs[i], get_devectorized_parameters(gradient_full, input_dim, hidden_dims, output_dim), gradient_list_all_epochs[i])
                    
    #                     print(torch.dot(Y_k_list[:,i-1], S_k_list[:,i-1]))
    #                     y=0
    #                     y+=1
                    alpha = learning_rate_all_epochs[i]
                    
                    
                    res_para.append(curr_para)
                    
                    res_grad.append(gradient_full)
                    
                    para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_para - learning_rate*gradient_full, full_shape_list, shape_list)
#                     para = get_devectorized_parameters(params, full_shape_list, shape_list)
#                     
#                     para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradient_full, input_dim, hidden_dims, output_dim)
        #                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
                    
    
                    
                    
                    recorded += 1
                    
                    
                    del gradient_full
                    
                    del gradient_remaining
                    
                    del expect_gradients
                    
                    del batch_remaining_X
                    
                    del batch_remaining_Y
                    
                    if to_add:
                        
                        del batch_delta_X
                        
                        del batch_delta_Y
                    
                    if k > 0 or (k == 0 and jj > 0):
                        del prev_para
                    
                        del curr_para
                    
                    if recorded >= length:
                        use_standard_way = False
                
                
            else:
                
#                 print('epoch::', i)
                
                
    #             delta_X = X[delta_ids]
    #             
    #             delta_Y = Y[delta_ids]
#                 t1 = time.time()
                gradient_dual = None
    
#                 if curr_matched_ids_size > 0:
                if to_add:
                
#                     t3 = time.time()
                    init_model(model, para)
                    
                    
                    
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                    
#                     t4 = time.time()
#                     
#                     overhead2 += (t4 - t3)
                    
                    gradient_dual = model.get_all_gradient()
                
                with torch.no_grad():
                
                    curr_vec_para = get_all_vectorized_parameters1(para)
#                 t5 = time.time()
#                     v_vec = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
                    
                    vec_para_diff = torch.t((curr_vec_para - para_list_GPU_tensor[cached_id]))
                    
#                     compute_diff_vectorized_parameters2(para, para_list_all_epochs[i], vec_para_diff, shape_list, is_GPU, device)
    #                 v_vec2 = compute_model_para_diff2(para, para_list_all_epochs[i])
                    
#                     if i/period >= 1:
#                         if i % period == 1:
                    if (i-curr_init_epochs)/period >= 1:
                        if (i-curr_init_epochs) % period == 1:
    #                         print(i)
    #                         
    #                         if i >= 370:
    #                             y = 0
    #                             y+=1
                            zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, is_GPU, device)
#                             zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2 = prepare_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, period)
                            
                            
                            
                            mat = np.linalg.inv(mat_prime.cpu().numpy())
                            mat = torch.from_numpy(mat)
                            if is_GPU:
                                
                                
                                mat = mat.to(device)
                            
                    
                        hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff, is_GPU, device)
                        
                    else:
                        
                        hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period, is_GPU, device)
#                         hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
    #                     hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms2(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2, v_vec2, shape_list)
                        
#                     else:
#                         hessian_para_prod = cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
                    
                    
        #             print('para_diff::', torch.norm(v_vec))
        #             
        #             print('para_angle::', torch.dot(get_all_vectorized_parameters(para).view(-1), get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))/(torch.norm(get_all_vectorized_parameters(para).view(-1))*torch.norm(get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))))
                    
                    
                    
        #             init_model(model, para_list_all_epochs[i])
                     
        #             curr_vec_gradient,_ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, gradient_list_all_epochs[i], beta)
                     
        #             hessian_matrix1 = compute_hessian_matrix(model, curr_vec_gradient, input_dim, hidden_dims, output_dim)
                    
                    
                    
    #                 cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, v_vec.view(-1,1), period)
                    
                    
                    
                    
        #             hessian_para_prod = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1))
                    
        #             hessian_para_prod = cal_approx_hessian_vec_prod2(S_k_list, Y_k_list, i, m-1, v_vec.view(-1,1), last_v_vec.view(-1,1), last_gradient_full.view(-1, 1) - get_all_vectorized_parameters(gradient_list_all_epochs[i-1]).view(-1,1))
                    
        #             hessian_para_prod = cal_approx_hessian_vec_prod3(i, m, v_vec.view(-1,1), para_list_all_epochs, gradient_list_all_epochs)
                    
        #             hessian_para_prod, tmp_res = cal_approx_hessian_vec_prod4(truncted_s, extended_Y_k_list[i], i, m, v_vec.view(-1,1))
                    exp_gradient, exp_param = None, None
                    
                    delta_const = 0#compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
                    
        #             para_diff = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
                    
                    alpha = learning_rate_all_epochs[i]
                    
                    
                    if gradient_dual is not None:
                        is_positive, final_gradient_list = compute_grad_final3(curr_vec_para, torch.t(hessian_para_prod), get_all_vectorized_parameters1(gradient_dual), grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_added_size, learning_rate, regularization_coeff, is_GPU, device)
                        
                    else:
                        is_positive, final_gradient_list = compute_grad_final3(curr_vec_para, torch.t(hessian_para_prod), None, grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_added_size, learning_rate, regularization_coeff, is_GPU, device)
                    
#                     update_para_final(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, exp_gradient, exp_param)
                    vec_para = update_para_final2(curr_vec_para, final_gradient_list, learning_rate, regularization_coeff, exp_gradient, exp_param)
                    
                    
                    res_para.append(curr_vec_para)
                    
                    res_grad.append(final_gradient_list - regularization_coeff*curr_vec_para)
                    
                    para = get_devectorized_parameters(vec_para, full_shape_list, shape_list)
                    
#                     update_para_final(para, get_devectorized_parameters(hessian_para_prod, input_dim, hidden_dims, output_dim), gradient_dual, gradient_list_all_epochs[i], end_id - j, curr_added_size, alpha)
#                     gradient_full = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i])
                    
        #             delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1), error, model, beta)
        #             gradient_full2 = torch.mm(v_vec.view(1,-1), hessian_matrix1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
        #             gradient_full2 = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) - delta_const*v_vec.view(1,-1)
                    
#                     gradients = (gradient_full*(end_id - j) - gradient_dual*curr_matched_ids_size)/(end_id - j - curr_matched_ids_size)
                    
        #             gradients2 = (gradient_full2*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
                    
        #             print('hessian_vector_prod_diff::', torch.norm(torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1) - hessian_para_prod))
                    
        #             print('gradient_diff::', torch.norm(gradients - expect_gradients))
                    
        #             print('gradient2_diff::', torch.norm(gradients2 - expect_gradients))
                    
        #             compute_model_para_diff(exp_para_list_all_epochs[i], para)
                    
    #                 S_k_list[:,i-1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
                    
#                     para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradients, input_dim, hidden_dims, output_dim)
            
#                 t6 = time.time()
#                     
#                 overhead3 += (t6 - t5)
#                 Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
#                 print(torch.norm(get_all_vectorized_parameters(para)))
                
#                 print(Y_k_list[:,i-1])
#                 t2 = time.time()
                    
#                 overhead += (t2 - t1)
                 
                
            i = i + 1
            
            
            cached_id += 1
            
            if cached_id%cached_size == 0:
                
                GPU_tensor_end_id = (batch_id + 1)*cached_size
                
                if GPU_tensor_end_id > para_list_all_epochs_tensor.shape[0]:
                    GPU_tensor_end_id = para_list_all_epochs_tensor.shape[0] 
                print("end_tensor_id::", GPU_tensor_end_id)
                
                para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(para_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(gradient_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                batch_id += 1
                
                cached_id = 0
                
            
            id_start = id_end
            
            j += batch_size
            
#             last_gradient_full = gradient_full
#             
#             last_v_vec = v_vec.clone()
        
        
#         else:
#             if i >= 1:
#                 S_k_list[:,i - 1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
#             last_para = para
#             
#             para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*expect_gradients, input_dim, hidden_dims, output_dim)
#         
#             if i == m-1:
#                 init_model(model, para)
#                 
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             
#                 last_gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
#                 
#                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
#                 
#             if i >= 1:
#                 
#                 init_model(model, para)
#                 
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             
#                 gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
#                 
# #                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
#                 
#                 Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
        
#         last_gradient = expect_gradients
#             init_model(model, get_devectorized_parameters(updated_para, input_dim, hidden_dims, output_dim))
            
            
            
            
    print('overhead::', overhead)
    
    print('overhead2::', overhead2)
    
    print('overhead3::', overhead3)
    
    print('overhead4::', overhead4)
    
    print('overhead5::', overhead5)
    
            
    return para, res_para, res_grad

def model_update_provenance_test3_0(origin_train_set_len, period, length, init_epochs, truncted_s, extended_Y_k_list, exp_gradient_list_all_epochs, exp_para_list_all_epochs, dataset_train, model, gradient_list_all_epochs_tensor, para_list_all_epochs_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, epoch, m, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, added_random_ids_multi_super_iteration, X_to_add, Y_to_add, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device):
    
    
#     para_num = S_k_list.shape[0]
    
    
#     model_dual = DNNModel(input_dim, hidden_dims, output_dim)
#     selected_rows_set = set(selected_rows.view(-1).tolist())
    
    para = list(model.parameters())
    
#     expected_para = list(model.parameters())
#     
#     last_gradient_full = None
# 
#     last_para = None
    
    use_standard_way = False
    
    recorded = 0
    
    overhead = 0
    
#     vecorized_paras = get_all_vectorized_parameters_with_gradient(model.parameters())
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
#     shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    if not is_GPU:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    else:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double, device=device)
    
#     remaining_shape_num = 0
#     
#     for i in range(len(shape_list) - first_few_layer_num):
#         remaining_shape_num += shape_list[i+first_few_layer_num]
#         
#     S_k_list = torch.zeros(remaining_shape_num, len(random_ids_multi_super_iterations))
#     
#     
#     Y_k_list = torch.zeros(remaining_shape_num, len(random_ids_multi_super_iterations))
    
    i = 0
    
    S_k_list = deque()
    
    Y_k_list = deque()
    
    overhead2 = 0
    
    overhead3 = 0
    
    overhead4 = 0
    
    overhead5 = 0
    
    old_lr = 0
    
    cached_id = 0
    
    batch_id = 1
    
    res_para = []
    
    res_grad = []
    
    for k in range(len(random_ids_multi_super_iterations)):
    
        random_ids = random_ids_multi_super_iterations[k]
        
        added_to_random_ids = added_random_ids_multi_super_iteration[k]
        
#         sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
#         all_indexes = np.sort(sort_idx[delta_ids])
                
        id_start = 0
    
        id_end = 0
        
        j = 0
        
        to_add = True
        
        
        if k == 0:
             
            for p in range(len(added_to_random_ids)):
                if len(added_to_random_ids[p]) > 0:
                    i = p
                    break
                else:
                    res_para.append(None)
                    
                    res_grad.append(None)
                 
            added_to_random_ids = added_to_random_ids[i:]
#             i = torch.nonzero(torch.tensor(removed_batch_empty_list).view(-1) == False)[0].item()
            init_model(model, get_devectorized_parameters(para_list_all_epochs_tensor[i], full_shape_list, shape_list))
         
            para = list(model.parameters())
         
#             random_ids_list = random_ids_list[i:]
             
#             remaining_ids_list = remaining_ids_list[i:]
         
            j = batch_size*i
             
            cached_id = i
             
            curr_init_epochs = init_epochs + i
             
             
            if cached_id >= cached_size:
                 
                batch_id = cached_id/cached_size
                 
                GPU_tensor_end_id = (batch_id + 1)*cached_size
                 
                if GPU_tensor_end_id > para_list_all_epochs_tensor.shape[0]:
                    GPU_tensor_end_id = para_list_all_epochs_tensor.shape[0] 
                print("end_tensor_id::", GPU_tensor_end_id)
                 
                para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(para_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                 
                grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(gradient_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                 
                batch_id += 1
                 
                cached_id = 0
        
        
        
#         for j in range(0, dim[0], batch_size):
            
        for jj in range(len(added_to_random_ids)):
        
            end_id = j + batch_size
            
#             added_end_id = jj + added_batch_size
            curr_added_random_ids = added_to_random_ids[jj]
            
            if end_id > dim[0]:
                end_id = dim[0]
            
#             if added_end_id >= X_to_add.shape[0]:
#                 added_end_id = X_to_add.shape[0]
            
            
            if curr_added_random_ids.shape[0] <= 0:
                to_add = False
            else:
                to_add = True

            curr_added_size = 0

            

            if to_add:
                
#                 curr_added_random_ids = added_to_random_ids[jj:added_end_id]
                
                batch_delta_X = X_to_add[curr_added_random_ids]
                
                batch_delta_Y = Y_to_add[curr_added_random_ids]
            
                curr_added_size = curr_added_random_ids.shape[0]
                
                
                if is_GPU:
                    batch_delta_X = batch_delta_X.to(device)
                    
                    batch_delta_Y = batch_delta_Y.to(device)
                
            
            
            learning_rate = learning_rate_all_epochs[i]
            
            
#             if end_id - j - curr_matched_ids_size <= 0:
#                 
#                 i += 1
#                 
#                 continue
            
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate    
                
                      
            if (i-curr_init_epochs)%period == 0:
                
                recorded = 0
                
                use_standard_way = True
                
                
            if i< curr_init_epochs or use_standard_way == True:
                t7 = time.time()
                
                curr_rand_ids = random_ids[j:end_id]
            
            
                batch_remaining_X = dataset_train.data[curr_rand_ids]
                
                batch_remaining_Y = dataset_train.labels[curr_rand_ids]
                
                if is_GPU:
                    batch_remaining_X = batch_remaining_X.to(device)
                    
                    batch_remaining_Y = batch_remaining_Y.to(device)
                
                
                
                t8 = time.time()
            
                overhead4 += (t8 - t7)
                
                
                t5 = time.time()
                
                init_model(model, para)
                
                compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
                
                

                
                
                expect_gradients = get_all_vectorized_parameters1(model.get_all_gradient())
                
                t6 = time.time()

                overhead3 += (t6 - t5)
                
                gradient_remaining = 0
#                 if curr_matched_ids_size > 0:
                if to_add:
                    
                    t3 = time.time()
                    
                    clear_gradients(model.parameters())
                        
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                
                
                    gradient_remaining = get_all_vectorized_parameters1(model.get_all_gradient())     
                    
                    
                    t4 = time.time()
                
                
                    overhead2 += (t4  -t3)
                
                with torch.no_grad():
                               
                
                    curr_para = get_all_vectorized_parameters1(para)
                
                    if k > 0 or (k == 0 and jj > 0):
                        
                        
                        
                        
                        

#                         torch.cuda.synchronize()
                        
                        
#                         prev_para = get_all_vectorized_parameters1(para_list_all_epochs[i])
                        prev_para = para_list_GPU_tensor[cached_id]
                        
#                         t9 = time.time()
                        
#                         if is_GPU:
#                             prev_para = prev_para.to(device)
                        
#                         torch.cuda.synchronize()
#                         t10 = time.time()
#                         overhead5 += (t10 - t9)
                        
                        curr_s_list = (curr_para - prev_para)
                        
                        
#                         if is_GPU:
#                             curr_s_list = curr_s_list.to(device)
                        S_k_list.append(curr_s_list)
                        if len(S_k_list) > m:
                            removed_s_k = S_k_list.popleft()
                            
                            del removed_s_k
                        
                        
                            
    #                     print(i-1)
    #                     
    #                     print(S_k_list[:,i - 1])
                    
        #             init_model(model, para)
                    
#                     gradient_full = (expect_gradients*curr_rand_ids.shape[0] + gradient_remaining*curr_added_random_id_size)/(curr_added_random_id_size + curr_rand_ids.shape[0])

                    gradient_full = (expect_gradients*curr_rand_ids.shape[0] + gradient_remaining*curr_added_size)/(curr_rand_ids.shape[0] + curr_added_size)
                    
                    if k > 0 or (k == 0 and jj > 0):
                        
                        
#                         Y_k_list.append((expect_gradients - get_all_vectorized_parameters1(gradient_list_all_epochs[i]).to(device)).view(-1))
                        
                        Y_k_list.append((expect_gradients - grad_list_GPU_tensor[cached_id] + regularization_coeff*curr_s_list))
                        
                        if len(Y_k_list) > m:
                            removed_y_k = Y_k_list.popleft()
                            
                            del removed_y_k
                    
                    
                    
    #                 batch_X = X[curr_rand_ids]
    #                 
    #                 batch_Y = Y[curr_rand_ids]
    #                 clear_gradients(model.parameters())
    #                     
    #                 compute_derivative_one_more_step(model, error, batch_X, batch_Y, beta)
    #                 
    #                 expect_gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
                    
    #                 decompose_model_paras2(para, para_list_all_epochs[i], get_devectorized_parameters(gradient_full, input_dim, hidden_dims, output_dim), gradient_list_all_epochs[i])
                    
    #                     print(torch.dot(Y_k_list[:,i-1], S_k_list[:,i-1]))
    #                     y=0
    #                     y+=1
                    alpha = learning_rate_all_epochs[i]
                    
                    
                    res_para.append(curr_para)
                    
                    res_grad.append(gradient_full)
                    
                    para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_para - learning_rate*gradient_full, full_shape_list, shape_list)
#                     para = get_devectorized_parameters(params, full_shape_list, shape_list)
#                     
#                     para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradient_full, input_dim, hidden_dims, output_dim)
        #                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
                    
    
                    
                    
                    recorded += 1
                    
                    
                    del gradient_full
                    
                    del gradient_remaining
                    
                    del expect_gradients
                    
                    del batch_remaining_X
                    
                    del batch_remaining_Y
                    
                    if to_add:
                        
                        del batch_delta_X
                        
                        del batch_delta_Y
                    
                    if k > 0 or (k == 0 and jj > 0):
                        del prev_para
                    
                        del curr_para
                    
                    if recorded >= length:
                        use_standard_way = False
                
                
            else:
                
#                 print('epoch::', i)
                
                
    #             delta_X = X[delta_ids]
    #             
    #             delta_Y = Y[delta_ids]
#                 t1 = time.time()
                gradient_dual = None
    
#                 if curr_matched_ids_size > 0:
                if to_add:
                
#                     t3 = time.time()
                    init_model(model, para)
                    
                    
                    
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                    
#                     t4 = time.time()
#                     
#                     overhead2 += (t4 - t3)
                    
                    gradient_dual = model.get_all_gradient()
                
                with torch.no_grad():
                
                    curr_vec_para = get_all_vectorized_parameters1(para)
#                 t5 = time.time()
#                     v_vec = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
                    
                    vec_para_diff = torch.t((curr_vec_para - para_list_GPU_tensor[cached_id]))
                    
#                     compute_diff_vectorized_parameters2(para, para_list_all_epochs[i], vec_para_diff, shape_list, is_GPU, device)
    #                 v_vec2 = compute_model_para_diff2(para, para_list_all_epochs[i])
                    
#                     if i/period >= 1:
#                         if i % period == 1:
                    if (i-curr_init_epochs)/period >= 1:
                        if (i-curr_init_epochs) % period == 1:
    #                         print(i)
    #                         
    #                         if i >= 370:
    #                             y = 0
    #                             y+=1
                            zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, is_GPU, device)
#                             zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2 = prepare_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, period)
                            
                            
                            
                            mat = np.linalg.inv(mat_prime.cpu().numpy())
                            mat = torch.from_numpy(mat)
                            if is_GPU:
                                
                                
                                mat = mat.to(device)
                            
                    
                        hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff, is_GPU, device)
                        
                    else:
                        
                        hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period, is_GPU, device)
#                         hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
    #                     hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms2(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2, v_vec2, shape_list)
                        
#                     else:
#                         hessian_para_prod = cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
                    
                    
        #             print('para_diff::', torch.norm(v_vec))
        #             
        #             print('para_angle::', torch.dot(get_all_vectorized_parameters(para).view(-1), get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))/(torch.norm(get_all_vectorized_parameters(para).view(-1))*torch.norm(get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))))
                    
                    
                    
        #             init_model(model, para_list_all_epochs[i])
                     
        #             curr_vec_gradient,_ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, gradient_list_all_epochs[i], beta)
                     
        #             hessian_matrix1 = compute_hessian_matrix(model, curr_vec_gradient, input_dim, hidden_dims, output_dim)
                    
                    
                    
    #                 cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, v_vec.view(-1,1), period)
                    
                    
                    
                    
        #             hessian_para_prod = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1))
                    
        #             hessian_para_prod = cal_approx_hessian_vec_prod2(S_k_list, Y_k_list, i, m-1, v_vec.view(-1,1), last_v_vec.view(-1,1), last_gradient_full.view(-1, 1) - get_all_vectorized_parameters(gradient_list_all_epochs[i-1]).view(-1,1))
                    
        #             hessian_para_prod = cal_approx_hessian_vec_prod3(i, m, v_vec.view(-1,1), para_list_all_epochs, gradient_list_all_epochs)
                    
        #             hessian_para_prod, tmp_res = cal_approx_hessian_vec_prod4(truncted_s, extended_Y_k_list[i], i, m, v_vec.view(-1,1))
                    exp_gradient, exp_param = None, None
                    
                    delta_const = 0#compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
                    
        #             para_diff = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
                    
                    alpha = learning_rate_all_epochs[i]
                    
                    
                    if gradient_dual is not None:
                        is_positive, final_gradient_list = compute_grad_final3(curr_vec_para, torch.t(hessian_para_prod), get_all_vectorized_parameters1(gradient_dual), grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_added_size, learning_rate, regularization_coeff, is_GPU, device)
                        
                    else:
                        is_positive, final_gradient_list = compute_grad_final3(curr_vec_para, torch.t(hessian_para_prod), None, grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_added_size, learning_rate, regularization_coeff, is_GPU, device)
                    
#                     update_para_final(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, exp_gradient, exp_param)
                    vec_para = update_para_final2(curr_vec_para, final_gradient_list, learning_rate, regularization_coeff, exp_gradient, exp_param)
                    
                    
                    res_para.append(curr_vec_para)
                    
                    res_grad.append(final_gradient_list - regularization_coeff*curr_vec_para)
                    
                    para = get_devectorized_parameters(vec_para, full_shape_list, shape_list)
                    
#                     update_para_final(para, get_devectorized_parameters(hessian_para_prod, input_dim, hidden_dims, output_dim), gradient_dual, gradient_list_all_epochs[i], end_id - j, curr_added_size, alpha)
#                     gradient_full = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i])
                    
        #             delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1), error, model, beta)
        #             gradient_full2 = torch.mm(v_vec.view(1,-1), hessian_matrix1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
        #             gradient_full2 = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) - delta_const*v_vec.view(1,-1)
                    
#                     gradients = (gradient_full*(end_id - j) - gradient_dual*curr_matched_ids_size)/(end_id - j - curr_matched_ids_size)
                    
        #             gradients2 = (gradient_full2*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
                    
        #             print('hessian_vector_prod_diff::', torch.norm(torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1) - hessian_para_prod))
                    
        #             print('gradient_diff::', torch.norm(gradients - expect_gradients))
                    
        #             print('gradient2_diff::', torch.norm(gradients2 - expect_gradients))
                    
        #             compute_model_para_diff(exp_para_list_all_epochs[i], para)
                    
    #                 S_k_list[:,i-1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
                    
#                     para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradients, input_dim, hidden_dims, output_dim)
            
#                 t6 = time.time()
#                     
#                 overhead3 += (t6 - t5)
#                 Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
#                 print(torch.norm(get_all_vectorized_parameters(para)))
                
#                 print(Y_k_list[:,i-1])
#                 t2 = time.time()
                    
#                 overhead += (t2 - t1)
                 
                
            i = i + 1
            
            
            cached_id += 1
            
            if cached_id%cached_size == 0:
                
                GPU_tensor_end_id = (batch_id + 1)*cached_size
                
                if GPU_tensor_end_id > para_list_all_epochs_tensor.shape[0]:
                    GPU_tensor_end_id = para_list_all_epochs_tensor.shape[0] 
                print("end_tensor_id::", GPU_tensor_end_id)
                
                para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(para_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(gradient_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                batch_id += 1
                
                cached_id = 0
                
            
            id_start = id_end
            
            j += batch_size
            
#             last_gradient_full = gradient_full
#             
#             last_v_vec = v_vec.clone()
        
        
#         else:
#             if i >= 1:
#                 S_k_list[:,i - 1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
#             last_para = para
#             
#             para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*expect_gradients, input_dim, hidden_dims, output_dim)
#         
#             if i == m-1:
#                 init_model(model, para)
#                 
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             
#                 last_gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
#                 
#                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
#                 
#             if i >= 1:
#                 
#                 init_model(model, para)
#                 
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             
#                 gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
#                 
# #                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
#                 
#                 Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
        
#         last_gradient = expect_gradients
#             init_model(model, get_devectorized_parameters(updated_para, input_dim, hidden_dims, output_dim))
            
            
            
            
    print('overhead::', overhead)
    
    print('overhead2::', overhead2)
    
    print('overhead3::', overhead3)
    
    print('overhead4::', overhead4)
    
    print('overhead5::', overhead5)
    
            
    return para, res_para, res_grad


def model_update_provenance_test3_multi(all_res, period, length, init_epochs, res_para, res_grad, exp_gradient_list_all_epochs, exp_para_list_all_epochs, dataset_train, model, gradient_list_all_epochs_tensor, para_list_all_epochs_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, epoch, m, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, X_to_add, Y_to_add, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device, all_added_random_ids_list_all_samples):
    
    
#     para_num = S_k_list.shape[0]
    
    
#     model_dual = DNNModel(input_dim, hidden_dims, output_dim)
#     selected_rows_set = set(selected_rows.view(-1).tolist())
    
    para = list(model.parameters())
    
#     expected_para = list(model.parameters())
#     
#     last_gradient_full = None
# 
#     last_para = None
    
    use_standard_way = False
    
    recorded = 0
    
    overhead = 0
    
#     vecorized_paras = get_all_vectorized_parameters_with_gradient(model.parameters())
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
#     shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    if not is_GPU:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    else:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double, device=device)
    
#     remaining_shape_num = 0
#     
#     for i in range(len(shape_list) - first_few_layer_num):
#         remaining_shape_num += shape_list[i+first_few_layer_num]
#         
#     S_k_list = torch.zeros(remaining_shape_num, len(random_ids_multi_super_iterations))
#     
#     
#     Y_k_list = torch.zeros(remaining_shape_num, len(random_ids_multi_super_iterations))
    
    i = 0
    
    overhead2 = 0
    
    overhead3 = 0
    
    overhead4 = 0
    
    overhead5 = 0
    
    old_lr = 0
    
    cached_id = 0
    
    batch_id = 1
    
    prev_random_ids_list = None
    
    for r in range(X_to_add.shape[0]):
        
        
        S_k_list = deque()
    
        Y_k_list = deque()
        
        t5 = time.time()
        
        curr_X_add = X_to_add[r:r+1]
        
        curr_Y_add = Y_to_add[r:r+1]
        
        curr_delta_data_id = torch.tensor([r])
    
        curr_added_random_ids_list = all_added_random_ids_list_all_samples[r]
    
        last_explicit_training_iteration = 0           

#         curr_exp_gradient_list_all_epochs = exp_gradient_list_all_epochs[r]
#         
#         curr_exp_para_list_all_epochs = exp_para_list_all_epochs[r]
    
        for k in range(len(random_ids_multi_super_iterations)):
        
            random_ids = random_ids_multi_super_iterations[k]
            
            curr_added_random_ids_list_this_epoch = curr_added_random_ids_list[k]
            
            
            next_random_ids = []
    #         added_to_random_ids = added_random_ids_multi_super_iteration[k]
            
    #         sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
            
    #         all_indexes = np.sort(sort_idx[delta_ids])
                    
            id_start = 0
        
            id_end = 0
            
            j = 0
            
            to_add = True
            
            
            if k == 0:
             
                for p in range(len(curr_added_random_ids_list_this_epoch)):
                    if curr_delta_data_id[0].item() in curr_added_random_ids_list_this_epoch[p]:
                        i = p
                        break
                
                
#                 print("init_iters::", i)
                
                curr_added_random_ids_list_this_epoch = curr_added_random_ids_list_this_epoch[i:]
    #             i = torch.nonzero(torch.tensor(removed_batch_empty_list).view(-1) == False)[0].item()
                init_model(model, get_devectorized_parameters(para_list_all_epochs_tensor[i], full_shape_list, shape_list))
             
                para = list(model.parameters())
             
    #             random_ids_list = random_ids_list[i:]
                 
    #             remaining_ids_list = remaining_ids_list[i:]
             
                j = batch_size*i
                 
                cached_id = i
                 
                curr_init_epochs = init_epochs + i
                 
                 
                if cached_id >= cached_size:
                     
                    batch_id = cached_id/cached_size
                     
                    GPU_tensor_end_id = (batch_id + 1)*cached_size
                     
                    if GPU_tensor_end_id > para_list_all_epochs_tensor.shape[0]:
                        GPU_tensor_end_id = para_list_all_epochs_tensor.shape[0] 
                    print("end_tensor_id::", GPU_tensor_end_id)
                     
                    para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(para_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                     
                    grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(gradient_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                     
                    batch_id += 1
                     
                    cached_id = 0
            
#             for j in range(0, dim[0], batch_size):
            for jj in range(len(curr_added_random_ids_list_this_epoch)):
            
                end_id = j + batch_size
                
    #             added_end_id = jj + added_batch_size
                curr_added_random_ids = curr_added_random_ids_list_this_epoch[jj]
                
                if end_id > dim[0]:
                    end_id = dim[0]
                
    #             if added_end_id >= X_to_add.shape[0]:
    #                 added_end_id = X_to_add.shape[0]
                
                
#                 if curr_added_random_ids.shape[0] <= 0:
                if r not in curr_added_random_ids:
                    to_add = False
                else:
                    to_add = True
    
                curr_added_size = 0
    
                
    
                if to_add:
                    
    #                 curr_added_random_ids = added_to_random_ids[jj:added_end_id]
                    
                    batch_delta_X = curr_X_add
                    
                    batch_delta_Y = curr_Y_add
                
                    curr_added_size = 1#curr_added_random_ids.shape[0]
                    
                    
                    if is_GPU:
                        batch_delta_X = batch_delta_X.to(device)
                        
                        batch_delta_Y = batch_delta_Y.to(device)
                    
                
                
                learning_rate = learning_rate_all_epochs[i]
                
                
    #             if end_id - j - curr_matched_ids_size <= 0:
    #                 
    #                 i += 1
    #                 
    #                 continue
                
                if not learning_rate == old_lr:
                    update_learning_rate(optimizer, learning_rate)
                
                old_lr = learning_rate    
                
                
#                 if i == 15:
#                     print(curr_added_random_ids)
#                     print("here")
                
                
                
                          
                if (i-last_explicit_training_iteration)%period == 0:
                    
                    recorded = 0
                    
                    use_standard_way = True
                    
                    
                if i<= curr_init_epochs or use_standard_way == True:
                    
                    
#                     t7 = time.time()
                    
                    
                    last_explicit_training_iteration = i
                    
                    curr_rand_ids = random_ids[j:end_id]
                
                    
                    
                
                    batch_remaining_X = dataset_train.data[curr_rand_ids]
                    
                    batch_remaining_Y = dataset_train.labels[curr_rand_ids]
                    
                    if is_GPU:
                        batch_remaining_X = batch_remaining_X.to(device)
                        
                        batch_remaining_Y = batch_remaining_Y.to(device)
                    
                    
#                     t8 = time.time()
#                 
#                     overhead4 += (t8 - t7)
                    
                    
#                     t5 = time.time()
                    
                    init_model(model, para)
                    
#                     print(i, curr_rand_ids.shape, batch_remaining_X.shape)
                    
                    compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
                    
                    
    
                    
                    
                    expect_gradients1 = get_all_vectorized_parameters1(model.get_all_gradient())
                    
                    expect_gradients2 = 0
                    
                    
                    if curr_delta_data_id.shape[0] > 0:
                        extra_remaining_ids = get_remaining_subset_data_per_epoch(curr_added_random_ids, curr_delta_data_id)
                    else:
                        extra_remaining_ids = curr_added_random_ids
                    
                    if extra_remaining_ids.shape[0] > 0:
                        batch_extra_remaining_X = X_to_add[extra_remaining_ids]
                        
                        batch_extra_remaining_Y = Y_to_add[extra_remaining_ids]
                        
                        if is_GPU:
                            batch_extra_remaining_X = batch_extra_remaining_X.to(device)
                            
                            batch_extra_remaining_Y = batch_extra_remaining_Y.to(device)
                        
                        init_model(model, para)
                    
                        compute_derivative_one_more_step(model, batch_extra_remaining_X, batch_extra_remaining_Y, criterion, optimizer)
                        
                        expect_gradients2 = get_all_vectorized_parameters1(model.get_all_gradient())
                        
                    
                    expect_gradients = (expect_gradients1*curr_rand_ids.shape[0] + expect_gradients2*extra_remaining_ids.shape[0])/(curr_rand_ids.shape[0] + extra_remaining_ids.shape[0])
                    
#                     t6 = time.time()
#     
#                     overhead3 += (t6 - t5)
                    
                    gradient_remaining = 0
    #                 if curr_matched_ids_size > 0:
                    if to_add:
                        
                        t3 = time.time()
                        
                        clear_gradients(model.parameters())
                            
                        compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                    
                    
                        gradient_remaining = get_all_vectorized_parameters1(model.get_all_gradient())     
                        
                        
                        t4 = time.time()
                    
                    
                        overhead2 += (t4  -t3)
                    
                    with torch.no_grad():
                                   
                    
                        curr_para = get_all_vectorized_parameters1(para)
                    
                        if k > 0 or (k == 0 and jj > 0):
                            
                            
                            
                            
                            
    
    #                         torch.cuda.synchronize()
                            
                            
    #                         prev_para = get_all_vectorized_parameters1(para_list_all_epochs[i])
                            prev_para = para_list_GPU_tensor[cached_id]
                            
    #                         t9 = time.time()
                            
    #                         if is_GPU:
    #                             prev_para = prev_para.to(device)
                            
    #                         torch.cuda.synchronize()
    #                         t10 = time.time()
    #                         overhead5 += (t10 - t9)
                            
                            curr_s_list = (curr_para - prev_para)
                            
                            
    #                         if is_GPU:
    #                             curr_s_list = curr_s_list.to(device)
                            S_k_list.append(curr_s_list)
                            if len(S_k_list) > m:
                                removed_s_k = S_k_list.popleft()
                                
                                del removed_s_k
                            
                            
                                
        #                     print(i-1)
        #                     
        #                     print(S_k_list[:,i - 1])
                        
            #             init_model(model, para)
                        
    #                     gradient_full = (expect_gradients*curr_rand_ids.shape[0] + gradient_remaining*curr_added_random_id_size)/(curr_added_random_id_size + curr_rand_ids.shape[0])
    
                        gradient_full = (expect_gradients*(curr_rand_ids.shape[0] + extra_remaining_ids.shape[0]) + gradient_remaining*curr_added_size)/(curr_rand_ids.shape[0] + extra_remaining_ids.shape[0] + curr_added_size)
                        
                        if k > 0 or (k == 0 and jj > 0):
                            
                            
    #                         Y_k_list.append((expect_gradients - get_all_vectorized_parameters1(gradient_list_all_epochs[i]).to(device)).view(-1))
                            
                            Y_k_list.append((expect_gradients - grad_list_GPU_tensor[cached_id] + regularization_coeff*curr_s_list))
                            
                            if len(Y_k_list) > m:
                                removed_y_k = Y_k_list.popleft()
                                
                                del removed_y_k
                        
                        
                        
        #                 batch_X = X[curr_rand_ids]
        #                 
        #                 batch_Y = Y[curr_rand_ids]
        #                 clear_gradients(model.parameters())
        #                     
        #                 compute_derivative_one_more_step(model, error, batch_X, batch_Y, beta)
        #                 
        #                 expect_gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
                        
        #                 decompose_model_paras2(para, para_list_all_epochs[i], get_devectorized_parameters(gradient_full, input_dim, hidden_dims, output_dim), gradient_list_all_epochs[i])
                        
        #                     print(torch.dot(Y_k_list[:,i-1], S_k_list[:,i-1]))
        #                     y=0
        #                     y+=1
                        alpha = learning_rate_all_epochs[i]
                        
                        
#                         if r > 0:
#                             print("iteration1::", i)
#                             print("para diff::")
#                
#                             compute_model_para_diff(para, curr_exp_para_list_all_epochs[i])
#                               
#     #                             print(torch.norm(get_all_vectorized_parameters1(para) - para_list_all_epochs_tensor[i]))
#                                     
#                             print("gradient diff::") 
#                                    
#                             compute_model_para_diff(get_devectorized_parameters(gradient_full, full_shape_list, shape_list), curr_exp_gradient_list_all_epochs[i])
#                              
#                              
#                              
# #                             print("para diff2::")
# #                              
# #                             print(torch.norm(res_para[i] - get_all_vectorized_parameters1(para)))
# #                              
# #                             print("grad diff2::")
# #                              
# #                             print(torch.norm(res_grad[i] - gradient_full))
#                              
#                             print("here")
                        
                        
                        
                        para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_para - learning_rate*gradient_full, full_shape_list, shape_list)
    #                     para = get_devectorized_parameters(params, full_shape_list, shape_list)
    #                     
    #                     para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradient_full, input_dim, hidden_dims, output_dim)
            #                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
                        
                        para_list_GPU_tensor[cached_id] = curr_para
                        
                        grad_list_GPU_tensor[cached_id] = gradient_full
                        
                        
                        recorded += 1
                        
                        
                        del gradient_full
                        
                        del gradient_remaining
                        
                        del expect_gradients
                        
                        del batch_remaining_X
                        
                        del batch_remaining_Y
                        
                        if to_add:
                            
                            del batch_delta_X
                            
                            del batch_delta_Y
                        
                        if k > 0 or (k == 0 and jj > 0):
                            del prev_para
                        
                            del curr_para
                        
                        if recorded >= length:
                            use_standard_way = False
                    
                    
                else:
                    
    #                 print('epoch::', i)
                    
                    
        #             delta_X = X[delta_ids]
        #             
        #             delta_Y = Y[delta_ids]
    #                 t1 = time.time()
                    gradient_dual = None
        
                    curr_vec_para = get_all_vectorized_parameters1(para)
                    
        
    #                 if curr_matched_ids_size > 0:
                    if to_add:
                    
    #                     t3 = time.time()
                        init_model(model, para)
                        
                        
                        
                        compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                        
    #                     t4 = time.time()
    #                     
    #                     overhead2 += (t4 - t3)
                        
                        gradient_dual = model.get_all_gradient()
                    
                    
                    if curr_delta_data_id.shape[0] > 0:
                        extra_remaining_ids = get_remaining_subset_data_per_epoch(curr_added_random_ids, curr_delta_data_id)
                    else:
                        extra_remaining_ids = curr_added_random_ids
                    
                    
#                         extra_remaining_ids = get_remaining_subset_data_per_epoch(curr_added_random_ids, curr_delta_data_id)
                    
                    with torch.no_grad():
                    
                    
    #                 t5 = time.time()
    #                     v_vec = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
                        
                        vec_para_diff = torch.t((get_all_vectorized_parameters1(para) - para_list_GPU_tensor[cached_id]))
                        
    #                     compute_diff_vectorized_parameters2(para, para_list_all_epochs[i], vec_para_diff, shape_list, is_GPU, device)
        #                 v_vec2 = compute_model_para_diff2(para, para_list_all_epochs[i])
                        
    #                     if i/period >= 1:
    #                         if i % period == 1:
#                         if (i-last_explicit_training_iteration)/period >= 1:
                        if (i-last_explicit_training_iteration) % period == 1:
    #                         print(i)
    #                         
    #                         if i >= 370:
    #                             y = 0
    #                             y+=1
                            zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, is_GPU, device)
#                             zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2 = prepare_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, period)
                            
                            
                            
                            mat = np.linalg.inv(mat_prime.cpu().numpy())
                            mat = torch.from_numpy(mat)
                            if is_GPU:
                                
                                
                                mat = mat.to(device)
                            
                    
                        hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff, is_GPU, device)
                            
#                         else:
#                             
#                             hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period, is_GPU, device)
    #                         hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
        #                     hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms2(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2, v_vec2, shape_list)
                            
    #                     else:
    #                         hessian_para_prod = cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
                        
                        
            #             print('para_diff::', torch.norm(v_vec))
            #             
            #             print('para_angle::', torch.dot(get_all_vectorized_parameters(para).view(-1), get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))/(torch.norm(get_all_vectorized_parameters(para).view(-1))*torch.norm(get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))))
                        
                        
                        
            #             init_model(model, para_list_all_epochs[i])
                         
            #             curr_vec_gradient,_ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, gradient_list_all_epochs[i], beta)
                         
            #             hessian_matrix1 = compute_hessian_matrix(model, curr_vec_gradient, input_dim, hidden_dims, output_dim)
                        
                        
                        
        #                 cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, v_vec.view(-1,1), period)
                        
                        
                        
                        
            #             hessian_para_prod = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1))
                        
            #             hessian_para_prod = cal_approx_hessian_vec_prod2(S_k_list, Y_k_list, i, m-1, v_vec.view(-1,1), last_v_vec.view(-1,1), last_gradient_full.view(-1, 1) - get_all_vectorized_parameters(gradient_list_all_epochs[i-1]).view(-1,1))
                        
            #             hessian_para_prod = cal_approx_hessian_vec_prod3(i, m, v_vec.view(-1,1), para_list_all_epochs, gradient_list_all_epochs)
                        
            #             hessian_para_prod, tmp_res = cal_approx_hessian_vec_prod4(truncted_s, extended_Y_k_list[i], i, m, v_vec.view(-1,1))
                        exp_gradient, exp_param = None, None
                        
                        delta_const = 0#compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
                        
            #             para_diff = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
                        
                        alpha = learning_rate_all_epochs[i]
                        
                        
                        if gradient_dual is not None:
                            is_positive, final_gradient_list, exp_grad_list_full = compute_grad_final5(get_all_vectorized_parameters1(para), torch.t(hessian_para_prod), get_all_vectorized_parameters1(gradient_dual), grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j + extra_remaining_ids.shape[0], curr_added_size, learning_rate, regularization_coeff, is_GPU, device)
                            
                        else:
                            is_positive, final_gradient_list, exp_grad_list_full = compute_grad_final5(get_all_vectorized_parameters1(para), torch.t(hessian_para_prod), None, grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j + extra_remaining_ids.shape[0], curr_added_size, learning_rate, regularization_coeff, is_GPU, device)
                    
#                     if r>0:
#                         print("iteration2::", i)
#                         print("para diff::")
#            
#                         compute_model_para_diff(para, curr_exp_para_list_all_epochs[i])
#                           
#     #                             print(torch.norm(get_all_vectorized_parameters1(para) - para_list_all_epochs_tensor[i]))
#                                 
#                         print("gradient diff::") 
#                                
#                         compute_model_para_diff(get_devectorized_parameters(final_gradient_list - regularization_coeff*get_all_vectorized_parameters1(para), full_shape_list, shape_list), curr_exp_gradient_list_all_epochs[i])
#                            
#                          
#                         print("para diff2::")
#                          
#                         print(torch.norm(res_para[i] - get_all_vectorized_parameters1(para)))
#                          
#                         print("grad diff2::")
#                          
#                         print(torch.norm(res_grad[i] - (final_gradient_list - regularization_coeff*get_all_vectorized_parameters1(para))))  
#                            
#                          
#                         print("here")    
    #                     update_para_final(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, exp_gradient, exp_param)
#                     is_positive = True
                    if not is_positive:
                        last_explicit_training_iteration = i
                    
                        curr_rand_ids = random_ids[j:end_id]
                    
                        
                        
                    
                        batch_remaining_X = dataset_train.data[curr_rand_ids]
                        
                        batch_remaining_Y = dataset_train.labels[curr_rand_ids]
                        
                        if is_GPU:
                            batch_remaining_X = batch_remaining_X.to(device)
                            
                            batch_remaining_Y = batch_remaining_Y.to(device)
                        
                        
#                         t8 = time.time()
#                     
#                         overhead4 += (t8 - t7)
#                         
#                         
#                         t5 = time.time()
                        
                        init_model(model, para)
                        
#                         print(i, curr_rand_ids.shape, batch_remaining_X.shape)
                        
                        compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
                        
                        
        
                        
                        
                        expect_gradients1 = get_all_vectorized_parameters1(model.get_all_gradient())
                        
                        expect_gradients2 = 0
                        
#                         extra_remaining_ids = get_remaining_subset_data_per_epoch(curr_added_random_ids, curr_delta_data_id)
                        
                        if extra_remaining_ids.shape[0] > 0:
                            batch_extra_remaining_X = X_to_add[extra_remaining_ids]
                            
                            batch_extra_remaining_Y = Y_to_add[extra_remaining_ids]
                            
                            if is_GPU:
                                batch_extra_remaining_X = batch_extra_remaining_X.to(device)
                                
                                batch_extra_remaining_Y = batch_extra_remaining_Y.to(device)
                            
                            init_model(model, para)
                        
                            compute_derivative_one_more_step(model, batch_extra_remaining_X, batch_extra_remaining_Y, criterion, optimizer)
                            
                            expect_gradients2 = get_all_vectorized_parameters1(model.get_all_gradient())
                            
                        
                        expect_gradients = (expect_gradients1*curr_rand_ids.shape[0] + expect_gradients2*extra_remaining_ids.shape[0])/(curr_rand_ids.shape[0] + extra_remaining_ids.shape[0])
                        
#                         t6 = time.time()
#         
#                         overhead3 += (t6 - t5)
                        
                        gradient_remaining = 0
        #                 if curr_matched_ids_size > 0:
                        if to_add:
                            
                            t3 = time.time()
                            
                            clear_gradients(model.parameters())
                                
                            compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                        
                        
                            gradient_remaining = get_all_vectorized_parameters1(model.get_all_gradient())     
                            
                            
                            t4 = time.time()
                        
                        
                            overhead2 += (t4  -t3)
                        
                        with torch.no_grad():
                                       
                        
#                             curr_para = get_all_vectorized_parameters1(para)
                            curr_para = curr_vec_para
                        
                            if k > 0 or (k == 0 and jj > 0):
                                
                                
                                
                                
                                
        
        #                         torch.cuda.synchronize()
                                
                                
        #                         prev_para = get_all_vectorized_parameters1(para_list_all_epochs[i])
                                prev_para = para_list_GPU_tensor[cached_id]
                                
        #                         t9 = time.time()
                                
        #                         if is_GPU:
        #                             prev_para = prev_para.to(device)
                                
        #                         torch.cuda.synchronize()
        #                         t10 = time.time()
        #                         overhead5 += (t10 - t9)
                                
                                curr_s_list = (curr_para - prev_para)
                                
                                
        #                         if is_GPU:
        #                             curr_s_list = curr_s_list.to(device)
                                S_k_list.append(curr_s_list)
                                if len(S_k_list) > m:
                                    removed_s_k = S_k_list.popleft()
                                    
                                    del removed_s_k
                                
                                
                                    
            #                     print(i-1)
            #                     
            #                     print(S_k_list[:,i - 1])
                            
                #             init_model(model, para)
                            
        #                     gradient_full = (expect_gradients*curr_rand_ids.shape[0] + gradient_remaining*curr_added_random_id_size)/(curr_added_random_id_size + curr_rand_ids.shape[0])
        
                            gradient_full = (expect_gradients*(curr_rand_ids.shape[0] + extra_remaining_ids.shape[0]) + gradient_remaining*curr_added_size)/(curr_rand_ids.shape[0] + extra_remaining_ids.shape[0] + curr_added_size)
                            
                            if k > 0 or (k == 0 and jj > 0):
                                
                                
        #                         Y_k_list.append((expect_gradients - get_all_vectorized_parameters1(gradient_list_all_epochs[i]).to(device)).view(-1))
                                
                                Y_k_list.append((expect_gradients - grad_list_GPU_tensor[cached_id] + regularization_coeff*curr_s_list))
                                
                                if len(Y_k_list) > m:
                                    removed_y_k = Y_k_list.popleft()
                                    
                                    del removed_y_k
                            
                            
                            
            #                 batch_X = X[curr_rand_ids]
            #                 
            #                 batch_Y = Y[curr_rand_ids]
            #                 clear_gradients(model.parameters())
            #                     
            #                 compute_derivative_one_more_step(model, error, batch_X, batch_Y, beta)
            #                 
            #                 expect_gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
                            
            #                 decompose_model_paras2(para, para_list_all_epochs[i], get_devectorized_parameters(gradient_full, input_dim, hidden_dims, output_dim), gradient_list_all_epochs[i])
                            
            #                     print(torch.dot(Y_k_list[:,i-1], S_k_list[:,i-1]))
            #                     y=0
            #                     y+=1
                            alpha = learning_rate_all_epochs[i]
                            
                            
#                             if r > 0:
#                                 print("iteration1::", i)
#                                 print("para diff::")
#                    
#                                 compute_model_para_diff(para, curr_exp_para_list_all_epochs[i])
#                                   
#         #                             print(torch.norm(get_all_vectorized_parameters1(para) - para_list_all_epochs_tensor[i]))
#                                         
#                                 print("gradient diff::") 
#                                        
#                                 compute_model_para_diff(get_devectorized_parameters(gradient_full, full_shape_list, shape_list), curr_exp_gradient_list_all_epochs[i])
#                                  
#                                  
#                                  
#                                 print("para diff2::")
#                                  
#                                 print(torch.norm(res_para[i] - get_all_vectorized_parameters1(para)))
#                                  
#                                 print("grad diff2::")
#                                  
#                                 print(torch.norm(res_grad[i] - gradient_full))
#                                  
#                                 print("here")
                            
                            
                            
                            para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_para - learning_rate*gradient_full, full_shape_list, shape_list)
        #                     para = get_devectorized_parameters(params, full_shape_list, shape_list)
        #                     
        #                     para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradient_full, input_dim, hidden_dims, output_dim)
                #                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
                            
                            para_list_GPU_tensor[cached_id] = curr_para
                            
                            grad_list_GPU_tensor[cached_id] = gradient_full
                            
                            
                            recorded += 1
                            
                            
                            del gradient_full
                            
                            del gradient_remaining
                            
                            del expect_gradients
                            
                            del batch_remaining_X
                            
                            del batch_remaining_Y
                            
                            if to_add:
                                
                                del batch_delta_X
                                
                                del batch_delta_Y
                            
                            if k > 0 or (k == 0 and jj > 0):
                                del prev_para
                            
                                del curr_para
                            
                            if recorded >= length:
                                use_standard_way = False
                        
                    else:
                    
                    
                        para_list_GPU_tensor[cached_id] = curr_vec_para
                            
                        grad_list_GPU_tensor[cached_id] = exp_grad_list_full
                    
                    
                        
                        vec_para = update_para_final2(curr_vec_para, final_gradient_list, learning_rate, regularization_coeff, exp_gradient, exp_param)
                        
                        para = get_devectorized_parameters(vec_para, full_shape_list, shape_list)

                        
    #                     update_para_final(para, get_devectorized_parameters(hessian_para_prod, input_dim, hidden_dims, output_dim), gradient_dual, gradient_list_all_epochs[i], end_id - j, curr_added_size, alpha)
    #                     gradient_full = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i])
                        
            #             delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1), error, model, beta)
            #             gradient_full2 = torch.mm(v_vec.view(1,-1), hessian_matrix1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
            #             gradient_full2 = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) - delta_const*v_vec.view(1,-1)
                        
    #                     gradients = (gradient_full*(end_id - j) - gradient_dual*curr_matched_ids_size)/(end_id - j - curr_matched_ids_size)
                        
            #             gradients2 = (gradient_full2*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
                        
            #             print('hessian_vector_prod_diff::', torch.norm(torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1) - hessian_para_prod))
                        
            #             print('gradient_diff::', torch.norm(gradients - expect_gradients))
                        
            #             print('gradient2_diff::', torch.norm(gradients2 - expect_gradients))
                        
            #             compute_model_para_diff(exp_para_list_all_epochs[i], para)
                        
        #                 S_k_list[:,i-1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
                        
    #                     para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradients, input_dim, hidden_dims, output_dim)
                
    #                 t6 = time.time()
    #                     
    #                 overhead3 += (t6 - t5)
    #                 Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
    #                 print(torch.norm(get_all_vectorized_parameters(para)))
                    
    #                 print(Y_k_list[:,i-1])
    #                 t2 = time.time()
                        
    #                 overhead += (t2 - t1)
                     
                    
                i = i + 1
                
                
                cached_id += 1
                
                if cached_id%cached_size == 0:
                    
                    GPU_tensor_end_id = (batch_id + 1)*cached_size
                    
                    if GPU_tensor_end_id > para_list_all_epochs_tensor.shape[0]:
                        GPU_tensor_end_id = para_list_all_epochs_tensor.shape[0] 
                    print("end_tensor_id::", GPU_tensor_end_id)
                    
                    para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(para_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                    
                    grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(gradient_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                    
                    batch_id += 1
                    
                    cached_id = 0
                    
                
                id_start = id_end
                
                j += batch_size
        
        
        
        t6 = time.time()
        
        overhead3 += (t6  -t5)
        
        if r % 10 == 0:
            
#             print('Train - Epoch %d, Batch: %d, Loss: %f' % (j, i, loss.detach().cpu().item()))
            print("Num of deletion:: %d, running time provenance::%f" %(r, overhead3))
        
        
        
#         print(all_res)
#         compute_model_para_diff(para, all_res[r])
#         print(torch.norm(all_res[r] - get_all_vectorized_parameters1(para)))
           
#             last_gradient_full = gradient_full
#             
#             last_v_vec = v_vec.clone()
        
        
#         else:
#             if i >= 1:
#                 S_k_list[:,i - 1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
#             last_para = para
#             
#             para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*expect_gradients, input_dim, hidden_dims, output_dim)
#         
#             if i == m-1:
#                 init_model(model, para)
#                 
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             
#                 last_gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
#                 
#                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
#                 
#             if i >= 1:
#                 
#                 init_model(model, para)
#                 
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             
#                 gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
#                 
# #                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
#                 
#                 Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
        
#         last_gradient = expect_gradients
#             init_model(model, get_devectorized_parameters(updated_para, input_dim, hidden_dims, output_dim))
            
            
            
            
    print('overhead::', overhead)
    
    print('overhead2::', overhead2)
    
    print('overhead3::', overhead3)
    
    print('overhead4::', overhead4)
    
    print('overhead5::', overhead5)
    
            
    return para

def model_update_provenance_test3_multi0(origin_train_data_size, all_res, period, length, init_epochs, res_para, res_grad, exp_gradient_list_all_epochs, exp_para_list_all_epochs, dataset_train, model, gradient_list_all_epochs_tensor, para_list_all_epochs_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, max_epoch, m, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, X_to_add, Y_to_add, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device, all_added_random_ids_list_all_samples):
    
    
#     para_num = S_k_list.shape[0]
    
    
#     model_dual = DNNModel(input_dim, hidden_dims, output_dim)
#     selected_rows_set = set(selected_rows.view(-1).tolist())
    
    para = list(model.parameters())
    
#     expected_para = list(model.parameters())
#     
#     last_gradient_full = None
# 
#     last_para = None
    
    use_standard_way = False
    
    recorded = 0
    
    overhead = 0
    
#     vecorized_paras = get_all_vectorized_parameters_with_gradient(model.parameters())
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
#     shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    if not is_GPU:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    else:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double, device=device)
    
#     remaining_shape_num = 0
#     
#     for i in range(len(shape_list) - first_few_layer_num):
#         remaining_shape_num += shape_list[i+first_few_layer_num]
#         
#     S_k_list = torch.zeros(remaining_shape_num, len(random_ids_multi_super_iterations))
#     
#     
#     Y_k_list = torch.zeros(remaining_shape_num, len(random_ids_multi_super_iterations))
    
    
    
    overhead2 = 0
    
    overhead3 = 0
    
    overhead4 = 0
    
    overhead5 = 0
    
    old_lr = 0
    
    cached_id = 0
    
    batch_id = 1
    
    prev_random_ids_list = None
    
    for r in range(X_to_add.shape[0]):
        
        i = 0
        
        cached_id = 0
        
        S_k_list = deque()
    
        Y_k_list = deque()
        
        t5 = time.time()
        
        curr_X_add = X_to_add[r:r+1]
        
        curr_Y_add = Y_to_add[r:r+1]
        
        curr_delta_data_id = torch.tensor([r+origin_train_data_size])
    
        curr_added_random_ids_list = all_added_random_ids_list_all_samples[r]
    
        last_explicit_training_iteration = 0           

#         curr_exp_gradient_list_all_epochs = exp_gradient_list_all_epochs[r]
#         
#         curr_exp_para_list_all_epochs = exp_para_list_all_epochs[r]
    
#         for k in range(len(random_ids_multi_super_iterations)):
        for k in range(max_epoch):
        
            random_ids = random_ids_multi_super_iterations[k]
            
            curr_added_random_ids_list_this_epoch = curr_added_random_ids_list[k]
            
            
            next_random_ids = []
    #         added_to_random_ids = added_random_ids_multi_super_iteration[k]
            
    #         sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
            
    #         all_indexes = np.sort(sort_idx[delta_ids])
                    
            id_start = 0
        
            id_end = 0
            
            j = 0
            
            to_add = True
            
            curr_init_epochs = init_epochs
            if k == 0:
              
#                 for p in range(len(curr_added_random_ids_list_this_epoch)):
#                     if curr_delta_data_id[0].item() in curr_added_random_ids_list_this_epoch[p]:
#                         i = p
#                         break
#                  
#                  
# #                 print("init_iters::", i)
#                  
#                 curr_added_random_ids_list_this_epoch = curr_added_random_ids_list_this_epoch[i:]
    #             i = torch.nonzero(torch.tensor(removed_batch_empty_list).view(-1) == False)[0].item()
                init_model(model, get_devectorized_parameters(para_list_all_epochs_tensor[i], full_shape_list, shape_list))
              
                para = list(model.parameters())
              
    #             random_ids_list = random_ids_list[i:]
                  
    #             remaining_ids_list = remaining_ids_list[i:]
              
                j = batch_size*i
                  
                cached_id = i
                  
                curr_init_epochs = init_epochs + i
                  
                  
                if cached_id >= cached_size:
                      
                    batch_id = cached_id/cached_size
                      
                    GPU_tensor_end_id = (batch_id + 1)*cached_size
                      
                    if GPU_tensor_end_id > para_list_all_epochs_tensor.shape[0]:
                        GPU_tensor_end_id = para_list_all_epochs_tensor.shape[0] 
                    print("end_tensor_id::", GPU_tensor_end_id)
                      
                    para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(para_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                      
                    grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(gradient_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                      
                    batch_id += 1
                      
                    cached_id = 0
            
#             for j in range(0, dim[0], batch_size):
            for jj in range(len(curr_added_random_ids_list_this_epoch)):
            
                end_id = j + batch_size
                
    #             added_end_id = jj + added_batch_size
                curr_added_random_ids = curr_added_random_ids_list_this_epoch[jj]
                
                if end_id > dim[0]:
                    end_id = dim[0]
                
    #             if added_end_id >= X_to_add.shape[0]:
    #                 added_end_id = X_to_add.shape[0]
                
                
#                 if curr_added_random_ids.shape[0] <= 0:
                if curr_delta_data_id[0].item() not in curr_added_random_ids:
                    to_add = False
                else:
                    to_add = True
    
                curr_added_size = 0
    
                
    
                if to_add:
                    
    #                 curr_added_random_ids = added_to_random_ids[jj:added_end_id]
                    
                    batch_delta_X = curr_X_add
                    
                    batch_delta_Y = curr_Y_add
                
                    curr_added_size = 1#curr_added_random_ids.shape[0]
                    
                    
                    if is_GPU:
                        batch_delta_X = batch_delta_X.to(device)
                        
                        batch_delta_Y = batch_delta_Y.to(device)
                    
                
                
                learning_rate = learning_rate_all_epochs[i]
                
                
    #             if end_id - j - curr_matched_ids_size <= 0:
    #                 
    #                 i += 1
    #                 
    #                 continue
                
                if not learning_rate == old_lr:
                    update_learning_rate(optimizer, learning_rate)
                
                old_lr = learning_rate    
                
                
#                 if i == 15:
#                     print(curr_added_random_ids)
#                     print("here")
                
                
                
                          
                if (i-last_explicit_training_iteration)%period == 0:
                    
                    recorded = 0
                    
                    use_standard_way = True
                    
                    
                if i<= curr_init_epochs or use_standard_way == True:
                    
                    
#                     t7 = time.time()
                    
                    
                    last_explicit_training_iteration = i
                    
                    curr_rand_ids = random_ids[j:end_id]
                    
                    if curr_delta_data_id.shape[0] > 0:
                        extra_remaining_ids = get_remaining_subset_data_per_epoch(curr_added_random_ids, curr_delta_data_id)
                    else:
                        extra_remaining_ids = curr_added_random_ids
                    
                    
                    
                    if extra_remaining_ids.shape[0] > 0:
                
                        full_random_ids = torch.cat([curr_rand_ids, extra_remaining_ids], 0)
                    else:
                        full_random_ids = curr_rand_ids
                    
                
                    batch_remaining_X = dataset_train.data[full_random_ids]
                    
                    batch_remaining_Y = dataset_train.labels[full_random_ids]
                    
                    if is_GPU:
                        batch_remaining_X = batch_remaining_X.to(device)
                        
                        batch_remaining_Y = batch_remaining_Y.to(device)
                    
                    
#                     t8 = time.time()
#                 
#                     overhead4 += (t8 - t7)
                    
                    
#                     t5 = time.time()
                    
                    init_model(model, para)
                    
#                     print(i, curr_rand_ids.shape, batch_remaining_X.shape)
                    
                    compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
                    
                    
    
                    
                    
                    expect_gradients = get_all_vectorized_parameters1(model.get_all_gradient())
                    
#                     expect_gradients2 = 0
#                     
#                     
#                     if curr_delta_data_id.shape[0] > 0:
#                         extra_remaining_ids = get_remaining_subset_data_per_epoch(curr_added_random_ids, curr_delta_data_id)
#                     else:
#                         extra_remaining_ids = curr_added_random_ids
#                     
#                     if extra_remaining_ids.shape[0] > 0:
#                         batch_extra_remaining_X = X_to_add[extra_remaining_ids]
#                         
#                         batch_extra_remaining_Y = Y_to_add[extra_remaining_ids]
#                         
#                         if is_GPU:
#                             batch_extra_remaining_X = batch_extra_remaining_X.to(device)
#                             
#                             batch_extra_remaining_Y = batch_extra_remaining_Y.to(device)
#                         
#                         init_model(model, para)
#                     
#                         compute_derivative_one_more_step(model, batch_extra_remaining_X, batch_extra_remaining_Y, criterion, optimizer)
#                         
#                         expect_gradients2 = get_all_vectorized_parameters1(model.get_all_gradient())
#                         
#                     
#                     expect_gradients = (expect_gradients1*curr_rand_ids.shape[0] + expect_gradients2*extra_remaining_ids.shape[0])/(curr_rand_ids.shape[0] + extra_remaining_ids.shape[0])
                    
#                     t6 = time.time()
#     
#                     overhead3 += (t6 - t5)
                    
                    gradient_remaining = 0
    #                 if curr_matched_ids_size > 0:
                    if to_add:
                        
                        t3 = time.time()
                        
                        clear_gradients(model.parameters())
                            
                        compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                    
                    
                        gradient_remaining = get_all_vectorized_parameters1(model.get_all_gradient())     
                        
                        
                        t4 = time.time()
                    
                    
                        overhead2 += (t4  -t3)
                    
                    with torch.no_grad():
                                   
                    
                        curr_para = get_all_vectorized_parameters1(para)
                    
                        if k > 0 or (k == 0 and jj > 0):
                            
                            
                            
                            
                            
    
    #                         torch.cuda.synchronize()
                            
                            
    #                         prev_para = get_all_vectorized_parameters1(para_list_all_epochs[i])
                            prev_para = para_list_GPU_tensor[cached_id]
                            
    #                         t9 = time.time()
                            
    #                         if is_GPU:
    #                             prev_para = prev_para.to(device)
                            
    #                         torch.cuda.synchronize()
    #                         t10 = time.time()
    #                         overhead5 += (t10 - t9)
                            
                            curr_s_list = (curr_para - prev_para) + 1e-16
                            
                            
    #                         if is_GPU:
    #                             curr_s_list = curr_s_list.to(device)
                            S_k_list.append(curr_s_list)
                            if len(S_k_list) > m:
                                removed_s_k = S_k_list.popleft()
                                
                                del removed_s_k
                            
                            
                                
        #                     print(i-1)
        #                     
        #                     print(S_k_list[:,i - 1])
                        
            #             init_model(model, para)
                        
    #                     gradient_full = (expect_gradients*curr_rand_ids.shape[0] + gradient_remaining*curr_added_random_id_size)/(curr_added_random_id_size + curr_rand_ids.shape[0])
    
                        gradient_full = (expect_gradients*full_random_ids.shape[0] + gradient_remaining*curr_added_size)/(full_random_ids.shape[0] + curr_added_size)
                        
                        if k > 0 or (k == 0 and jj > 0):
                            
                            
    #                         Y_k_list.append((expect_gradients - get_all_vectorized_parameters1(gradient_list_all_epochs[i]).to(device)).view(-1))
                            
                            Y_k_list.append((expect_gradients - grad_list_GPU_tensor[cached_id] + regularization_coeff*curr_s_list) + 1e-16)
                            
                            if len(Y_k_list) > m:
                                removed_y_k = Y_k_list.popleft()
                                
                                del removed_y_k
                        
                        
                        
        #                 batch_X = X[curr_rand_ids]
        #                 
        #                 batch_Y = Y[curr_rand_ids]
        #                 clear_gradients(model.parameters())
        #                     
        #                 compute_derivative_one_more_step(model, error, batch_X, batch_Y, beta)
        #                 
        #                 expect_gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
                        
        #                 decompose_model_paras2(para, para_list_all_epochs[i], get_devectorized_parameters(gradient_full, input_dim, hidden_dims, output_dim), gradient_list_all_epochs[i])
                        
        #                     print(torch.dot(Y_k_list[:,i-1], S_k_list[:,i-1]))
        #                     y=0
        #                     y+=1
                        alpha = learning_rate_all_epochs[i]
                        
                        
#                         if r > 0:
#                             print("iteration1::", i)
#                             print("para diff::")
#                
#                             compute_model_para_diff(para, curr_exp_para_list_all_epochs[i])
#                               
#     #                             print(torch.norm(get_all_vectorized_parameters1(para) - para_list_all_epochs_tensor[i]))
#                                     
#                             print("gradient diff::") 
#                                    
#                             compute_model_para_diff(get_devectorized_parameters(gradient_full, full_shape_list, shape_list), curr_exp_gradient_list_all_epochs[i])
#                              
#                              
#                              
# #                             print("para diff2::")
# #                              
# #                             print(torch.norm(res_para[i] - get_all_vectorized_parameters1(para)))
# #                              
# #                             print("grad diff2::")
# #                              
# #                             print(torch.norm(res_grad[i] - gradient_full))
#                              
#                             print("here")
                        
                        
                        
                        para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_para - learning_rate*gradient_full, full_shape_list, shape_list)
    #                     para = get_devectorized_parameters(params, full_shape_list, shape_list)
    #                     
    #                     para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradient_full, input_dim, hidden_dims, output_dim)
            #                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
                        
                        para_list_GPU_tensor[cached_id] = curr_para
                        
                        grad_list_GPU_tensor[cached_id] = gradient_full
                        
                        
                        recorded += 1
                        
                        
                        del gradient_full
                        
                        del gradient_remaining
                        
                        del expect_gradients
                        
                        del batch_remaining_X
                        
                        del batch_remaining_Y
                        
                        if to_add:
                            
                            del batch_delta_X
                            
                            del batch_delta_Y
                        
                        if k > 0 or (k == 0 and jj > 0):
                            del prev_para
                        
                            del curr_para
                        
                        if recorded >= length:
                            use_standard_way = False
                    
                    
                else:
                    
    #                 print('epoch::', i)
                    
                    
        #             delta_X = X[delta_ids]
        #             
        #             delta_Y = Y[delta_ids]
    #                 t1 = time.time()
                    gradient_dual = None
        
                    curr_vec_para = get_all_vectorized_parameters1(para)
                    
        
    #                 if curr_matched_ids_size > 0:
                    if to_add:
                    
    #                     t3 = time.time()
                        init_model(model, para)
                        
                        
                        
                        compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                        
    #                     t4 = time.time()
    #                     
    #                     overhead2 += (t4 - t3)
                        
                        gradient_dual = model.get_all_gradient()
                    
                    
                    if curr_delta_data_id.shape[0] > 0:
                        extra_remaining_ids = get_remaining_subset_data_per_epoch(curr_added_random_ids, curr_delta_data_id)
                    else:
                        extra_remaining_ids = curr_added_random_ids
                    
                    
#                         extra_remaining_ids = get_remaining_subset_data_per_epoch(curr_added_random_ids, curr_delta_data_id)
                    
                    with torch.no_grad():
                    
                    
    #                 t5 = time.time()
    #                     v_vec = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
                        
                        vec_para_diff = torch.t((get_all_vectorized_parameters1(para) - para_list_GPU_tensor[cached_id]))
                        
    #                     compute_diff_vectorized_parameters2(para, para_list_all_epochs[i], vec_para_diff, shape_list, is_GPU, device)
        #                 v_vec2 = compute_model_para_diff2(para, para_list_all_epochs[i])
                        
    #                     if i/period >= 1:
    #                         if i % period == 1:
#                         if (i-last_explicit_training_iteration)/period >= 1:
                        if (i-last_explicit_training_iteration) % period == 1:
    #                         print(i)
    #                         
    #                         if i >= 370:
    #                             y = 0
    #                             y+=1
                            zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, is_GPU, device)
#                             zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2 = prepare_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, period)
                            
                            
                            
                            mat = np.linalg.inv(mat_prime.cpu().numpy())
                            mat = torch.from_numpy(mat)
                            if is_GPU:
                                
                                
                                mat = mat.to(device)
                            
                    
                        hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff, is_GPU, device)
                            
#                         else:
#                             
#                             hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period, is_GPU, device)
    #                         hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
        #                     hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms2(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2, v_vec2, shape_list)
                            
    #                     else:
    #                         hessian_para_prod = cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
                        
                        
            #             print('para_diff::', torch.norm(v_vec))
            #             
            #             print('para_angle::', torch.dot(get_all_vectorized_parameters(para).view(-1), get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))/(torch.norm(get_all_vectorized_parameters(para).view(-1))*torch.norm(get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))))
                        
                        
                        
            #             init_model(model, para_list_all_epochs[i])
                         
            #             curr_vec_gradient,_ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, gradient_list_all_epochs[i], beta)
                         
            #             hessian_matrix1 = compute_hessian_matrix(model, curr_vec_gradient, input_dim, hidden_dims, output_dim)
                        
                        
                        
        #                 cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, v_vec.view(-1,1), period)
                        
                        
                        
                        
            #             hessian_para_prod = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1))
                        
            #             hessian_para_prod = cal_approx_hessian_vec_prod2(S_k_list, Y_k_list, i, m-1, v_vec.view(-1,1), last_v_vec.view(-1,1), last_gradient_full.view(-1, 1) - get_all_vectorized_parameters(gradient_list_all_epochs[i-1]).view(-1,1))
                        
            #             hessian_para_prod = cal_approx_hessian_vec_prod3(i, m, v_vec.view(-1,1), para_list_all_epochs, gradient_list_all_epochs)
                        
            #             hessian_para_prod, tmp_res = cal_approx_hessian_vec_prod4(truncted_s, extended_Y_k_list[i], i, m, v_vec.view(-1,1))
                        exp_gradient, exp_param = None, None
                        
                        delta_const = 0#compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
                        
            #             para_diff = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
                        
                        alpha = learning_rate_all_epochs[i]
                        
                        
                        if gradient_dual is not None:
                            is_positive, final_gradient_list, exp_grad_list_full = compute_grad_final5(get_all_vectorized_parameters1(para), torch.t(hessian_para_prod), get_all_vectorized_parameters1(gradient_dual), grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j + extra_remaining_ids.shape[0], curr_added_size, learning_rate, regularization_coeff, is_GPU, device)
                            
                        else:
                            is_positive, final_gradient_list, exp_grad_list_full = compute_grad_final5(get_all_vectorized_parameters1(para), torch.t(hessian_para_prod), None, grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j + extra_remaining_ids.shape[0], curr_added_size, learning_rate, regularization_coeff, is_GPU, device)
                    
#                     if r>0:
#                         print("iteration2::", i)
#                         print("para diff::")
#            
#                         compute_model_para_diff(para, curr_exp_para_list_all_epochs[i])
#                           
#     #                             print(torch.norm(get_all_vectorized_parameters1(para) - para_list_all_epochs_tensor[i]))
#                                 
#                         print("gradient diff::") 
#                                
#                         compute_model_para_diff(get_devectorized_parameters(final_gradient_list - regularization_coeff*get_all_vectorized_parameters1(para), full_shape_list, shape_list), curr_exp_gradient_list_all_epochs[i])
#                            
#                          
#                         print("para diff2::")
#                          
#                         print(torch.norm(res_para[i] - get_all_vectorized_parameters1(para)))
#                          
#                         print("grad diff2::")
#                          
#                         print(torch.norm(res_grad[i] - (final_gradient_list - regularization_coeff*get_all_vectorized_parameters1(para))))  
#                            
#                          
#                         print("here")    
    #                     update_para_final(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, exp_gradient, exp_param)
#                     is_positive = True
                    if not is_positive:
                        last_explicit_training_iteration = i
                    
                        curr_rand_ids = random_ids[j:end_id]
                        
#                         if curr_delta_data_id.shape[0] > 0:
#                             extra_remaining_ids = get_remaining_subset_data_per_epoch(curr_added_random_ids, curr_delta_data_id)
#                         else:
#                             extra_remaining_ids = curr_added_random_ids
                    
#                         full_random_ids = torch.cat([curr_rand_ids, curr_added_random_ids], 0)
                        if extra_remaining_ids.shape[0] > 0:
                    
                            full_random_ids = torch.cat([curr_rand_ids, extra_remaining_ids], 0)
                        else:
                            full_random_ids = curr_rand_ids
                        
                    
                        batch_remaining_X = dataset_train.data[full_random_ids]
                        
                        batch_remaining_Y = dataset_train.labels[full_random_ids]
                        
                        if is_GPU:
                            batch_remaining_X = batch_remaining_X.to(device)
                            
                            batch_remaining_Y = batch_remaining_Y.to(device)
                        
                        
#                         t8 = time.time()
#                     
#                         overhead4 += (t8 - t7)
#                         
#                         
#                         t5 = time.time()
                        
                        init_model(model, para)
                        
#                         print(i, curr_rand_ids.shape, batch_remaining_X.shape)
                        
                        compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
                        
                        
        
                        
                        
                        expect_gradients = get_all_vectorized_parameters1(model.get_all_gradient())
                        
#                         expect_gradients2 = 0
#                         
# #                         extra_remaining_ids = get_remaining_subset_data_per_epoch(curr_added_random_ids, curr_delta_data_id)
#                         
#                         if extra_remaining_ids.shape[0] > 0:
#                             batch_extra_remaining_X = X_to_add[extra_remaining_ids]
#                             
#                             batch_extra_remaining_Y = Y_to_add[extra_remaining_ids]
#                             
#                             if is_GPU:
#                                 batch_extra_remaining_X = batch_extra_remaining_X.to(device)
#                                 
#                                 batch_extra_remaining_Y = batch_extra_remaining_Y.to(device)
#                             
#                             init_model(model, para)
#                         
#                             compute_derivative_one_more_step(model, batch_extra_remaining_X, batch_extra_remaining_Y, criterion, optimizer)
#                             
#                             expect_gradients2 = get_all_vectorized_parameters1(model.get_all_gradient())
#                             
#                         
#                         expect_gradients = (expect_gradients1*curr_rand_ids.shape[0] + expect_gradients2*extra_remaining_ids.shape[0])/(curr_rand_ids.shape[0] + extra_remaining_ids.shape[0])
                        
#                         t6 = time.time()
#         
#                         overhead3 += (t6 - t5)
                        
                        gradient_remaining = 0
        #                 if curr_matched_ids_size > 0:
                        if to_add:
                            
                            t3 = time.time()
                            
                            clear_gradients(model.parameters())
                                
                            compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                        
                        
                            gradient_remaining = get_all_vectorized_parameters1(model.get_all_gradient())     
                            
                            
                            t4 = time.time()
                        
                        
                            overhead2 += (t4  -t3)
                        
                        with torch.no_grad():
                                       
                        
#                             curr_para = get_all_vectorized_parameters1(para)
                            curr_para = curr_vec_para
                        
                            if k > 0 or (k == 0 and jj > 0):
                                
                                
                                
                                
                                
        
        #                         torch.cuda.synchronize()
                                
                                
        #                         prev_para = get_all_vectorized_parameters1(para_list_all_epochs[i])
                                prev_para = para_list_GPU_tensor[cached_id]
                                
        #                         t9 = time.time()
                                
        #                         if is_GPU:
        #                             prev_para = prev_para.to(device)
                                
        #                         torch.cuda.synchronize()
        #                         t10 = time.time()
        #                         overhead5 += (t10 - t9)
                                
                                curr_s_list = (curr_para - prev_para)
                                
                                
        #                         if is_GPU:
        #                             curr_s_list = curr_s_list.to(device)
                                S_k_list.append(curr_s_list)
                                if len(S_k_list) > m:
                                    removed_s_k = S_k_list.popleft()
                                    
                                    del removed_s_k
                                
                                
                                    
            #                     print(i-1)
            #                     
            #                     print(S_k_list[:,i - 1])
                            
                #             init_model(model, para)
                            
        #                     gradient_full = (expect_gradients*curr_rand_ids.shape[0] + gradient_remaining*curr_added_random_id_size)/(curr_added_random_id_size + curr_rand_ids.shape[0])
        
                            gradient_full = (expect_gradients*full_random_ids.shape[0] + gradient_remaining*curr_added_size)/(full_random_ids.shape[0] + curr_added_size)
                            
                            if k > 0 or (k == 0 and jj > 0):
                                
                                
        #                         Y_k_list.append((expect_gradients - get_all_vectorized_parameters1(gradient_list_all_epochs[i]).to(device)).view(-1))
                                
                                Y_k_list.append((expect_gradients - grad_list_GPU_tensor[cached_id] + regularization_coeff*curr_s_list))
                                
                                if len(Y_k_list) > m:
                                    removed_y_k = Y_k_list.popleft()
                                    
                                    del removed_y_k
                            
                            
                            
            #                 batch_X = X[curr_rand_ids]
            #                 
            #                 batch_Y = Y[curr_rand_ids]
            #                 clear_gradients(model.parameters())
            #                     
            #                 compute_derivative_one_more_step(model, error, batch_X, batch_Y, beta)
            #                 
            #                 expect_gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
                            
            #                 decompose_model_paras2(para, para_list_all_epochs[i], get_devectorized_parameters(gradient_full, input_dim, hidden_dims, output_dim), gradient_list_all_epochs[i])
                            
            #                     print(torch.dot(Y_k_list[:,i-1], S_k_list[:,i-1]))
            #                     y=0
            #                     y+=1
                            alpha = learning_rate_all_epochs[i]
                            
                            
#                             if r > 0:
#                                 print("iteration1::", i)
#                                 print("para diff::")
#                    
#                                 compute_model_para_diff(para, curr_exp_para_list_all_epochs[i])
#                                   
#         #                             print(torch.norm(get_all_vectorized_parameters1(para) - para_list_all_epochs_tensor[i]))
#                                         
#                                 print("gradient diff::") 
#                                        
#                                 compute_model_para_diff(get_devectorized_parameters(gradient_full, full_shape_list, shape_list), curr_exp_gradient_list_all_epochs[i])
#                                  
#                                  
#                                  
#                                 print("para diff2::")
#                                  
#                                 print(torch.norm(res_para[i] - get_all_vectorized_parameters1(para)))
#                                  
#                                 print("grad diff2::")
#                                  
#                                 print(torch.norm(res_grad[i] - gradient_full))
#                                  
#                                 print("here")
                            
                            
                            
                            para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_para - learning_rate*gradient_full, full_shape_list, shape_list)
        #                     para = get_devectorized_parameters(params, full_shape_list, shape_list)
        #                     
        #                     para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradient_full, input_dim, hidden_dims, output_dim)
                #                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
                            
                            para_list_GPU_tensor[cached_id] = curr_para
                            
                            grad_list_GPU_tensor[cached_id] = gradient_full
                            
                            
                            recorded += 1
                            
                            
                            del gradient_full
                            
                            del gradient_remaining
                            
                            del expect_gradients
                            
                            del batch_remaining_X
                            
                            del batch_remaining_Y
                            
                            if to_add:
                                
                                del batch_delta_X
                                
                                del batch_delta_Y
                            
                            if k > 0 or (k == 0 and jj > 0):
                                del prev_para
                            
                                del curr_para
                            
                            if recorded >= length:
                                use_standard_way = False
                        
                    else:
                    
                    
                        para_list_GPU_tensor[cached_id] = curr_vec_para
                            
                        grad_list_GPU_tensor[cached_id] = exp_grad_list_full
                    
                    
                        
                        vec_para = update_para_final2(curr_vec_para, final_gradient_list, learning_rate, regularization_coeff, exp_gradient, exp_param)
                        
                        para = get_devectorized_parameters(vec_para, full_shape_list, shape_list)

                        
    #                     update_para_final(para, get_devectorized_parameters(hessian_para_prod, input_dim, hidden_dims, output_dim), gradient_dual, gradient_list_all_epochs[i], end_id - j, curr_added_size, alpha)
    #                     gradient_full = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i])
                        
            #             delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1), error, model, beta)
            #             gradient_full2 = torch.mm(v_vec.view(1,-1), hessian_matrix1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
            #             gradient_full2 = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) - delta_const*v_vec.view(1,-1)
                        
    #                     gradients = (gradient_full*(end_id - j) - gradient_dual*curr_matched_ids_size)/(end_id - j - curr_matched_ids_size)
                        
            #             gradients2 = (gradient_full2*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
                        
            #             print('hessian_vector_prod_diff::', torch.norm(torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1) - hessian_para_prod))
                        
            #             print('gradient_diff::', torch.norm(gradients - expect_gradients))
                        
            #             print('gradient2_diff::', torch.norm(gradients2 - expect_gradients))
                        
            #             compute_model_para_diff(exp_para_list_all_epochs[i], para)
                        
        #                 S_k_list[:,i-1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
                        
    #                     para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradients, input_dim, hidden_dims, output_dim)
                
    #                 t6 = time.time()
    #                     
    #                 overhead3 += (t6 - t5)
    #                 Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
    #                 print(torch.norm(get_all_vectorized_parameters(para)))
                    
    #                 print(Y_k_list[:,i-1])
    #                 t2 = time.time()
                        
    #                 overhead += (t2 - t1)
                     
                    
                i = i + 1
                
                
                cached_id += 1
                
                if cached_id%cached_size == 0:
                    
                    GPU_tensor_end_id = (batch_id + 1)*cached_size
                    
                    if GPU_tensor_end_id > para_list_all_epochs_tensor.shape[0]:
                        GPU_tensor_end_id = para_list_all_epochs_tensor.shape[0] 
                    print("end_tensor_id::", GPU_tensor_end_id)
                    
                    para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(para_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                    
                    grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(gradient_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                    
                    batch_id += 1
                    
                    cached_id = 0
                    
                
                id_start = id_end
                
                j += batch_size
        
        
        
        t6 = time.time()
        
        overhead3 += (t6  -t5)
        
        if r % 10 == 0:
            
            print("Num of deletion:: %d, running time provenance::%f" %(r, overhead3))
            
            
            
    print('overhead::', overhead)
    
    print('overhead2::', overhead2)
    
    print('overhead3::', overhead3)
    
    print('overhead4::', overhead4)
    
    print('overhead5::', overhead5)
    
            
    return para

def model_update_provenance_test1_skipnet(period, length, init_epochs, truncted_s, extended_Y_k_list, exp_gradient_list_all_epochs, exp_para_list_all_epochs, dataset_train, model, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, m, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device, all_ids_list_all_epochs):
    
    para = list(model.parameters())
    
    use_standard_way = False
    
    recorded = 0
    
    overhead = 0
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    
    i = 0
    
    S_k_list = deque()
    
    Y_k_list = deque()
    
    overhead2 = 0
    
    overhead3 = 0
    
    old_lr = -1
    
    random_ids_list_all_epochs = []

    removed_batch_empty_list = []
    
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    drop_out_removed_random_ids_list_all_epochs = []
    
#     drop_out_removed_random_ids_list2_all_epochs = []
    
    for k in range(len(random_ids_multi_super_iterations)):
        

    
        random_ids = random_ids_multi_super_iterations[k]
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
        if delta_ids.shape[0] > 1:
            all_indexes = np.sort(sort_idx[delta_ids])
        else:
            all_indexes = torch.tensor([sort_idx[delta_ids]])
                
        id_start = 0
    
        id_end = 0
        
        random_ids_list = []
        
        for j in range(0, dim[0], batch_size):
        
            end_id = j + batch_size
            
            curr_rand_ids = random_ids[j:end_id]
            
#             print(j, end_id)
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]

            curr_matched_id_num = curr_matched_ids.shape[0]

            if curr_matched_id_num > 0:
                random_ids_list.append(curr_matched_ids)
                
                all_curr_removed_ids_list = get_remaining_subset_data_per_epoch_skipnet(curr_rand_ids, curr_matched_ids, all_ids_list_all_epochs[i])
                
                drop_out_removed_random_ids_list_all_epochs.append(all_curr_removed_ids_list)
                
#                 drop_out_removed_random_ids_list2_all_epochs.append(curr_removed_ids_list2)
                
                removed_batch_empty_list.append(False)
            else:
                random_ids_list.append(random_ids[0:1])
                removed_batch_empty_list.append(True)
                
                drop_out_removed_random_ids_list_all_epochs.append(None)
                
#                 drop_out_removed_random_ids_list2_all_epochs.append([])
            
            if (i-init_epochs)%period == 0:
                
                recorded = 0
                
                use_standard_way = True
                
                
            if i< init_epochs or use_standard_way == True:
                
            
                curr_matched_ids2 = (get_remaining_subset_data_per_epoch(curr_rand_ids, curr_matched_ids))
            
            
                random_ids_list.append(curr_matched_ids2)
                
                recorded += 1
                    
                    
                if recorded >= length:
                    use_standard_way = False
            
            i += 1
            
            id_start = id_end
                
        random_ids_list_all_epochs.append(random_ids_list)        
    
    i = 0
    
    curr_batch_sampler = Batch_sampler(random_ids_list_all_epochs)
    
        
    data_train_loader = DataLoader(dataset_train, num_workers=0, shuffle=False, batch_sampler = curr_batch_sampler)
    
    for k in range(len(random_ids_multi_super_iterations)):            
            
        print("epoch ", k)
        random_ids_list = random_ids_list_all_epochs[k]
            
        j = 0
        
        enum_loader = enumerate(data_train_loader)
        
        for t, items in enum_loader:
            
            curr_matched_ids = items[2]        
            
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]        
                    
            curr_matched_ids_size = 0
            if not removed_batch_empty_list[i]:
                
                if not is_GPU:
                
                    batch_delta_X = items[0]
                    
                    batch_delta_Y = items[1]
                
                else:
                    batch_delta_X = items[0].to(device)
                    
                    batch_delta_Y = items[1].to(device)
                
                curr_matched_ids_size = items[2].shape[0]
            
            learning_rate = learning_rate_all_epochs[i]
            
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate
            
                      
            if (i-init_epochs)%period == 0:
                
                recorded = 0
                
                use_standard_way = True
                
                
            if i< init_epochs or use_standard_way == True:
                
                _,next_items = enum_loader.__next__()
                
                if not is_GPU:
                
                    ids = next_items[2]
                
                    sorted_ids, sorted_id_seq = torch.sort(ids)
                
                    batch_remaining_X = next_items[0]
                    
                    batch_remaining_X = batch_remaining_X[sorted_id_seq]
                    
                    batch_remaining_Y = next_items[1]
                    
                    batch_remaining_Y = batch_remaining_Y[sorted_id_seq]
                    
                else:
                    
                    ids = next_items[2]
                
                    sorted_ids, sorted_id_seq = torch.sort(ids)
                    
                    batch_remaining_X = next_items[0].to(device)
                    batch_remaining_X = batch_remaining_X[sorted_id_seq]
                    
                    batch_remaining_Y = next_items[1].to(device)
                    
                    batch_remaining_Y = batch_remaining_Y[sorted_id_seq]
                
                init_model(model, para)
                                
                compute_derivative_one_more_step_skipnet(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer, all_ids_list_all_epochs[i])
                 
                expect_gradients = get_all_vectorized_parameters(model.get_all_gradient())


                gradient_remaining = 0
                if curr_matched_ids_size > 0:
                    clear_gradients(model.parameters())
                        
                    compute_derivative_one_more_step_skipnet(model, batch_delta_X, batch_delta_Y, criterion, optimizer, drop_out_removed_random_ids_list_all_epochs[i])
                
                
                    gradient_remaining = get_all_vectorized_parameters(model.get_all_gradient())     
                
                with torch.no_grad():
                               
                
                    if i>0:
                        
                        
    
                        S_k_list.append((get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1))
                        if len(S_k_list) > m:
                            S_k_list.popleft()
                    
                    gradient_full = (expect_gradients*next_items[2].shape[0] + gradient_remaining*curr_matched_ids_size)/(next_items[2].shape[0] + curr_matched_ids_size)
                    
                    if i>0:
                        
                        
                        Y_k_list.append((gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i]) + regularization_coeff*S_k_list[-1]).view(-1))
                        
                        print("period::", i)
                          
                        print("secont condition::", torch.dot(Y_k_list[-1].view(-1), S_k_list[-1].view(-1)))
                          
                        print("batch size check::", curr_matched_ids_size + next_items[2].shape[0])
                        
                        
                        if len(Y_k_list) > m:
                            Y_k_list.popleft()
                    
                    exp_gradient = None
                    
                    exp_gradient = exp_gradient_list_all_epochs[i]
# #               
#  
#                      
                    exp_param = exp_para_list_all_epochs[i]
                            
                    print("para_diff::")
                    compute_model_para_diff(exp_param, para)
                    
                    print("gradient_diff:")
                    
                    print(torch.norm(expect_gradients -  get_all_vectorized_parameters(exp_gradient)))
                    
                    para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*get_all_vectorized_parameters(para) - learning_rate*expect_gradients, full_shape_list, shape_list)
                    
                    recorded += 1
                    
                    
                    if recorded >= length:
                        use_standard_way = False
                
                
            else:
                gradient_dual = None
    
                if curr_matched_ids_size > 0:
                
                    init_model(model, para)
                    
                    compute_derivative_one_more_step_skipnet(model, batch_delta_X, batch_delta_Y, criterion, optimizer, drop_out_removed_random_ids_list_all_epochs[i])
                                        
                    gradient_dual = model.get_all_gradient()
                
                with torch.no_grad():                    
                    compute_diff_vectorized_parameters(para, para_list_all_epochs[i], vec_para_diff, shape_list)                    
                    
                    if (i-init_epochs)/period >= 1:
                        if (i-init_epochs) % period == 1:
                            
                            zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m)
                            
                            mat = torch.inverse(mat_prime)
                    
                        hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff)
                        
                    else:
                        hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
                         
                    delta_const = 0#compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
                    
                    exp_gradient = None
                    
                    exp_param = None
                    
                    exp_gradient = exp_gradient_list_all_epochs[i]
# #               
#  
#                      
                    exp_param = exp_para_list_all_epochs[i]
                            
                    print("para_diff::")
                    compute_model_para_diff(exp_param, para)
                        
                    print(curr_matched_ids_size)
                      
                    is_positive, final_gradient_list = compute_grad_final(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff)
                    
#                     update_para_final(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, exp_gradient, exp_param)
                    update_para_final(para, final_gradient_list, learning_rate, regularization_coeff, exp_gradient, exp_param)

                 
            
                
            i = i + 1
            
            j += batch_size
            
            
        data_train_loader.batch_sampler.increm_ids()   
            
            
    print('overhead::', overhead)
    
    print('overhead2::', overhead2)
    
    print('overhead3::', overhead3)
    
            
    return para



def get_batch_train_data(dataset_train, ids):
    
#     batch_x_train = []
#     
#     batch_y_train = []
#     
#     batch_ids_train = []
    
#     batch_x_train, batch_y_train, batch_ids_train  = zip(*[dataset_train[i] for i in ids])
#     
#     
# #     
#     batch_y_train = list(batch_y_train)
#      
#     batch_ids_train = list(batch_ids_train)
#         
#     if isinstance(batch_y_train[0], torch.Tensor):
#         batch_x_train, batch_y_train, batch_ids_train = torch.stack(batch_x_train, 0), torch.stack(batch_y_train), torch.tensor(batch_ids_train)
#     else:
#         batch_x_train, batch_y_train, batch_ids_train =  torch.stack(batch_x_train, 0), torch.tensor(batch_y_train), torch.tensor(batch_ids_train)
    
    
    batch_x_train_cp, batch_y_train_cp = dataset_train.data[ids],dataset_train.labels[ids] 
    
    return batch_x_train_cp, batch_y_train_cp, ids
#     batch_x_train = dataset_train.data.data[ids]
#     
#     batch_y_train = dataset_train.data.targets[ids]
    
#     for i in ids:
#         items = dataset_train[i]
#         
#         
#         batch_x_train.append(items[0])
#         
#         batch_y_train.append(items[1])
#         
#         batch_ids_train.append(items[2])
        
#     batch_x_train = list(batch_x_train)
#     
#     batch_y_train = list(batch_y_train)
#     
#     batch_ids_train = list(batch_ids_train)
#        
#     if isinstance(batch_y_train[0], torch.Tensor):
#         return torch.stack(batch_x_train, 0), torch.stack(batch_y_train), torch.tensor(batch_ids_train)
#     else:
#         return torch.stack(batch_x_train, 0), torch.tensor(batch_y_train), torch.tensor(batch_ids_train)
    
        
    
    

def model_update_provenance_test1_3(period, length, init_epochs, truncted_s, extended_Y_k_list, exp_gradient_list_all_epochs, exp_para_list_all_epochs, dataset_train, model, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, m, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device):
    
    para = list(model.parameters())
    
    use_standard_way = False
    
    recorded = 0
    
    overhead = 0
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    
    i = 0
    
    S_k_list = deque()
    
    Y_k_list = deque()
    
    queue_id_list = deque()
    
    overhead2 = 0
    
    overhead3 = 0
    
    old_lr = -1
    
    random_ids_list_all_epochs = []

    removed_batch_empty_list = []
    
    last_explicit_training_iteration = 0
    
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    
    for k in range(len(random_ids_multi_super_iterations)):
        

    
        random_ids = random_ids_multi_super_iterations[k]
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
#         all_indexes = np.sort(sort_idx[delta_ids])
        if delta_ids.shape[0] > 1:
            all_indexes = np.sort(sort_idx[delta_ids])
        else:
            all_indexes = torch.tensor([sort_idx[delta_ids]])
                
        id_start = 0
    
        id_end = 0
        
        random_ids_list = []
        
        for j in range(0, dim[0], batch_size):
        
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]

            curr_matched_id_num = curr_matched_ids.shape[0]

            if curr_matched_id_num > 0:
                random_ids_list.append(curr_matched_ids)
                removed_batch_empty_list.append(False)
            else:
                random_ids_list.append(random_ids[0:1])
                removed_batch_empty_list.append(True)
            
#             if (i-init_epochs)%period == 0:
#                 
#                 recorded = 0
#                 
#                 use_standard_way = True
#                 
#                 
#             if i< init_epochs or use_standard_way == True:
#                 
#                 curr_rand_ids = random_ids[j:end_id]
#             
#             
#                 curr_matched_ids2 = (get_remaining_subset_data_per_epoch(curr_rand_ids, curr_matched_ids))
#             
#                 random_ids_list.append(curr_matched_ids2)
#                 
#                 recorded += 1
#                     
#                     
#                 if recorded >= length:
#                     use_standard_way = False
            
            i += 1
            
            id_start = id_end
                
        random_ids_list_all_epochs.append(random_ids_list)        
    
    i = 0
    
    curr_batch_sampler = Batch_sampler(random_ids_list_all_epochs)
    
        
    data_train_loader = DataLoader(dataset_train, num_workers=0, shuffle=False, batch_sampler = curr_batch_sampler, pin_memory=True)
    
    for k in range(len(random_ids_multi_super_iterations)):            
            
        print("epoch ", k)
        random_ids_list = random_ids_list_all_epochs[k]
        
        random_ids = random_ids_multi_super_iterations[k]
            
        j = 0
        
#         enum_loader = enumerate(data_train_loader)
#         
#         for t, items in enum_loader:
        for p in range(len(random_ids_list)):
            
#             curr_matched_ids = items[2]        
            curr_matched_ids = random_ids_list[p]
            
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]        
                    
            curr_matched_ids_size = 0
            if not removed_batch_empty_list[i]:
                
                if not is_GPU:
                
#                     batch_delta_X = items[0]
#                     
#                     batch_delta_Y = items[1]
                    batch_delta_X = dataset_train.data[curr_matched_ids]
                    
                    batch_delta_Y = dataset_train.labels[curr_matched_ids]
                
                else:
#                     batch_delta_X = items[0].to(device)
#                     
#                     batch_delta_Y = items[1].to(device)

                    batch_delta_X = dataset_train.data[curr_matched_ids].to(device)
                    
                    batch_delta_Y = dataset_train.labels[curr_matched_ids].to(device)
                
#                 curr_matched_ids_size = items[2].shape[0]
                curr_matched_ids_size = len(curr_matched_ids)
            
            learning_rate = learning_rate_all_epochs[i]
            
            if end_id - j - curr_matched_ids_size <= 0:
                
                i += 1
                
                continue
            
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate
            
                      
            if (i-last_explicit_training_iteration)%period == 0:
                
#                 recorded = 0
                
                use_standard_way = True
                
                
            if i< init_epochs or use_standard_way == True:
                
                
                last_explicit_training_iteration = i
                
                curr_rand_ids = random_ids[j:end_id]
                
                if not removed_batch_empty_list[i]:
                
                    curr_matched_ids2 = (get_remaining_subset_data_per_epoch(curr_rand_ids, curr_matched_ids))
                else:
                    curr_matched_ids2 = curr_rand_ids
                
                batch_remaining_X, batch_remaining_Y, batch_remaining_ids = get_batch_train_data(dataset_train, curr_matched_ids2)
                
                
                
#                 _,next_items = enum_loader.__next__()
                
#                 if not is_GPU:
#                 
#                     batch_remaining_X = next_items[0]
#                     
#                     batch_remaining_Y = next_items[1]
#                     
#                 else:
                if is_GPU:
                    batch_remaining_X = batch_remaining_X.to(device)
                    
                    batch_remaining_Y = batch_remaining_Y.to(device)
                
                init_model(model, para)
                                
                compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
                 
                expect_gradients = get_all_vectorized_parameters(model.get_all_gradient())


                gradient_remaining = 0
                if curr_matched_ids_size > 0:
                    clear_gradients(model.parameters())
                        
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                
                
                    gradient_remaining = get_all_vectorized_parameters(model.get_all_gradient())     
                
                with torch.no_grad():
                               
                
                    if i>0:
                        
                        curr_S_k = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
    
                        
#                         if len(S_k_list) > m:
#                             S_k_list.popleft()
                    
                    gradient_full = (expect_gradients*batch_remaining_ids.shape[0] + gradient_remaining*curr_matched_ids_size)/(batch_remaining_ids.shape[0] + curr_matched_ids_size)
                    
                    if i>0:
                        
                        curr_Y_k = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i]) + regularization_coeff*curr_S_k).view(-1)
                        
                        dot_res = torch.dot(curr_S_k, curr_Y_k)
                        
#                         print("curr_secont condition::", dot_res)
                        if dot_res > 0:
                            Y_k_list.append(curr_Y_k)
                            S_k_list.append(curr_S_k)
                            queue_id_list.append(i)
                            
#                             print("secont condition::", dot_res)
                            
                            
                            if len(Y_k_list) > m or i - queue_id_list[0] > 2 :
                                Y_k_list.popleft()
                                S_k_list.popleft()
                                queue_id_list.popleft()
                                
                            if len(Y_k_list) == m:
                                use_standard_way = False
                            
#                         print("explicit_evaluation epoch::", i)
#                         
#                         print("batch size check::", curr_matched_ids_size + batch_remaining_ids.shape[0])
                    
                    exp_gradient = None
                    exp_param = None
                    
#                     exp_gradient = exp_gradient_list_all_epochs[i]
# #               
#   
#                       
#                     exp_param = exp_para_list_all_epochs[i]
#                             
#                     print("para_diff::")
#                     compute_model_para_diff(exp_param, para)
                    
                    
#                     print("para_diff2::")
#                     compute_model_para_diff(para_list_all_epochs[i], para)
                    
                    
#                     if i >= 115:
#                         print("here!!")
                    
                    para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*get_all_vectorized_parameters(para) - learning_rate*expect_gradients, full_shape_list, shape_list)
                    
#                     recorded += 1
#                     
#                     
#                     if recorded >= length:
#                         use_standard_way = False
                
                
            else:
                gradient_dual = None
    
                if curr_matched_ids_size > 0:
                
                    init_model(model, para)
                    
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                                        
                    gradient_dual = model.get_all_gradient()
                
                with torch.no_grad():                    
                    compute_diff_vectorized_parameters(para, para_list_all_epochs[i], vec_para_diff, shape_list)                    
                    
#                     if (i-last_explicit_training_iteration)/period >= 1:
                    if (i-last_explicit_training_iteration) % period == 1:
                        
                        zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m)
                        
                        mat = torch.inverse(mat_prime)
                
                    hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff)
                        
#                     else:
#                         hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
                         
                    delta_const = 0#compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
                    
#                     S_k_list.clear()
#                     
#                     Y_k_list.clear()
#                     
#                     queue_id_list.clear()
                    
#                     del S_k_list[:]
# 
#                     del Y_k_list[:]
                    
                    exp_gradient = None
                    
                    exp_param = None
                    
#                     exp_gradient = exp_gradient_list_all_epochs[i]
# #               
#    
#                        
#                     exp_param = exp_para_list_all_epochs[i]
#                              
#                     print("para_diff::")
#                     compute_model_para_diff(exp_param, para)
#                          
#                     print(curr_matched_ids_size)
                      
                    
                    
#                     update_para_final(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, exp_gradient, exp_param)

                    is_positive, final_gradient_list = compute_grad_final(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff)

                    
                if not is_positive:
                     
                    use_standard_way = True
                     
                    last_explicit_training_iteration = i
             
                    curr_rand_ids = random_ids[j:end_id]
                     
                    if not removed_batch_empty_list[i]:
                     
                        curr_matched_ids2 = (get_remaining_subset_data_per_epoch(curr_rand_ids, curr_matched_ids))
                    else:
                        curr_matched_ids2 = curr_rand_ids
                     
                    batch_remaining_X, batch_remaining_Y, batch_remaining_ids = get_batch_train_data(dataset_train, curr_matched_ids2)
                     
                     
                     
    #                 _,next_items = enum_loader.__next__()
                     
    #                 if not is_GPU:
    #                 
    #                     batch_remaining_X = next_items[0]
    #                     
    #                     batch_remaining_Y = next_items[1]
    #                     
    #                 else:
                    if is_GPU:
                        batch_remaining_X = batch_remaining_X.to(device)
                         
                        batch_remaining_Y = batch_remaining_Y.to(device)
                     
                    init_model(model, para)
                                     
                    compute_derivative_one_more_step(model, Variable(batch_remaining_X), Variable(batch_remaining_Y), criterion, optimizer)
                      
                    expect_gradients = get_all_vectorized_parameters(model.get_all_gradient())
     
     
                    gradient_remaining = 0
                    if curr_matched_ids_size > 0:
                        clear_gradients(model.parameters())
                             
                        compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                     
                     
                        gradient_remaining = get_all_vectorized_parameters(model.get_all_gradient())     
                     
                    with torch.no_grad():
                                    
                     
                        if i>0:
                             
                            curr_S_k = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
         
                             
    #                         if len(S_k_list) > m:
    #                             S_k_list.popleft()
                         
                        gradient_full = (expect_gradients*batch_remaining_ids.shape[0] + gradient_remaining*curr_matched_ids_size)/(batch_remaining_ids.shape[0] + curr_matched_ids_size)
                         
                        if i>0:
                             
                            curr_Y_k = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i]) + regularization_coeff*curr_S_k).view(-1)
                             
                            dot_res = torch.dot(curr_S_k, curr_Y_k)
                             
#                             print("curr_secont condition::", dot_res)
                            if dot_res > 0:
                                Y_k_list.append(curr_Y_k)
                                S_k_list.append(curr_S_k)
                                queue_id_list.append(i)
                                 
#                                 print("secont condition::", dot_res)
                                 
                                 
                                while len(Y_k_list) > m:
                                    Y_k_list.popleft()
                                    S_k_list.popleft()
                                    queue_id_list.popleft()
                                     
                                if len(Y_k_list) == m:
                                    use_standard_way = False
                                 
#                             print("explicit_evaluation epoch::", i)
                             
#                             print("batch size check::", curr_matched_ids_size + batch_remaining_ids.shape[0])
                         
                        exp_gradient = None
                        exp_param = None
                         
#                         exp_gradient = exp_gradient_list_all_epochs[i]
#     #               
#         
#                             
#                         exp_param = exp_para_list_all_epochs[i]
#                                   
#                         print("para_diff::")
#                         compute_model_para_diff(exp_param, para)
                         
                         
    #                     print("para_diff2::")
    #                     compute_model_para_diff(para_list_all_epochs[i], para)
                         
                         
#                         if i >= 115:
#                             print("here!!")
                         
                        para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*get_all_vectorized_parameters(para) - learning_rate*expect_gradients, full_shape_list, shape_list)
                else:
                    update_para_final(para, final_gradient_list, learning_rate, regularization_coeff, exp_gradient, exp_param) 

                 
            
                
            i = i + 1
            
            j += batch_size
            
            
        data_train_loader.batch_sampler.increm_ids()   
            
            
    print('overhead::', overhead)
    
    print('overhead2::', overhead2)
    
    print('overhead3::', overhead3)
    
            
    return para

def model_update_provenance_test1_3_skipnet(period, length, init_epochs, truncted_s, extended_Y_k_list, exp_gradient_list_all_epochs, exp_para_list_all_epochs, dataset_train, model, model_cp, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, m, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device, all_ids_list_all_epochs):
    
    para = list(model.parameters())
    
    use_standard_way = False
    
    recorded = 0
    
    overhead = 0
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    
    i = 0
    
    S_k_list = deque()
    
    Y_k_list = deque()
    
    queue_id_list = deque()
    
    overhead2 = 0
    
    overhead3 = 0
    
    old_lr = -1
    
    random_ids_list_all_epochs = []

    removed_batch_empty_list = []
    
    last_explicit_training_iteration = 0
    
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    drop_out_removed_random_ids_list_all_epochs = []
    
#     model_cp = model.clone()
    
    for k in range(len(random_ids_multi_super_iterations)):
        

    
        random_ids = random_ids_multi_super_iterations[k]
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
#         if delta_ids.shape[0] > 1:
#             all_indexes = np.sort(sort_idx[delta_ids])
#         else:
#             all_indexes = torch.tensor(sort_idx[delta_ids])
            
        if delta_ids.shape[0] > 1:
            all_indexes = np.sort(sort_idx[delta_ids])
        else:
            all_indexes = torch.tensor([sort_idx[delta_ids]])    
        
                
        id_start = 0
    
        id_end = 0
        
        random_ids_list = []
        
        for j in range(0, dim[0], batch_size):
        
            end_id = j + batch_size
            
            curr_rand_ids = random_ids[j:end_id]
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            
            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
            
#             if all_indexes[-1] < end_id:
#                 id_end = all_indexes.shape[0]
#             else:
#                 id_end = np.argmax(all_indexes >= end_id)
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]

            curr_matched_id_num = curr_matched_ids.shape[0]

            if curr_matched_id_num > 0:
                random_ids_list.append(curr_matched_ids)
                
                all_curr_removed_ids_list = get_remaining_subset_data_per_epoch_skipnet(curr_rand_ids, curr_matched_ids, all_ids_list_all_epochs[i])
                
                drop_out_removed_random_ids_list_all_epochs.append(all_curr_removed_ids_list)
                
                removed_batch_empty_list.append(False)
            else:
                random_ids_list.append(random_ids[0:1])
                removed_batch_empty_list.append(True)
                
                drop_out_removed_random_ids_list_all_epochs.append(None)
            
#             if (i-init_epochs)%period == 0:
#                 
#                 recorded = 0
#                 
#                 use_standard_way = True
#                 
#                 
#             if i< init_epochs or use_standard_way == True:
#                 
#                 curr_rand_ids = random_ids[j:end_id]
#             
#             
#                 curr_matched_ids2 = (get_remaining_subset_data_per_epoch(curr_rand_ids, curr_matched_ids))
#             
#                 random_ids_list.append(curr_matched_ids2)
#                 
#                 recorded += 1
#                     
#                     
#                 if recorded >= length:
#                     use_standard_way = False
            
            i += 1
            
            id_start = id_end
                
        random_ids_list_all_epochs.append(random_ids_list)        
    
    i = 0
    
    curr_batch_sampler = Batch_sampler(random_ids_list_all_epochs)
    
        
    data_train_loader = DataLoader(dataset_train, num_workers=0, shuffle=False, batch_sampler = curr_batch_sampler)
    
    for k in range(len(random_ids_multi_super_iterations)):            
            
        print("epoch ", k)
        random_ids_list = random_ids_list_all_epochs[k]
        
        random_ids = random_ids_multi_super_iterations[k]
            
        j = 0
        
        enum_loader = enumerate(data_train_loader)
        
        for t, items in enum_loader:
            
            curr_matched_ids = items[2]        
            
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]        
                    
            curr_matched_ids_size = 0
            if not removed_batch_empty_list[i]:
                
                if not is_GPU:
                
                    batch_delta_X = items[0]
                    
                    batch_delta_Y = items[1]
                
                else:
                    batch_delta_X = items[0].to(device)
                    
                    batch_delta_Y = items[1].to(device)
                
                curr_matched_ids_size = items[2].shape[0]
            
            learning_rate = learning_rate_all_epochs[i]
            
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate
            
                      
            if (i-last_explicit_training_iteration)%period == 0:
                
#                 recorded = 0
                
                use_standard_way = True
            
            
#             if i >= 59:
#                 print("here")
            
                
            if i< init_epochs or use_standard_way == True:
                
                
                last_explicit_training_iteration = i
                
                curr_rand_ids = random_ids[j:end_id]
                
                if not removed_batch_empty_list[i]:
                
                    curr_matched_ids2 = (get_remaining_subset_data_per_epoch(curr_rand_ids, curr_matched_ids))
                else:
                    curr_matched_ids2 = curr_rand_ids
                
                batch_remaining_X, batch_remaining_Y, batch_remaining_ids = get_batch_train_data(dataset_train, curr_matched_ids2)
                
                
                
#                 _,next_items = enum_loader.__next__()
                
#                 if not is_GPU:
#                 
#                     batch_remaining_X = next_items[0]
#                     
#                     batch_remaining_Y = next_items[1]
#                     
#                 else:
                if is_GPU:
                    batch_remaining_X = batch_remaining_X.to(device)
                    
                    batch_remaining_Y = batch_remaining_Y.to(device)
                
                init_model(model, para)
                                
                compute_derivative_one_more_step_skipnet(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer, all_ids_list_all_epochs[i], is_GPU, device)
                 
                expect_gradients = get_all_vectorized_parameters(model.get_all_gradient())


                gradient_remaining = 0
                if curr_matched_ids_size > 0:
#                     clear_gradients(model.parameters())
                    init_model(model, para)
                    compute_derivative_one_more_step_skipnet(model, batch_delta_X, batch_delta_Y, criterion, optimizer, drop_out_removed_random_ids_list_all_epochs[i], is_GPU, device)
                
                
                    gradient_remaining = get_all_vectorized_parameters(model.get_all_gradient())     
                
                with torch.no_grad():
                               
                
                    if i>0:
                        
                        curr_S_k = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
    
                        
#                         if len(S_k_list) > m:
#                             S_k_list.popleft()
                    
                    gradient_full = (expect_gradients*batch_remaining_ids.shape[0] + gradient_remaining*curr_matched_ids_size)/(batch_remaining_ids.shape[0] + curr_matched_ids_size)
                    
                    if i>0:
                        
                        curr_Y_k = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i]) + regularization_coeff*curr_S_k).view(-1)
                        
                        dot_res = torch.dot(curr_S_k, curr_Y_k)
                        
                        print("curr_secont condition::", dot_res)
                        if dot_res > 0:
                            Y_k_list.append(curr_Y_k)
                            S_k_list.append(curr_S_k)
                            queue_id_list.append(i)
                            
                            print("secont condition::", dot_res)
                            
                            
                            while len(Y_k_list) > m :
                                Y_k_list.popleft()
                                S_k_list.popleft()
                                queue_id_list.popleft()
                                
                            if len(Y_k_list) == m:
                                use_standard_way = False
                        
                        
                        else:
                            use_standard_way = True
                            
#                         print("explicit_evaluation epoch::", i)
#                         
#                         print("batch size check::", curr_matched_ids_size + batch_remaining_ids.shape[0])
                    
                    exp_gradient = None
                    exp_param = None
                    
#                     exp_gradient = exp_gradient_list_all_epochs[i]
# #               
#   
#                       
#                     exp_param = exp_para_list_all_epochs[i]
#                             
#                     print("para_diff::")
#                     compute_model_para_diff(exp_param, para)
                    
                    
#                     print("para_diff2::")
#                     compute_model_para_diff(para_list_all_epochs[i], para)
                    
                    
#                     if i >= 115:
#                         print("here!!")
                    
                    para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*get_all_vectorized_parameters(para) - learning_rate*expect_gradients, full_shape_list, shape_list)
                    
#                     recorded += 1
#                     
#                     
#                     if recorded >= length:
#                         use_standard_way = False
                
                
            else:
                gradient_dual = None
    
                if curr_matched_ids_size > 0:
                
                    init_model(model, para)
                    
                    compute_derivative_one_more_step_skipnet(model, batch_delta_X, batch_delta_Y, criterion, optimizer, drop_out_removed_random_ids_list_all_epochs[i], is_GPU, device)
                                        
                    gradient_dual = model.get_all_gradient()
                
                with torch.no_grad():                    
                    compute_diff_vectorized_parameters(para, para_list_all_epochs[i], vec_para_diff, shape_list)                    
                    
#                     if (i-last_explicit_training_iteration)/period >= 1:
                    if (i-last_explicit_training_iteration) % period == 1:
                        
                        zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m)
                        
                        mat = torch.inverse(mat_prime)
                
                    hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff)
                        
#                     else:
#                         hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
                         
                    delta_const = 0#compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
                    
#                     S_k_list.clear()
#                     
#                     Y_k_list.clear()
#                     
#                     queue_id_list.clear()
                    
#                     del S_k_list[:]
# 
#                     del Y_k_list[:]
                    
                    exp_gradient = None
                    
                    exp_param = None
                    
#                     exp_gradient = exp_gradient_list_all_epochs[i]
# #               
#   
#                       
#                     exp_param = exp_para_list_all_epochs[i]
#                             
#                     print("para_diff::")
#                     compute_model_para_diff(exp_param, para)
#                         
#                     print(curr_matched_ids_size)
                      
                    
                    
#                     is_positive, old_para = update_para_final(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, exp_gradient, exp_param)


                    is_positive, final_gradient_list = compute_grad_final(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff)

                    
                if not is_positive:
                     
                    use_standard_way = True
                     
                    last_explicit_training_iteration = i
             
                    curr_rand_ids = random_ids[j:end_id]
                     
                    if not removed_batch_empty_list[i]:
                     
                        curr_matched_ids2 = (get_remaining_subset_data_per_epoch(curr_rand_ids, curr_matched_ids))
                    else:
                        curr_matched_ids2 = curr_rand_ids
                     
                    batch_remaining_X, batch_remaining_Y, batch_remaining_ids = get_batch_train_data(dataset_train, curr_matched_ids2)
                     
                     
                     
    #                 _,next_items = enum_loader.__next__()
                     
    #                 if not is_GPU:
    #                 
    #                     batch_remaining_X = next_items[0]
    #                     
    #                     batch_remaining_Y = next_items[1]
    #                     
    #                 else:
                    if is_GPU:
                        batch_remaining_X = batch_remaining_X.to(device)
                         
                        batch_remaining_Y = batch_remaining_Y.to(device)
                     
                    init_model(model, para)
                                     
                    compute_derivative_one_more_step_skipnet(model, Variable(batch_remaining_X), Variable(batch_remaining_Y), criterion, optimizer, all_ids_list_all_epochs[i], is_GPU, device)
                      
                    expect_gradients = get_all_vectorized_parameters(model.get_all_gradient())
     
     
                    gradient_remaining = 0
                    if curr_matched_ids_size > 0:
                        clear_gradients(model.parameters())
                             
                        compute_derivative_one_more_step_skipnet(model, batch_delta_X, batch_delta_Y, criterion, optimizer, drop_out_removed_random_ids_list_all_epochs[i], is_GPU, device)
                     
                     
                        gradient_remaining = get_all_vectorized_parameters(model.get_all_gradient())     
                     
                    with torch.no_grad():
                                    
                     
                        if i>0:
                             
                            curr_S_k = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
         
                             
    #                         if len(S_k_list) > m:
    #                             S_k_list.popleft()
                         
                        gradient_full = (expect_gradients*batch_remaining_ids.shape[0] + gradient_remaining*curr_matched_ids_size)/(batch_remaining_ids.shape[0] + curr_matched_ids_size)
                         
                        if i>0:
                             
                            curr_Y_k = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i]) + regularization_coeff*curr_S_k).view(-1)
                             
                            dot_res = torch.dot(curr_S_k, curr_Y_k)
                             
                            print("curr_secont condition::", dot_res)
                            if dot_res > 0:
                                Y_k_list.append(curr_Y_k)
                                S_k_list.append(curr_S_k)
                                queue_id_list.append(i)
                                 
#                                 print("secont condition::", dot_res)
                                 
                                 
                                while len(Y_k_list) > m:
                                    Y_k_list.popleft()
                                    S_k_list.popleft()
                                    queue_id_list.popleft()
                                     
                                if len(Y_k_list) == m:
                                    use_standard_way = False
                                 
                            print("explicit_evaluation epoch::", i)
                             
#                             print("batch size check::", curr_matched_ids_size + batch_remaining_ids.shape[0])
                         
                        exp_gradient = None
                        exp_param = None
                         
#                         exp_gradient = exp_gradient_list_all_epochs[i]
#     #               
#        
#                            
#                         exp_param = exp_para_list_all_epochs[i]
#                                  
#                         print("para_diff::")
#                         compute_model_para_diff(exp_param, para)
                         
                         
    #                     print("para_diff2::")
    #                     compute_model_para_diff(para_list_all_epochs[i], para)
                         
                         
#                         if i >= 115:
#                             print("here!!")
                         
                        para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*get_all_vectorized_parameters(para) - learning_rate*expect_gradients, full_shape_list, shape_list)
                else:
                    update_para_final(para, final_gradient_list, learning_rate, regularization_coeff, exp_gradient, exp_param)        
            
                
            i = i + 1
            
            j += batch_size
            
            
        data_train_loader.batch_sampler.increm_ids()   
            
            
    print('overhead::', overhead)
    
    print('overhead2::', overhead2)
    
    print('overhead3::', overhead3)
    
            
    return para


def model_update_provenance_test1_3_quantize(period, length, init_epochs, truncted_s, extended_Y_k_list, exp_gradient_list_all_epochs, exp_para_list_all_epochs, dataset_train, model, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, m, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device, epsilon):
    
    para = list(model.parameters())
    
    use_standard_way = False
    
    recorded = 0
    
    overhead = 0
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    
    i = 0
    
    S_k_list = deque()
    
    Y_k_list = deque()
    
    queue_id_list = deque()
    
    overhead2 = 0
    
    overhead3 = 0
    
    old_lr = -1
    
    random_ids_list_all_epochs = []

    removed_batch_empty_list = []
    
    last_explicit_training_iteration = 0
    
    for k in range(len(random_ids_multi_super_iterations)):
        

    
        random_ids = random_ids_multi_super_iterations[k]
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
        all_indexes = np.sort(sort_idx[delta_ids])
                
        id_start = 0
    
        id_end = 0
        
        random_ids_list = []
        
        for j in range(0, dim[0], batch_size):
        
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]

            curr_matched_id_num = curr_matched_ids.shape[0]

            if curr_matched_id_num > 0:
                random_ids_list.append(curr_matched_ids)
                removed_batch_empty_list.append(False)
            else:
                random_ids_list.append(random_ids[0:1])
                removed_batch_empty_list.append(True)
            
#             if (i-init_epochs)%period == 0:
#                 
#                 recorded = 0
#                 
#                 use_standard_way = True
#                 
#                 
#             if i< init_epochs or use_standard_way == True:
#                 
#                 curr_rand_ids = random_ids[j:end_id]
#             
#             
#                 curr_matched_ids2 = (get_remaining_subset_data_per_epoch(curr_rand_ids, curr_matched_ids))
#             
#                 random_ids_list.append(curr_matched_ids2)
#                 
#                 recorded += 1
#                     
#                     
#                 if recorded >= length:
#                     use_standard_way = False
            
            i += 1
            
            id_start = id_end
                
        random_ids_list_all_epochs.append(random_ids_list)        
    
    i = 0
    
    curr_batch_sampler = Batch_sampler(random_ids_list_all_epochs)
    
        
    data_train_loader = DataLoader(dataset_train, num_workers=0, shuffle=False, batch_sampler = curr_batch_sampler, pin_memory=True)
    
    for k in range(len(random_ids_multi_super_iterations)):            
            
        print("epoch ", k)
        random_ids_list = random_ids_list_all_epochs[k]
        
        random_ids = random_ids_multi_super_iterations[k]
            
        j = 0
        
        enum_loader = enumerate(data_train_loader)
        
        for t, items in enum_loader:
            
            curr_matched_ids = items[2]        
            
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]        
                    
            curr_matched_ids_size = 0
            if not removed_batch_empty_list[i]:
                
                if not is_GPU:
                
                    batch_delta_X = items[0]
                    
                    batch_delta_Y = items[1]
                
                else:
                    batch_delta_X = items[0].to(device)
                    
                    batch_delta_Y = items[1].to(device)
                
                curr_matched_ids_size = items[2].shape[0]
            
            learning_rate = learning_rate_all_epochs[i]
            
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate
            
                      
            if (i-last_explicit_training_iteration)%period == 0:
                
#                 recorded = 0
                
                use_standard_way = True
                
                
            if i< init_epochs or use_standard_way == True:
                
                
                last_explicit_training_iteration = i
                
                curr_rand_ids = random_ids[j:end_id]
                
                if not removed_batch_empty_list[i]:
                
                    curr_matched_ids2 = (get_remaining_subset_data_per_epoch(curr_rand_ids, curr_matched_ids))
                else:
                    curr_matched_ids2 = curr_rand_ids
                
                batch_remaining_X, batch_remaining_Y, batch_remaining_ids = get_batch_train_data(dataset_train, curr_matched_ids2)
                
                
                
#                 _,next_items = enum_loader.__next__()
                
#                 if not is_GPU:
#                 
#                     batch_remaining_X = next_items[0]
#                     
#                     batch_remaining_Y = next_items[1]
#                     
#                 else:
                if is_GPU:
                    batch_remaining_X = batch_remaining_X.to(device)
                    
                    batch_remaining_Y = batch_remaining_Y.to(device)
                
                init_model(model, para)
                                
                compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
                 
                expect_gradients = get_all_vectorized_parameters(model.get_all_gradient())


                gradient_remaining = 0
                if curr_matched_ids_size > 0:
                    clear_gradients(model.parameters())
                        
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                
                
                    gradient_remaining = get_all_vectorized_parameters(model.get_all_gradient())     
                
                with torch.no_grad():
                               
                
                    if i>0:
                        
                        curr_S_k = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
    
                        
#                         if len(S_k_list) > m:
#                             S_k_list.popleft()
                    
                    gradient_full = (expect_gradients*batch_remaining_ids.shape[0] + gradient_remaining*curr_matched_ids_size)/(batch_remaining_ids.shape[0] + curr_matched_ids_size)
                    
                    if i>0:
                        
                        curr_Y_k = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i]) + regularization_coeff*curr_S_k).view(-1)
                        
                        dot_res = torch.dot(curr_S_k, curr_Y_k)
                        
                        print("curr_secont condition::", dot_res)
                        if dot_res > 0:
                            Y_k_list.append(curr_Y_k)
                            S_k_list.append(curr_S_k)
                            queue_id_list.append(i)
                            
                            print("secont condition::", dot_res)
                            
                            
                            if len(Y_k_list) > m or i - queue_id_list[0] > 2 :
                                Y_k_list.popleft()
                                S_k_list.popleft()
                                queue_id_list.popleft()
                                
                            if len(Y_k_list) == m:
                                use_standard_way = False
                            
                        print("explicit_evaluation epoch::", i)
                        
                        print("batch size check::", curr_matched_ids_size + batch_remaining_ids.shape[0])
                    
                    exp_gradient = None
                    exp_param = None
                    
                    exp_gradient = exp_gradient_list_all_epochs[i]
#               
  
                      
                    exp_param = exp_para_list_all_epochs[i]
                            
                    print("para_diff::")
                    compute_model_para_diff(exp_param, para)
                    
                    print("gradient_diff:")
                    print(torch.norm(get_all_vectorized_parameters(exp_gradient) - expect_gradients))
                    
                    
                    
                    if i >= 115:
                        print("here!!")
                    
                    para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*get_all_vectorized_parameters(para) - learning_rate*expect_gradients, full_shape_list, shape_list)
                    
                    para = quantize_model_param2(para, epsilon)
                    
#                     recorded += 1
#                     
#                     
#                     if recorded >= length:
#                         use_standard_way = False
                
                
            else:
                gradient_dual = None
    
                if curr_matched_ids_size > 0:
                
                    init_model(model, para)
                    
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                                        
                    gradient_dual = model.get_all_gradient()
                
                with torch.no_grad():                    
                    compute_diff_vectorized_parameters(para, para_list_all_epochs[i], vec_para_diff, shape_list)                    
                    
#                     if (i-last_explicit_training_iteration)/period >= 1:
                    if (i-last_explicit_training_iteration) % period == 1:
                        
                        zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m)
                        
                        mat = torch.inverse(mat_prime)
                
                    hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff)
                        
#                     else:
#                         hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
                         
                    delta_const = 0#compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
                    
                    S_k_list.clear()
                    
                    Y_k_list.clear()
                    
                    queue_id_list.clear()
                    
#                     del S_k_list[:]
# 
#                     del Y_k_list[:]
                    
                    exp_gradient = None
                    
                    exp_param = None
                    
                    exp_gradient = exp_gradient_list_all_epochs[i]
#               
  
                      
                    exp_param = exp_para_list_all_epochs[i]
                            
                    print("para_diff::")
                    compute_model_para_diff(exp_param, para)
                        
                    print(curr_matched_ids_size)
                      
                    
                    
                    update_para_final(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, exp_gradient, exp_param)

                    para = quantize_model_param2(para, epsilon)
            
                
            i = i + 1
            
            j += batch_size
            
            
        data_train_loader.batch_sampler.increm_ids()   
            
            
    print('overhead::', overhead)
    
    print('overhead2::', overhead2)
    
    print('overhead3::', overhead3)
    
            
    return para





def model_update_provenance_test1_2(period, length, init_epochs, truncted_s, extended_Y_k_list, exp_gradient_list_all_epochs, exp_para_list_all_epochs, dataset_train, model, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, m, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device, lambda_value):

    
    para = list(model.parameters())
            
    use_standard_way = False
    
    recorded = 0
    
    overhead = 0
        
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    
    i = 0
    
    S_k_list = deque()
    
    Y_k_list = deque()
    
    overhead2 = 0
    
    overhead3 = 0
    
    old_lr = -1
    
    random_ids_list_all_epochs = []
    
    removed_batch_empty_list = []
    
    for k in range(len(random_ids_multi_super_iterations)):
        
        random_ids = random_ids_multi_super_iterations[k]
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
        all_indexes = np.sort(sort_idx[delta_ids])
                
        id_start = 0
    
        id_end = 0
        
        random_ids_list = []
        
        for j in range(0, dim[0], batch_size):
        
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]

            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]

            curr_matched_id_num = curr_matched_ids.shape[0]

            if curr_matched_id_num > 0:
                random_ids_list.append(curr_matched_ids)
                removed_batch_empty_list.append(False)
            else:
                random_ids_list.append(random_ids[0:1])
                removed_batch_empty_list.append(True)
            
            if i%period == 0:
                
                recorded = 0
                
                use_standard_way = True
                
                
            if i< init_epochs or use_standard_way == True:
                
                curr_rand_ids = random_ids[j:end_id]
            
            
                curr_matched_ids2 = (get_remaining_subset_data_per_epoch(curr_rand_ids, curr_matched_ids))
            
                random_ids_list.append(curr_matched_ids2)
                
                recorded += 1
                    
                    
                if recorded >= length:
                    use_standard_way = False
            
            i += 1
            
            id_start = id_end               
        random_ids_list_all_epochs.append(random_ids_list)        
    
    i = 0
    
    curr_batch_sampler = Batch_sampler(random_ids_list_all_epochs)
    
    data_train_loader = DataLoader(dataset_train, num_workers=0, shuffle=False, batch_sampler = curr_batch_sampler)
    
    for k in range(len(random_ids_multi_super_iterations)):            
            
        print("epoch ", k)
        random_ids_list = random_ids_list_all_epochs[k]        
        
        j = 0
        
        enum_loader = enumerate(data_train_loader)
                
        for t, items in enum_loader:
            
            curr_matched_ids = items[2]        
            
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]        
                
            curr_matched_ids_size = 0
            if not removed_batch_empty_list[i]:
                
                if not is_GPU:
                
                    batch_delta_X = items[0]
                    
                    batch_delta_Y = items[1]
                
                else:
                    batch_delta_X = items[0].to(device)
                    
                    batch_delta_Y = items[1].to(device)
                
                curr_matched_ids_size = items[2].shape[0]
            
            learning_rate = learning_rate_all_epochs[i]
                        
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate
                
                      
            if i%period == 0:
                
                recorded = 0
                
                use_standard_way = True
                
                
            if i< init_epochs or use_standard_way == True:

                _,next_items = enum_loader.__next__()
                
                if not is_GPU:
                
                    batch_remaining_X = next_items[0]
                    
                    batch_remaining_Y = next_items[1]
                    
                else:
                    batch_remaining_X = next_items[0].to(device)
                    
                    batch_remaining_Y = next_items[1].to(device)
                
                init_model(model, para)
                                
                compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
                 
                expect_gradients = get_all_vectorized_parameters(model.get_all_gradient())


                gradient_remaining = 0
                if curr_matched_ids_size > 0:
                    clear_gradients(model.parameters())
                        
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                
                
                    gradient_remaining = get_all_vectorized_parameters(model.get_all_gradient())     
                
                with torch.no_grad():
                               
                
                    if i>0:
                        
                        
    
                        S_k_list.append((get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1))
                        if len(S_k_list) > m:
                            S_k_list.popleft()
                    
                    gradient_full = (expect_gradients*next_items[2].shape[0] + gradient_remaining*curr_matched_ids_size)/(next_items[2].shape[0] + curr_matched_ids_size)
                    
                    if i>0:
                        
                        
                        Y_k_list.append((gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1) + (lambda_value + regularization_coeff)*S_k_list[-1])
                        if len(Y_k_list) > m:
                            Y_k_list.popleft()
                    
                        print("secand condition check::", torch.dot(S_k_list[-1], Y_k_list[-1]))
                        
                        if curr_matched_ids_size > 0:
                            print("batch_size check::", len(list(set(items[2].tolist()).union(set(next_items[2].tolist())))))
                        else:
                            print("batch_size check::", next_items[2].shape[0])
                            
                    para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*get_all_vectorized_parameters(para) - learning_rate*expect_gradients, full_shape_list, shape_list)
                    
                    recorded += 1
                    
                    
                    if recorded >= length:
                        use_standard_way = False
                
                
            else:
                gradient_dual = None
    
                if curr_matched_ids_size > 0:
                
                    init_model(model, para)
                    
                    
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)

                    gradient_dual = model.get_all_gradient()
                
                with torch.no_grad():
                    compute_diff_vectorized_parameters(para, para_list_all_epochs[i], vec_para_diff, shape_list)                    
                    
                    if i/period >= 1:
                        if i % period == 1:

                            zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, period)

                            mat = torch.inverse(mat_prime)
                    
                        
                        hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff)
                        
                    else:                        
                        hessian_para_prod = cal_approx_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
                    
                    delta_const = 0#compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
                    
                    exp_gradient = None
                    exp_gradient = exp_gradient_list_all_epochs[i]
#              
                    exp_param = exp_para_list_all_epochs[i]
                        
                    print("para_diff::")
                    compute_model_para_diff(exp_param, para)
                    
                    update_para_final(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_matched_ids_size, learning_rate, lambda_value + regularization_coeff, exp_gradient)
            
                
            i = i + 1
            
            j += batch_size
            
            
        data_train_loader.batch_sampler.increm_ids()   
            
            
    print('overhead::', overhead)
    
    print('overhead2::', overhead2)
    
    print('overhead3::', overhead3)
    
            
    return para


def get_random_id_list_with_evaluation_iteration_fixed_period(random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, delta_ids, dim, init_epochs, period, length, batch_size):
    
    random_ids_list_all_epochs = []

    removed_batch_empty_list = []
    
    explicit_eval_iterations = []
    
    i = 0
    
    for k in range(len(random_ids_multi_super_iterations)):
        

    
        random_ids = random_ids_multi_super_iterations[k]
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
        all_indexes = np.sort(sort_idx[delta_ids])
                
        id_start = 0
    
        id_end = 0
        
        random_ids_list = []
        
        for j in range(0, dim[0], batch_size):
        
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]

            curr_matched_id_num = curr_matched_ids.shape[0]

            if curr_matched_id_num > 0:
                random_ids_list.append(curr_matched_ids)
                removed_batch_empty_list.append(False)
            else:
                random_ids_list.append(random_ids[0:1])
                removed_batch_empty_list.append(True)
            
            if (i-init_epochs)%period == 0:
                
                recorded = 0
                
                use_standard_way = True
                
                
            if i< init_epochs or use_standard_way == True:
                
                curr_rand_ids = random_ids[j:end_id]
            
            
                curr_matched_ids2 = (get_remaining_subset_data_per_epoch(curr_rand_ids, curr_matched_ids))
            
                random_ids_list.append(curr_matched_ids2)
                
                explicit_eval_iterations.append(i)
                
                recorded += 1
                    
                    
                if recorded >= length:
                    use_standard_way = False
            
            i += 1
            
            id_start = id_end
                
        random_ids_list_all_epochs.append(random_ids_list)   
        
    return random_ids_list_all_epochs, removed_batch_empty_list, set(explicit_eval_iterations)



def get_random_id_list_with_evaluation_iteration_varied_period(random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, delta_ids, dim, init_epochs, max_period, length, batch_size):
    
    gap_count = 0
    
    period = 2
    
    period_count = 1
        
    random_ids_list_all_epochs = []

    removed_batch_empty_list = []
    
    explicit_eval_iterations = []
    
    i = 0
    
    recorded = 0
    
    for k in range(len(random_ids_multi_super_iterations)):
        

    
        random_ids = random_ids_multi_super_iterations[k]
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
        all_indexes = np.sort(sort_idx[delta_ids])
                
        id_start = 0
    
        id_end = 0
        
        random_ids_list = []
        
        for j in range(0, dim[0], batch_size):
        
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]

            curr_matched_id_num = curr_matched_ids.shape[0]

            if curr_matched_id_num > 0:
                random_ids_list.append(curr_matched_ids)
                removed_batch_empty_list.append(False)
            else:
                random_ids_list.append(random_ids[0:1])
                removed_batch_empty_list.append(True)
            
            
            
            
            
            if gap_count >= period:
                
                recorded = 0
                
                use_standard_way = True
                
                if i >= init_epochs:
                    
                    if period_count % 8 == 0:
                        period = min(int(period *1.5), max_period)
                    
                    
                    period_count += 1
                
#                 print(i, period)
                
                
            if i< init_epochs or use_standard_way == True:
                
                gap_count = 0
                
                curr_rand_ids = random_ids[j:end_id]
            
            
                curr_matched_ids2 = (get_remaining_subset_data_per_epoch(curr_rand_ids, curr_matched_ids))
                
                random_ids_list.append(curr_matched_ids2)
                
                explicit_eval_iterations.append(i)
                
                recorded += 1

                if recorded >= length:
                    use_standard_way = False
                    
            else:
                gap_count += 1
            
            i += 1
            
            id_start = id_end
                
        random_ids_list_all_epochs.append(random_ids_list)   
        
    return random_ids_list_all_epochs, removed_batch_empty_list, set(explicit_eval_iterations)


def model_update_provenance_test1_varied_period(max_period, length, init_epochs, truncted_s, extended_Y_k_list, exp_gradient_list_all_epochs, exp_para_list_all_epochs, dataset_train, model, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, m, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device):
    
    
    para = list(model.parameters())
    
#     use_standard_way = False
#     
#     recorded = 0
    
    overhead = 0
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    
    i = 0
    
    first_in_period = True
    
    S_k_list = deque()
    
    Y_k_list = deque()
    
    overhead2 = 0
    
    overhead3 = 0
    
    old_lr = -1
    
    
#     random_ids_list_all_epochs, removed_batch_empty_list, explicit_eval_iterations = get_random_id_list_with_evaluation_iteration_fixed_period(random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, delta_ids, dim, init_epochs, max_period, length, batch_size)
    
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    random_ids_list_all_epochs, removed_batch_empty_list, explicit_eval_iterations = get_random_id_list_with_evaluation_iteration_varied_period(random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, delta_ids, dim, init_epochs, max_period, length, batch_size)
    
#     random_ids_list_all_epochs = []
# 
#     removed_batch_empty_list = []
#     
#     for k in range(len(random_ids_multi_super_iterations)):
#         
# 
#     
#         random_ids = random_ids_multi_super_iterations[k]
#         
#         sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
#         
#         all_indexes = np.sort(sort_idx[delta_ids])
#                 
#         id_start = 0
#     
#         id_end = 0
#         
#         random_ids_list = []
#         
#         for j in range(0, dim[0], batch_size):
#         
#             end_id = j + batch_size
#             
#             if end_id > dim[0]:
#                 end_id = dim[0]
#             
#             if all_indexes[-1] < end_id:
#                 id_end = all_indexes.shape[0]
#             else:
#                 id_end = np.argmax(all_indexes >= end_id)
#             
#             curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
# 
#             curr_matched_id_num = curr_matched_ids.shape[0]
# 
#             if curr_matched_id_num > 0:
#                 random_ids_list.append(curr_matched_ids)
#                 removed_batch_empty_list.append(False)
#             else:
#                 random_ids_list.append(random_ids[0:1])
#                 removed_batch_empty_list.append(True)
#             
#             if (i-init_epochs)%period == 0:
#                 
#                 recorded = 0
#                 
#                 use_standard_way = True
#                 
#                 
#             if i< init_epochs or use_standard_way == True:
#                 
#                 curr_rand_ids = random_ids[j:end_id]
#             
#             
#                 curr_matched_ids2 = (get_remaining_subset_data_per_epoch(curr_rand_ids, curr_matched_ids))
#             
#                 random_ids_list.append(curr_matched_ids2)
#                 
#                 recorded += 1
#                     
#                     
#                 if recorded >= length:
#                     use_standard_way = False
#             
#             i += 1
#             
#             id_start = id_end
#                 
#         random_ids_list_all_epochs.append(random_ids_list)        
    
    i = 0
    
    curr_batch_sampler = Batch_sampler(random_ids_list_all_epochs)
    
        
    data_train_loader = DataLoader(dataset_train, num_workers=0, shuffle=False, batch_sampler = curr_batch_sampler, pin_memory=True)
    
    for k in range(len(random_ids_multi_super_iterations)):            
            
        print("epoch ", k)
        random_ids_list = random_ids_list_all_epochs[k]
            
        j = 0
        
        enum_loader = enumerate(data_train_loader)
        
        for t, items in enum_loader:
            
            curr_matched_ids = items[2]        
            
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]        
                    
            curr_matched_ids_size = 0
            if not removed_batch_empty_list[i]:
                
                if not is_GPU:
                
                    batch_delta_X = items[0]
                    
                    batch_delta_Y = items[1]
                
                else:
                    batch_delta_X = items[0].to(device)
                    
                    batch_delta_Y = items[1].to(device)
                
                curr_matched_ids_size = items[2].shape[0]
            
            learning_rate = learning_rate_all_epochs[i]
            
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate
            
            if i in explicit_eval_iterations:
            
#             if (i-init_epochs)%period == 0:
#                 
#                 recorded = 0
#                 
#                 use_standard_way = True
#                 
#                 
#             if i< init_epochs or use_standard_way == True:
                
                _,next_items = enum_loader.__next__()
                
                if not is_GPU:
                
                    batch_remaining_X = next_items[0]
                    
                    batch_remaining_Y = next_items[1]
                    
                else:
                    batch_remaining_X = next_items[0].to(device)
                    
                    batch_remaining_Y = next_items[1].to(device)
                
                init_model(model, para)
                                
                compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
                 
                expect_gradients = get_all_vectorized_parameters(model.get_all_gradient())


                gradient_remaining = 0
                if curr_matched_ids_size > 0:
                    clear_gradients(model.parameters())
                        
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                
                
                    gradient_remaining = get_all_vectorized_parameters(model.get_all_gradient())     
                
                with torch.no_grad():
                               
                
                    if i>0:
                        
                        
    
                        S_k_list.append((get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1))
                        if len(S_k_list) > m:
                            S_k_list.popleft()
                    
                    gradient_full = (expect_gradients*next_items[2].shape[0] + gradient_remaining*curr_matched_ids_size)/(next_items[2].shape[0] + curr_matched_ids_size)
                    
                    if i>0:
                        
                        
                        Y_k_list.append((gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i]) + regularization_coeff*S_k_list[-1]).view(-1))
                        if len(Y_k_list) > m:
                            Y_k_list.popleft()
                    
#                         print("period::", i)
#                         
#                         print("secont condition::", torch.dot(Y_k_list[-1].view(-1), S_k_list[-1].view(-1)))
#                         
#                         print("batch size check::", curr_matched_ids_size + next_items[2].shape[0])
                        
                    
                    
                    exp_gradient = None
                    
                    para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*get_all_vectorized_parameters(para) - learning_rate*expect_gradients, full_shape_list, shape_list)
                 
                 
                first_in_period = True   
#                     recorded += 1
#                     
#                     
#                     if recorded >= length:
#                         use_standard_way = False
                
                
            else:
                gradient_dual = None
    
                if curr_matched_ids_size > 0:
                
                    init_model(model, para)
                    
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                                        
                    gradient_dual = model.get_all_gradient()
                
                with torch.no_grad():                    
                    compute_diff_vectorized_parameters(para, para_list_all_epochs[i], vec_para_diff, shape_list)                    
                    
#                     if (i-init_epochs)/period >= 1:
                    if first_in_period:
                        
                        zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m)
                        
                        mat = torch.inverse(mat_prime)
                        
                        first_in_period = False
                
                    hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff)
                        
#                     else:
#                         hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
                         
                    delta_const = 0#compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
                    
                    exp_gradient = None
                    
                    exp_param = None
                    
#                     exp_gradient = exp_gradient_list_all_epochs[i]
# #               
#  
#                      
#                     exp_param = exp_para_list_all_epochs[i]
#                            
#                     print("para_diff::")
#                     compute_model_para_diff(exp_param, para)
#                        
#                     print(curr_matched_ids_size)
                      
                    
                    
                    update_para_final(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, exp_gradient, exp_param)

                 
            
                
            i = i + 1
            
            j += batch_size
            
            
        data_train_loader.batch_sampler.increm_ids()   
            
            
    print('overhead::', overhead)
    
    print('overhead2::', overhead2)
    
    print('overhead3::', overhead3)
    
            
    return para
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     para = list(model.parameters())
#     
#     expected_para = list(model.parameters())
#     
#     last_gradient_full = None
# 
#     last_para = None
#     
#     use_standard_way = False
#     
#     recorded = 0
#     
#     overhead = 0
#     
#     vecorized_paras = get_all_vectorized_parameters_with_gradient(model.parameters())
#     
#     shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
#     
#     vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
#     
#     i = 0
#     
#     
#     gap_count = 0
#     
#     period = 2
#     
#     period_count = 1
#     
#     first_in_period = True
#     
#     
#     S_k_list = deque()
#     
#     Y_k_list = deque()
#     
#     overhead2 = 0
#     
#     overhead3 = 0
#     
#     
#     total_iteration = len(para_list_all_epochs)
#     
#     
#     
#     
#     
#     
#     
#     
#     
#     
#     
#     for k in range(len(random_ids_multi_super_iterations)):
#     
#         random_ids = random_ids_multi_super_iterations[k]
#         
#         sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
#         
#         all_indexes = np.sort(sort_idx[delta_ids])
#                 
#         id_start = 0
#     
#         id_end = 0
#         
#         
#         for j in range(0, dim[0], batch_size):
#         
#             end_id = j + batch_size
#             
#             if end_id > dim[0]:
#                 end_id = dim[0]
#             
#             if all_indexes[-1] < end_id:
#                 id_end = all_indexes.shape[0]
#             else:
#                 id_end = np.argmax(all_indexes >= end_id)
#             
#             curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
#             
#             curr_matched_ids_size = curr_matched_ids.shape[0]
#             
#             if (end_id - j - curr_matched_ids_size) <= 0:
#                 i = i + 1
#                 
#                 continue
#             
#             if curr_matched_ids_size > 0:
#                 
#                 
#                 batch_delta_X = Variable(X[curr_matched_ids])
#                 
#                 batch_delta_Y = Variable(Y[curr_matched_ids])  
#                 
#                 
#                 
#             expect_para = exp_para_list_all_epochs[i]
#                     
#                     
# #             compute_model_para_diff(list(expect_para), list(para))   
#                 
#                       
# #             if i%period == 0:
#             if gap_count >= period:
#                 
#                 recorded = 0
#                 
# #                 print(i, period)
#                 
#                 use_standard_way = True
#                 
#                 if i >= init_epochs:
#                     
# #                     if period <= max_period:
#                     if period_count % 8 == 0:
#                         period = min(period *2, max_period)
# #                     if i > total_iteration/5 and i < total_iteration/2:
# #                         period = min(period + 2, max_period)
# #                     else:
# #                         if i >= total_iteration/2:
# #                             period = min(period*2, max_period)
#                     
#                     
#                     period_count += 1
#                 
# #                 print(i, period)
#                 
#                 
#             if i< init_epochs or use_standard_way == True:
#                 
#                 gap_count = 0
#                 
#                 curr_rand_ids = random_ids[j:end_id]
#             
#             
#                 curr_matched_ids2 = (get_subset_data_per_epoch(curr_rand_ids, selected_rows_set))
#                 
#                 batch_remaining_X = X[curr_matched_ids2]
#                 
#                 batch_remaining_Y = Y[curr_matched_ids2]
#                 
#                 init_model(model, para)
#                 
#                 compute_derivative_one_more_step(model, error, batch_remaining_X, batch_remaining_Y, beta)
#                  
#                 expect_gradients = get_all_vectorized_parameters(model.get_all_gradient())
# 
# 
#                 gradient_remaining = 0
#                 if curr_matched_ids_size > 0:
#                     clear_gradients(model.parameters())
#                         
#                     compute_derivative_one_more_step(model, error, batch_delta_X, batch_delta_Y, beta)
#                 
#                 
#                     gradient_remaining = get_all_vectorized_parameters(model.get_all_gradient())     
#                 
#                 with torch.no_grad():
#                                
#                 
#                     if i>0:
#                         
#                         
#     
#                         S_k_list.append((get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1))
#                         if len(S_k_list) > m:
#                             S_k_list.popleft()
#                     
#                     gradient_full = (expect_gradients*curr_matched_ids2.shape[0] + gradient_remaining*curr_matched_ids.shape[0])/(curr_matched_ids2.shape[0] + curr_matched_ids.shape[0])
#                     
#                     if i>0:
#                         
#                         
#                         Y_k_list.append((gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1))
#                         if len(Y_k_list) > m:
#                             Y_k_list.popleft()
#                     
#                     
#                     
#                     
#                     alpha = learning_rate_all_epochs[i]
#                     
#                     para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*expect_gradients, input_dim, hidden_dims, output_dim)
#                     
#                     recorded += 1
#                     
#                 
#                 
#                 
#                    
#                     if recorded >= length:
#                         use_standard_way = False
#                         
#                         first_in_period = True
#                 
#             else:
#                 
#                 gradient_dual = None
#     
#                 if curr_matched_ids_size > 0:
#                 
# #                     t3 = time.time()
#                     init_model(model, para)
#                     
#                     
#                     
#                     compute_derivative_one_more_step(model, error, batch_delta_X, batch_delta_Y, beta)
#                     
# #                     t4 = time.time()
# #                     
# #                     overhead2 += (t4 - t3)
#                     
#                     gradient_dual = model.get_all_gradient()
#                 
#                 with torch.no_grad():
#                 
#                 
# #                 t5 = time.time()
# #                     v_vec = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
#                     
#                     compute_diff_vectorized_parameters(para, para_list_all_epochs[i], vec_para_diff, shape_list)
#     #                 v_vec2 = compute_model_para_diff2(para, para_list_all_epochs[i])
#                     
#                     
#                     if period_count > 1:
# #                         if i % period == 1:
#                         if first_in_period:
# #                             print("prepare lbfgs ", i)
#     #                         
#     #                         if i >= 370:
#     #                             y = 0
#     #                             y+=1
#                             
#                             zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2 = prepare_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, period)
#                             
#                             mat_1_inverse = torch.inverse(mat_1)
#                             
#                             mat_2_inverse = torch.inverse(mat_2)
#                             
#                             first_in_period = False
#                     
#                         
#                         hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1_inverse, mat_2_inverse, vec_para_diff)
#     #                     hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms2(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2, v_vec2, shape_list)
#                         
#                     else:
#                         hessian_para_prod = cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
#                     
#                                         
#                     
#                     delta_const = 0#compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
#                     
#         #             para_diff = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
#                     
#                     alpha = learning_rate_all_epochs[i]
#                     
#                     update_para_final(para, get_devectorized_parameters(hessian_para_prod, input_dim, hidden_dims, output_dim), gradient_dual, gradient_list_all_epochs[i], end_id - j, curr_matched_ids_size, alpha)
# 
#                 gap_count += 1
#                 
#             i = i + 1
#             
#             id_start = id_end    
#             
# #         period = min(period*2, max_period)
#             
#     print('overhead::', overhead)
#     
#     print('overhead2::', overhead2)
#     
#     print('overhead3::', overhead3)
#     
#             
#     return para







def model_training_quantized_baseline(origin_model, epsilon, X, Y, model, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, input_dim, hidden_dims, output_dim, alpha, beta, selected_rows, error, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, random_theta_list_all_epochs):
    
    
#     para_num = S_k_list.shape[0]
    
    
#     model_dual = DNNModel(input_dim, hidden_dims, output_dim)
    selected_rows_set = set(selected_rows.view(-1).tolist())
    
    para = list(model.parameters())
    
    use_standard_way = False
    
    recorded = 0
    
    overhead = 0
    
    shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
        
    i = 0
    
    overhead2 = 0
    
    overhead3 = 0
    
    for k in range(len(random_ids_multi_super_iterations)):
    
        random_ids = random_ids_multi_super_iterations[k]
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
        all_indexes = np.sort(sort_idx[delta_ids])
                
        id_start = 0
    
        id_end = 0
        
        for j in range(0, dim[0], batch_size):
        
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
            
            curr_matched_ids_size = curr_matched_ids.shape[0]
        
        
            curr_rand_ids = random_ids[j:end_id]
            
            curr_selected_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
        
        
        
        
            
            if (end_id - j - curr_matched_ids_size) <= 0:
                i = i + 1
                print('iteration::', i)
                continue
            
#             if i >= 394:
#                 y = 0
#                 
#                 y += 1
#             
#             
#             if curr_matched_ids_size > 0:
            else:
                
                batch_delta_X = Variable(X[curr_selected_ids])
                
                batch_delta_Y = Variable(Y[curr_selected_ids])  
                
                compute_derivative_one_more_step(model, error, batch_delta_X, batch_delta_Y, beta)
                
                
                
                
                random_theta_list = random_theta_list_all_epochs[i]
                
                
                matches = update_and_zero_model_gradient_quantized_baseline(model, alpha, epsilon, gradient_list_all_epochs, para_list_all_epochs, i, random_theta_list, end_id - j, curr_matched_ids_size)
                
                if matches == False:
                    return False
            
#             else:
#                 random_theta_list = random_theta_list_all_epochs[i]
#                 
#                 
#                 
#                 
#                 
#                 matches = update_and_zero_model_gradient_quantized_incremental_no_delta(model, alpha, epsilon, gradient_list_all_epochs, para_list_all_epochs, i, random_theta_list, end_id - j, curr_matched_ids_size)

            
            
            print('iteration::', i)
            
            
            i = i + 1
            
            id_start = id_end            
            
            
    print('overhead::', overhead)
    
    print('overhead2::', overhead2)
    
    print('overhead3::', overhead3)
    
    compute_model_para_diff(list(origin_model.parameters()), para)
    
    return para




# def model_update_provenance_test2(exp_gradient_list_all_epochs, exp_para_list_all_epochs, X, Y, model, S_k_list, Y_k_list, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, input_dim, hidden_dims, output_dim, m, alpha, beta, selected_rows, error):
#     
#     
#     para_num = S_k_list.shape[0]
#     
#     
#     model_dual = DNNModel(input_dim, hidden_dims, output_dim)
#     
#     
#     para = list(model.parameters())
#     
#     expected_para = list(model.parameters())
#     
#     
#     gradient_full = None
#     
#     last_para = None
#     
#     for i in range(epoch):
#         
#         init_model(model, para)
#         
#         compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
#         
#         expect_gradients = get_all_vectorized_parameters(model.get_all_gradient())
#         
#         
#         init_model(model, expected_para)
#         
#         compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
#         
# #         compute_model_para_diff(expected_para, exp_para_list_all_epochs[i])
# 
#         
#         expected_para = get_devectorized_parameters(get_all_vectorized_parameters(expected_para) - alpha*get_all_vectorized_parameters(model.get_all_gradient()), input_dim, hidden_dims, output_dim)
#         
#         
#         if i >= m + 1:
#         
#             init_model(model_dual, para)
#              
#             compute_derivative_one_more_step(model_dual, error, X[delta_ids], Y[delta_ids], beta)
#              
#             gradient_dual = get_all_vectorized_parameters(model_dual.get_all_gradient())
# #             
# #             
# #             
# #             
# #             print('para_diff::', torch.norm(v_vec))
# #             
# #             
# #             
# #             init_model(model, para_list_all_epochs[i])
# #              
# #             curr_vec_gradient,_ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, gradient_list_all_epochs[i], beta)
# #              
# # #             hessian_matrix1 = compute_hessian_matrix(model, curr_vec_gradient, input_dim, hidden_dims, output_dim)
# #             
# #             
# #             
# #             
# #             hessian_para_prod = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1))
#             
# #             delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
# #             
# # #             para_diff = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
# #             
# #             gradient_full = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
# #             
# #             delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1), error, model, beta)
# #             gradient_full2 = torch.mm(v_vec.view(1,-1), hessian_matrix1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
# # #             gradient_full2 = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) - delta_const*v_vec.view(1,-1)
# #             
#             gradients = (gradient_full*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
# #             
# #             gradients2 = (gradient_full2*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
# #             
# #             print('gradient_diff::', torch.norm(gradients - expect_gradients))
# #             
# #             print('gradient2_diff::', torch.norm(gradients2 - expect_gradients))
# #             
# #             compute_model_para_diff(exp_para_list_all_epochs[i], para)
# 
# 
#             last_para = para
# 
# 
# #             gradients = (gradient_full*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
# 
#             
#             para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradients, input_dim, hidden_dims, output_dim)
#         
#             compute_model_para_diff(exp_para_list_all_epochs[i], para)
#         
#         
#             v_vec = get_all_vectorized_parameters(para) - get_all_vectorized_parameters(last_para)
#         
#             
#             gradient_full = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1)) + gradient_full
#             
#             
#             init_model(model, para)
#         
#             compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             expected_gradient_full = model.get_all_gradient()
#             
#             print('gradient_diff::')
#             
#             compute_model_para_diff(get_devectorized_parameters(gradient_full, input_dim, hidden_dims, output_dim), expected_gradient_full)
#         
#         else:
#             
# #             last_para = para.clone()
#             
#             if i == m:
#                 
# #                 gradients = expect_gradients
#                 last_para = para
#             
#             para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*expect_gradients, input_dim, hidden_dims, output_dim)
#         
#         
#             if i == m:
# #                 v_vec = get_all_vectorized_parameters(para) - get_all_vectorized_parameters(last_para)
#                 
#                 init_model(model, para)
#         
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#                 
#                 gradient_full = get_all_vectorized_parameters(model.get_all_gradient())
#                 
# #                 gradient_full = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1)) + expect_gradients
#         
#         
#         
# #             init_model(model, get_devectorized_parameters(updated_para, input_dim, hidden_dims, output_dim))
#             
#             
#             
#             
#             
#             
#             
#     return para



def model_training_quantized_incremental(origin_model, epsilon, X, Y, model, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, input_dim, hidden_dims, output_dim, alpha, beta, selected_rows, error, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, random_theta_list_all_epochs):
    
    
#     para_num = S_k_list.shape[0]
    
    
#     model_dual = DNNModel(input_dim, hidden_dims, output_dim)
    selected_rows_set = set(selected_rows.view(-1).tolist())
    
    para = list(model.parameters())
    
    use_standard_way = False
    
    recorded = 0
    
    overhead = 0
    
    shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
        
    i = 0
    
    overhead2 = 0
    
    overhead3 = 0
    
    for k in range(len(random_ids_multi_super_iterations)):
    
        random_ids = random_ids_multi_super_iterations[k]
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
        all_indexes = np.sort(sort_idx[delta_ids])
                
        id_start = 0
    
        id_end = 0
        
        for j in range(0, dim[0], batch_size):
        
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
            
            curr_matched_ids_size = curr_matched_ids.shape[0]
        
        
            curr_rand_ids = random_ids[i:end_id]
            
            curr_selected_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
        
        
        
        
            
            if (end_id - j - curr_matched_ids_size) <= 0:
                i = i + 1
#                 print('iteration::', i)
                continue

            
            
            if curr_matched_ids_size > 0:
                
                
                batch_delta_X = Variable(X[curr_matched_ids])
                
                batch_delta_Y = Variable(Y[curr_matched_ids])  
                
                compute_derivative_one_more_step(model, error, batch_delta_X, batch_delta_Y, beta)
                
                
#                 print(i)
                
                random_theta_list = random_theta_list_all_epochs[i]
                
                
                matches = update_and_zero_model_gradient_quantized_incremental(model, alpha, epsilon, gradient_list_all_epochs, para_list_all_epochs, i, random_theta_list, end_id - j, curr_matched_ids_size)
                
#                 if matches == False:
#                     return False
            
            else:
                random_theta_list = random_theta_list_all_epochs[i]
                
                
                
                if i + 1 < len(para_list_all_epochs):
                    init_model(model, para_list_all_epochs[i+1])
                else:
                    update_and_zero_model_gradient_quantized_incremental_no_delta(model, alpha, epsilon, gradient_list_all_epochs, para_list_all_epochs, i, random_theta_list, end_id - j, curr_matched_ids_size)

            
            
#             print('iteration::', i)
            
            
            i = i + 1
            
            id_start = id_end            
            
            
#     print('overhead::', overhead)
#     
#     print('overhead2::', overhead2)
#     
#     print('overhead3::', overhead3)
    
#     compute_model_para_diff(list(origin_model.parameters()), para)
    
    return True




# def model_update_provenance_test2(exp_gradient_list_all_epochs, exp_para_list_all_epochs, X, Y, model, S_k_list, Y_k_list, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, input_dim, hidden_dims, output_dim, m, alpha, beta, selected_rows, error):
#     
#     
#     para_num = S_k_list.shape[0]
#     
#     
#     model_dual = DNNModel(input_dim, hidden_dims, output_dim)
#     
#     
#     para = list(model.parameters())
#     
#     expected_para = list(model.parameters())
#     
#     
#     gradient_full = None
#     
#     last_para = None
#     
#     for i in range(epoch):
#         
#         init_model(model, para)
#         
#         compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
#         
#         expect_gradients = get_all_vectorized_parameters(model.get_all_gradient())
#         
#         
#         init_model(model, expected_para)
#         
#         compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
#         
# #         compute_model_para_diff(expected_para, exp_para_list_all_epochs[i])
# 
#         
#         expected_para = get_devectorized_parameters(get_all_vectorized_parameters(expected_para) - alpha*get_all_vectorized_parameters(model.get_all_gradient()), input_dim, hidden_dims, output_dim)
#         
#         
#         if i >= m + 1:
#         
#             init_model(model_dual, para)
#              
#             compute_derivative_one_more_step(model_dual, error, X[delta_ids], Y[delta_ids], beta)
#              
#             gradient_dual = get_all_vectorized_parameters(model_dual.get_all_gradient())
# #             
# #             
# #             
# #             
# #             print('para_diff::', torch.norm(v_vec))
# #             
# #             
# #             
# #             init_model(model, para_list_all_epochs[i])
# #              
# #             curr_vec_gradient,_ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, gradient_list_all_epochs[i], beta)
# #              
# # #             hessian_matrix1 = compute_hessian_matrix(model, curr_vec_gradient, input_dim, hidden_dims, output_dim)
# #             
# #             
# #             
# #             
# #             hessian_para_prod = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1))
#             
# #             delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
# #             
# # #             para_diff = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
# #             
# #             gradient_full = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
# #             
# #             delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1), error, model, beta)
# #             gradient_full2 = torch.mm(v_vec.view(1,-1), hessian_matrix1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
# # #             gradient_full2 = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) - delta_const*v_vec.view(1,-1)
# #             
#             gradients = (gradient_full*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
# #             
# #             gradients2 = (gradient_full2*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
# #             
# #             print('gradient_diff::', torch.norm(gradients - expect_gradients))
# #             
# #             print('gradient2_diff::', torch.norm(gradients2 - expect_gradients))
# #             
# #             compute_model_para_diff(exp_para_list_all_epochs[i], para)
# 
# 
#             last_para = para
# 
# 
# #             gradients = (gradient_full*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
# 
#             
#             para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradients, input_dim, hidden_dims, output_dim)
#         
#             compute_model_para_diff(exp_para_list_all_epochs[i], para)
#         
#         
#             v_vec = get_all_vectorized_parameters(para) - get_all_vectorized_parameters(last_para)
#         
#             
#             gradient_full = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1)) + gradient_full
#             
#             
#             init_model(model, para)
#         
#             compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             expected_gradient_full = model.get_all_gradient()
#             
#             print('gradient_diff::')
#             
#             compute_model_para_diff(get_devectorized_parameters(gradient_full, input_dim, hidden_dims, output_dim), expected_gradient_full)
#         
#         else:
#             
# #             last_para = para.clone()
#             
#             if i == m:
#                 
# #                 gradients = expect_gradients
#                 last_para = para
#             
#             para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*expect_gradients, input_dim, hidden_dims, output_dim)
#         
#         
#             if i == m:
# #                 v_vec = get_all_vectorized_parameters(para) - get_all_vectorized_parameters(last_para)
#                 
#                 init_model(model, para)
#         
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#                 
#                 gradient_full = get_all_vectorized_parameters(model.get_all_gradient())
#                 
# #                 gradient_full = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1)) + expect_gradients
#         
#         
#         
# #             init_model(model, get_devectorized_parameters(updated_para, input_dim, hidden_dims, output_dim))
#             
#             
#             
#             
#             
#             
#             
#     return para


def compute_para_dif_by_layers(para1, para2, S_k_list, k):
    
    for i in range(len(para1)):
        
        curr_para1 = para1[i]
        
        curr_para2 = para2[i]


        S_k_list[i][:,k] = (curr_para1 - curr_para2).detach().view(-1)
        
    
def compute_grad_diff_by_layers(grad_full, grad_list, Y_k_list, k):
    
    for i in range(len(grad_full)):
        
        
        Y_k_list[i][:,k] = (grad_full[i] - grad_list[i]).detach().view(-1)
        


def get_full_gradients_by_layer(expect_grad, size1, grad_remaining, size2):
    
    full_gradients = []
    
    for i in range(len(expect_grad)):
        full_gradients.append((expect_grad[i]*size1 + grad_remaining[i]*size2)/(size1 + size2))
        
    return full_gradients
        


def update_para_by_layer(para_list, gradient_list, alpha):
    
    res_para = []
    
    for i in range(len(para_list)):
        
        res_para.append(para_list[i] - alpha*gradient_list[i])
        
    return res_para
        
    
def compute_gradients_with_hessian_prod(hessian_para_prod_list, grad_list, grad_dual_list, size1, size2):
    
    
    gradients = []
    
    for i in range(len(hessian_para_prod_list)):
    
        hessian_para_prod = hessian_para_prod_list[i]
    
#         gradient_full = 
        
        grad_dual = grad_dual_list[i]
                    
#         gradients = (gradient_full*size1 - grad_dual*curr_matched_ids_size)/(end_id - j - curr_matched_ids_size)
        
        gradients.append(((hessian_para_prod.view(grad_list[i].shape) + grad_list[i])*size1 - grad_dual*size2)/(size1 - size2))

    return gradients
    

def model_update_provenance_test4(period, length, init_epochs, truncted_s, extended_Y_k_list, exp_gradient_list_all_epochs, exp_para_list_all_epochs, X, Y, model, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, input_dim, hidden_dims, output_dim, m, alpha, beta, selected_rows, error, delta_X,delta_Y, update_X, update_Y, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim):
    
    
#     para_num = S_k_list.shape[0]
    
    
#     model_dual = DNNModel(input_dim, hidden_dims, output_dim)
    selected_rows_set = set(selected_rows.view(-1).tolist())
    
    para = list(model.parameters())
    
    expected_para = list(model.parameters())
    
    last_gradient_full = None

    last_para = None
    
    use_standard_way = False
    
    recorded = 0
    
    overhead = 0
    
    vecorized_paras = get_all_vectorized_parameters_with_gradient(model.parameters())
    
    shape_list = get_model_para_shape_list(model.parameters())
    
#     remaining_shape_num = 0
#     
#     for i in range(len(shape_list) - first_few_layer_num):
#         remaining_shape_num += shape_list[i+first_few_layer_num]
#         
#     S_k_list = torch.zeros(remaining_shape_num, len(random_ids_multi_super_iterations))
#     
#     
#     Y_k_list = torch.zeros(remaining_shape_num, len(random_ids_multi_super_iterations))
    
    
    S_k_list = []
    
    Y_k_list = []
    
    
    for shape in shape_list:
        
        S_k_list.append(torch.zeros([shape, len(para_list_all_epochs)], dtype = torch.double))
        
        Y_k_list.append(torch.zeros([shape, len(para_list_all_epochs)], dtype = torch.double))
    
    
    
    i = 0
    
    for k in range(len(random_ids_multi_super_iterations)):
    
        random_ids = random_ids_multi_super_iterations[k]
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
        all_indexes = np.sort(sort_idx[delta_ids])
                
        id_start = 0
    
        id_end = 0
        
        for j in range(0, dim[0], batch_size):
        
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            
            
#             while 1:
#                 if id_end >= all_indexes.shape[0] or all_indexes[id_end] >= end_id:
#                     break
#                 
#                 id_end = id_end + 1

            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
            
            curr_matched_ids_size = curr_matched_ids.shape[0]
            
            if (end_id - j - curr_matched_ids_size) <= 0:
                i = i + 1
#                 theta_list.append(vectorized_theta)
#                 
#                 output_list.append(0)
#     
#                 sub_term2_list.append(0)
#                 
#                 x_sum_by_list.append(0)
#                 
#                 sub_term_1_theta_list.append(0)
                
                
                continue
            
#             print(i, torch.norm(torch.sort(curr_matched_ids_2)[0].type(torch.DoubleTensor) - torch.sort(curr_matched_ids)[0].type(torch.DoubleTensor)))
            
#             curr_rand_id_set = set(curr_rand_ids.view(-1).tolist())
            
#             curr_matched_ids = (curr_rand_ids.view(-1,1) == delta_ids.view(1,-1))
#             curr_matched_ids = torch.tensor(list(delta_ids_set.intersection(curr_rand_id_set)))
            
            
#             curr_nonzero_ids = torch.nonzero(((nonzero_ids[:, 0] >= i)*(nonzero_ids[:, 0] < end_id))).view(-1)
#             
#             curr_nonzero_ids_this_batch0 = nonzero_ids[curr_nonzero_ids][:, 1]
            
#             curr_nonzero_ids_this_batch = torch.nonzero(curr_matched_ids)[:, 1]
#             print(curr_matched_ids)
            if curr_matched_ids_size > 0:
                
                
                batch_delta_X = (X[curr_matched_ids])
                
                batch_delta_Y = (Y[curr_matched_ids])  
                
                
                
                
                
                      
            if i%period == 0:
                
                recorded = 0
                
                use_standard_way = True
                
                
            if i< init_epochs or use_standard_way == True:
                
#                 if i % period == 1:
#                     zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2 = prepare_hessian_vec_prod0(S_k_list, Y_k_list, i, m, k, period)

                
                
                curr_rand_ids = random_ids[j:end_id]
            
            
                curr_matched_ids2 = (get_subset_data_per_epoch(curr_rand_ids, selected_rows_set))
                
#                 if curr_matched_ids_size <= 0:
#                     continue
                
                batch_remaining_X = X[curr_matched_ids2]
                
                batch_remaining_Y = Y[curr_matched_ids2]
                
                init_model(model, para)
                
#                 print('epoch::', i)
                
                compute_derivative_one_more_step2(model, error, batch_remaining_X, batch_remaining_Y, beta, vecorized_paras)
                 
                expect_gradients = model.get_all_gradient()
                
                
                if i>0:
#                     S_k_list[:,i - 1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
                    compute_para_dif_by_layers(para, para_list_all_epochs[i], S_k_list, i-1)
                    
#                     print(i-1)
#                     
#                     print(S_k_list[:,i - 1])
                
    #             init_model(model, para)
                clear_gradients(model.parameters())
                    
                compute_derivative_one_more_step2(model, error, batch_delta_X, batch_delta_Y, beta, vecorized_paras)
            
            
                gradient_remaining = model.get_all_gradient()
                
                
#                 gradient_full = (expect_gradients*curr_matched_ids2.shape[0] + gradient_remaining*curr_matched_ids.shape[0])/(curr_matched_ids2.shape[0] + curr_matched_ids.shape[0])
                
                gradient_full = get_full_gradients_by_layer(expect_gradients, curr_matched_ids2.shape[0], gradient_remaining, curr_matched_ids.shape[0])
                
                if i>0:
#                     Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
                    compute_grad_diff_by_layers(gradient_full, gradient_list_all_epochs[i], Y_k_list, i-1)
                
#                 decompose_model_paras2(para, para_list_all_epochs[i], get_devectorized_parameters(gradient_full, input_dim, hidden_dims, output_dim), gradient_list_all_epochs[i])
                
#                     print(torch.dot(Y_k_list[:,i-1], S_k_list[:,i-1]))
#                     y=0
#                     y+=1
                para = update_para_by_layer(para, expect_gradients, alpha)
#                 para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*expect_gradients, input_dim, hidden_dims, output_dim)
    #                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
                

                
                recorded += 1
                
                
                if recorded >= length:
                    use_standard_way = False
                
                
            else:
                
#                 print('epoch::', i)
                
                
    #             delta_X = X[delta_ids]
    #             
    #             delta_Y = Y[delta_ids]
                
                gradient_dual = 0
    
                if curr_matched_ids_size > 0:
                
                    init_model(model, para)
                    
                    compute_derivative_one_more_step2(model, error, batch_delta_X, batch_delta_Y, beta, vecorized_paras)
                    
                    
                    gradient_dual = model.get_all_gradient()
                
                
#                 v_vec = get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])
                
                
                v_vec2 = compute_model_para_diff2(para, para_list_all_epochs[i])
                
                
                if i/period >= 1:
                    if i % period == 1:
                        zero_mat_dim, sigma_k, mat_1, mat_2, ids = prepare_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, period)
                
                    t1 = time.time()
#                     hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2, v_vec.view(-1,1))
                    hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms2(ids, zero_mat_dim, Y_k_list, S_k_list, sigma_k, mat_1, mat_2, v_vec2)
                    t2 = time.time()
                    
                    overhead += (t2 - t1)
                else:
                    
                    zero_mat_dim, sigma_k, mat_1, mat_2, ids = prepare_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, period)
                    hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms2(ids, zero_mat_dim, Y_k_list, S_k_list, sigma_k, mat_1, mat_2, v_vec2)
#                     hessian_para_prod = cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, v_vec.view(-1,1), period)
                
                
    #             print('para_diff::', torch.norm(v_vec))
    #             
    #             print('para_angle::', torch.dot(get_all_vectorized_parameters(para).view(-1), get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))/(torch.norm(get_all_vectorized_parameters(para).view(-1))*torch.norm(get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))))
                
                
                
    #             init_model(model, para_list_all_epochs[i])
                 
    #             curr_vec_gradient,_ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, gradient_list_all_epochs[i], beta)
                 
    #             hessian_matrix1 = compute_hessian_matrix(model, curr_vec_gradient, input_dim, hidden_dims, output_dim)
                
                
                
#                 cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, v_vec.view(-1,1), period)
                
                
                
                
    #             hessian_para_prod = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1))
                
    #             hessian_para_prod = cal_approx_hessian_vec_prod2(S_k_list, Y_k_list, i, m-1, v_vec.view(-1,1), last_v_vec.view(-1,1), last_gradient_full.view(-1, 1) - get_all_vectorized_parameters(gradient_list_all_epochs[i-1]).view(-1,1))
                
    #             hessian_para_prod = cal_approx_hessian_vec_prod3(i, m, v_vec.view(-1,1), para_list_all_epochs, gradient_list_all_epochs)
                
    #             hessian_para_prod, tmp_res = cal_approx_hessian_vec_prod4(truncted_s, extended_Y_k_list[i], i, m, v_vec.view(-1,1))
                
                
                delta_const = 0#compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
                
    #             para_diff = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
                
                
                gradients = compute_gradients_with_hessian_prod(hessian_para_prod, gradient_list_all_epochs[i], gradient_dual, end_id - j, curr_matched_ids_size)
                
                
    #             gradients2 = (gradient_full2*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
                
    #             print('hessian_vector_prod_diff::', torch.norm(torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1) - hessian_para_prod))
                
    #             print('gradient_diff::', torch.norm(gradients - expect_gradients))
                
    #             print('gradient2_diff::', torch.norm(gradients2 - expect_gradients))
                
    #             compute_model_para_diff(exp_para_list_all_epochs[i], para)
                
#                 S_k_list[:,i-1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
                
                para = update_para_by_layer(para, gradients, alpha)# get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradients, input_dim, hidden_dims, output_dim)
            
            
#                 Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
#                 print(torch.norm(get_all_vectorized_parameters(para)))
                
#                 print(Y_k_list[:,i-1])
                
                
                
            i = i + 1
            
            id_start = id_end
            
#             last_gradient_full = gradient_full
#             
#             last_v_vec = v_vec.clone()
        
        
#         else:
#             if i >= 1:
#                 S_k_list[:,i - 1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
#             last_para = para
#             
#             para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*expect_gradients, input_dim, hidden_dims, output_dim)
#         
#             if i == m-1:
#                 init_model(model, para)
#                 
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             
#                 last_gradient_full = get_all_vectorized_parameters(model.get_all_gradient())
#                 
#                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
#                 
#             if i >= 1:
#                 
#                 init_model(model, para)
#                 
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             
#                 gradient_full = get_all_vectorized_parameters(model.get_all_gradient())
#                 
# #                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
#                 
#                 Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
        
#         last_gradient = expect_gradients
#             init_model(model, get_devectorized_parameters(updated_para, input_dim, hidden_dims, output_dim))
            
            
            
            
    print('overhead::', overhead)
    
            
    return para




# def model_update_provenance_test2(exp_gradient_list_all_epochs, exp_para_list_all_epochs, X, Y, model, S_k_list, Y_k_list, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, input_dim, hidden_dims, output_dim, m, alpha, beta, selected_rows, error):
#     
#     
#     para_num = S_k_list.shape[0]
#     
#     
#     model_dual = DNNModel(input_dim, hidden_dims, output_dim)
#     
#     
#     para = list(model.parameters())
#     
#     expected_para = list(model.parameters())
#     
#     
#     gradient_full = None
#     
#     last_para = None
#     
#     for i in range(epoch):
#         
#         init_model(model, para)
#         
#         compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
#         
#         expect_gradients = get_all_vectorized_parameters(model.get_all_gradient())
#         
#         
#         init_model(model, expected_para)
#         
#         compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
#         
# #         compute_model_para_diff(expected_para, exp_para_list_all_epochs[i])
# 
#         
#         expected_para = get_devectorized_parameters(get_all_vectorized_parameters(expected_para) - alpha*get_all_vectorized_parameters(model.get_all_gradient()), input_dim, hidden_dims, output_dim)
#         
#         
#         if i >= m + 1:
#         
#             init_model(model_dual, para)
#              
#             compute_derivative_one_more_step(model_dual, error, X[delta_ids], Y[delta_ids], beta)
#              
#             gradient_dual = get_all_vectorized_parameters(model_dual.get_all_gradient())
# #             
# #             
# #             
# #             
# #             print('para_diff::', torch.norm(v_vec))
# #             
# #             
# #             
# #             init_model(model, para_list_all_epochs[i])
# #              
# #             curr_vec_gradient,_ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, gradient_list_all_epochs[i], beta)
# #              
# # #             hessian_matrix1 = compute_hessian_matrix(model, curr_vec_gradient, input_dim, hidden_dims, output_dim)
# #             
# #             
# #             
# #             
# #             hessian_para_prod = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1))
#             
# #             delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
# #             
# # #             para_diff = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
# #             
# #             gradient_full = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
# #             
# #             delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1), error, model, beta)
# #             gradient_full2 = torch.mm(v_vec.view(1,-1), hessian_matrix1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
# # #             gradient_full2 = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) - delta_const*v_vec.view(1,-1)
# #             
#             gradients = (gradient_full*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
# #             
# #             gradients2 = (gradient_full2*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
# #             
# #             print('gradient_diff::', torch.norm(gradients - expect_gradients))
# #             
# #             print('gradient2_diff::', torch.norm(gradients2 - expect_gradients))
# #             
# #             compute_model_para_diff(exp_para_list_all_epochs[i], para)
# 
# 
#             last_para = para
# 
# 
# #             gradients = (gradient_full*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
# 
#             
#             para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradients, input_dim, hidden_dims, output_dim)
#         
#             compute_model_para_diff(exp_para_list_all_epochs[i], para)
#         
#         
#             v_vec = get_all_vectorized_parameters(para) - get_all_vectorized_parameters(last_para)
#         
#             
#             gradient_full = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1)) + gradient_full
#             
#             
#             init_model(model, para)
#         
#             compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             expected_gradient_full = model.get_all_gradient()
#             
#             print('gradient_diff::')
#             
#             compute_model_para_diff(get_devectorized_parameters(gradient_full, input_dim, hidden_dims, output_dim), expected_gradient_full)
#         
#         else:
#             
# #             last_para = para.clone()
#             
#             if i == m:
#                 
# #                 gradients = expect_gradients
#                 last_para = para
#             
#             para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*expect_gradients, input_dim, hidden_dims, output_dim)
#         
#         
#             if i == m:
# #                 v_vec = get_all_vectorized_parameters(para) - get_all_vectorized_parameters(last_para)
#                 
#                 init_model(model, para)
#         
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#                 
#                 gradient_full = get_all_vectorized_parameters(model.get_all_gradient())
#                 
# #                 gradient_full = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1)) + expect_gradients
#         
#         
#         
# #             init_model(model, get_devectorized_parameters(updated_para, input_dim, hidden_dims, output_dim))
#             
#             
#             
#             
#             
#             
#             
#     return para



def model_update_provenance_test1_advanced(first_few_layer_num, period, length, init_epochs, truncted_s, extended_Y_k_list, exp_gradient_list_all_epochs, exp_para_list_all_epochs, X, Y, model, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, input_dim, hidden_dims, output_dim, m, alpha, beta, selected_rows, error, delta_X,delta_Y, update_X, update_Y, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim):
    
    
    selected_rows_set = set(selected_rows.view(-1).tolist())
    
    para = list(model.parameters())
    
    expected_para = list(model.parameters())
    
    last_gradient_full = None

    last_para = None
    
    use_standard_way = False
    
    recorded = 0
    
    overhead = 0
    
    vecorized_paras = get_all_vectorized_parameters_with_gradient(model.parameters())
    
    shape_list = get_model_para_shape_list(model.parameters())
    
    remaining_shape_num = 0
    
    for i in range(len(shape_list) - 2*first_few_layer_num):
        remaining_shape_num += shape_list[i+2*first_few_layer_num]
        
    S_k_list = torch.zeros([remaining_shape_num, len(para_list_all_epochs)], dtype = torch.double)
    
    
    Y_k_list = torch.zeros([remaining_shape_num, len(para_list_all_epochs)], dtype = torch.double)
    
    i = 0
    
    for k in range(len(random_ids_multi_super_iterations)):
    
        random_ids = random_ids_multi_super_iterations[k]
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
        all_indexes = np.sort(sort_idx[delta_ids])
                
        id_start = 0
    
        id_end = 0
        
        for j in range(0, dim[0], batch_size):
        
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]

            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
            
            curr_matched_ids_size = curr_matched_ids.shape[0]
            
            if (end_id - j - curr_matched_ids_size) <= 0:
                i = i + 1
                continue
            
            if curr_matched_ids_size > 0:
                batch_delta_X = (X[curr_matched_ids])
                
                batch_delta_Y = (Y[curr_matched_ids])  
                
                      
            if i%period == 0:
                
                recorded = 0
                
                use_standard_way = True
                
                
            if i< init_epochs or use_standard_way == True:
                
                curr_rand_ids = random_ids[j:end_id]
            
            
                curr_matched_ids2 = (get_subset_data_per_epoch(curr_rand_ids, selected_rows_set))

                batch_remaining_X = X[curr_matched_ids2]
                
                batch_remaining_Y = Y[curr_matched_ids2]
                
                init_model(model, para)
                
                compute_derivative_one_more_step2(model, error, batch_remaining_X, batch_remaining_Y, beta, vecorized_paras)
                 
                expect_gradients = get_all_vectorized_parameters_by_layers(model.get_all_gradient(), first_few_layer_num)
                
                
                if i>0:
                    S_k_list[:,i - 1] = (get_all_vectorized_parameters_by_layers(para, first_few_layer_num) - get_all_vectorized_parameters_by_layers(para_list_all_epochs[i], first_few_layer_num)).view(-1)

                clear_gradients(model.parameters())
                    
                compute_derivative_one_more_step2(model, error, batch_delta_X, batch_delta_Y, beta, vecorized_paras)
            
            
                gradient_remaining = get_all_vectorized_parameters_by_layers(model.get_all_gradient(), first_few_layer_num)
                
                
                gradient_full = (expect_gradients*curr_matched_ids2.shape[0] + gradient_remaining*curr_matched_ids.shape[0])/(curr_matched_ids2.shape[0] + curr_matched_ids.shape[0])
                
                if i>0:
                    Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters_by_layers(gradient_list_all_epochs[i], first_few_layer_num)).view(-1)
                
                
#                 decompose_model_paras2(para, para_list_all_epochs[i], get_devectorized_parameters(gradient_full, input_dim, hidden_dims, output_dim), gradient_list_all_epochs[i])

                last_few_model_para = get_all_vectorized_parameters_by_layers(para, first_few_layer_num) - alpha*expect_gradients
                
                para = get_devectorized_parameters_by_layers(para_list_all_epochs[i], last_few_model_para, input_dim, hidden_dims, output_dim, first_few_layer_num)
                
                
                recorded += 1
                
                
                if recorded >= length:
                    use_standard_way = False
                
                
            else:

                gradient_dual = 0
    
                if curr_matched_ids_size > 0:
                
                    init_model(model, para)
                    
                    compute_derivative_one_more_step2(model, error, batch_delta_X, batch_delta_Y, beta, vecorized_paras)
                    
                    
                    gradient_dual = get_all_vectorized_parameters_by_layers(model.get_all_gradient(), first_few_layer_num)
                
                
                v_vec = get_all_vectorized_parameters_by_layers(para, first_few_layer_num) - get_all_vectorized_parameters_by_layers(para_list_all_epochs[i], first_few_layer_num)
                
                
                v_vec2 = compute_model_para_diff2(para, para_list_all_epochs[i])
                
                
                if i/period >= 1:
                    if i % period == 1:
                        zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2 = prepare_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, period)
                
                    t1 = time.time()
                    hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2, v_vec.view(-1,1))
#                     hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms2(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2, v_vec2, shape_list)
                    t2 = time.time()
                    
                    overhead += (t2 - t1)
                else:
                    hessian_para_prod = cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, v_vec.view(-1,1), period)
                
                
                gradient_full = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters_by_layers(gradient_list_all_epochs[i], first_few_layer_num)
                
                gradients = (gradient_full*(end_id - j) - gradient_dual*curr_matched_ids_size)/(end_id - j - curr_matched_ids_size)
                
                
                last_few_model_para = get_all_vectorized_parameters_by_layers(para, first_few_layer_num) - alpha*gradients
                
                para = get_devectorized_parameters_by_layers(para_list_all_epochs[i], last_few_model_para, input_dim, hidden_dims, output_dim, first_few_layer_num)
                
                
#                 para = get_devectorized_parameters(get_all_vectorized_parameters_by_layers(para, first_few_layer_num) - alpha*gradients, input_dim, hidden_dims, output_dim)
            
            
              
                
            i = i + 1
            
            id_start = id_end          
            
    print('overhead::', overhead)
    
            
    return para




# def model_update_provenance_test2(exp_gradient_list_all_epochs, exp_para_list_all_epochs, X, Y, model, S_k_list, Y_k_list, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, input_dim, hidden_dims, output_dim, m, alpha, beta, selected_rows, error):
#     
#     
#     para_num = S_k_list.shape[0]
#     
#     
#     model_dual = DNNModel(input_dim, hidden_dims, output_dim)
#     
#     
#     para = list(model.parameters())
#     
#     expected_para = list(model.parameters())
#     
#     
#     gradient_full = None
#     
#     last_para = None
#     
#     for i in range(epoch):
#         
#         init_model(model, para)
#         
#         compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
#         
#         expect_gradients = get_all_vectorized_parameters(model.get_all_gradient())
#         
#         
#         init_model(model, expected_para)
#         
#         compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
#         
# #         compute_model_para_diff(expected_para, exp_para_list_all_epochs[i])
# 
#         
#         expected_para = get_devectorized_parameters(get_all_vectorized_parameters(expected_para) - alpha*get_all_vectorized_parameters(model.get_all_gradient()), input_dim, hidden_dims, output_dim)
#         
#         
#         if i >= m + 1:
#         
#             init_model(model_dual, para)
#              
#             compute_derivative_one_more_step(model_dual, error, X[delta_ids], Y[delta_ids], beta)
#              
#             gradient_dual = get_all_vectorized_parameters(model_dual.get_all_gradient())
# #             
# #             
# #             
# #             
# #             print('para_diff::', torch.norm(v_vec))
# #             
# #             
# #             
# #             init_model(model, para_list_all_epochs[i])
# #              
# #             curr_vec_gradient,_ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, gradient_list_all_epochs[i], beta)
# #              
# # #             hessian_matrix1 = compute_hessian_matrix(model, curr_vec_gradient, input_dim, hidden_dims, output_dim)
# #             
# #             
# #             
# #             
# #             hessian_para_prod = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1))
#             
# #             delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
# #             
# # #             para_diff = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
# #             
# #             gradient_full = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
# #             
# #             delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1), error, model, beta)
# #             gradient_full2 = torch.mm(v_vec.view(1,-1), hessian_matrix1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
# # #             gradient_full2 = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) - delta_const*v_vec.view(1,-1)
# #             
#             gradients = (gradient_full*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
# #             
# #             gradients2 = (gradient_full2*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
# #             
# #             print('gradient_diff::', torch.norm(gradients - expect_gradients))
# #             
# #             print('gradient2_diff::', torch.norm(gradients2 - expect_gradients))
# #             
# #             compute_model_para_diff(exp_para_list_all_epochs[i], para)
# 
# 
#             last_para = para
# 
# 
# #             gradients = (gradient_full*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
# 
#             
#             para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradients, input_dim, hidden_dims, output_dim)
#         
#             compute_model_para_diff(exp_para_list_all_epochs[i], para)
#         
#         
#             v_vec = get_all_vectorized_parameters(para) - get_all_vectorized_parameters(last_para)
#         
#             
#             gradient_full = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1)) + gradient_full
#             
#             
#             init_model(model, para)
#         
#             compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             expected_gradient_full = model.get_all_gradient()
#             
#             print('gradient_diff::')
#             
#             compute_model_para_diff(get_devectorized_parameters(gradient_full, input_dim, hidden_dims, output_dim), expected_gradient_full)
#         
#         else:
#             
# #             last_para = para.clone()
#             
#             if i == m:
#                 
# #                 gradients = expect_gradients
#                 last_para = para
#             
#             para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*expect_gradients, input_dim, hidden_dims, output_dim)
#         
#         
#             if i == m:
# #                 v_vec = get_all_vectorized_parameters(para) - get_all_vectorized_parameters(last_para)
#                 
#                 init_model(model, para)
#         
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#                 
#                 gradient_full = get_all_vectorized_parameters(model.get_all_gradient())
#                 
# #                 gradient_full = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1)) + expect_gradients
#         
#         
#         
# #             init_model(model, get_devectorized_parameters(updated_para, input_dim, hidden_dims, output_dim))
#             
#             
#             
#             
#             
#             
#             
#     return para


def model_update_provenance_cp0(alpha, X, Y, hessian_matrix, origin_gradient_list, vectorized_orign_params, epoch, model, dim, w_list, b_list, input_dim, hidden_dims, output_dim, delta_ids, expected_gradient_list_all_epochs, expected_para_list_all_epochs_all_epochs, selelcted_rows):

    vectorized_gradient = get_all_vectorized_parameters(origin_gradient_list) 
    
    t1  = time.time()
    
    para_list = list(model.parameters())
    
    depth = len(hidden_dims) + 1
    
    delta_id_num = delta_ids.shape[0]
    
    old_vec_gradient_list = None
    
    old_vec_para_list = None
    
    error = nn.CrossEntropyLoss()
    
    model_dual = DNNModel(input_dim, hidden_dims, output_dim)
    
    for i in range(epoch):
    
#         output_list,input_to_non_linear_layer_list = model.get_output_each_layer(X[delta_ids])
#         
#         input_to_non_linear_layer_list = input_to_non_linear_layer_list[1:]
        
        para_list = list(model.parameters())

        
        
        init_model(model_dual, para_list)

        
        compute_derivative_one_more_step(model_dual, error, X[delta_ids], Y[delta_ids])
        
        
        gradient_dual = model_dual.get_all_gradient()
        
        
        
        curr_vectorized_params = get_all_vectorized_parameters(para_list)        
        
        delta_vectorized_gradient_parameters = torch.mm(hessian_matrix, (curr_vectorized_params - vectorized_orign_params).view(-1, 1)).view(1,-1)
        
        gradient_list = get_devectorized_parameters(delta_vectorized_gradient_parameters + vectorized_gradient, input_dim, hidden_dims, output_dim)
        
        
#         old_vec_para_list = get_all_vectorized_parameters(para_list)
        '''delta: n*output_dim'''
        
        
        '''A: output_dim, hidden_dim[depth-2]^2'''    
        
#         print(depth)
        
        
#         pred = output_list[len(output_list) - 1]
       
        '''A: output_dim, hidden_dims[-1]*hiddem_dims[-1]'''
        
#         input_to_non_linear_layer = input_to_non_linear_layer_list[depth -1]



#         delta = Variable(softmax_func(pred) - get_onehot_y(Y[delta_ids], [delta_id_num, output_dim], output_dim))
#         
#         non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[depth-1][delta_ids] + b_list[depth-1][delta_ids]))
#         
#         delta_A = torch.mm(torch.t(non_linear_output), output_list[depth-1])
        
#         delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(non_linear_output))))
        
        gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
        
        gradient_dual_curr_layer = torch.cat((gradient_dual[2*depth - 2].data, gradient_dual[2*depth - 1].data.view(-1,1)), 1) 
        
        para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 

        curr_A = gradient_curr_layer - gradient_dual_curr_layer*delta_ids.shape[0]

        '''B: output_dim, hidden_dim[depth-2]'''
        
        curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
            
        para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
            
        para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
            
        para_list[2*depth - 1].data = para_curr_layer[:, -1]
        
        '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
    
        for i in range(depth - 2):
            
            '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
            
#             input_to_non_linear_layer = input_to_non_linear_layer_list[depth - i - 2]
#             delta = Variable(delta_para_prod)
# 
#             non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[depth- i - 2][delta_ids] + b_list[depth-i-2][delta_ids]))
#             
#             delta_A = torch.mm(torch.t(non_linear_output), output_list[depth-i-2])
            
            gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
             
            gradient_dual_curr_layer = torch.cat((gradient_dual[2*depth - 2*i - 4].data, gradient_dual[2*depth - 2*i - 3].data.view(-1,1)), 1) 
             
            para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
    
            curr_A = gradient_curr_layer - gradient_dual_curr_layer*delta_ids.shape[0]
            
            '''B: output_dim, hidden_dim[depth-2]'''
#             delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(non_linear_output))))
            
            curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
                
            para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
                
            para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                
            para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
        
#         input_to_non_linear_layer = input_to_non_linear_layer_list[0]
# 
#         delta = Variable(delta_para_prod)
# 
#         non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[0][delta_ids] + b_list[0][delta_ids]))
#         
#         delta_A = torch.mm(torch.t(non_linear_output), output_list[0])

        gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)), 1) 
        
        gradient_dual_curr_layer = torch.cat((gradient_dual[0].data, gradient_dual[1].data.view(-1,1)), 1) 
        
        para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)

        curr_A = gradient_curr_layer - gradient_dual_curr_layer*delta_ids.shape[0]
        
        '''B: output_dim, hidden_dim[depth-2]'''
        
        curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
            
        para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
    
        para_list[0].data = para_curr_layer[:, 0:-1]
            
        para_list[1].data = para_curr_layer[:, -1]
        
        init_model(model, para_list)
        
        '''B: output_dim, hidden_dim[depth-2]'''

    t2  = time.time()
    
    print("time0:", t2 - t1)   
    
    return para_list



def model_update_provenance_cp0_stochastic(batch_size, alpha, X, Y, hessian_matrix, origin_gradient_list, vectorized_orign_params, epoch, model, dim, w_list, b_list, input_dim, hidden_dims, output_dim, delta_ids, expected_gradient_list_all_epochs, expected_para_list_all_epochs_all_epochs, selelcted_rows):

    vectorized_gradient = get_all_vectorized_parameters(origin_gradient_list) 
    
    t1  = time.time()
    
    para_list = list(model.parameters())
    
    depth = len(hidden_dims) + 1
    
    delta_id_num = delta_ids.shape[0]
    
    old_vec_gradient_list = None
    
    old_vec_para_list = None
    
    error = nn.CrossEntropyLoss()
    
    model_dual = DNNModel(input_dim, hidden_dims, output_dim)
    
    
    delta_X = X[delta_ids]
    
    delta_Y = Y[delta_ids]
    
    for i in range(epoch):
        
        
        random_ids = torch.randperm(delta_ids.shape[0])
        
        delta_X = delta_X[random_ids]
        
        delta_Y = delta_Y[random_ids]
        
        
        for i in range(0, delta_ids.shape[0], batch_size):
            
            
            end_id = i + batch_size
            
            if end_id > delta_ids.shape[0]:
                end_id = delta_ids.shape[0]
        
        
        
        
        
        
        
            delta_X_curr_batch = delta_X[i: end_id]
            
            delta_Y_curr_batch = delta_Y[i: end_id]
        
        
        
        
        
        
    
#         output_list,input_to_non_linear_layer_list = model.get_output_each_layer(X[delta_ids])
#         
#         input_to_non_linear_layer_list = input_to_non_linear_layer_list[1:]
        
            para_list = list(model.parameters())
    
            
            
            init_model(model_dual, para_list)
    
            
            compute_derivative_one_more_step(model_dual, error, delta_X_curr_batch, delta_Y_curr_batch)
            
            
            gradient_dual = model_dual.get_all_gradient()
            
            
            
            curr_vectorized_params = get_all_vectorized_parameters(para_list)        
            
            delta_vectorized_gradient_parameters = torch.mm(hessian_matrix, (curr_vectorized_params - vectorized_orign_params).view(-1, 1)).view(1,-1)
            
            gradient_list = get_devectorized_parameters(delta_vectorized_gradient_parameters + vectorized_gradient, input_dim, hidden_dims, output_dim)
            
            '''delta: n*output_dim'''
            
            
            '''A: output_dim, hidden_dim[depth-2]^2'''    
           
            '''A: output_dim, hidden_dims[-1]*hiddem_dims[-1]'''
            
           
            gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
            
            gradient_dual_curr_layer = torch.cat((gradient_dual[2*depth - 2].data, gradient_dual[2*depth - 1].data.view(-1,1)), 1) 
            
            para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 
    
            curr_A = gradient_curr_layer - gradient_dual_curr_layer*delta_ids.shape[0]
    
            '''B: output_dim, hidden_dim[depth-2]'''
            
            curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
                
            para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
                
            para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
                
            para_list[2*depth - 1].data = para_curr_layer[:, -1]
            
            '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
        
            for i in range(depth - 2):
                
                '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
                gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
                 
                gradient_dual_curr_layer = torch.cat((gradient_dual[2*depth - 2*i - 4].data, gradient_dual[2*depth - 2*i - 3].data.view(-1,1)), 1) 
                 
                para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
        
                curr_A = gradient_curr_layer - gradient_dual_curr_layer*delta_ids.shape[0]
                
                '''B: output_dim, hidden_dim[depth-2]'''
                
                curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
                    
                para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
                    
                para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                    
                para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
            
            gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)), 1) 
            
            gradient_dual_curr_layer = torch.cat((gradient_dual[0].data, gradient_dual[1].data.view(-1,1)), 1) 
            
            para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)
    
            curr_A = gradient_curr_layer - gradient_dual_curr_layer*delta_ids.shape[0]
            
            '''B: output_dim, hidden_dim[depth-2]'''
            
            curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
                
            para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
        
            para_list[0].data = para_curr_layer[:, 0:-1]
                
            para_list[1].data = para_curr_layer[:, -1]
            
            init_model(model, para_list)
            
            '''B: output_dim, hidden_dim[depth-2]'''

    t2  = time.time()
    
    print("time0:", t2 - t1)   
    
    return para_list


def model_update_provenance_cp(alpha, X, Y, hessian_matrix, origin_gradient_list, vectorized_orign_params, epoch, model, dim, w_list, b_list, input_dim, hidden_dims, output_dim, delta_ids, expected_gradient_list_all_epochs, expected_para_list_all_epochs_all_epochs, selelcted_rows):
    
    
#     loss = np.infty
#     
#     count = 0
#     
#     
#     output_list = model.get_output_each_layer(X)
#     
#     para_list = model.parameters()
# 
#     while loss > loss_threshold and count < num_epochs:
#         
#         delta_list = []
#         
#         outer_gradient = softmax_func(output_list[len(output_list) - 1]) - get_onehot_y(Y, dim, num_class)       
#         
#         para_list_len = len(para_list)
#         
#         
#         for i in range(para_list_len):
#             
#             
#             '''w_res: n * hidden[len-i], b_res: n*1, para_list[2*para_list_len - 2*i]: hidden[len-i]*hidden[len-i-1], output_list[len-i]: n*hidden[len-i-1]'''
#             
#             
#             w_res[para_list_len - i]*torch.sum(para_list[2*para_list_len - 2*i]*output_list[para_list_len - i], 2) + b_res[para_list_len - i]
#             
#             
#             
#             softmax_func()

    vectorized_gradient = get_all_vectorized_parameters(origin_gradient_list) 
    
    t1  = time.time()
    
    para_list = list(model.parameters())
    
    depth = len(hidden_dims) + 1
    
#     curr_gradient_list = get_all_vectorized_parameters(para_list)
    
    delta_id_num = delta_ids.shape[0]
    
    old_vec_gradient_list = None
    
    old_vec_para_list = None
    
    for i in range(epoch):
        
    
    
#         curr_exp_gradient_list = expected_gradient_list_all_epochs[i]
#         
#         
#         curr_exp_para_list = expected_para_list_all_epochs_all_epochs[i]
#         
#         
#         
#         expected_full_gradient_list, _ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, para_list)
#         
#         hessian_matrix2 = compute_hessian_matrix(model, expected_full_gradient_list, input_dim, hidden_dims, output_dim)
#         
#         hessian_matrix = (hessian_matrix2)
#         
#         vectorized_gradient = get_all_vectorized_parameters(expected_full_gradient_list)
        
#         vectorized_expected_full_gradient = get_all_vectorized_parameters(expected_full_gradient_list)/X.shape[0]
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
#     pred = output_list[len(output_list) - 1]
    
        output_list,input_to_non_linear_layer_list = model.get_output_each_layer(X[delta_ids])
        
        input_to_non_linear_layer_list = input_to_non_linear_layer_list[1:]
        
        para_list = list(model.parameters())

        
        curr_vectorized_params = get_all_vectorized_parameters(para_list)        
        
        delta_vectorized_gradient_parameters = torch.mm(hessian_matrix, (curr_vectorized_params - vectorized_orign_params).view(-1, 1)).view(1,-1)
        
        gradient_list = get_devectorized_parameters(delta_vectorized_gradient_parameters + vectorized_gradient, input_dim, hidden_dims, output_dim)
        
        
        old_vec_para_list = get_all_vectorized_parameters(para_list)
#         vectorized_gradient_list = get_all_vectorized_parameters(gradient_list)/X.shape[0]
        
    
    #         loss = error(pred, Y)
        
    #     delta = softmax_func(pred) - get_onehot_y(Y, dim, num_class)
        
        
        '''delta: n*output_dim'''
        
        
        '''A: output_dim, hidden_dim[depth-2]^2'''    
    #     print(w_res[depth - 1])
        
        print(depth)
        
        
        pred = output_list[len(output_list) - 1]
#         delta_A_list = [None]*depth
#         
#     #     delta_A_list0 = [None]*depth
#         
#         
#         delta_B_list = [None]*depth
        
    #     delta_B_list0 = [None]*depth
        
    
#         w_delta_prod = w_delta_prod_list[depth - 1][delta_ids]
#         
#         b_delta_prod = b_delta_prod_list[depth - 1][delta_ids]
        
        '''A: output_dim, hidden_dims[-1]*hiddem_dims[-1]'''
        
        input_to_non_linear_layer = input_to_non_linear_layer_list[depth -1]



        delta = Variable(softmax_func(pred) - get_onehot_y(Y[delta_ids], [delta_id_num, output_dim], output_dim))
        
#         derivative_non_linear_layer = Variable(w_list[delta_ids]*input_to_non_linear_layer + b_list[delta_ids])
        
        
        non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[depth-1][delta_ids] + b_list[depth-1][delta_ids]))
        
        
        
        delta_A = torch.mm(torch.t(non_linear_output), output_list[depth-1])
        
        
#         non_linear_output = Variable(delta*derivative_non_linear_layer[:, torch.sum(hidden_dim_tensor):torch.sum(hidden_dim_tensor)+output_dim])
        
        
        
        delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(non_linear_output))))
        
#         input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
            
#         delta_A = Variable(torch.mm(torch.t((w_delta_prod*input_to_non_linear_layer + b_delta_prod)), output_list[depth-1]))
        
        gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
        
#         gradient_curr_layer_curr_epoch = torch.cat((curr_exp_gradient_list[2*depth - 2].data, curr_exp_gradient_list[2*depth - 1].data.view(-1,1)), 1)
#         
#         
#         exp_para_curr_layer_curr_epoch = torch.cat((curr_exp_para_list[2*depth - 2].data, curr_exp_para_list[2*depth - 1].data.view(-1,1)), 1)
        
        para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 

        curr_A = gradient_curr_layer - delta_A
#         delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
        
    #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
        
#         delta_A_list[depth - 1] = delta_A
        
    #     delta_A_list0.append(delta_A0)
        
        '''B: output_dim, hidden_dim[depth-2]'''
#         delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-1])
        
#         delta_B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-1]))
        
#         curr_B = Variable(0 - delta_B)
        
        curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
            
#         curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
        
        para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
            
    #         para_curr_layer = para_curr_layer - alpha/
    
        para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
            
        para_list[2*depth - 1].data = para_curr_layer[:, -1]
        
    #     delta_B0 = torch.t(b_delta_prod)
        
    #     delta_B_list0.append(delta_B0)
        
#         delta_B_list[depth - 1] = delta_B
        
        
    #     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1])))
    #     
    #     
    #     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
        
        
    #     delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
    #     
    #     deriv = torch.mm(torch.t(delta), output_list[depth - 1])
    
        '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
    
    #     delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(entire_delta_expression)))
        
    #     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
    #     
    #     para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
    #     
    #     para_list[2*depth - 1].data = para_curr_layer[:, -1]
            
        for i in range(depth - 2):
            
            '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
            
    #         para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
    #         delta = delta_para_prod#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
#             w_delta_prod = w_delta_prod_list[depth - i - 2][delta_ids]
#         
#             b_delta_prod = b_delta_prod_list[depth - i - 2][delta_ids]
            
            input_to_non_linear_layer = input_to_non_linear_layer_list[depth - i - 2]

        
            delta = Variable(delta_para_prod)
        
#         input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
            non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[depth- i - 2][delta_ids] + b_list[depth-i-2][delta_ids]))
            
            delta_A = torch.mm(torch.t(non_linear_output), output_list[depth-i-2])
            
#             delta_A = Variable(torch.mm(torch.t((w_delta_prod*input_to_non_linear_layer + b_delta_prod)), output_list[depth- i - 2]))
            
            gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
            
            
#             gradient_curr_layer_curr_epoch = torch.cat((curr_exp_gradient_list[2*depth - 2*i - 4].data, curr_exp_gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
            
             
            para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
    
            curr_A = gradient_curr_layer - delta_A
    #         delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
            
        #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
            
    #         delta_A_list[depth - 1] = delta_A
            
        #     delta_A_list0.append(delta_A0)
            
            '''B: output_dim, hidden_dim[depth-2]'''
    #         delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-1])
            
#             delta_B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth- i - 2]))
#             
#             curr_B = Variable(0 - delta_B)
            
            
            delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(non_linear_output))))
            
            
            curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
                
#             curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
            
            para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
                
        #         para_curr_layer = para_curr_layer - alpha/
        
            para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                
            para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
#             delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - i - 2].view(delta_id_num, hidden_dims[depth - i - 3] + 1, 1), output_list[depth - i - 2].view(delta_id_num, 1, hidden_dims[depth - i - 3] + 1)).view(delta_id_num, (hidden_dims[depth-i-3]+1)*(hidden_dims[depth-i-3]+1)))
#             
#     #         delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - i - 2][delta_ids])
#             
#             delta_A_list[depth - i - 2] = delta_A
#             
#     #         delta_A_list0.append(delta_A0)
#             
#             '''B: output_dim, hidden_dim[depth-2]'''
#             delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-i-2])
#             
#     #         delta_B0 = torch.t(b_delta_prod)
#             
#     #         delta_B_list0.append(delta_B0)
#             
#             
#             delta_B_list[depth - i - 2] = delta_B
            
    #         w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - i - 2])))
    #         
    #         
    #         entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    #         
    #         
    #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(entire_delta_expression)))
    #         
    #         A_list[depth - 2 - i] = A
    #         
    #         B_list[depth - 2 - i] = B
            
    #         deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])
    #         
    #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))                    
    #         
    #         '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
    #         para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
    #     
    #         para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
    #         
    #         para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
                
    #     para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1)
    
    #     delta = delta_para_prod#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
        
#         w_delta_prod = w_delta_prod_list[0][delta_ids]
#     
#         b_delta_prod = b_delta_prod_list[0][delta_ids]
        
        
        input_to_non_linear_layer = input_to_non_linear_layer_list[0]


        delta = Variable(delta_para_prod)
        
#         input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
        non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[0][delta_ids] + b_list[0][delta_ids]))
        
        delta_A = torch.mm(torch.t(non_linear_output), output_list[0])

        
#         input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
            
#         delta_A = Variable(torch.mm(torch.t((w_delta_prod*input_to_non_linear_layer + b_delta_prod)), output_list[0]))
        
        gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)), 1) 
        
#         gradient_curr_layer_curr_epoch = torch.cat((curr_exp_gradient_list[0].data, curr_exp_gradient_list[1].data.view(-1,1)), 1)
        
        para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)

        curr_A = gradient_curr_layer - delta_A
#         delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
        
    #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
        
#         delta_A_list[depth - 1] = delta_A
        
    #     delta_A_list0.append(delta_A0)
        
        '''B: output_dim, hidden_dim[depth-2]'''
#         delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-1])
        
#         delta_B = Variable(torch.mm(torch.t(b_delta_prod), output_list[0]))
#         
#         curr_B = Variable(0 - delta_B)
        
        curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
            
#         curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
        
        para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
            
    #         para_curr_layer = para_curr_layer - alpha/
    
        para_list[0].data = para_curr_layer[:, 0:-1]
            
        para_list[1].data = para_curr_layer[:, -1]
        
        init_model(model, para_list)
        
#         if old_vec_gradient_list is not None:
#             hessian_matrix = update_hessian(hessian_matrix, old_vec_para_list, get_all_vectorized_parameters(para_list), old_vec_gradient_list, get_all_vectorized_parameters(gradient_list))
#         
#         
#             vectorized_orign_params = curr_vectorized_params
#             
#             vectorized_gradient = get_all_vectorized_parameters(gradient_list)
#         
#         old_vec_gradient_list = get_all_vectorized_parameters(gradient_list)
        
        
        
        
        
#         delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[0].view(delta_id_num, input_dim + 1, 1), output_list[0].view(delta_id_num, 1, input_dim + 1)).view(delta_id_num, (input_dim+1)*(input_dim + 1)))
#     
#         delta_A = delta_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)
    
    #     print(w_delta_prod.shape)
    
    #     delta_A = torch.mm(torch.t(output_list[0][delta_ids]), torch.bmm(w_delta_prod.view(delta_id_num, hidden_dims[0], 1), output_list[0][delta_ids].view(delta_id_num, 1, input_dim + 1)).view(delta_id_num, hidden_dims[0]*(input_dim + 1)))
    #     
    #     delta_A = torch.transpose(delta_A.view(input_dim + 1, hidden_dims[0], input_dim + 1), 0, 1)
        
    #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[0][delta_ids])
        
    #     delta_A_list0.append(delta_A0)
        
        '''B: output_dim, hidden_dim[depth-2]'''
#         delta_B = torch.mm(torch.t(b_delta_prod), output_list[0])
#         
#     #     delta_B0 = torch.t(b_delta_prod)
#         
#     #     delta_B_list0.append(delta_B0)
#         
#         
#         delta_A_list[0] = delta_A
#         
#         delta_B_list[0] = delta_B
#     
#     
#     #     weights = para_list[2*depth - 2].data
#     #         
#     #     offsets = para_list[2*depth - 1].data
#         
#         
#         para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 
#         
#         
#         '''A: (output_dim*hidden_dims[depth-1])*hiddem_dims[depth-1]'''
#     
#         '''B: output_dim*hidden_dims[depth-2]'''
#         
#         '''weights: output_dim*hidden_dims[depth-1]'''
#         
#         curr_A = A_list[depth - 1] - delta_A_list[depth - 1] 
#         
#         curr_B = B_list[depth - 1] - delta_B_list[depth - 1]
#         
# #         for j in range(epoch):
#             
#         '''output_dim, 1, hidden_dims[depth-2]'''
#         
# #         curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
#         
#         curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
#         
#         curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
#         
#         
#         gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
#         
#         delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#         
#         para_curr_layer = para_curr_layer - alpha*curr_gradient
#             
#     #         para_curr_layer = para_curr_layer - alpha/
#     
#         para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
#             
#         para_list[2*depth - 1].data = para_curr_layer[:, -1]
#         
#         for i in range(depth-2):
#             
#             para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
#                     
#             '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
#     
#             '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#             
#             '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#             
#             curr_A = A_list[depth - i - 2] - delta_A_list[depth - i - 2] 
#             
#             curr_B = B_list[depth - i - 2] - delta_B_list[depth - i - 2]
#             
#             for j in range(epoch):
#             
#                 gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)),1)
#                 
#                 curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[depth-i-2], 1, hidden_dims[depth-i-3] + 1), curr_A.view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1, hidden_dims[depth-3-i] + 1)).view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1)
#                 
#                 curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
#                 
#                 delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#                 
#                 para_curr_layer = para_curr_layer - alpha*curr_gradient
#             
#             
#             para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
#             
#             para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
#         
#         
#         para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)
#                     
#         '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
#     
#         '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#         
#         '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#         
#         curr_A = A_list[0] - delta_A_list[0] 
#         
#         curr_B = B_list[0] - delta_B_list[0]
#         
#         for j in range(epoch):
#             
#             gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)),1)
#             
#             curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), curr_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
#             
#             curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
#             
#             delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#                     
#             para_curr_layer = para_curr_layer - alpha*curr_gradient
#         
#         
#         para_list[0].data = para_curr_layer[:, 0:-1]
#         
#         para_list[1].data = para_curr_layer[:, -1]

    
    
#     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[0])))
#     
#     
#     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    
#     deriv = torch.mm(torch.t(delta), output_list[0])
#     
#     '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
# 
#     para_list[0].data = para_curr_layer[:, 0:-1]
#     
#     para_list[1].data = para_curr_layer[:, -1]

    t2  = time.time()
    
    print("time0:", t2 - t1)   
    
    return para_list

def model_update_provenance_cp2(alpha, X, Y, epoch, model, dim, w_list_all_epochs, b_list_all_epochs, input_dim, hidden_dims, output_dim, selected_rows, exp_gradient_list_all_epochs, exp_para_list_all_epochs):
    
    
#     loss = np.infty
#     
#     count = 0
#     
#     
#     output_list = model.get_output_each_layer(X)
#     
#     para_list = model.parameters()
# 
#     while loss > loss_threshold and count < num_epochs:
#         
#         delta_list = []
#         
#         outer_gradient = softmax_func(output_list[len(output_list) - 1]) - get_onehot_y(Y, dim, num_class)       
#         
#         para_list_len = len(para_list)
#         
#         
#         for i in range(para_list_len):
#             
#             
#             '''w_res: n * hidden[len-i], b_res: n*1, para_list[2*para_list_len - 2*i]: hidden[len-i]*hidden[len-i-1], output_list[len-i]: n*hidden[len-i-1]'''
#             
#             
#             w_res[para_list_len - i]*torch.sum(para_list[2*para_list_len - 2*i]*output_list[para_list_len - i], 2) + b_res[para_list_len - i]
#             
#             
#             
#             softmax_func()

#     vectorized_gradient = get_all_vectorized_parameters(origin_gradient_list) 
    
    t1  = time.time()
    
    para_list = list(model.parameters())
    
    depth = len(hidden_dims) + 1
    
#     curr_gradient_list = get_all_vectorized_parameters(para_list)
    
#     delta_id_num = delta_ids.shape[0]
    
    selected_row_num = selected_rows.shape[0]
    
    for i in range(epoch):
        
    
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
#     pred = output_list[len(output_list) - 1]

        if i >= cut_off_epoch:
            w_list = w_list_all_epochs[cut_off_epoch-1]
            
            b_list = b_list_all_epochs[cut_off_epoch-1]
        else:
            w_list = w_list_all_epochs[i]
            
            b_list = b_list_all_epochs[i]
    
        output_list,input_to_non_linear_layer_list = model.get_output_each_layer(X[selected_rows])
        
        input_to_non_linear_layer_list = input_to_non_linear_layer_list[1:]
        
        para_list = list(model.parameters())

        
#         curr_vectorized_params = get_all_vectorized_parameters(para_list)
#         
#         
#         
#         delta_vectorized_gradient_parameters = torch.mm(curr_vectorized_params - vectorized_orign_params, hessian_matrix)
#         
#         gradient_list = get_devectorized_parameters(delta_vectorized_gradient_parameters + vectorized_gradient, input_dim, hidden_dims, output_dim)
#         gradient_list = exp_gradient_list_all_epochs[i]
        
    
    #         loss = error(pred, Y)
        
    #     delta = softmax_func(pred) - get_onehot_y(Y, dim, num_class)
        
        
        '''delta: n*output_dim'''
        
        
        '''A: output_dim, hidden_dim[depth-2]^2'''    
    #     print(w_res[depth - 1])
        
#         print(depth)
        
        
        pred = output_list[len(output_list) - 1]
#         delta_A_list = [None]*depth
#         
#     #     delta_A_list0 = [None]*depth
#         
#         
#         delta_B_list = [None]*depth
        
    #     delta_B_list0 = [None]*depth
        
    
#         w_delta_prod = w_delta_prod_list[depth - 1][delta_ids]
#         
#         b_delta_prod = b_delta_prod_list[depth - 1][delta_ids]
        
        '''A: output_dim, hidden_dims[-1]*hiddem_dims[-1]'''
        
        input_to_non_linear_layer = input_to_non_linear_layer_list[depth -1]



        delta = Variable(softmax_func(pred) - get_onehot_y(Y[selected_rows], [selected_row_num, output_dim], output_dim))
        
#         derivative_non_linear_layer = Variable(w_list[delta_ids]*input_to_non_linear_layer + b_list[delta_ids])
        
        
        non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[depth-1][selected_rows] + b_list[depth-1][selected_rows]))
        
        
        
        delta_A = torch.mm(torch.t(non_linear_output), output_list[depth-1])
        
        
#         non_linear_output = Variable(delta*derivative_non_linear_layer[:, torch.sum(hidden_dim_tensor):torch.sum(hidden_dim_tensor)+output_dim])
        
        
        
        delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(non_linear_output))))
        
#         input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
            
#         delta_A = Variable(torch.mm(torch.t((w_delta_prod*input_to_non_linear_layer + b_delta_prod)), output_list[depth-1]))
        
#         gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
        para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 

        curr_A = delta_A
#         delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
        
    #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
        
#         delta_A_list[depth - 1] = delta_A
        
    #     delta_A_list0.append(delta_A0)
        
        '''B: output_dim, hidden_dim[depth-2]'''
#         delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-1])
        
#         delta_B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-1]))
        
#         curr_B = Variable(0 - delta_B)
        
        curr_gradient = Variable((1.0/(selected_row_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
            
#         curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
        
        para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
            
    #         para_curr_layer = para_curr_layer - alpha/
    
        para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
            
        para_list[2*depth - 1].data = para_curr_layer[:, -1]
        
    #     delta_B0 = torch.t(b_delta_prod)
        
    #     delta_B_list0.append(delta_B0)
        
#         delta_B_list[depth - 1] = delta_B
        
        
    #     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1])))
    #     
    #     
    #     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
        
        
    #     delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
    #     
    #     deriv = torch.mm(torch.t(delta), output_list[depth - 1])
    
        '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
    
    #     delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(entire_delta_expression)))
        
    #     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
    #     
    #     para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
    #     
    #     para_list[2*depth - 1].data = para_curr_layer[:, -1]
            
        for i in range(depth - 2):
            
            '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
            
    #         para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
    #         delta = delta_para_prod#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
#             w_delta_prod = w_delta_prod_list[depth - i - 2][delta_ids]
#         
#             b_delta_prod = b_delta_prod_list[depth - i - 2][delta_ids]
            
            input_to_non_linear_layer = input_to_non_linear_layer_list[depth - i - 2]

        
            delta = Variable(delta_para_prod)
        
#         input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
            non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[depth- i - 2][selected_rows] + b_list[depth-i-2][selected_rows]))
            
            delta_A = torch.mm(torch.t(non_linear_output), output_list[depth-i-2])
            
#             delta_A = Variable(torch.mm(torch.t((w_delta_prod*input_to_non_linear_layer + b_delta_prod)), output_list[depth- i - 2]))
            
#             gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1) 
            para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
    
            curr_A = delta_A
    #         delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
            
        #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
            
    #         delta_A_list[depth - 1] = delta_A
            
        #     delta_A_list0.append(delta_A0)
            
            '''B: output_dim, hidden_dim[depth-2]'''
    #         delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-1])
            
#             delta_B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth- i - 2]))
#             
#             curr_B = Variable(0 - delta_B)
            
            
            delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(non_linear_output))))
            
            
            curr_gradient = Variable((1.0/(selected_row_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
                
#             curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
            
            para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
                
        #         para_curr_layer = para_curr_layer - alpha/
        
            para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                
            para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
#             delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - i - 2].view(delta_id_num, hidden_dims[depth - i - 3] + 1, 1), output_list[depth - i - 2].view(delta_id_num, 1, hidden_dims[depth - i - 3] + 1)).view(delta_id_num, (hidden_dims[depth-i-3]+1)*(hidden_dims[depth-i-3]+1)))
#             
#     #         delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - i - 2][delta_ids])
#             
#             delta_A_list[depth - i - 2] = delta_A
#             
#     #         delta_A_list0.append(delta_A0)
#             
#             '''B: output_dim, hidden_dim[depth-2]'''
#             delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-i-2])
#             
#     #         delta_B0 = torch.t(b_delta_prod)
#             
#     #         delta_B_list0.append(delta_B0)
#             
#             
#             delta_B_list[depth - i - 2] = delta_B
            
    #         w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - i - 2])))
    #         
    #         
    #         entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    #         
    #         
    #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(entire_delta_expression)))
    #         
    #         A_list[depth - 2 - i] = A
    #         
    #         B_list[depth - 2 - i] = B
            
    #         deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])
    #         
    #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))                    
    #         
    #         '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
    #         para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
    #     
    #         para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
    #         
    #         para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
                
    #     para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1)
    
    #     delta = delta_para_prod#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
        
#         w_delta_prod = w_delta_prod_list[0][delta_ids]
#     
#         b_delta_prod = b_delta_prod_list[0][delta_ids]
        
        
        input_to_non_linear_layer = input_to_non_linear_layer_list[0]


        delta = Variable(delta_para_prod)
        
#         input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
        non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[0][selected_rows] + b_list[0][selected_rows]))
        
        delta_A = torch.mm(torch.t(non_linear_output), output_list[0])

        
#         input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
            
#         delta_A = Variable(torch.mm(torch.t((w_delta_prod*input_to_non_linear_layer + b_delta_prod)), output_list[0]))
        
#         gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)), 1) 
        para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)

        curr_A = delta_A
#         delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
        
    #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
        
#         delta_A_list[depth - 1] = delta_A
        
    #     delta_A_list0.append(delta_A0)
        
        '''B: output_dim, hidden_dim[depth-2]'''
#         delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-1])
        
#         delta_B = Variable(torch.mm(torch.t(b_delta_prod), output_list[0]))
#         
#         curr_B = Variable(0 - delta_B)
        
        curr_gradient = Variable((1.0/(selected_row_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
            
#         curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
        
        para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
            
    #         para_curr_layer = para_curr_layer - alpha/
    
        para_list[0].data = para_curr_layer[:, 0:-1]
            
        para_list[1].data = para_curr_layer[:, -1]
        
        init_model(model, para_list)
        
        
        
        
        
        
        
#         delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[0].view(delta_id_num, input_dim + 1, 1), output_list[0].view(delta_id_num, 1, input_dim + 1)).view(delta_id_num, (input_dim+1)*(input_dim + 1)))
#     
#         delta_A = delta_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)
    
    #     print(w_delta_prod.shape)
    
    #     delta_A = torch.mm(torch.t(output_list[0][delta_ids]), torch.bmm(w_delta_prod.view(delta_id_num, hidden_dims[0], 1), output_list[0][delta_ids].view(delta_id_num, 1, input_dim + 1)).view(delta_id_num, hidden_dims[0]*(input_dim + 1)))
    #     
    #     delta_A = torch.transpose(delta_A.view(input_dim + 1, hidden_dims[0], input_dim + 1), 0, 1)
        
    #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[0][delta_ids])
        
    #     delta_A_list0.append(delta_A0)
        
        '''B: output_dim, hidden_dim[depth-2]'''
#         delta_B = torch.mm(torch.t(b_delta_prod), output_list[0])
#         
#     #     delta_B0 = torch.t(b_delta_prod)
#         
#     #     delta_B_list0.append(delta_B0)
#         
#         
#         delta_A_list[0] = delta_A
#         
#         delta_B_list[0] = delta_B
#     
#     
#     #     weights = para_list[2*depth - 2].data
#     #         
#     #     offsets = para_list[2*depth - 1].data
#         
#         
#         para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 
#         
#         
#         '''A: (output_dim*hidden_dims[depth-1])*hiddem_dims[depth-1]'''
#     
#         '''B: output_dim*hidden_dims[depth-2]'''
#         
#         '''weights: output_dim*hidden_dims[depth-1]'''
#         
#         curr_A = A_list[depth - 1] - delta_A_list[depth - 1] 
#         
#         curr_B = B_list[depth - 1] - delta_B_list[depth - 1]
#         
# #         for j in range(epoch):
#             
#         '''output_dim, 1, hidden_dims[depth-2]'''
#         
# #         curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
#         
#         curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
#         
#         curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
#         
#         
#         gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
#         
#         delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#         
#         para_curr_layer = para_curr_layer - alpha*curr_gradient
#             
#     #         para_curr_layer = para_curr_layer - alpha/
#     
#         para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
#             
#         para_list[2*depth - 1].data = para_curr_layer[:, -1]
#         
#         for i in range(depth-2):
#             
#             para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
#                     
#             '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
#     
#             '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#             
#             '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#             
#             curr_A = A_list[depth - i - 2] - delta_A_list[depth - i - 2] 
#             
#             curr_B = B_list[depth - i - 2] - delta_B_list[depth - i - 2]
#             
#             for j in range(epoch):
#             
#                 gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)),1)
#                 
#                 curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[depth-i-2], 1, hidden_dims[depth-i-3] + 1), curr_A.view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1, hidden_dims[depth-3-i] + 1)).view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1)
#                 
#                 curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
#                 
#                 delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#                 
#                 para_curr_layer = para_curr_layer - alpha*curr_gradient
#             
#             
#             para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
#             
#             para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
#         
#         
#         para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)
#                     
#         '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
#     
#         '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#         
#         '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#         
#         curr_A = A_list[0] - delta_A_list[0] 
#         
#         curr_B = B_list[0] - delta_B_list[0]
#         
#         for j in range(epoch):
#             
#             gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)),1)
#             
#             curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), curr_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
#             
#             curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
#             
#             delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#                     
#             para_curr_layer = para_curr_layer - alpha*curr_gradient
#         
#         
#         para_list[0].data = para_curr_layer[:, 0:-1]
#         
#         para_list[1].data = para_curr_layer[:, -1]

    
    
#     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[0])))
#     
#     
#     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    
#     deriv = torch.mm(torch.t(delta), output_list[0])
#     
#     '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
# 
#     para_list[0].data = para_curr_layer[:, 0:-1]
#     
#     para_list[1].data = para_curr_layer[:, -1]

    t2  = time.time()
    
    print("time0:", t2 - t1)   
    
    return para_list



def model_update_provenance(alpha, epoch, model, dim, output_list, A_list, B_list, w_delta_prod_list, b_delta_prod_list, input_dim, hidden_dims, output_dim, delta_ids, gradient_list):
    
    
#     loss = np.infty
#     
#     count = 0
#     
#     
#     output_list = model.get_output_each_layer(X)
#     
#     para_list = model.parameters()
# 
#     while loss > loss_threshold and count < num_epochs:
#         
#         delta_list = []
#         
#         outer_gradient = softmax_func(output_list[len(output_list) - 1]) - get_onehot_y(Y, dim, num_class)       
#         
#         para_list_len = len(para_list)
#         
#         
#         for i in range(para_list_len):
#             
#             
#             '''w_res: n * hidden[len-i], b_res: n*1, para_list[2*para_list_len - 2*i]: hidden[len-i]*hidden[len-i-1], output_list[len-i]: n*hidden[len-i-1]'''
#             
#             
#             w_res[para_list_len - i]*torch.sum(para_list[2*para_list_len - 2*i]*output_list[para_list_len - i], 2) + b_res[para_list_len - i]
#             
#             
#             
#             softmax_func()


    t1  = time.time()
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
#     pred = output_list[len(output_list) - 1]
    
    para_list = list(model.parameters())
    
    
    depth = len(hidden_dims) + 1

#         loss = error(pred, Y)
    
#     delta = softmax_func(pred) - get_onehot_y(Y, dim, num_class)
    
    
    '''delta: n*output_dim'''
    
    
    '''A: output_dim, hidden_dim[depth-2]^2'''    
#     print(w_res[depth - 1])
    
    print(depth)
    
    delta_A_list = [None]*depth
    
#     delta_A_list0 = [None]*depth
    
    
    delta_B_list = [None]*depth
    
#     delta_B_list0 = [None]*depth
    

    delta_id_num = delta_ids.shape[0]
    
    w_delta_prod = w_delta_prod_list[depth - 1][delta_ids]
    
    b_delta_prod = b_delta_prod_list[depth - 1][delta_ids]
    
    '''A: output_dim, hidden_dims[-1]*hiddem_dims[-1]'''
    
    delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1][delta_ids].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1][delta_ids].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
    
#     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
    
    delta_A_list[depth - 1] = delta_A
    
#     delta_A_list0.append(delta_A0)
    
    '''B: output_dim, hidden_dim[depth-2]'''
    delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-1][delta_ids])
    
#     delta_B0 = torch.t(b_delta_prod)
    
#     delta_B_list0.append(delta_B0)
    
    delta_B_list[depth - 1] = delta_B
    
    
#     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1])))
#     
#     
#     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    
    
#     delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
#     
#     deriv = torch.mm(torch.t(delta), output_list[depth - 1])

    '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''

#     delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(entire_delta_expression)))
    
#     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
#     
#     para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
#     
#     para_list[2*depth - 1].data = para_curr_layer[:, -1]
        
    for i in range(depth - 2):
        
        '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
        
#         para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
#         delta = delta_para_prod#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
        w_delta_prod = w_delta_prod_list[depth - i - 2][delta_ids]
    
        b_delta_prod = b_delta_prod_list[depth - i - 2][delta_ids]
        
        delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - i - 2][delta_ids].view(delta_id_num, hidden_dims[depth - i - 3] + 1, 1), output_list[depth - i - 2][delta_ids].view(delta_id_num, 1, hidden_dims[depth - i - 3] + 1)).view(delta_id_num, (hidden_dims[depth-i-3]+1)*(hidden_dims[depth-i-3]+1)))
        
#         delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - i - 2][delta_ids])
        
        delta_A_list[depth - i - 2] = delta_A
        
#         delta_A_list0.append(delta_A0)
        
        '''B: output_dim, hidden_dim[depth-2]'''
        delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-i-2][delta_ids])
        
#         delta_B0 = torch.t(b_delta_prod)
        
#         delta_B_list0.append(delta_B0)
        
        
        delta_B_list[depth - i - 2] = delta_B
        
#         w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - i - 2])))
#         
#         
#         entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
#         
#         
#         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(entire_delta_expression)))
#         
#         A_list[depth - 2 - i] = A
#         
#         B_list[depth - 2 - i] = B
        
#         deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])
#         
#         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))                    
#         
#         '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#         para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
#     
#         para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
#         
#         para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
            
#     para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1)

#     delta = delta_para_prod#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
    
    w_delta_prod = w_delta_prod_list[0][delta_ids]

    b_delta_prod = b_delta_prod_list[0][delta_ids]
    
    delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[0][delta_ids].view(delta_id_num, input_dim + 1, 1), output_list[0][delta_ids].view(delta_id_num, 1, input_dim + 1)).view(delta_id_num, (input_dim+1)*(input_dim + 1)))

    delta_A = delta_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)

#     print(w_delta_prod.shape)

#     delta_A = torch.mm(torch.t(output_list[0][delta_ids]), torch.bmm(w_delta_prod.view(delta_id_num, hidden_dims[0], 1), output_list[0][delta_ids].view(delta_id_num, 1, input_dim + 1)).view(delta_id_num, hidden_dims[0]*(input_dim + 1)))
#     
#     delta_A = torch.transpose(delta_A.view(input_dim + 1, hidden_dims[0], input_dim + 1), 0, 1)
    
#     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[0][delta_ids])
    
#     delta_A_list0.append(delta_A0)
    
    '''B: output_dim, hidden_dim[depth-2]'''
    delta_B = torch.mm(torch.t(b_delta_prod), output_list[0][delta_ids])
    
#     delta_B0 = torch.t(b_delta_prod)
    
#     delta_B_list0.append(delta_B0)
    
    
    delta_A_list[0] = delta_A
    
    delta_B_list[0] = delta_B


#     weights = para_list[2*depth - 2].data
#         
#     offsets = para_list[2*depth - 1].data
    
    
    para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 
    
    
    '''A: (output_dim*hidden_dims[depth-1])*hiddem_dims[depth-1]'''

    '''B: output_dim*hidden_dims[depth-2]'''
    
    '''weights: output_dim*hidden_dims[depth-1]'''
    
    curr_A = A_list[depth - 1] - delta_A_list[depth - 1] 
    
    curr_B = B_list[depth - 1] - delta_B_list[depth - 1]
    
    for j in range(epoch):
        
        '''output_dim, 1, hidden_dims[depth-2]'''
        
        curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
        
        curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
        
        
        gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
        
        delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
        
        para_curr_layer = para_curr_layer - alpha*curr_gradient
        
#         para_curr_layer = para_curr_layer - alpha/

    para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
        
    para_list[2*depth - 1].data = para_curr_layer[:, -1]
    
    for i in range(depth-2):
        
        para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
                
        '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''

        '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
        
        '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
        
        curr_A = A_list[depth - i - 2] - delta_A_list[depth - i - 2] 
        
        curr_B = B_list[depth - i - 2] - delta_B_list[depth - i - 2]
        
        for j in range(epoch):
        
            gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)),1)
            
            curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[depth-i-2], 1, hidden_dims[depth-i-3] + 1), curr_A.view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1, hidden_dims[depth-3-i] + 1)).view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1)
            
            curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
            
            delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
        
        
        para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
        
        para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
    
    
    para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)
                
    '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''

    '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
    
    '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
    
    curr_A = A_list[0] - delta_A_list[0] 
    
    curr_B = B_list[0] - delta_B_list[0]
    
    for j in range(epoch):
        
        gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)),1)
        
        curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), curr_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
        
        curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
        
        delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
                
        para_curr_layer = para_curr_layer - alpha*curr_gradient
    
    
    para_list[0].data = para_curr_layer[:, 0:-1]
    
    para_list[1].data = para_curr_layer[:, -1]

    
    
#     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[0])))
#     
#     
#     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    
#     deriv = torch.mm(torch.t(delta), output_list[0])
#     
#     '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
# 
#     para_list[0].data = para_curr_layer[:, 0:-1]
#     
#     para_list[1].data = para_curr_layer[:, -1]

    t2  = time.time()
    
    print("time0:", t2 - t1)   
    
    return para_list

def model_update_provenance1_2(alpha, init_para, selected_rows, model, X, Y, w_list_all_epochs, b_list_all_epochs, num_class, input_dim, hidden_dims, output_dim, gradient_list_all_epochs):
    
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
#     A_list_all_epochs = []
#     
#     B_list_all_epochs = []
#     
#     w_delta_prod_list_all_epochs = []
#     
#     b_delta_prod_list_all_epochs = []
    
    
    curr_X = X[selected_rows]
    
    
    curr_Y = Y[selected_rows]
    
    
    selected_row_num = selected_rows.shape[0]
    
    init_model(model, init_para)
    
    t1  = time.time()
    
    depth = len(hidden_dims) + 1
    
    overhead = 0.0
    
    with torch.no_grad():
        
        for k in range(len(w_list_all_epochs)):
            w_res = w_list_all_epochs[k]
            
            b_res = b_list_all_epochs[k]
            
#             expected_gradient_list = gradient_list_all_epochs[k]
            
            output_list, input_to_non_linear_layer_list = model.get_output_each_layer(curr_X)
            
            input_to_non_linear_layer_list = input_to_non_linear_layer_list[1:]
            
            pred = output_list[len(output_list) - 1]
            
            para_list = list(model.parameters())
        
        
#             A_list = [None]*depth
#         
#         
#             B_list = [None]*depth
            
            
        #     A0_list = [None]*depth
        #     
        #     B0_list = [None]*depth
        #         loss = error(pred, Y)
            
                   
            
            delta = Variable(softmax_func(pred) - get_onehot_y(curr_Y, [selected_row_num, num_class], num_class))
            
            
            '''delta: n*output_dim'''
            
            para_curr_layer = Variable(torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1)) 
            
            
            '''output_list[depth-1]: n*hidden_dims[depth-2], para_curr_layer: output_dim*hidden_dims[depth-2] -> output_dim*n'''
            
            input_to_non_linear_layer = input_to_non_linear_layer_list[depth -1]#Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth-1]))))
            
            '''n*output_dim'''
            
            t3 = time.time()
            
            non_linear_output = Variable(delta*(input_to_non_linear_layer*w_res[depth-1][selected_rows] + b_res[depth-1][selected_rows]))
            
            t4 = time.time()
            
            curr_overhead = (t4 - t3)
            
            overhead = overhead + curr_overhead
#             w_delta_prod = Variable(w_res[depth - 1][selected_rows]*delta)
#              
#             b_delta_prod = Variable(b_res[depth - 1][selected_rows]*delta)
            
            
#             w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
            
            
#             entire_delta_expression = non_linear_output#w_delta_prod*input_to_non_linear_layer + b_delta_prod
            
            
#             entire_delta_expression2 = w_delta_prod*w_input_prod + b_delta_prod
            
        #     delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
        #     
        #     deriv = torch.mm(torch.t(delta), output_list[depth - 1])
        
            '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
        
            delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(non_linear_output))))
            
#             delta_para_prod2 = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(entire_delta_expression2)))) 
            
            curr_gradient = Variable(torch.mm(torch.t(non_linear_output), output_list[depth-1])/selected_row_num) 
            
#             expected_gradient = Variable(torch.cat((expected_gradient_list[2*depth - 2].data, expected_gradient_list[2*depth - 1].data.view(-1,1)), 1))
            
            
#             gradient_diff = torch.norm(expected_gradient - curr_gradient)
            
#             print("gradient_diff::", gradient_diff)
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
                
                
            para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
            
            para_list[2*depth - 1].data = para_curr_layer[:, -1]
            
            
            
        #     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        #     
        #     para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
        #     
        #     para_list[2*depth - 1].data = para_curr_layer[:, -1         
                
            for i in range(depth - 2):
                
                '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
                
                para_curr_layer = Variable(torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1))
                
                delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
                        
                        
                input_to_non_linear_layer = input_to_non_linear_layer_list[depth-i-2]#Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth- i - 2]))))
                        
                non_linear_output = Variable(delta*(input_to_non_linear_layer*w_res[depth-i-2][selected_rows] + b_res[depth-i-2][selected_rows]))

#                 w_delta_prod = Variable(w_res[depth - i - 2][selected_rows]*delta)
#             
#                 b_delta_prod = Variable(b_res[depth - i - 2][selected_rows]*delta)
                
               
#                 w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - i - 2]))))
                
                
#                 entire_delta_expression = non_linear_output#w_delta_prod*input_to_non_linear_layer + b_delta_prod
                
                
                delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(non_linear_output))))
                                                
                
                curr_gradient = Variable(torch.mm(torch.t(delta*(input_to_non_linear_layer*w_res[depth-i-2][selected_rows] + b_res[depth-i-2][selected_rows])),output_list[depth-i-2])/selected_row_num) 
                        
#                 expected_gradient = Variable(torch.cat((expected_gradient_list[2*depth - 2*i - 4].data, expected_gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1))
            
            
#                 gradient_diff = torch.norm(expected_gradient - curr_gradient)
                
#                 print("gradient_diff::", gradient_diff)
                
                para_curr_layer = para_curr_layer - alpha*curr_gradient
                    
                    
                para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                
                para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
                
#                 A_list[depth - 2 - i] = A
#                 
#                 B_list[depth - 2 - i] = B
                
        #         deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])
        #         
        #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))                    
        #         
        #         '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
        #         para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        #     
        #         para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
        #         
        #         para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
            
            
            
            para_curr_layer = Variable(torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1))
        
            delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
            
            

            input_to_non_linear_layer = input_to_non_linear_layer_list[0]#Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
            
            t3 = time.time()
            
            non_linear_output = Variable(delta*(input_to_non_linear_layer*w_res[0][selected_rows] + b_res[0][selected_rows]))
            t4 = time.time()
            
            curr_overhead = (t4 - t3)
            
            overhead = overhead + curr_overhead
            
            curr_gradient = Variable(torch.mm(torch.t(non_linear_output),output_list[0])/selected_row_num)
                        
                        
            
#             expected_gradient = Variable(torch.cat((expected_gradient_list[0].data, expected_gradient_list[1].data.view(-1,1)), 1))
        
        
#             gradient_diff = torch.norm(expected_gradient - curr_gradient)
            
#             print("gradient_diff::", gradient_diff)
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
                
                
            para_list[0].data = para_curr_layer[:, 0:-1]
            
            para_list[1].data = para_curr_layer[:, -1]

            init_model(model, para_list)
            
#             t4 = time.time()
            
            
            
#             print("curr_overhead::", (t4 - t3))
#             
#             print("overhead::", overhead)
            
#             A_list[0] = A
#             
#             B_list[0] = B
#         
#             A_list_all_epochs.append(A_list)
#             
#             B_list_all_epochs.append(B_list)
    
#     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[0])))
#     
#     
#     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    
#     deriv = torch.mm(torch.t(delta), output_list[0])
#     
#     '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
# 
#     para_list[0].data = para_curr_layer[:, 0:-1]
#     
#     para_list[1].data = para_curr_layer[:, -1]

    t2  = time.time()
    
    print("time0:", t2 - t1)   
    
    print("overhead::", overhead)
    
    
    return list(model.parameters())
    
    
    
    
#     for param in param_list:
#         
#         delta =  


def model_update_provenance1_3(alpha, init_para, selected_rows, model, X, Y, w_list_all_epochs, b_list_all_epochs, num_class, input_dim, hidden_dims, output_dim, gradient_list_all_epochs):
    
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
#     A_list_all_epochs = []
#     
#     B_list_all_epochs = []
#     
#     w_delta_prod_list_all_epochs = []
#     
#     b_delta_prod_list_all_epochs = []
    
    
    curr_X = X[selected_rows]
    
    
    curr_Y = Y[selected_rows]
    
    
    selected_row_num = selected_rows.shape[0]
    
    init_model(model, init_para)
    
    t1  = time.time()
    
    depth = len(hidden_dims) + 1
    
    hidden_dim_tensor = torch.tensor(hidden_dims)
    
    overhead = 0.0
    
    with torch.no_grad():
        
        for k in range(len(w_list_all_epochs)):
            
            
            w_res_cat = torch.cat(w_list_all_epochs[k], 1)
            
            b_res_cat = torch.cat(b_list_all_epochs[k], 1)
            
#             expected_gradient_list = gradient_list_all_epochs[k]
            
            
            
            output_list, input_to_non_linear_layer_list = model.get_output_each_layer(curr_X)
            
            input_to_non_linear_layer_list = input_to_non_linear_layer_list[1:]
            t3 = time.time()

            
#             output_list_cat = torch.cat(output_list, 1)
            
            input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
            
            derivative_non_linear_layer = Variable(w_res_cat[selected_rows]*input_to_non_linear_layer_list_cat + b_res_cat[selected_rows])
            
            
            t4 = time.time()
            
            overhead += (t4 - t3)
            
            pred = output_list[len(output_list) - 1]
            
            para_list = list(model.parameters())
        
        
#             A_list = [None]*depth
#         
#         
#             B_list = [None]*depth
            
            
        #     A0_list = [None]*depth
        #     
        #     B0_list = [None]*depth
        #         loss = error(pred, Y)
            
            delta = Variable(softmax_func(pred) - get_onehot_y(curr_Y, [selected_row_num, num_class], num_class))
            
            
            '''delta: n*output_dim'''
            
            para_curr_layer = Variable(torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1)) 
            
            
            '''output_list[depth-1]: n*hidden_dims[depth-2], para_curr_layer: output_dim*hidden_dims[depth-2] -> output_dim*n'''
            
#             input_to_non_linear_layer = input_to_non_linear_layer_list[depth -1]#Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth-1]))))
            
            '''n*output_dim'''
            
            non_linear_output = Variable(delta*derivative_non_linear_layer[:, torch.sum(hidden_dim_tensor):torch.sum(hidden_dim_tensor)+output_dim])#Variable(delta*(input_to_non_linear_layer*w_res[depth-1][selected_rows] + b_res[depth-1][selected_rows]))
            
            
#             w_delta_prod = Variable(w_res[depth - 1][selected_rows]*delta)
#              
#             b_delta_prod = Variable(b_res[depth - 1][selected_rows]*delta)
            
            
#             w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
            
            
#             entire_delta_expression = non_linear_output#w_delta_prod*input_to_non_linear_layer + b_delta_prod
            
            
#             entire_delta_expression2 = w_delta_prod*w_input_prod + b_delta_prod
            
        #     delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
        #     
        #     deriv = torch.mm(torch.t(delta), output_list[depth - 1])
        
            '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
        
            delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(non_linear_output))))
            
#             delta_para_prod2 = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(entire_delta_expression2)))) 
            
            
            curr_gradient = Variable(torch.mm(torch.t(non_linear_output), output_list[depth-1])/selected_row_num) 
            
            
#             expected_gradient = Variable(torch.cat((expected_gradient_list[2*depth - 2].data, expected_gradient_list[2*depth - 1].data.view(-1,1)), 1))
            
            
#             gradient_diff = torch.norm(expected_gradient - curr_gradient)
            
#             print("gradient_diff::", gradient_diff)
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
                
                
            para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
            
            para_list[2*depth - 1].data = para_curr_layer[:, -1]
            
            
            
        #     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        #     
        #     para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
        #     
        #     para_list[2*depth - 1].data = para_curr_layer[:, -1         
                
            for i in range(depth - 2):
                
                '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
                
                para_curr_layer = Variable(torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1))
                
                delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
                        
                        
#                 input_to_non_linear_layer = input_to_non_linear_layer_list[depth-i-2]#Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth- i - 2]))))
                        
                non_linear_output = Variable(delta*derivative_non_linear_layer[:, torch.sum(hidden_dim_tensor[0:depth - i - 2]):torch.sum(hidden_dim_tensor[0:depth-i-1])])#Variable(delta*(input_to_non_linear_layer*w_res[depth-i-2][selected_rows] + b_res[depth-i-2][selected_rows]))

#                 w_delta_prod = Variable(w_res[depth - i - 2][selected_rows]*delta)
#             
#                 b_delta_prod = Variable(b_res[depth - i - 2][selected_rows]*delta)
                
               
#                 w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - i - 2]))))
                
                
#                 entire_delta_expression = non_linear_output#w_delta_prod*input_to_non_linear_layer + b_delta_prod
                
                
                delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(non_linear_output))))
                                                
                
                curr_gradient = Variable(torch.mm(torch.t(non_linear_output),output_list[depth-i-2])/selected_row_num) 
                        
#                 expected_gradient = Variable(torch.cat((expected_gradient_list[2*depth - 2*i - 4].data, expected_gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1))
            
            
#                 gradient_diff = torch.norm(expected_gradient - curr_gradient)
                
#                 print("gradient_diff::", gradient_diff)
                
                para_curr_layer = para_curr_layer - alpha*curr_gradient
                    
                    
                para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                
                para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
                
#                 A_list[depth - 2 - i] = A
#                 
#                 B_list[depth - 2 - i] = B
                
        #         deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])
        #         
        #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))                    
        #         
        #         '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
        #         para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        #     
        #         para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
        #         
        #         para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
                    
            para_curr_layer = Variable(torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1))
        
            delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
            
            
#             input_to_non_linear_layer = input_to_non_linear_layer_list[0]#Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
            
            non_linear_output = Variable(delta*derivative_non_linear_layer[:, 0:torch.sum(hidden_dim_tensor[0:1])])#Variable(delta*(input_to_non_linear_layer*w_res[0][selected_rows] + b_res[0][selected_rows]))
            
            curr_gradient = Variable(torch.mm(torch.t(non_linear_output),output_list[0])/selected_row_num)
                        
#             expected_gradient = Variable(torch.cat((expected_gradient_list[0].data, expected_gradient_list[1].data.view(-1,1)), 1))
        
        
#             gradient_diff = torch.norm(expected_gradient - curr_gradient)
            
#             print("gradient_diff::", gradient_diff)
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
                
                
            para_list[0].data = para_curr_layer[:, 0:-1]
            
            para_list[1].data = para_curr_layer[:, -1]

            init_model(model, para_list)
            
            
            
#             A_list[0] = A
#             
#             B_list[0] = B
#         
#             A_list_all_epochs.append(A_list)
#             
#             B_list_all_epochs.append(B_list)
    
#     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[0])))
#     
#     
#     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    
#     deriv = torch.mm(torch.t(delta), output_list[0])
#     
#     '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
# 
#     para_list[0].data = para_curr_layer[:, 0:-1]
#     
#     para_list[1].data = para_curr_layer[:, -1]

    t2  = time.time()
    
    print("time0:", t2 - t1)   
    
    print("overhead::", overhead)
    
    
    return list(model.parameters())
    
    
    
    
#     for param in param_list:
#         
#         delta =  


def model_update_provenance1(alpha, init_para, selected_rows, model, X, Y, w_list_all_epochs, b_list_all_epochs, num_class, input_dim, hidden_dims, output_dim, gradient_list_all_epochs):
    
#     t1  = time.time()
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
#     A_list_all_epochs = []
#     
#     B_list_all_epochs = []
#     
#     w_delta_prod_list_all_epochs = []
#     
#     b_delta_prod_list_all_epochs = []
    
    
    curr_X = X[selected_rows]
    
    
    curr_Y = Y[selected_rows]
    
    
    selected_row_num = selected_rows.shape[0]
    
    init_model(model, init_para)
    
    with torch.no_grad():
        
        for k in range(len(w_list_all_epochs)):
            w_res = w_list_all_epochs[k]
            
            b_res = b_list_all_epochs[k]
            
            expected_gradient_list = gradient_list_all_epochs[k]
            
            
            output_list = model.get_output_each_layer(curr_X)
            
            pred = output_list[len(output_list) - 1]
            
            para_list = list(model.parameters())
        
        
            depth = len(hidden_dims) + 1
        
#             A_list = [None]*depth
#         
#         
#             B_list = [None]*depth
            
            
        #     A0_list = [None]*depth
        #     
        #     B0_list = [None]*depth
        #         loss = error(pred, Y)
            
            delta = Variable(softmax_func(pred) - get_onehot_y(curr_Y, [selected_row_num, num_class], num_class))
            
            
            '''delta: n*output_dim'''
            
            para_curr_layer = Variable(torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1)) 
            
            
            '''A: output_dim, hidden_dim[depth-2]^2'''
            
        #     print(len(w_res))
            
        #     print(w_res[depth - 1])
            
        #     print(depth)
            
            
#             w_delta_prod_list = [None]*depth
#             
#             b_delta_prod_list = [None]*depth
            
            
            '''w_delta_prod: n*output_dim'''
            
            w_delta_prod = Variable(w_res[depth - 1][selected_rows]*delta)
            
            b_delta_prod = Variable(b_res[depth - 1][selected_rows]*delta)
            
#             w_delta_prod_list[depth - 1] = w_delta_prod
#             
#             b_delta_prod_list[depth - 1] = b_delta_prod
            
            '''A: output_dim*(hidden_dims[-1]*hiddem_dims[-1])'''
            A = Variable(torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(selected_row_num, hidden_dims[depth-2] + 1, 1), output_list[depth - 1].view(selected_row_num, 1, hidden_dims[depth-2]+ 1)).view(selected_row_num, -1)))
            
        #     A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1])
            
        #     A0_list.append(A0)
            
            '''B: output_dim, hidden_dim[depth-2]'''
            B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-1]))
            
        #     B0 = torch.t(torch.sum(b_delta_prod, 0))
            
        #     B0_list.append(B0)
            
            
            
#             A_list[depth - 1] = A
#             
#             B_list[depth - 1] = B
            
            
            
            w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
            
            
            entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
            
            
        #     delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
        #     
        #     deriv = torch.mm(torch.t(delta), output_list[depth - 1])
        
            '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
        
            delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(entire_delta_expression))))
            
        #     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        #     
        #     para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
        #     
        #     para_list[2*depth - 1].data = para_curr_layer[:, -1]
        
            curr_gradient = (1.0/(selected_row_num))*torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth-2] + 1), A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
                
            curr_gradient += (1.0/(selected_row_num))*B
            
            
            expected_gradient = Variable(torch.cat((expected_gradient_list[2*depth - 2].data, expected_gradient_list[2*depth - 1].data.view(-1,1)), 1))
            
            
            gradient_diff = torch.norm(expected_gradient - curr_gradient)
            
            print("gradient_diff::", gradient_diff)
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
                
                
            para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
            
            para_list[2*depth - 1].data = para_curr_layer[:, -1]
        
                
            for i in range(depth - 2):
                
                '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
                
                para_curr_layer = Variable(torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1))
                
                delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
                        
                w_delta_prod = Variable(w_res[depth - i - 2][selected_rows]*delta)
            
                b_delta_prod = Variable(b_res[depth - i - 2][selected_rows]*delta)
                
#                 w_delta_prod_list[depth - i - 2] = w_delta_prod
#             
#                 b_delta_prod_list[depth - i - 2] = b_delta_prod
                
                A = Variable(torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - i - 2].view(selected_row_num, hidden_dims[depth - i - 3]+ 1, 1), output_list[depth - i - 2].view(selected_row_num, 1, hidden_dims[depth - i - 3]+ 1)).view(selected_row_num, -1)))
                
        #         A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - i - 2])
                
        #         A0_list.append(A0)
                
                '''B: output_dim, hidden_dim[depth-2]'''
                B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-i-2]))
                
        #         B0 = torch.t(torch.sum(b_delta_prod, 0))
                
        #         B0_list.append(B0)
                
                w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - i - 2]))))
                
                
                entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
                
                
                delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(entire_delta_expression))))
                
                
                expected_gradient = Variable(torch.cat((expected_gradient_list[2*depth - 2*i - 4].data, expected_gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1))
                
                
                curr_gradient = (1.0/(selected_row_num))*torch.bmm(para_curr_layer.view(hidden_dims[depth-i-2], 1, hidden_dims[depth-i-3] + 1), A.view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1, hidden_dims[depth-3-i] + 1)).view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1)
                
                curr_gradient += (1.0/(selected_row_num))*B
                
                
                gradient_diff = torch.norm(curr_gradient-expected_gradient)
                
                print("gradient_diff::", gradient_diff)
                
                para_curr_layer = para_curr_layer - alpha*curr_gradient
                    
                    
                para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                
                para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
                
                
#                 A_list[depth - 2 - i] = A
#                 
#                 B_list[depth - 2 - i] = B
                
        #         deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])
        #         
        #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))                    
        #         
        #         '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
        #         para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        #     
        #         para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
        #         
        #         para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
                    
            para_curr_layer = Variable(torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1))
        
            delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
            
            w_delta_prod = Variable(w_res[0][selected_rows]*delta)
        
            b_delta_prod = Variable(b_res[0][selected_rows]*delta)
            
#             w_delta_prod_list[0] = w_delta_prod
#             
#             b_delta_prod_list[0] = b_delta_prod
#             
#             
#             w_delta_prod_list_all_epochs.append(w_delta_prod_list)
#             
#             b_delta_prod_list_all_epochs.append(b_delta_prod_list)
            
        #     A = torch.mm(torch.bmm(w_delta_prod.view(dim[0], )))
            
        #     A = torch.zeros([hidden_dims[0], (input_dim+1)*(input_dim+1)])
            
            
            A = Variable(torch.mm(torch.t(output_list[0]), torch.bmm(w_delta_prod.view(selected_row_num, hidden_dims[0], 1), output_list[0].view(selected_row_num, 1, input_dim + 1)).view(selected_row_num, hidden_dims[0]*(input_dim + 1))))
            
            A = Variable(torch.transpose(A.view(input_dim + 1, hidden_dims[0], input_dim + 1), 0, 1))
            
        #     for k in range(dim[0]):
        #         A += torch.mm(torch.t(w_delta_prod)[:,k].view(-1,1), torch.mm(output_list[0][k].view(input_dim + 1, 1), output_list[0][k].view(1, input_dim + 1)).view(1,-1))
            
        #     A = torch.mm( temp)
            
            '''B: output_dim, hidden_dim[depth-2]'''
            B = Variable(torch.mm(torch.t(b_delta_prod), output_list[0]))
            
            
            curr_gradient = (1.0/(selected_row_num))*torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
            
            curr_gradient += (1.0/(selected_row_num))*B
            
            
            expected_gradient = Variable(torch.cat((expected_gradient_list[0].data, expected_gradient_list[1].data.view(-1,1)), 1))
            
            gradient_diff = torch.norm(expected_gradient - curr_gradient)
            
            print("gradient_diff::", gradient_diff)
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
                
                
            para_list[0].data = para_curr_layer[:, 0:-1]
            
            para_list[1].data = para_curr_layer[:, -1]
            
            
            init_model(model, para_list)
            
#             A_list[0] = A
#             
#             B_list[0] = B
#         
#             A_list_all_epochs.append(A_list)
#             
#             B_list_all_epochs.append(B_list)
    
#     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[0])))
#     
#     
#     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    
#     deriv = torch.mm(torch.t(delta), output_list[0])
#     
#     '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
# 
#     para_list[0].data = para_curr_layer[:, 0:-1]
#     
#     para_list[1].data = para_curr_layer[:, -1]

#     t2  = time.time()
    
#     print("time0:", t2 - t1)   
    
    
    return list(model.parameters())
    
    
    
    
#     for param in param_list:
#         
#         delta =  
        


def model_update_provenance2(alpha, model, dim, output_list_all_epochs, A_list_all_epochs, B_list_all_epochs, w_delta_prod_list_all_epochs, b_delta_prod_list_all_epochs, input_dim, hidden_dims, output_dim, delta_ids, expected_gradient_list_all_epochs):
    
    
#     loss = np.infty
#     
#     count = 0
#     
#     
#     output_list = model.get_output_each_layer(X)
#     
#     para_list = model.parameters()
# 
#     while loss > loss_threshold and count < num_epochs:
#         
#         delta_list = []
#         
#         outer_gradient = softmax_func(output_list[len(output_list) - 1]) - get_onehot_y(Y, dim, num_class)       
#         
#         para_list_len = len(para_list)
#         
#         
#         for i in range(para_list_len):
#             
#             
#             '''w_res: n * hidden[len-i], b_res: n*1, para_list[2*para_list_len - 2*i]: hidden[len-i]*hidden[len-i-1], output_list[len-i]: n*hidden[len-i-1]'''
#             
#             
#             w_res[para_list_len - i]*torch.sum(para_list[2*para_list_len - 2*i]*output_list[para_list_len - i], 2) + b_res[para_list_len - i]
#             
#             
#             
#             softmax_func()


    t1  = time.time()
    
    
    para_list = list(model.parameters())
    
    depth = len(hidden_dims) + 1
    
    '''delta: n*output_dim'''
    
    
    '''A: output_dim, hidden_dim[depth-2]^2'''    
    
    print(depth)
    
    delta_id_num = delta_ids.shape[0]
    
    curr_training_data = output_list_all_epochs[0][0][:, 0:-1]
    
    
    with torch.no_grad():
        
    
        for k in range(len(output_list_all_epochs)):
        
            expected_gradient_list = expected_gradient_list_all_epochs[k]
        
            
            delta_A_list = [None]*depth
            
        #     delta_A_list0 = [None]*depth
            
            
            delta_B_list = [None]*depth
            
        #     delta_B_list0 = [None]*depth
            
            w_delta_prod_list = w_delta_prod_list_all_epochs[k]
            
            b_delta_prod_list = b_delta_prod_list_all_epochs[k]
            
            A_list = A_list_all_epochs[k]
            
            B_list = B_list_all_epochs[k]
            
            
            output_list,_ = model.get_output_each_layer(curr_training_data)#output_list_all_epochs[k]
            
        
            w_delta_prod = w_delta_prod_list[depth - 1][delta_ids]
            
            b_delta_prod = b_delta_prod_list[depth - 1][delta_ids]
            
            '''A: output_dim, hidden_dims[-1]*hiddem_dims[-1]'''
            
            delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1][delta_ids].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1][delta_ids].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
            
        #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
            
            delta_A_list[depth - 1] = delta_A
            
        #     delta_A_list0.append(delta_A0)
            
            '''B: output_dim, hidden_dim[depth-2]'''
            delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-1][delta_ids])
            
        #     delta_B0 = torch.t(b_delta_prod)
            
        #     delta_B_list0.append(delta_B0)
            
            delta_B_list[depth - 1] = delta_B
            
            for i in range(depth - 2):
                
                '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
                
        #         para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
        #         delta = delta_para_prod#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
                w_delta_prod = w_delta_prod_list[depth - i - 2][delta_ids]
            
                b_delta_prod = b_delta_prod_list[depth - i - 2][delta_ids]
                
                delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - i - 2][delta_ids].view(delta_id_num, hidden_dims[depth - i - 3] + 1, 1), output_list[depth - i - 2][delta_ids].view(delta_id_num, 1, hidden_dims[depth - i - 3] + 1)).view(delta_id_num, (hidden_dims[depth-i-3]+1)*(hidden_dims[depth-i-3]+1)))
                
        #         delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - i - 2][delta_ids])
                
                delta_A_list[depth - i - 2] = delta_A
                
        #         delta_A_list0.append(delta_A0)
                
                '''B: output_dim, hidden_dim[depth-2]'''
                delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-i-2][delta_ids])
                
        #         delta_B0 = torch.t(b_delta_prod)
                
        #         delta_B_list0.append(delta_B0)
                
                
                delta_B_list[depth - i - 2] = delta_B
            
            w_delta_prod = w_delta_prod_list[0][delta_ids]
        
            b_delta_prod = b_delta_prod_list[0][delta_ids]
            
            delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[0][delta_ids].view(delta_id_num, input_dim + 1, 1), output_list[0][delta_ids].view(delta_id_num, 1, input_dim + 1)).view(delta_id_num, (input_dim+1)*(input_dim + 1)))
        
            delta_A = delta_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)
        
        #    
            '''B: output_dim, hidden_dim[depth-2]'''
            delta_B = torch.mm(torch.t(b_delta_prod), output_list[0][delta_ids])
            
        #     delta_B0 = torch.t(b_delta_prod)
            
        #     delta_B_list0.append(delta_B0)
            
            
            delta_A_list[0] = delta_A
            
            delta_B_list[0] = delta_B
        
        
        #     weights = para_list[2*depth - 2].data
        #         
        #     offsets = para_list[2*depth - 1].data
            
            
            para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 
            
            '''A: (output_dim*hidden_dims[depth-1])*hiddem_dims[depth-1]'''
        
            '''B: output_dim*hidden_dims[depth-2]'''
            
            '''weights: output_dim*hidden_dims[depth-1]'''
            
            curr_A = A_list[depth - 1] - delta_A_list[depth - 1] 
            
            curr_B = B_list[depth - 1] - delta_B_list[depth - 1]
            
    #         for j in range(epoch):
                
            '''output_dim, 1, hidden_dims[depth-2]'''
                
            curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
            
            curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
            
            gradient_curr_layer = torch.cat((expected_gradient_list[2*depth - 2].data, expected_gradient_list[2*depth - 1].data.view(-1,1)), 1) 

            
    #         gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
    #         
            delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
                
        #         para_curr_layer = para_curr_layer - alpha/
        
            para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
                
            para_list[2*depth - 1].data = para_curr_layer[:, -1]
            
            for i in range(depth-2):
                
                para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
                        
                '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
        
                '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
                
                '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
                
                curr_A = A_list[depth - i - 2] - delta_A_list[depth - i - 2] 
                
                curr_B = B_list[depth - i - 2] - delta_B_list[depth - i - 2]
                
    #             for j in range(epoch):
                
    #             gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)),1)
                
                curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[depth-i-2], 1, hidden_dims[depth-i-3] + 1), curr_A.view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1, hidden_dims[depth-3-i] + 1)).view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1)
                
                curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
                
                gradient_curr_layer = torch.cat((expected_gradient_list[2*depth - 2*i - 4].data, expected_gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1) 

                
                delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
                
                para_curr_layer = para_curr_layer - alpha*curr_gradient
                
                
                para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                
                para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
            
            
            para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)
                        
            '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
        
            '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
            
            '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
            
            curr_A = A_list[0] - delta_A_list[0] 
            
            curr_B = B_list[0] - delta_B_list[0]
            
    #         for j in range(epoch):
                
    #         gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)),1)
            
            curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), curr_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
            
            curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
            
            gradient_curr_layer = torch.cat((expected_gradient_list[0].data, expected_gradient_list[1].data.view(-1,1)), 1) 
            
            delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
                    
            para_curr_layer = para_curr_layer - alpha*curr_gradient
            
            
            para_list[0].data = para_curr_layer[:, 0:-1]
            
            para_list[1].data = para_curr_layer[:, -1] 
            
            init_model(model, para_list)
    
    t2  = time.time()
    
    print("time0:", t2 - t1)   
    
    return para_list


def model_update_provenance3(X, Y, alpha, model, dim, output_list_all_epochs, A_list_all_epochs, B_list_all_epochs, w_delta_prod_list_all_epochs, b_delta_prod_list_all_epochs, input_dim, hidden_dims, output_dim, delta_ids, expected_gradient_list_all_epochs):
    
    
#     loss = np.infty
#     
#     count = 0
#     
#     
#     output_list = model.get_output_each_layer(X)
#     
#     para_list = model.parameters()
# 
#     while loss > loss_threshold and count < num_epochs:
#         
#         delta_list = []
#         
#         outer_gradient = softmax_func(output_list[len(output_list) - 1]) - get_onehot_y(Y, dim, num_class)       
#         
#         para_list_len = len(para_list)
#         
#         
#         for i in range(para_list_len):
#             
#             
#             '''w_res: n * hidden[len-i], b_res: n*1, para_list[2*para_list_len - 2*i]: hidden[len-i]*hidden[len-i-1], output_list[len-i]: n*hidden[len-i-1]'''
#             
#             
#             w_res[para_list_len - i]*torch.sum(para_list[2*para_list_len - 2*i]*output_list[para_list_len - i], 2) + b_res[para_list_len - i]
#             
#             
#             
#             softmax_func()


    t1  = time.time()
    
    
    para_list = list(model.parameters())
    
    depth = len(hidden_dims) + 1
    
    '''delta: n*output_dim'''
    
    
    '''A: output_dim, hidden_dim[depth-2]^2'''    
    
    print(depth)
    
    delta_id_num = delta_ids.shape[0]
    
    curr_training_data = X[delta_ids]#output_list_all_epochs[0][0][:, 0:-1]
    
    
    overhead = 0
    
    with torch.no_grad():
        
    
        for k in range(len(output_list_all_epochs)):
        
            expected_gradient_list = expected_gradient_list_all_epochs[k]
        
            
#             delta_A_list = [None]*depth
            
        #     delta_A_list0 = [None]*depth
            
            
#             delta_B_list = [None]*depth
            
        #     delta_B_list0 = [None]*depth
            
            w_delta_prod_list = w_delta_prod_list_all_epochs[k]
            
            b_delta_prod_list = b_delta_prod_list_all_epochs[k]
            
            A_list = A_list_all_epochs[k]
            
            B_list = B_list_all_epochs[k]
            
            
            output_list,input_to_non_linear_layer_list = model.get_output_each_layer(curr_training_data)#output_list_all_epochs[k]
            
            
        
            input_to_non_linear_layer_list = input_to_non_linear_layer_list[1:]
        
            w_delta_prod = Variable(w_delta_prod_list[depth - 1][delta_ids])
            
            b_delta_prod = Variable(b_delta_prod_list[depth - 1][delta_ids])
            
            '''A: output_dim, hidden_dims[-1]*hiddem_dims[-1]'''
            
            para_curr_layer = Variable(torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1)) 


            input_to_non_linear_layer = input_to_non_linear_layer_list[depth -1]

            t3 = time.time()

            delta_A = Variable(torch.mm(torch.t((w_delta_prod*input_to_non_linear_layer + b_delta_prod)), output_list[depth-1]))


            

            
            curr_A = Variable(torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), A_list[depth - 1].view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1) - delta_A) 
            
            
            
            
            delta_B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-1]))
            
            curr_B = Variable(B_list[depth - 1] - delta_B)
            
    #         for j in range(epoch):
                
            '''output_dim, 1, hidden_dims[depth-2]'''
                
            curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
            
            curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
            
            para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
                
        #         para_curr_layer = para_curr_layer - alpha/
        
            para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
                
            para_list[2*depth - 1].data = para_curr_layer[:, -1]
            
#             delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1][delta_ids].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1][delta_ids].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
            
        #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
                        
        #     delta_A_list0.append(delta_A0)
            
            '''B: output_dim, hidden_dim[depth-2]'''
            
        #     delta_B0 = torch.t(b_delta_prod)
            
        #     delta_B_list0.append(delta_B0)
                        
            for i in range(depth - 2):
                
                '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
                para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
        #         para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
        #         delta = delta_para_prod#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
                w_delta_prod = w_delta_prod_list[depth - i - 2][delta_ids]
            
                b_delta_prod = b_delta_prod_list[depth - i - 2][delta_ids]
                
                delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - i - 2].view(delta_id_num, hidden_dims[depth - i - 3] + 1, 1), output_list[depth - i - 2].view(delta_id_num, 1, hidden_dims[depth - i - 3] + 1)).view(delta_id_num, (hidden_dims[depth-i-3]+1)*(hidden_dims[depth-i-3]+1)))
                
        #         delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - i - 2][delta_ids])
                
#                 delta_A_list[depth - i - 2] = delta_A
                
        #         delta_A_list0.append(delta_A0)
                
                '''B: output_dim, hidden_dim[depth-2]'''
                delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-i-2][delta_ids])


                curr_A = torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), A_list[depth - i - 2].view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1) - delta_A 
            
                curr_B = B_list[depth - i - 2] - delta_B
                
                
                curr_gradient = (1.0/(dim[0]-delta_id_num))*curr_A #torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), curr_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
            
                curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
                
                para_curr_layer = para_curr_layer - alpha*curr_gradient
                
                para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                
                para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]

                
        #         delta_B0 = torch.t(b_delta_prod)
                
        #         delta_B_list0.append(delta_B0)
                
                
#                 delta_B_list[depth - i - 2] = delta_B
            
            w_delta_prod = Variable(w_delta_prod_list[0][delta_ids])
        
            b_delta_prod = Variable(b_delta_prod_list[0][delta_ids])
            
#             delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[0][delta_ids].view(delta_id_num, input_dim + 1, 1), output_list[0][delta_ids].view(delta_id_num, 1, input_dim + 1)).view(delta_id_num, (input_dim+1)*(input_dim + 1)))
#         
#             delta_A = delta_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)
        
        #
            para_curr_layer = Variable(torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1))

        
            input_to_non_linear_layer = input_to_non_linear_layer_list[0]
        
            delta_A = Variable(torch.mm(torch.t((w_delta_prod*input_to_non_linear_layer + b_delta_prod)), output_list[0]))


            '''B: output_dim, hidden_dim[depth-2]'''
            delta_B = Variable(torch.mm(torch.t(b_delta_prod), output_list[0]))

            t4 = time.time()
            
            overhead += (t4 - t3)
            
            curr_A = Variable(torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), A_list[0].view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1) - delta_A)

            

            
            curr_B = Variable(B_list[0] - delta_B)
            
    #         for j in range(epoch):
                
    #         gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)),1)
            
            curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A) #torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), curr_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
            
            curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
            
            
            para_list[0].data = para_curr_layer[:, 0:-1]
            
            para_list[1].data = para_curr_layer[:, -1] 
            
            init_model(model, para_list)

        #     delta_B0 = torch.t(b_delta_prod)
            
        #     delta_B_list0.append(delta_B0)
            
            
        #         
        #     offsets = para_list[2*depth - 1].data
            
            
#             para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 
            
            '''A: (output_dim*hidden_dims[depth-1])*hiddem_dims[depth-1]'''
        
            '''B: output_dim*hidden_dims[depth-2]'''
            
            '''weights: output_dim*hidden_dims[depth-1]'''
            
            
            
#             gradient_curr_layer = torch.cat((expected_gradient_list[2*depth - 2].data, expected_gradient_list[2*depth - 1].data.view(-1,1)), 1) 
# 
#             
#     #         gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
#     #         
#             delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#             
#             para_curr_layer = para_curr_layer - alpha*curr_gradient
#                 
#         #         para_curr_layer = para_curr_layer - alpha/
#         
#             para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
#                 
#             para_list[2*depth - 1].data = para_curr_layer[:, -1]
#             
#             for i in range(depth-2):
#                 
#                 para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
#                         
#                 '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
#         
#                 '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#                 
#                 '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#                 
#                 curr_A = A_list[depth - i - 2] - delta_A_list[depth - i - 2] 
#                 
#                 curr_B = B_list[depth - i - 2] - delta_B_list[depth - i - 2]
#                 
#     #             for j in range(epoch):
#                 
#     #             gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)),1)
#                 
#                 curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[depth-i-2], 1, hidden_dims[depth-i-3] + 1), curr_A.view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1, hidden_dims[depth-3-i] + 1)).view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1)
#                 
#                 curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
#                 
#                 gradient_curr_layer = torch.cat((expected_gradient_list[2*depth - 2*i - 4].data, expected_gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1) 
# 
#                 
#                 delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#                 
#                 para_curr_layer = para_curr_layer - alpha*curr_gradient
#                 
#                 
#                 para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
#                 
#                 para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
#             
#             
#             para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)
#                         
#             '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
#         
#             '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#             
#             '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#             
#             curr_A = torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), A_list[0].view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1) - delta_A_list[0] 
#             
#             curr_B = B_list[0] - delta_B_list[0]
#             
#     #         for j in range(epoch):
#                 
#     #         gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)),1)
#             
#             curr_gradient = (1.0/(dim[0]-delta_id_num))*curr_A #torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), curr_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
#             
#             curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
#             
#             gradient_curr_layer = torch.cat((expected_gradient_list[0].data, expected_gradient_list[1].data.view(-1,1)), 1) 
#             
#             delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#                     
#             para_curr_layer = para_curr_layer - alpha*curr_gradient
#             
#             
#             para_list[0].data = para_curr_layer[:, 0:-1]
#             
#             para_list[1].data = para_curr_layer[:, -1] 
            
    
    t2  = time.time()
    
    print("time0:", t2 - t1)  
    
    print("overhead:", overhead) 
    
    return para_list


# def compute_derivative_with_provenance(dim, hidden_dims, A_list, B_list, para_list):
#      
# #     para_list = list(model.parameters())
#      
#     depth = len(hidden_dims) + 1
#      
#     para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 
#      
#     curr_A = A_list[depth  -1]
#     
#     curr_B = B_list[depth - 1]
#      
#     der_list = [None]*(2*depth)
#      
#     derivative1 =1.0/(dim[0])*torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
#     
#     
#     derivative1 += 1.0/(dim[0])*curr_B
#  
#     der_list[2*depth - 2] = derivative1[:,0:-1]
#     
#     der_list[2*depth - 1] = derivative1[:,-1]
#  
#     for i in range(depth-2):
#          
#         para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
#                  
#         '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
#  
#         '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#          
#         '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#          
#         curr_A = A_list[depth - i - 2]
#          
#         curr_B = B_list[depth - i - 2]
#          
#         derivative1 = 1.0/(dim[0])*torch.bmm(para_curr_layer.view(hidden_dims[depth-i-2], 1, hidden_dims[depth-i-3] + 1), curr_A.view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1, hidden_dims[depth-3-i] + 1)).view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1)
#          
#         derivative1 += 1.0/(dim[0])*curr_B
#         
#         der_list[2*depth - 2*i - 4] = derivative1[:,0:-1]
#     
#         der_list[2*depth - 2*i - 3] = derivative1[:,-1]
#         
#      
#     para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)
#                  
#     '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
#  
#     '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#      
#     '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#      
#     curr_A = A_list[0] 
#      
#     curr_B = B_list[0]
#     
#     derivative1 = 1./(dim[0])*torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), curr_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
#      
#     derivative1 += 1.0/(dim[0])*curr_B
#     
#     der_list[0] = derivative1[:,0:-1]
#     
#     der_list[1] = derivative1[:,-1]
#     
#     return der_list
    


def model_update_provenance_test1_4(max_epoch, period, length, init_epochs, res_para_list, res_grad_list, exp_gradient_list_all_epochs, exp_para_list_all_epochs, dataset_train, model, grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, m, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, added_random_ids_multi_super_iteration, X_to_add, Y_to_add, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device):
    
    para = list(model.parameters())
    
    use_standard_way = False
    
    recorded = 0
    
    overhead = 0
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    
    if is_GPU:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double, device = device)
    else:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    
    i = 0
    
    S_k_list = deque()
    
    Y_k_list = deque()
    
    queue_id_list = deque()
    
    overhead2 = 0
    
    overhead3 = 0
    
    old_lr = -1
    
    cached_id = 0
    
    batch_id = 1
    
    
    
#     random_ids_list_all_epochs = []
# 
#     removed_batch_empty_list = []
    
    last_explicit_training_iteration = 0
    
#     torch.manual_seed(random_seed)
#     torch.cuda.manual_seed(random_seed)
#     np.random.seed(random_seed)
#     random.seed(random_seed)
    
    
#     for k in range(len(random_ids_multi_super_iterations)):
#         
# 
#     
#         random_ids = random_ids_multi_super_iterations[k]
#         
#         sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
#         
# #         all_indexes = np.sort(sort_idx[delta_ids])
#         if delta_ids.shape[0] > 1:
#             all_indexes = np.sort(sort_idx[delta_ids])
#         else:
#             all_indexes = torch.tensor([sort_idx[delta_ids]])
#                 
#         id_start = 0
#     
#         id_end = 0
#         
#         random_ids_list = []
#         
#         for j in range(0, dim[0], batch_size):
#         
#             end_id = j + batch_size
#             
#             if end_id > dim[0]:
#                 end_id = dim[0]
#             
#             if all_indexes[-1] < end_id:
#                 id_end = all_indexes.shape[0]
#             else:
#                 id_end = np.argmax(all_indexes >= end_id)
#             
#             curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
# 
#             curr_matched_id_num = curr_matched_ids.shape[0]
# 
#             if curr_matched_id_num > 0:
#                 random_ids_list.append(curr_matched_ids)
#                 removed_batch_empty_list.append(False)
#             else:
#                 random_ids_list.append(random_ids[0:1])
#                 removed_batch_empty_list.append(True)
#             
# #             if (i-init_epochs)%period == 0:
# #                 
# #                 recorded = 0
# #                 
# #                 use_standard_way = True
# #                 
# #                 
# #             if i< init_epochs or use_standard_way == True:
# #                 
# #                 curr_rand_ids = random_ids[j:end_id]
# #             
# #             
# #                 curr_matched_ids2 = (get_remaining_subset_data_per_epoch(curr_rand_ids, curr_matched_ids))
# #             
# #                 random_ids_list.append(curr_matched_ids2)
# #                 
# #                 recorded += 1
# #                     
# #                     
# #                 if recorded >= length:
# #                     use_standard_way = False
#             
#             i += 1
#             
#             id_start = id_end
#                 
#         random_ids_list_all_epochs.append(random_ids_list)        
#     
#     i = 0
#     
#     curr_batch_sampler = Batch_sampler(random_ids_list_all_epochs)
#     
#         
#     data_train_loader = DataLoader(dataset_train, num_workers=0, shuffle=False, batch_sampler = curr_batch_sampler, pin_memory=True)
    
#     for k in range(len(random_ids_multi_super_iterations)):            
    for k in range(max_epoch):
            
        print("epoch ", k)
#         random_ids_list = random_ids_list_all_epochs[k]
        
        added_to_random_ids = added_random_ids_multi_super_iteration[k]
        
        random_ids = random_ids_multi_super_iterations[k]
            
        j = 0
        
#         enum_loader = enumerate(data_train_loader)
#         
#         for t, items in enum_loader:

        if k == 0:
             
            for p in range(len(added_to_random_ids)):
                if len(added_to_random_ids[p]) > 0:
                    i = p
                    break
#                 else:
#                     res_para.append(None)
#                     
#                     res_grad.append(None)
                 
            added_to_random_ids = added_to_random_ids[i:]
#             i = torch.nonzero(torch.tensor(removed_batch_empty_list).view(-1) == False)[0].item()
            init_model(model, get_devectorized_parameters(para_list_all_epoch_tensor[i], full_shape_list, shape_list))
         
            para = list(model.parameters())
         
#             random_ids_list = random_ids_list[i:]
             
#             remaining_ids_list = remaining_ids_list[i:]
         
            j = batch_size*i
             
            cached_id = i
             
            curr_init_epochs = init_epochs + i
             
             
            if cached_id >= cached_size:
                 
                batch_id = cached_id/cached_size
                 
                GPU_tensor_end_id = (batch_id + 1)*cached_size
                 
                if GPU_tensor_end_id > para_list_all_epoch_tensor.shape[0]:
                    GPU_tensor_end_id = para_list_all_epoch_tensor.shape[0] 
                print("end_tensor_id::", GPU_tensor_end_id)
                 
                para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(para_list_all_epoch_tensor[batch_id*cached_size:GPU_tensor_end_id])
                 
                grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(grad_list_all_epoch_tensor[batch_id*cached_size:GPU_tensor_end_id])
                 
                batch_id += 1
                 
                cached_id = 0




        
        
        to_add = True
        
#         for j in range(0, dim[0], batch_size):
        for jj in range(len(added_to_random_ids)):
        
            end_id = j + batch_size
            
#             added_end_id = jj + added_batch_size
            curr_added_random_ids = added_to_random_ids[jj]
        
#             end_id = j + batch_size
#             
#             added_end_id = jj + added_batch_size
            
            
            if end_id > dim[0]:
                end_id = dim[0]
            
#             if added_end_id >= X_to_add.shape[0]:
#                 added_end_id = X_to_add.shape[0]
            
            
            if curr_added_random_ids.shape[0] <= 0:
                to_add = False
            else:
                to_add = True

            curr_added_size = 0

            

            if to_add:
                
#                 curr_added_random_ids = added_to_random_ids[jj:added_end_id]
                
                batch_delta_X = dataset_train.data[curr_added_random_ids]
                
                batch_delta_Y = dataset_train.labels[curr_added_random_ids]
            
                curr_added_size = curr_added_random_ids.shape[0]
                
                
                if is_GPU:
                    batch_delta_X = batch_delta_X.to(device)
                    
                    batch_delta_Y = batch_delta_Y.to(device)
            
            learning_rate = learning_rate_all_epochs[i]
            
#             if end_id - j - curr_matched_ids_size <= 0:
#                 
#                 i += 1
#                 
#                 continue
            
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate
            
                      
            if (i-last_explicit_training_iteration)%period == 0:
                
#                 recorded = 0
                
                use_standard_way = True
                
                
            if i<= curr_init_epochs or use_standard_way == True:
                
                
                last_explicit_training_iteration = i
                
                curr_rand_ids = random_ids[j:end_id]
            
            
                batch_remaining_X = dataset_train.data[curr_rand_ids]
                
                batch_remaining_Y = dataset_train.labels[curr_rand_ids]
                
                if is_GPU:
                    batch_remaining_X = batch_remaining_X.to(device)
                    
                    batch_remaining_Y = batch_remaining_Y.to(device)                
                
                
#                 _,next_items = enum_loader.__next__()
                
#                 if not is_GPU:
#                 
#                     batch_remaining_X = next_items[0]
#                     
#                     batch_remaining_Y = next_items[1]
#                     
#                 else:
#                 if is_GPU:
#                     batch_remaining_X = batch_remaining_X.to(device)
#                     
#                     batch_remaining_Y = batch_remaining_Y.to(device)
                
                init_model(model, para)
                                
                compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
                 
                expect_gradients = get_all_vectorized_parameters1(model.get_all_gradient())


                gradient_remaining = 0
                if to_add:
                    clear_gradients(model.parameters())
                        
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                
                
                    gradient_remaining = get_all_vectorized_parameters1(model.get_all_gradient())     
                
                with torch.no_grad():
                  
                    curr_para = get_all_vectorized_parameters1(para)         
                
                    if i>0:
                        
                        prev_para = para_list_GPU_tensor[cached_id]
                        
                        curr_S_k = (curr_para - prev_para)
                        
                        
                        
#                         curr_S_k = (get_all_vectorized_parameters1(para) - get_all_vectorized_parameters2(para_list_all_epochs[i], is_GPU, device)).view(-1)
    
                        
#                         if len(S_k_list) > m:
#                             S_k_list.popleft()
                    
#                     gradient_full = (expect_gradients*batch_remaining_ids.shape[0] + gradient_remaining*curr_matched_ids_size)/(batch_remaining_ids.shape[0] + curr_matched_ids_size)
                    gradient_full = (expect_gradients*curr_rand_ids.shape[0] + gradient_remaining*curr_added_size)/(curr_rand_ids.shape[0] + curr_added_size)
                    
                    if i>0:
                        
                        
                        curr_Y_k = expect_gradients - grad_list_GPU_tensor[cached_id] + regularization_coeff*curr_S_k
                        
                        
#                         curr_Y_k = (gradient_full - get_all_vectorized_parameters2(gradient_list_all_epochs[i], is_GPU, device) + regularization_coeff*curr_S_k).view(-1)
                        
                        dot_res = torch.mm(curr_S_k, torch.t(curr_Y_k))
                        
#                         print("curr_secont condition::", dot_res)
                        if dot_res > 0:
                            Y_k_list.append(curr_Y_k)
                            S_k_list.append(curr_S_k)
                            queue_id_list.append(i)
                            
#                             print("secont condition::", dot_res)
                            
                            
                            if len(Y_k_list) > m or i - queue_id_list[0] > 2 :
                                Y_k_list.popleft()
                                S_k_list.popleft()
                                queue_id_list.popleft()
                                
                            if len(Y_k_list) == m:
                                use_standard_way = False
                            
#                         print("explicit_evaluation epoch::", i)
#                         
#                         print("batch size check::", curr_matched_ids_size + batch_remaining_ids.shape[0])
                    
                    exp_gradient = None
                    exp_param = None
                    
#                     exp_gradient = exp_gradient_list_all_epochs[i]
# #               
#      
#                          
#                     exp_param = exp_para_list_all_epochs[i]
                     
#                     print(i, curr_rand_ids.shape[0], curr_added_size)
#                               
#                     print("para_diff::")
# #                     print(torch.norm(exp_param - get_all_vectorized_parameters1(para)))
#                     compute_model_para_diff(exp_param, para)
#                       
#                       
#                     print("gradient diff::")
#                       
#                     print(torch.norm(get_all_vectorized_parameters1(exp_gradient) - (gradient_full)))
#                     
#                     print("here")
#                     
#                     
#                     if i == 1:
#                         exp_batch_X = torch.load(git_ignore_folder + 'tmp_batch_x')
#                  
#                         exp_batch_Y = torch.load(git_ignore_folder + 'tmp_batch_y')
#                          
#                         exp_added_x = torch.load(git_ignore_folder + 'tmp_added_x')
#                          
#                         exp_added_y = torch.load(git_ignore_folder + 'tmp_added_y')
#                          
#                         exp_grad_remaining = torch.load(git_ignore_folder + 'tmp_grad_remaining')
#                          
#                         exp_grad_dual = torch.load(git_ignore_folder + 'tmp_grad_dual')
#                          
#                         exp_grad2 = (exp_grad_remaining*curr_rand_ids.shape[0] + exp_grad_dual*curr_added_size)/(curr_rand_ids.shape[0] + curr_added_size)
#                         print(torch.norm(exp_grad2 - gradient_full))
#                                
#                         print(torch.norm(exp_grad2 - get_all_vectorized_parameters1(exp_gradient)))   
#                          
#                         print(torch.norm(exp_batch_X - batch_remaining_X))
#                          
#                         print(torch.max(exp_batch_Y - batch_remaining_Y))
#                          
#                         print(torch.min(exp_batch_Y - batch_remaining_Y))
#                         if to_add:
#                             print(torch.norm(exp_added_x - batch_delta_X))
#                              
#                             print(torch.max(exp_added_y - batch_delta_Y))
#                              
#                             print(torch.min(exp_added_y - batch_delta_Y))
#                          
#                         print(torch.norm(exp_grad_remaining - expect_gradients))
#                          
#                         print(torch.norm(exp_grad_dual - gradient_remaining))
#                         
#                         print("here")
                    
                    
#                     compute_model_para_diff(exp_gradient, para)
#                     compute_model_para_diff(exp_gradient, )
                     
#                     print("para_diff2::")
#                     compute_model_para_diff(para_list_all_epochs[i], para)
                    
                    
#                     if i >= 115:
#                         print("here!!")
                    
#                     para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_para - learning_rate*expect_gradients, full_shape_list, shape_list)
                    para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_para - learning_rate*gradient_full, full_shape_list, shape_list)
                    
                    
                    del gradient_full
                    
                    del gradient_remaining
                    
                    del expect_gradients
                    
                    del batch_remaining_X
                    
                    del batch_remaining_Y
                    
                    if to_add:
                        
                        del batch_delta_X
                        
                        del batch_delta_Y
                    
                    if i > 0:
                        del prev_para
                    
                        del curr_para
                    
#                     del gradient_full, gradient_remaining, expect_gradients
#                     recorded += 1
#                     
#                     
#                     if recorded >= length:
#                         use_standard_way = False
                
                
            else:
                gradient_dual = None
                
                curr_vec_para = get_all_vectorized_parameters1(para)
    
                if to_add:
                
                    init_model(model, para)
                    
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                    
                    
                    gradient_dual = model.get_all_gradient()
                    
                    
                
                with torch.no_grad():         
                    
                    
                    vec_para_diff = torch.t((curr_vec_para - para_list_GPU_tensor[cached_id]))
#                     
#                     compute_diff_vectorized_parameters2(para, para_list_all_epochs[i], vec_para_diff, shape_list, is_GPU, device)                    
                    
#                     if (i-last_explicit_training_iteration)/period >= 1:
                    
                    
                    
                    if (i-last_explicit_training_iteration)/period >= 1:
                        if (i-last_explicit_training_iteration) % period == 1:
    #                         print(i)
    #                         
    #                         if i >= 370:
    #                             y = 0
    #                             y+=1
                            zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, is_GPU, device)
#                             zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2 = prepare_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, period)
                            
                            
                            
                            mat = np.linalg.inv(mat_prime.cpu().numpy())
                            mat = torch.from_numpy(mat)
                            if is_GPU:
                                
                                
                                mat = mat.to(device)
                            
                    
                        hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff, is_GPU, device)
                        
                    else:
                        
                        hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period, is_GPU, device)

                    
                    
                    
#                     if (i-last_explicit_training_iteration) % period == 1:
#                         
#                         zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, is_GPU,device)
#                         
#                         
#                         
#                         
#                         mat = np.linalg.inv(mat_prime.cpu().numpy())
#                         mat = torch.from_numpy(mat)
#                         if is_GPU:
#                             
#                             
#                             mat = mat.to(device)
#                         
#                         
#                         
#                         
# #                         mat = torch.inverse(mat_prime)
#                 
#                     hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms0(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff, is_GPU, device)
                        
#                     else:
#                         hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_2(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period)
                         
                    delta_const = 0#compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
                    
#                     S_k_list.clear()
#                     
#                     Y_k_list.clear()
#                     
#                     queue_id_list.clear()
                    
#                     del S_k_list[:]
# 
#                     del Y_k_list[:]
                    
                    exp_gradient = None
                    
                    exp_param = None
                    
#                     exp_gradient = exp_gradient_list_all_epochs[i]
# #               
#    
#                        
#                     exp_param = exp_para_list_all_epochs[i]
#                              
#                     print("para_diff::")
#                     compute_model_para_diff(exp_param, para)
#                          
#                     print(curr_matched_ids_size)
                      
                    
                    
#                     update_para_final(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, exp_gradient, exp_param)

                    if gradient_dual is not None:
                        is_positive, final_gradient_list = compute_grad_final4(curr_vec_para, torch.t(hessian_para_prod), get_all_vectorized_parameters1(gradient_dual), grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_added_size, learning_rate, regularization_coeff, is_GPU, device)
                        
                    else:
                        is_positive, final_gradient_list = compute_grad_final4(curr_vec_para, torch.t(hessian_para_prod), None, grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_added_size, learning_rate, regularization_coeff, is_GPU, device)


#                     if i == 12:
#                         exp_batch_delta_X = torch.load(git_ignore_folder + 'tmp_delat_x')
#                     
#                         exp_batch_delta_Y = torch.load(git_ignore_folder + 'tmp_delat_y')
#                         
# #                         exp_gradient_dual = torch.load(git_ignore_folder + 'grad_dual')
#                         
#                         exp_hessian_para_prod = torch.load(git_ignore_folder + 'hessian_para_prod')
#                         
#                         print(torch.norm(exp_batch_delta_X - batch_delta_X))
#                         
#                         print(torch.max(exp_batch_delta_Y - batch_delta_Y))
#                         
#                         
#                         print(torch.min(exp_batch_delta_Y - batch_delta_Y))
#                         
#                         print(torch.norm(exp_hessian_para_prod - hessian_para_prod))
#                         
# #                         compute_model_para_diff(gradient_dual, exp_gradient_dual)
#                     
#                         print("here")


#                     exp_gradient = exp_gradient_list_all_epochs[i]
#                          
#                     exp_param = exp_para_list_all_epochs[i]
#                                
#                     print("para_diff::")
#                     compute_model_para_diff(exp_param, para)
# #                     print(torch.norm(exp_param - get_all_vectorized_parameters1(para)))
#                      
#                      
#                     print("grad diff::")
#                      
#                     print(torch.norm(get_all_vectorized_parameters1(exp_gradient) - final_gradient_list))

                    

#                 print(i, curr_matched_ids_size, is_positive)


#                     is_positive, final_gradient_list = compute_grad_final2(para, get_devectorized_parameters(hessian_para_prod, full_shape_list, shape_list), gradient_dual, gradient_list_all_epochs[i], para_list_all_epochs[i], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, is_GPU, device)

                    
                if not is_positive:
                     
                    use_standard_way = True
                     
                    last_explicit_training_iteration = i
             
                    curr_rand_ids = random_ids[j:end_id]
            
            
                    batch_remaining_X = dataset_train.data[curr_rand_ids]
                    
                    batch_remaining_Y = dataset_train.labels[curr_rand_ids]
                    
                    if is_GPU:
                        batch_remaining_X = batch_remaining_X.to(device)
                        
                        batch_remaining_Y = batch_remaining_Y.to(device)    
                     
                     
                     
    #                 _,next_items = enum_loader.__next__()
                     
    #                 if not is_GPU:
    #                 
    #                     batch_remaining_X = next_items[0]
    #                     
    #                     batch_remaining_Y = next_items[1]
    #                     
    #                 else:
#                     if is_GPU:
#                         batch_remaining_X = batch_remaining_X.to(device)
#                          
#                         batch_remaining_Y = batch_remaining_Y.to(device)
                     
                    init_model(model, para)
                                     
                    compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
                      
                    expect_gradients = get_all_vectorized_parameters1(model.get_all_gradient())
     
     
                    gradient_remaining = 0
                    if to_add:
                        clear_gradients(model.parameters())
                             
                        compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                     
                     
                        gradient_remaining = get_all_vectorized_parameters1(model.get_all_gradient())     
                     
                    with torch.no_grad():
                                    
                     
                        if i>0:
                             
                            curr_S_k = (curr_vec_para - para_list_GPU_tensor[cached_id])
         
                             
    #                         if len(S_k_list) > m:
    #                             S_k_list.popleft()
                         
#                         gradient_full = (expect_gradients*batch_remaining_ids.shape[0] + gradient_remaining*curr_matched_ids_size)/(batch_remaining_ids.shape[0] + curr_matched_ids_size)
                        gradient_full = (expect_gradients*curr_rand_ids.shape[0] + gradient_remaining*curr_added_size)/(curr_rand_ids.shape[0] + curr_added_size)
                         
                        if i>0:
                             
                            curr_Y_k = (expect_gradients - grad_list_GPU_tensor[cached_id] + regularization_coeff*curr_S_k)
                             
                            dot_res = torch.mm(curr_S_k, torch.t(curr_Y_k))
                             
#                             print("curr_secont condition::", dot_res)
                            if dot_res > 0:
                                Y_k_list.append(curr_Y_k)
                                S_k_list.append(curr_S_k)
                                queue_id_list.append(i)
                                 
#                                 print("secont condition::", dot_res)
                                 
                                 
                                while len(Y_k_list) > m:
                                    Y_k_list.popleft()
                                    S_k_list.popleft()
                                    queue_id_list.popleft()
                                     
                                if len(Y_k_list) == m:
                                    use_standard_way = False
                                 
#                             print("explicit_evaluation epoch::", i)
                             
#                             print("batch size check::", curr_matched_ids_size + batch_remaining_ids.shape[0])
                         
                        exp_gradient = None
                        exp_param = None
                         
#                         exp_gradient = exp_gradient_list_all_epochs[i]
#     #               
#         
#                             
#                         exp_param = exp_para_list_all_epochs[i]
#                                   
#                         print("para_diff::")
#                         compute_model_para_diff(exp_param, para)
                         
                         
    #                     print("para_diff2::")
    #                     compute_model_para_diff(para_list_all_epochs[i], para)
                         
                         
#                         if i >= 115:
#                             print("here!!")
                         
                        para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_vec_para - learning_rate*expect_gradients, full_shape_list, shape_list)
                else:
#                     update_para_final(para, final_gradient_list, learning_rate, regularization_coeff, exp_gradient, exp_param)
                    vec_para = update_para_final2(curr_vec_para, final_gradient_list, learning_rate, regularization_coeff, None, None)
                    
                    
                    
                    para = get_devectorized_parameters(vec_para, full_shape_list, shape_list) 

                
                
                del final_gradient_list
            
                
            i = i + 1
            
            j += batch_size
            
            
            cached_id += 1
            
            if cached_id%cached_size == 0:
                
                GPU_tensor_end_id = (batch_id + 1)*cached_size
                
                if GPU_tensor_end_id > para_list_all_epoch_tensor.shape[0]:
                    GPU_tensor_end_id = para_list_all_epoch_tensor.shape[0] 
                print("end_tensor_id::", GPU_tensor_end_id)
                
                para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(para_list_all_epoch_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(grad_list_all_epoch_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                batch_id += 1
                
                cached_id = 0
            
            
#             jj += added_batch_size    
            
#         data_train_loader.batch_sampler.increm_ids()   
            
            
    print('overhead::', overhead)
    
    print('overhead2::', overhead2)
    
    print('overhead3::', overhead3)
    
            
    return para



def compute_predictions(X, para_list):
    
    output = X
    
    for para in para_list:
        output = torch.mm(para_list, output)
        
        output = sigmoid_func(output)
        
        
    return output
    


def sigmoid_func_derivitive(x):
    
    res = sigmoid_func(x)
    
    
    
    return (1-res)*res


    
def compute_gradient_diff(curr_gradient_list, gradient_list):
    
    print("gradient diff:")
    
    for i in range(len(curr_gradient_list)):
        
        print(curr_gradient_list[i])
        
        print(gradient_list[i])
        
        
        print("gradient_dff::", i, torch.norm(curr_gradient_list[i]-gradient_list[i]))



def compute_updated_parameters(old_delta, expected_delta, delta_gradient_list, old_para_list, para_list):
    
    
    new_delta = old_delta.clone()
    
    
    for i in range(len(old_para_list)):
        para_updates = para_list[i] - old_para_list[i]
        
        for j in range(20):
            
            for k in range(old_delta.shape[1]):
                new_delta[j][k] += torch.sum(delta_gradient_list[j][k][i]*para_updates)
            
    
    print("delta_diff::", torch.norm(new_delta[0:20]-expected_delta[0:20]))
    
        
            
        
    
    
    
    
    
   
def compute_model_parameter_iteration(num_epochs, model, X, Y, alpha, error, dim, num_class, input_dim, hidden_dims, output_dim, para_list_all_epochs, delta_gradient_all_epochs, delta_all_epochs):
    
    loss = np.infty
    
    count = 0
    
#     for para in init_para_list:
#         para.requires_grad = True
        
    
    
    
    
    '''hidden_dim len: depth - 1'''
        
    '''output_list len: depth'''
    
    depth = len(hidden_dims) + 1

    para_list = list(model.parameters())
    
    print("depth::", depth)
    
    print("para_len::", len(para_list))

    while count < num_epochs:    
        
        
        print("iteration::", count)
        
#         compute_model_para_diff2(para_lists_all_epochs[count], para_list)
        
        t1  = time.time()
        
        
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
        old_para_list = para_list_all_epochs[count]
        
        
        old_delta_list = delta_all_epochs[count]
        
        delta_gradient_list = delta_gradient_all_epochs[count]
        
        
        output_list,_ = model.get_output_each_layer(X)
        
        pred = output_list[len(output_list) - 1]
        
#         loss = error(pred, Y)
        
        delta = softmax_func(pred) - get_onehot_y(Y, dim, num_class)
        
        
        '''delta: n*output_dim'''
        
        para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 
        
        compute_updated_parameters(old_delta_list, delta, delta_gradient_list, old_para_list, para_list)

#         print(output_list[depth - 1].shape)
        t3 = time.time()
#         delta = delta*sigmoid_func_derivitive(torch.sum(para_curr_layer.view(1, output_dim, hidden_dims[depth - 2] + 1)*output_list[depth - 1].view(dim[0], 1, hidden_dims[depth - 2] + 1), 2))
        delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
        t4 = time.time()

        


#         deriv = torch.bmm(delta.view(dim[0], output_dim, 1), output_list[depth - 1].view(dim[0], 1, hidden_dims[depth - 2] + 1))
        
        deriv = torch.mm(torch.t(delta), output_list[depth - 1])

        '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
        
        
#         delta_para_prod = torch.sum(torch.t(para_list[2*depth - 2].data).view(1, hidden_dims[depth - 2], output_dim)*delta.view(dim[0], 1, output_dim), 2)

        delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(delta)))
        
        para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        
        para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
        
        para_list[2*depth - 1].data = para_curr_layer[:, -1]
        
#         this_gradient_list = [None]*len(para_list)
#         
#         this_gradient_list[2*depth - 2] = (deriv[:,0:-1]/dim[0])
#          
#         this_gradient_list[2*depth - 1] = (deriv[:,-1]/dim[0])
         
        print("iteration::", count)
         
#         print("diff1::", torch.norm(curr_para_list[2*depth - 2].data - para_list[2*depth - 2].data))
#          
#         print("diff2::", torch.norm(curr_para_list[2*depth - 1].data - para_list[2*depth - 1].data))
        
        for i in range(depth - 2):
            
            '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
            
            para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
            
            
#             delta = delta_para_prod*sigmoid_func_derivitive(torch.sum(para_curr_layer.view(1, hidden_dims[depth - 2 - i], hidden_dims[depth - 3 - i] + 1)*output_list[depth - 2 - i].view(dim[0], 1, hidden_dims[depth - 3 - i]), 2))
        
            delta = delta_para_prod*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
            
#             deriv = torch.bmm(delta.view(dim[0], hidden_dims[depth - 2 - i], 1), output_list[depth - 2 - i].view(dim[0], 1, hidden_dims[depth- 3 - i] + 1))



            deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])

#             delta_para_prod = torch.sum(torch.t(para_list[2*depth - 2*i - 4].data).view(1, hidden_dims[depth - 3 - i], hidden_dims[depth - 2 - i])*delta.view(dim[0], 1, hidden_dims[depth - 2 - i]), 2)
            
            
            delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))            
            
            
#             para_list[len(para_list) - 2 - i].data = para_list[len(para_list) - 2 - i].data - alpha/dim[0]*torch.sum(deriv, 0)
            
            
            '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
            para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        
            para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
            
            para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
            
            
#             this_gradient_list[2*depth - 2*i - 4] = (deriv[:,0:-1]/dim[0])
#           
#             this_gradient_list[2*depth - 2*i - 3] = (deriv[:,-1]/dim[0])
          
#             print("diff1::", torch.norm(curr_para_list[2*depth - 2*i - 4].data - para_list[2*depth - 2*i - 4].data))
#           
#             print("diff2::", torch.norm(curr_para_list[2*depth - 2*i - 3].data - para_list[2*depth - 2*i - 3].data))
        
        para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1)
            
                    
#         delta = delta_para_prod*sigmoid_func_derivitive(torch.sum(para_curr_layer.view(1, hidden_dims[0], input_dim + 1)*output_list[0].view(dim[0], 1, input_dim + 1), 2))
#         print(para_curr_layer.shape)
#         
#         print(output_list[0].shape)

        delta = delta_para_prod*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
        
#         deriv = torch.bmm(delta.view(dim[0], hidden_dims[0], 1), output_list[0].view(dim[0], 1, input_dim + 1))

        deriv = torch.mm(torch.t(delta), output_list[0])

#         delta_para_prod = torch.sum(torch.t(para_list[0].data).view(1, input_dim, hidden_dims[0])*delta.view(dim[0], 1, hidden_dims[0]), 2)            
        
        
#             para_list[len(para_list) - 2 - i].data = para_list[len(para_list) - 2 - i].data - alpha/dim[0]*torch.sum(deriv, 0)
        
        
        '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
        para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
    
        para_list[0].data = para_curr_layer[:, 0:-1]
        
        para_list[1].data = para_curr_layer[:, -1]
        
#         this_gradient_list[0] = (deriv[:,0:-1]/dim[0])
#          
#         this_gradient_list[1] = (deriv[:,-1]/dim[0])
        
        t2  = time.time()
        
        print("time0:", t2 - t1)
        
        print("time1:", t4 - t3)
        
        
#         compute_gradient_diff(curr_gradient_list, this_gradient_list)
#         print("diff1::", torch.norm(curr_para_list[0].data - para_list[0].data))
#          
#         print("diff2::", torch.norm(curr_para_list[1].data - para_list[1].data))
        
        
        
        
#         delta = delta_para_prod*sigmoid_func_derivitive(torch.sum(para_list[0].data.view(1, hidden_dims[0], input_dim)*X.view(dim[0], 1, input_dim), 2))
#     
#         deriv = torch.bmm(delta.view(dim[0], hidden_dims[0], 1), torch.transpose(X, 1, 2).view(dim[0], 1, input_dim))
#         
#         
#         para_list[0].data = para_list[0].data - alpha/dim[0]*torch.sum(deriv, 0)
        
        
        '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
        
#         delta_para_prod = torch.sum(torch.t(para_list[0].data).view(1, hidden_dims[0], input_dim)*delta.view(dim[0], 1, hidden_dims[0]), 2)
        
        
        count += 1
        
        
        
        
#         for i in range(len(para_list)):
#             print(para_list[i].data)
#             
#             print(para_list[i].data.shape)
        
        
    print("iteration::", num_epochs)
        
#     compute_model_para_diff2(expected_model_paras, para_list)    

def compute_x_sum_by_class(X, Y, num_class, dim, is_GPU, device):
    
#     x_sum_by_class = torch.zeros([num_class, dim[1]], dtype = torch.double)
    
#     if not is_GPU:
    y_onehot = torch.zeros([dim[0], num_class], dtype = torch.double)
        
#         one_tensor = torch.ones(1)
        
#     else:
#         y_onehot = torch.zeros([dim[0], num_class], dtype = torch.double, device = device)
        
#         one_tensor = torch.ones([1], device = device)

#     print(Y[0])

#     Y = Y.type(torch.LongTensor)

# In your for loop
#     y_onehot.zero_()
#     print(one_tensor)
    
#     print(Y[0])

    y_onehot.scatter_(1, Y.cpu().view(-1, 1), 1)
    
    
    x_sum_by_class = torch.mm(torch.t(y_onehot), X.cpu())
    
    
    del y_onehot
    
#     for i in range(num_class):
#         Y_mask = (Y == i)
#         
#         Y_mask = Y_mask.type(torch.DoubleTensor)
#         
#         x_sum_by_class[i] = torch.mm(torch.t(Y_mask), X) 
    
    
    if is_GPU:
        return x_sum_by_class.view(-1, 1).to(device)
    
    else:
        return x_sum_by_class.view(-1, 1)

def compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, expected_gradient, beta):     
     
    output_list,_ = model.get_output_each_layer(X)
    
    para_list = list(model.parameters())
    
    pred = output_list[len(output_list) - 1]
    delta = softmax_func(pred) - get_onehot_y(Y, X.shape, output_dim)
#     delta = softmax_func(pred) - compute_x_sum_by_class(X, Y, output_dim, X.shape)
    
    depth = len(hidden_dims) + 1
    
    para_curr_layer = torch.cat((para_list[2*depth - 2], para_list[2*depth - 1].view(-1,1)), 1) 
    
    
    expected_gradient_curr_layer = torch.cat((expected_gradient[2*depth - 2], expected_gradient[2*depth - 1].view(-1,1)), 1) 
    
    
    delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
    
    
    deriv_list = [None]*(len(para_list))
    
    deriv = torch.mm(torch.t(delta), output_list[depth - 1])
    
#     deriv_list[depth-1] = deriv
    
    deriv_list[2*depth - 2] = deriv[:, 0:-1]/X.shape[0] + 2*beta*para_list[2*depth - 2]
            
    deriv_list[2*depth - 1] = deriv[:, -1]/X.shape[0] + 2*beta*para_list[2*depth - 1]
    
    
    
    
    delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2]),torch.t(delta)))

    for i in range(depth - 2):
        
        '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
        
        para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4], para_list[2*depth - 2*i - 3].view(-1,1)), 1)
            
#             delta = delta_para_prod*sigmoid_func_derivitive(torch.sum(para_curr_layer.view(1, hidden_dims[depth - 2 - i], hidden_dims[depth - 3 - i] + 1)*output_list[depth - 2 - i].view(dim[0], 1, hidden_dims[depth - 3 - i]), 2))
    
        delta = delta_para_prod*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
        
#             deriv = torch.bmm(delta.view(dim[0], hidden_dims[depth - 2 - i], 1), output_list[depth - 2 - i].view(dim[0], 1, hidden_dims[depth- 3 - i] + 1))



        deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])

#             delta_para_prod = torch.sum(torch.t(para_list[2*depth - 2*i - 4].data).view(1, hidden_dims[depth - 3 - i], hidden_dims[depth - 2 - i])*delta.view(dim[0], 1, hidden_dims[depth - 2 - i]), 2)
#         deriv_list[depth-2-i] = deriv
        
        deriv_list[2*depth - 2*i - 4] = deriv[:, 0:-1]/X.shape[0] + 2*beta*para_list[2*depth - 2*i - 4]
            
        deriv_list[2*depth - 2*i - 3] = deriv[:, -1]/X.shape[0] + 2*beta*para_list[2*depth - 2*i - 3]
        
        delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4]), torch.t(delta)))            
        
    para_curr_layer = torch.cat((para_list[0], para_list[1].view(-1,1)), 1)
    
    expected_gradient_curr_layer = torch.cat((expected_gradient[0], expected_gradient[1].view(-1,1)), 1) 

#         delta = delta_para_prod*sigmoid_func_derivitive(torch.sum(para_curr_layer.view(1, hidden_dims[0], input_dim + 1)*output_list[0].view(dim[0], 1, input_dim + 1), 2))
#         print(para_curr_layer.shape)
#         
#         print(output_list[0].shape)

    delta = delta_para_prod*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
    
#         deriv = torch.bmm(delta.view(dim[0], hidden_dims[0], 1), output_list[0].view(dim[0], 1, input_dim + 1))

    deriv = torch.mm(torch.t(delta), output_list[0])

#     deriv_list[0] = deriv

    deriv_list[0] = deriv[:, 0:-1]/X.shape[0] + 2*beta*para_list[0]
            
    deriv_list[1] = deriv[:, -1]/X.shape[0] + 2*beta*para_list[1]
    
    return deriv_list, para_list
#             para_list[len(para_list) - 2 - i].data = para_list[len(para_list) - 2 - i].data - alpha/dim[0]*torch.sum(deriv, 0)
    

     
def compute_model_derivitive_iteration(X, Y, alpha, error, dim, num_class, input_dim, hidden_dims, output_dim, expected_model_paras, gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, init_para_list):
    
    loss = np.infty
    
    count = 0


    para_list = []
    
    for para in init_para_list:
        
        para_list.append(para.data.clone())
        
        
#         para.requires_grad = True
        
    
    
    
    
    '''hidden_dim len: depth - 1'''
        
    '''output_list len: depth'''
    
    depth = len(hidden_dims) + 1

    print("depth::", depth)
    
    while count < num_epochs:    
        
        
        print("iteration::", count)
        
#         compute_model_para_diff2(para_lists_all_epochs[count], para_list)
        
        t1  = time.time()
        
        
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
        
        
        output_list = output_list_all_epochs[count]
        
        
#         para_list = para_list_all_epochs[count]
        
        pred = output_list[len(output_list) - 1]
        
        
        gradient_list = gradient_list_all_epochs[count]
        
#         loss = error(pred, Y)
        
        delta = softmax_func(pred) - get_onehot_y(Y, dim, num_class)
        
        
        expected_para_list = para_list_all_epochs[count]
        
        
        '''delta: n*output_dim'''
        
        para_curr_layer = torch.cat((para_list[2*depth - 2], para_list[2*depth - 1].view(-1,1)), 1) 
        
        curr_expected_para = torch.cat((expected_para_list[2*depth - 2], expected_para_list[2*depth - 1].view(-1,1)), 1)
        
        print(torch.norm(curr_expected_para - para_curr_layer))
        
#         print(output_list[depth - 1].shape)
        t3 = time.time()
#         delta = delta*sigmoid_func_derivitive(torch.sum(para_curr_layer.view(1, output_dim, hidden_dims[depth - 2] + 1)*output_list[depth - 1].view(dim[0], 1, hidden_dims[depth - 2] + 1), 2))
        delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
        t4 = time.time()

#         deriv = torch.bmm(delta.view(dim[0], output_dim, 1), output_list[depth - 1].view(dim[0], 1, hidden_dims[depth - 2] + 1))
        
        deriv = torch.mm(torch.t(delta), output_list[depth - 1])
        
        
        
        
        

        '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
        
        
#         delta_para_prod = torch.sum(torch.t(para_list[2*depth - 2].data).view(1, hidden_dims[depth - 2], output_dim)*delta.view(dim[0], 1, output_dim), 2)

        delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2]),torch.t(delta)))
        
        
        gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1)

      
        gradient_delta = torch.norm(gradient_curr_layer - deriv/dim[0])
        
        print("gradient delta::", gradient_delta)
        
        
        para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        
        para_list[2*depth - 2] = para_curr_layer[:, 0:-1]
        
        para_list[2*depth - 1] = para_curr_layer[:, -1]
        
#         this_gradient_list = [None]*len(para_list)
#         
#         this_gradient_list[2*depth - 2] = (deriv[:,0:-1]/dim[0])
#          
#         this_gradient_list[2*depth - 1] = (deriv[:,-1]/dim[0])
         
        print("iteration::", count)
         
#         print("diff1::", torch.norm(curr_para_list[2*depth - 2].data - para_list[2*depth - 2].data))
#          
#         print("diff2::", torch.norm(curr_para_list[2*depth - 1].data - para_list[2*depth - 1].data))
        
        for i in range(depth - 2):
            
            '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
            
            para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4], para_list[2*depth - 2*i - 3].view(-1,1)), 1)
            
            curr_expected_para = torch.cat((expected_para_list[2*depth - 2*i - 4], expected_para_list[2*depth - 2*i - 3].view(-1,1)), 1)
        
            print(torch.norm(curr_expected_para - para_curr_layer))
#             delta = delta_para_prod*sigmoid_func_derivitive(torch.sum(para_curr_layer.view(1, hidden_dims[depth - 2 - i], hidden_dims[depth - 3 - i] + 1)*output_list[depth - 2 - i].view(dim[0], 1, hidden_dims[depth - 3 - i]), 2))
        
            delta = delta_para_prod*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
            
#             deriv = torch.bmm(delta.view(dim[0], hidden_dims[depth - 2 - i], 1), output_list[depth - 2 - i].view(dim[0], 1, hidden_dims[depth- 3 - i] + 1))



            deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])


            gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1)

      
            gradient_delta = torch.norm(gradient_curr_layer - deriv/dim[0])
            
            print("gradient delta::", gradient_delta)


#             delta_para_prod = torch.sum(torch.t(para_list[2*depth - 2*i - 4].data).view(1, hidden_dims[depth - 3 - i], hidden_dims[depth - 2 - i])*delta.view(dim[0], 1, hidden_dims[depth - 2 - i]), 2)
            
            
            delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4]), torch.t(delta)))            
            
            
#             para_list[len(para_list) - 2 - i].data = para_list[len(para_list) - 2 - i].data - alpha/dim[0]*torch.sum(deriv, 0)
            
            
            '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
            para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        
            para_list[2*depth - 2*i - 4] = para_curr_layer[:, 0:-1]
            
            para_list[2*depth - 2*i - 3] = para_curr_layer[:, -1]
            
            
#             this_gradient_list[2*depth - 2*i - 4] = (deriv[:,0:-1]/dim[0])
#           
#             this_gradient_list[2*depth - 2*i - 3] = (deriv[:,-1]/dim[0])
          
#             print("diff1::", torch.norm(curr_para_list[2*depth - 2*i - 4].data - para_list[2*depth - 2*i - 4].data))
#           
#             print("diff2::", torch.norm(curr_para_list[2*depth - 2*i - 3].data - para_list[2*depth - 2*i - 3].data))
        
        para_curr_layer = torch.cat((para_list[0], para_list[1].view(-1,1)), 1)
        
        curr_expected_para = torch.cat((expected_para_list[0], expected_para_list[1].view(-1,1)), 1)
        
        print(torch.norm(curr_expected_para - para_curr_layer))

        
                    
#         delta = delta_para_prod*sigmoid_func_derivitive(torch.sum(para_curr_layer.view(1, hidden_dims[0], input_dim + 1)*output_list[0].view(dim[0], 1, input_dim + 1), 2))
#         print(para_curr_layer.shape)
#         
#         print(output_list[0].shape)

        delta = delta_para_prod*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
        
#         deriv = torch.bmm(delta.view(dim[0], hidden_dims[0], 1), output_list[0].view(dim[0], 1, input_dim + 1))

        deriv = torch.mm(torch.t(delta), output_list[0])


        gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)), 1)

      
        gradient_delta = torch.norm(gradient_curr_layer - deriv/dim[0])
        
        print("gradient delta::", gradient_delta)


#         delta_para_prod = torch.sum(torch.t(para_list[0].data).view(1, input_dim, hidden_dims[0])*delta.view(dim[0], 1, hidden_dims[0]), 2)            
        
        
#             para_list[len(para_list) - 2 - i].data = para_list[len(para_list) - 2 - i].data - alpha/dim[0]*torch.sum(deriv, 0)
        
        
        '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
        para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
    
        para_list[0] = para_curr_layer[:, 0:-1]
        
        para_list[1] = para_curr_layer[:, -1]
        
#         this_gradient_list[0] = (deriv[:,0:-1]/dim[0])
#          
#         this_gradient_list[1] = (deriv[:,-1]/dim[0])
        
        t2  = time.time()
        
        print("time0:", t2 - t1)
        
        print("time1:", t4 - t3)
        
        
#         compute_gradient_diff(curr_gradient_list, this_gradient_list)
#         print("diff1::", torch.norm(curr_para_list[0].data - para_list[0].data))
#          
#         print("diff2::", torch.norm(curr_para_list[1].data - para_list[1].data))
        
        
        
        
#         delta = delta_para_prod*sigmoid_func_derivitive(torch.sum(para_list[0].data.view(1, hidden_dims[0], input_dim)*X.view(dim[0], 1, input_dim), 2))
#     
#         deriv = torch.bmm(delta.view(dim[0], hidden_dims[0], 1), torch.transpose(X, 1, 2).view(dim[0], 1, input_dim))
#         
#         
#         para_list[0].data = para_list[0].data - alpha/dim[0]*torch.sum(deriv, 0)
        
        
        '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
        
#         delta_para_prod = torch.sum(torch.t(para_list[0].data).view(1, hidden_dims[0], input_dim)*delta.view(dim[0], 1, hidden_dims[0]), 2)
        
        
        count += 1
        
        
        
        
#         for i in range(len(para_list)):
#             print(para_list[i].data)
#             
#             print(para_list[i].data.shape)
        
        
    print("iteration::", num_epochs)
        
    compute_model_para_diff2(expected_model_paras, para_list)    
   


# def capture_provenance(X, Y, w_list_all_epochs, b_list_all_epochs, output_list_all_epochs, para_list_all_epochs, dim, num_class, input_dim, hidden_dims, output_dim, gradient_list_all_epochs):
#     
# #     t1  = time.time()
# #         curr_gradient_list = gradient_lists_all_epochs[count]
#         
#     A_list_all_epochs = []
#     
#     B_list_all_epochs = []
#     
#     w_delta_prod_list_all_epochs = []
#     
#     b_delta_prod_list_all_epochs = []
#     
#     with torch.no_grad():
#         
#         for k in range(len(w_list_all_epochs)):
#             w_res = w_list_all_epochs[k]
#             
#             b_res = b_list_all_epochs[k]
#             
#             
#             
#             output_list = output_list_all_epochs[k]
#             
#             pred = output_list[len(output_list) - 1]
#             
#             para_list = para_list_all_epochs[k]
#         
#         
#             depth = len(hidden_dims) + 1
#         
#             A_list = [None]*depth
#         
#         
#             B_list = [None]*depth
#             
#             
#         #     A0_list = [None]*depth
#         #     
#         #     B0_list = [None]*depth
#         #         loss = error(pred, Y)
#             
#             delta = Variable(softmax_func(pred) - get_onehot_y(Y, dim, num_class))
#             
#             
#             '''delta: n*output_dim'''
#             
#             para_curr_layer = Variable(torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1)) 
#             
#             
#             '''A: output_dim, hidden_dim[depth-2]^2'''
#             
#         #     print(len(w_res))
#             
#         #     print(w_res[depth - 1])
#             
#         #     print(depth)
#             
#             
#             w_delta_prod_list = [None]*depth
#             
#             b_delta_prod_list = [None]*depth
#             
#             
#             '''w_delta_prod: n*output_dim'''
#             
#             w_delta_prod = Variable(w_res[depth - 1]*delta)
#             
#             b_delta_prod = Variable(b_res[depth - 1]*delta)
#             
#             w_delta_prod_list[depth - 1] = w_delta_prod
#             
#             b_delta_prod_list[depth - 1] = b_delta_prod
#             
#             '''A: output_dim*(hidden_dims[-1]*hiddem_dims[-1])'''
#             A = Variable(torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(dim[0], hidden_dims[depth-2] + 1, 1), output_list[depth - 1].view(dim[0], 1, hidden_dims[depth-2]+ 1)).view(dim[0], -1)))
#             
#         #     A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1])
#             
#         #     A0_list.append(A0)
#             
#             '''B: output_dim, hidden_dim[depth-2]'''
#             B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-1]))
#             
#         #     B0 = torch.t(torch.sum(b_delta_prod, 0))
#             
#         #     B0_list.append(B0)
#             
#             
#             
#             A_list[depth - 1] = A
#             
#             B_list[depth - 1] = B
#             
#             
#             
#             w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
#             
#             
#             entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
#             
#             
#         #     delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
#         #     
#         #     deriv = torch.mm(torch.t(delta), output_list[depth - 1])
#         
#             '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
#         
#             delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(entire_delta_expression))))
#             
#         #     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
#         #     
#         #     para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
#         #     
#         #     para_list[2*depth - 1].data = para_curr_layer[:, -1]
#                 
#             for i in range(depth - 2):
#                 
#                 '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
#                 
#                 para_curr_layer = Variable(torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1))
#                 
#                 delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
#                         
#                 w_delta_prod = Variable(w_res[depth - i - 2]*delta)
#             
#                 b_delta_prod = Variable(b_res[depth - i - 2]*delta)
#                 
#                 w_delta_prod_list[depth - i - 2] = w_delta_prod
#             
#                 b_delta_prod_list[depth - i - 2] = b_delta_prod
#                 
#                 A = Variable(torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - i - 2].view(dim[0], hidden_dims[depth - i - 3]+ 1, 1), output_list[depth - i - 2].view(dim[0], 1, hidden_dims[depth - i - 3]+ 1)).view(dim[0], -1)))
#                 
#         #         A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - i - 2])
#                 
#         #         A0_list.append(A0)
#                 
#                 '''B: output_dim, hidden_dim[depth-2]'''
#                 B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-i-2]))
#                 
#         #         B0 = torch.t(torch.sum(b_delta_prod, 0))
#                 
#         #         B0_list.append(B0)
#                 
#                 w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - i - 2]))))
#                 
#                 
#                 entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
#                 
#                 
#                 delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(entire_delta_expression))))
#                 
#                 A_list[depth - 2 - i] = A
#                 
#                 B_list[depth - 2 - i] = B
#                 
#         #         deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])
#         #         
#         #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))                    
#         #         
#         #         '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#         #         para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
#         #     
#         #         para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
#         #         
#         #         para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
#                     
#             para_curr_layer = Variable(torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1))
#         
#             delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
#             
#             w_delta_prod = Variable(w_res[0]*delta)
#         
#             b_delta_prod = Variable(b_res[0]*delta)
#             
#             w_delta_prod_list[0] = w_delta_prod
#             
#             b_delta_prod_list[0] = b_delta_prod
#             
#             
#             w_delta_prod_list_all_epochs.append(w_delta_prod_list)
#             
#             b_delta_prod_list_all_epochs.append(b_delta_prod_list)
#             
#         #     A = torch.mm(torch.bmm(w_delta_prod.view(dim[0], )))
#             
#         #     A = torch.zeros([hidden_dims[0], (input_dim+1)*(input_dim+1)])
#             
#             
#             A = Variable(torch.mm(torch.t(output_list[0]), torch.bmm(w_delta_prod.view(dim[0], hidden_dims[0], 1), output_list[0].view(dim[0], 1, input_dim + 1)).view(dim[0], hidden_dims[0]*(input_dim + 1))))
#             
#             A = Variable(torch.transpose(A.view(input_dim + 1, hidden_dims[0], input_dim + 1), 0, 1))
#             
#         #     for k in range(dim[0]):
#         #         A += torch.mm(torch.t(w_delta_prod)[:,k].view(-1,1), torch.mm(output_list[0][k].view(input_dim + 1, 1), output_list[0][k].view(1, input_dim + 1)).view(1,-1))
#             
#         #     A = torch.mm( temp)
#             
#             '''B: output_dim, hidden_dim[depth-2]'''
#             B = Variable(torch.mm(torch.t(b_delta_prod), output_list[0]))
#             
#             
#             A_list[0] = A
#             
#             B_list[0] = B
#         
#             A_list_all_epochs.append(A_list)
#             
#             B_list_all_epochs.append(B_list)
#     
# #     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[0])))
# #     
# #     
# #     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
#     
# #     deriv = torch.mm(torch.t(delta), output_list[0])
# #     
# #     '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
# #     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
# # 
# #     para_list[0].data = para_curr_layer[:, 0:-1]
# #     
# #     para_list[1].data = para_curr_layer[:, -1]
# 
# #     t2  = time.time()
#     
# #     print("time0:", t2 - t1)   
#     
#     
#     return A_list_all_epochs, B_list_all_epochs, w_delta_prod_list_all_epochs, b_delta_prod_list_all_epochs
#     
#     
#     
#     
# #     for param in param_list:
# #         
# #         delta =  
        
    
    
def capture_provenance2(X, Y, w_list, b_list, output_list, para_list, dim, num_class, input_dim, hidden_dims, output_dim, gradient_list):
     
#     t1  = time.time()
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
#     hessian_matrix = compute_hessian_matrix(model, para_list, gradient_list, input_dim, hidden_dims, output_dim)    
        
    
#     A_list_all_epochs = []
#      
#     B_list_all_epochs = []
#      
#     w_delta_prod_list_all_epochs = []
#      
#     b_delta_prod_list_all_epochs = []
     
     
    depth = len(hidden_dims) + 1
    
    w_delta_prod_list = [None]*depth
     
    b_delta_prod_list = [None]*depth
     
    with torch.no_grad():
         
#         for k in range(len(w_list_all_epochs)):
        w_res = w_list
         
        b_res = b_list
         
         
         
#         output_list = output_list_all_epochs[k]
         
        pred = output_list[len(output_list) - 1]
         
#         para_list = para_list_all_epochs[k]
     
     
        A_list = [None]*depth
     
     
        B_list = [None]*depth
         
         
    #     A0_list = [None]*depth
    #     
    #     B0_list = [None]*depth
    #         loss = error(pred, Y)
         
        delta = Variable(softmax_func(pred) - get_onehot_y(Y, dim, num_class))
         
         
        '''delta: n*output_dim'''
         
        para_curr_layer = Variable(torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1)) 
         
         
        '''A: output_dim, hidden_dim[depth-2]^2'''
         
    #     print(len(w_res))
         
    #     print(w_res[depth - 1])
         
    #     print(depth)
         
         
         
        '''w_delta_prod: n*output_dim'''
         
        w_delta_prod = Variable(w_res[depth - 1]*delta)
         
        b_delta_prod = Variable(b_res[depth - 1]*delta)
         
        w_delta_prod_list[depth - 1] = w_delta_prod
         
        b_delta_prod_list[depth - 1] = b_delta_prod
         
        '''A: output_dim*(hidden_dims[-1]*hiddem_dims[-1])'''
#         A = Variable(torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(dim[0], hidden_dims[depth-2] + 1, 1), output_list[depth - 1].view(dim[0], 1, hidden_dims[depth-2]+ 1)).view(dim[0], -1)))
         
    #     A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1])
         
    #     A0_list.append(A0)
         
        '''B: output_dim, hidden_dim[depth-2]'''
#         B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-1]))
         
    #     B0 = torch.t(torch.sum(b_delta_prod, 0))
         
    #     B0_list.append(B0)
         
         
         
#         A_list[depth - 1] = A
#          
#         B_list[depth - 1] = B
         
         
         
        w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
         
         
        entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
         
         
    #     delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
    #     
    #     deriv = torch.mm(torch.t(delta), output_list[depth - 1])
     
        '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
     
        delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(entire_delta_expression))))
         
    #     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
    #     
    #     para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
    #     
    #     para_list[2*depth - 1].data = para_curr_layer[:, -1]
             
        for i in range(depth - 2):
             
            '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
             
            para_curr_layer = Variable(torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1))
             
            delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
                     
            w_delta_prod = Variable(w_res[depth - i - 2]*delta)
         
            b_delta_prod = Variable(b_res[depth - i - 2]*delta)
             
            w_delta_prod_list[depth - i - 2] = w_delta_prod
         
            b_delta_prod_list[depth - i - 2] = b_delta_prod
             
            A = Variable(torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - i - 2].view(dim[0], hidden_dims[depth - i - 3]+ 1, 1), output_list[depth - i - 2].view(dim[0], 1, hidden_dims[depth - i - 3]+ 1)).view(dim[0], -1)))
             
    #         A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - i - 2])
             
    #         A0_list.append(A0)
             
            '''B: output_dim, hidden_dim[depth-2]'''
            B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-i-2]))
             
    #         B0 = torch.t(torch.sum(b_delta_prod, 0))
             
    #         B0_list.append(B0)
             
            w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - i - 2]))))
             
             
            entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
             
             
            delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(entire_delta_expression))))
             
            A_list[depth - 2 - i] = A
             
            B_list[depth - 2 - i] = B
             
    #         deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])
    #         
    #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))                    
    #         
    #         '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
    #         para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
    #     
    #         para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
    #         
    #         para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
                 
        para_curr_layer = Variable(torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1))
     
        delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
         
        w_delta_prod = Variable(w_res[0]*delta)
     
        b_delta_prod = Variable(b_res[0]*delta)
         
        w_delta_prod_list[0] = w_delta_prod
         
        b_delta_prod_list[0] = b_delta_prod
         
    
    
    return w_delta_prod_list, b_delta_prod_list
       
#         w_delta_prod_list_all_epochs.append(w_delta_prod_list)
         
#         b_delta_prod_list_all_epochs.append(b_delta_prod_list)
         
    #     A = torch.mm(torch.bmm(w_delta_prod.view(dim[0], )))
         
    #     A = torch.zeros([hidden_dims[0], (input_dim+1)*(input_dim+1)])
         
         
#         A = Variable(torch.mm(torch.t(output_list[0]), torch.bmm(w_delta_prod.view(dim[0], hidden_dims[0], 1), output_list[0].view(dim[0], 1, input_dim + 1)).view(dim[0], hidden_dims[0]*(input_dim + 1))))
#          
#         A = Variable(torch.transpose(A.view(input_dim + 1, hidden_dims[0], input_dim + 1), 0, 1))
#          
#     #     for k in range(dim[0]):
#     #         A += torch.mm(torch.t(w_delta_prod)[:,k].view(-1,1), torch.mm(output_list[0][k].view(input_dim + 1, 1), output_list[0][k].view(1, input_dim + 1)).view(1,-1))
#          
#     #     A = torch.mm( temp)
#          
#         '''B: output_dim, hidden_dim[depth-2]'''
#         B = Variable(torch.mm(torch.t(b_delta_prod), output_list[0]))
#          
#          
#         A_list[0] = A
#          
#         B_list[0] = B
#      
#         A_list_all_epochs.append(A_list)
#          
#         B_list_all_epochs.append(B_list)
     
#     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[0])))
#     
#     
#     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
     
#     deriv = torch.mm(torch.t(delta), output_list[0])
#     
#     '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
# 
#     para_list[0].data = para_curr_layer[:, 0:-1]
#     
#     para_list[1].data = para_curr_layer[:, -1]
 
#     t2  = time.time()
     
#     print("time0:", t2 - t1)   
     
     
     
#     for param in param_list:
#         
#         delta =  


def compute_linearized_coefficieent_each_epoch(depth, para_list, output_list, Pi):
    w_list = [None]*depth
            
    b_list = [None]*depth
            
        #     for m in model.children():
    for i in range(depth):
        
        
#         if type(m) == nn.modules.activation.Sigmoid: 
        
#             curr_weight = para_list[2*i].view(1, para_list[2*i].shape[0], para_list[2*i].shape[1])

            '''para_curr_layer:: hidden_dims[i], hidden_dims[i-1]'''
            
            para_curr_layer = torch.cat((para_list[2*depth - 2*i - 2].data, para_list[2*depth - 2*i - 1].data.view(-1,1)), 1)
            
#             curr_weight = para_list[2*i]
            
#             curr_input = output_list[i].view(output_list[i].shape[0], 1, output_list[i].shape[1])
            '''curr_input:: n, hidden_dims[i-1]'''
            curr_input = Variable(output_list[depth - i - 1])
            
            
#             print(para_curr_layer.shape)
            
#             print(curr_input.shape)
            
#             curr_offset = para_list[2*i+1]
            
#             print(curr_offset.shape)
            
#             print(torch.sum(curr_weight*curr_input, 2).shape)
            '''w_paras: n, hidden_dims[i]'''
            w_paras, b_paras = Pi.piecewise_linear_interpolate_coeff_batch2(torch.t(torch.mm(para_curr_layer, torch.t(curr_input))))
        
        
            w_list[depth - i - 1] = w_paras
            
            b_list[depth - i - 1] = b_paras
            
    return w_list, b_list


def compute_linearized_coeffcient_single_epoch(X, input_dim, hidden_dims, output_dim, output_list, para_list):
    Pi = create_piecewise_linea_class(sigmoid_diff_function)
        
        
    depth = len(hidden_dims) + 1
    
    
    return compute_linearized_coefficieent_each_epoch(depth, para_list, output_list, Pi)
    


def compute_linearized_coefficient(X, input_dim, hidden_dims, output_dim, output_list_all_epochs, para_list_all_epochs):
    
    
    w_list_all_epochs = []
    
    b_list_all_epochs = []


    Pi = create_piecewise_linea_class(sigmoid_diff_function)
        
        
    depth = len(hidden_dims) + 1

    with torch.no_grad():
    
        for k in range(len(output_list_all_epochs)):
            
        
        
            para_list = para_list_all_epochs[k]#list(model.parameters())
            
            
            output_list = output_list_all_epochs[k]#model.get_output_each_layer(X)
            

            w_list, b_list = compute_linearized_coefficieent_each_epoch(depth, para_list, output_list, Pi)
            
            w_list_all_epochs.append(w_list)
            
            b_list_all_epochs.append(b_list)
#                 i += 1
            
            
            
    return w_list_all_epochs, b_list_all_epochs
            
#         print(type(m))
# 
#         if t

#         print(param.grad)
        
        
#         for grd in param.grad:
#             for single_grad in grd:
#                 print(single_grad)
#                 
#                 model.zero_grad()
#                 
#                 single_grad.backward()
#                 
#                 print(single_grad.grad)


def compute_hessian_matrix(model, gradient_list, input_dim, hidden_dims, output_dim):
    
    
    total_attr_num = 0
    
    para_list = list(model.parameters())
    
    for k in range(len(hidden_dims) -1):
        total_attr_num += hidden_dims[k] * hidden_dims[k+1]
    
    total_attr_num += hidden_dims[0]*(input_dim + 1)
    
    total_attr_num += output_dim*(hidden_dims[-1]+1)
    
    print("attribute num::", total_attr_num)
    
    hessian_matrix = torch.zeros([total_attr_num, total_attr_num], dtype = torch.float64)
    
    print("init_tensor_done!!")
    
    clear_gradients(para_list)
    
    
    i = 0
    
    for k in range(len(gradient_list)):
        
        curr_gradient = gradient_list[k]
        
        
        
        for m in range(curr_gradient.shape[0]):
            
            if(len(curr_gradient.shape) > 1):
            
                for l in range(curr_gradient.shape[1]):
                
#                     print(i)
                
                    curr_gradient[m][l].backward(retain_graph = True)
                    
                    vectorized_gradient = get_all_vectorized_gradients(para_list)
                    
                    clear_gradients(para_list)
                    
                    hessian_matrix[:,i] = vectorized_gradient
                
                    i = i + 1
        
    
            else:
#                 print(i)
                
                curr_gradient[m].backward(retain_graph = True)
                
                vectorized_gradient = get_all_vectorized_gradients(para_list)
                
                clear_gradients(para_list)
                
                hessian_matrix[:,i] = vectorized_gradient
            
                i = i + 1
    
    return hessian_matrix
    

# def compute_near_hessian_matrix(model, gradient_list, input_dim, hidden_dims, output_dim):
#     
#     
#     total_attr_num = 0
#     
#     para_list = list(model.parameters())
#     
#     for k in range(len(hidden_dims) -1):
#         total_attr_num += hidden_dims[k] * hidden_dims[k+1]
#     
#     total_attr_num += hidden_dims[0]*(input_dim + 1)
#     
#     total_attr_num += output_dim*(hidden_dims[-1]+1)
#     
#     print("attribute num::", total_attr_num)
#     
#     hessian_matrix = torch.zeros([total_attr_num, total_attr_num], dtype = torch.float32)
#     
#     print("init_tensor_done!!")
#     
#     clear_gradients(para_list)
#     
#     
#     i = 0
#     
#     for k in range(len(gradient_list)):
#         
#         curr_gradient = gradient_list[k]
#         
#         
#         
#         for m in range(curr_gradient.shape[0]):
#             
#             if(len(curr_gradient.shape) > 1):
#             
#                 for l in range(curr_gradient.shape[1]):
#                 
#                     print(i)
#                 
#                     curr_gradient[m][l].backward(retain_graph = True)
#                     
#                     vectorized_gradient = get_all_vectorized_gradients(para_list)
#                     
#                     clear_gradients(para_list)
#                     
#                     hessian_matrix[:,i] = vectorized_gradient
#                 
#                     i = i + 1
#         
#     
#             else:
#                 print(i)
#                 
#                 curr_gradient[m].backward(retain_graph = True)
#                 
#                 vectorized_gradient = get_all_vectorized_gradients(para_list)
#                 
#                 clear_gradients(para_list)
#                 
#                 hessian_matrix[:,i] = vectorized_gradient
#             
#                 i = i + 1
#     
#     return hessian_matrix

def compute_loss(model, error, X, Y, regularization_coeff):
    
#     train = Variable(X)
    labels = Y.view(-1)
    
    # Clear gradients
#         optimizer.zero_grad()
    
    # Forward propagation
    outputs = model.forward(X)
    
    # Calculate softmax and ross entropy loss
    
#         print(outputs)
#         
#         print(labels)
    
    labels = labels.type(torch.LongTensor)
    
    loss = error(outputs, labels)
    
    loss += regularization_coeff*get_regularization_term(model.parameters())
    
    return loss

def compute_loss2(model, error, X, Y, regularization_coeff, vecorized_paras):
    
#     train = Variable(X)
    labels = Y.view(-1)
    
    # Clear gradients
#         optimizer.zero_grad()
    
    # Forward propagation
    outputs = model.forward(X)
    
    para_square2 = torch.sum(torch.pow(get_all_vectorized_parameters_with_gradient(model.parameters()), 2))
    
    # Calculate softmax and ross entropy loss
    
#         print(outputs)
#         
#         print(labels)
    
    labels = labels.type(torch.LongTensor)
    
    loss = error(outputs, labels)
    
    loss += regularization_coeff*para_square2#get_regularization_term(model.parameters())
    
    return loss


def compute_delta_constant(X, Y, para1, para2, gradient1, hessian_para_prod, error, model, beta):
    
    init_model(model, para1)
    
    loss1 = compute_loss(model, error, X, Y, beta)
    
    init_model(model, para2)
    
    loss2 = compute_loss(model, error, X, Y, beta)
    
    para_diff = (get_all_vectorized_parameters(para2) - get_all_vectorized_parameters(para1))
    
    delta_loss = loss2 - loss1 - torch.mm(gradient1.view(1,-1), para_diff.view(-1,1)) - 0.5*torch.mm(para_diff.view(1,-1), hessian_para_prod)
    
    epsilon = delta_loss/(torch.mm(para_diff.view(1,-1), para_diff.view(-1,1)))
    
    return epsilon
    
    
    
def compute_derivative_one_more_step(model, batch_X, batch_Y, criterion, optimizer):
    
    
    optimizer.zero_grad()

#             batch_X = dataset_train.data.data[curr_matched_ids]
#             
#             batch_Y = dataset_train.data.targets[curr_matched_ids]
    
    output = model(batch_X)

    loss = criterion(output, batch_Y)
    
    loss.backward()
    
    
    return loss
    
def compute_derivative_one_more_step_skipnet(model, batch_X, batch_Y, criterion, optimizer, all_ids_list, is_GPU, device):
    
    
    if is_GPU:
        for i in range(len(all_ids_list)):
            all_ids_list[i] = all_ids_list[i].to(device)
            
    
    
    optimizer.zero_grad()

    output = model.forward_with_known_dropout(batch_X, [0], all_ids_list, False)

    loss = criterion(output, batch_Y)
    
#     for i in range(len(all_ids_list)):
#         all_ids_list[i]
    
    
#     torch.save(output, git_ignore_folder + 'tmp_output')
#     
#     torch.save(batch_X, git_ignore_folder + 'tmp_batch_X')
#     
#     torch.save(batch_Y, git_ignore_folder + 'tmp_batch_Y')
#     
#     torch.save(all_ids_list, git_ignore_folder + 'tmp_ids_list')
    
    loss.backward()
    
    return loss    
    
    

def compute_derivative_one_more_step2(model, error, X, Y, beta, vecorized_paras):
    
    
    loss = compute_loss2(model, error, X, Y, beta, vecorized_paras)
    
#     train = Variable(X)
#     labels = Variable(Y.view(-1))
#     
#     # Clear gradients
# #         optimizer.zero_grad()
#     
#     # Forward propagation
#     outputs = model(train)
#     
#     # Calculate softmax and ross entropy loss
#     
# #         print(outputs)
# #         
# #         print(labels)
#     
#     labels = labels.type(torch.LongTensor)
    
#     loss = error(outputs, labels)
    
    
#         print("loss0::", loss)
# 
#         loss = loss_function2(outputs, labels, X.shape)
    
    # Calculating gradients
#         loss.backward(retain_graph = True, create_graph=True)
    
    loss.backward()
    
    return loss



def add_small_variation(para_list):
    
    res_para_list = []
    
    
    for i in range(len(para_list)):
        res_para_list.append(para_list[i] + 0.00001)

    return res_para_list

def verify_hessian_matrix(hessian_mat, para_list, model, input_dim, hidden_dims, output_dim, X, Y, gradient_list):
    
    
    updated_para_list = add_small_variation(para_list)
    
    
    init_model(model, updated_para_list)
    
    udpated_gradient_list, _ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, gradient_list)
    
    expected_gradient_delta = torch.mm(hessian_mat, torch.t(get_all_vectorized_parameters(updated_para_list) - get_all_vectorized_parameters(para_list)))/X.shape[0]
    
    real_gradient_delta = get_all_vectorized_parameters(udpated_gradient_list)/X.shape[0] - get_all_vectorized_parameters(gradient_list)


    print(expected_gradient_delta - real_gradient_delta)
    
    
    
    
    

def compute_first_derivative(model, X, Y, error, beta):
    
    outputs = model(X)
    
    labels = Y.view(-1).type(torch.LongTensor)
        
#     loss = error(outputs, labels)
    loss = compute_loss(model, error, X, Y, beta)
    
    loss.backward()
    
    first_derivative = get_all_vectorized_gradients(list(model.parameters()))
    
    
#     zero_model_gradient(model)
    return first_derivative



def add_noise_data(X, Y, num, num_class, model):
    
    
#     X_distance = torch.sqrt(torch.bmm(X.view(dim[0], 1, dim[1]), X.view(dim[0],dim[1], 1))).view(-1,1)
    
    expected_selected_label =0
    
    updated_selected_label = 0
    
    max_count = -1
    
    min_count = Y.shape[0] + 1
    
    
    mean_list = [] 
    
    for i in range(num_class):
        
        curr_mean = torch.mean(X[Y.view(-1) == i], 0)
        
        mean_list.append(curr_mean)
        
        
        label_count = torch.sum((Y == i)) 
        if label_count > max_count:
            max_count = label_count
            expected_selected_label = i
        
        if label_count < min_count:
            min_count = label_count
            updated_selected_label = i
     
     
    '''n*q'''
    multi_res = model(X)
    
    prob, predict_labels = torch.max(multi_res, 1)
    
    print(prob)
    
#     predict_labels = torch.argmax(multi_res, 1)
    
    
    sorted_prob, indices = torch.sort(prob.view(-1), descending = True)
#     sorted, indices = torch.sort(torch.abs(multi_res), dim = 0, descending = True)
    selected_point = None
    
    selected_label = None
    
    selected_id = 0
    
    
    selected_points = []
    
    
    noise_data_X = torch.zeros((num, X.shape[1]), dtype = torch.float64)

    noise_data_Y = torch.zeros((num, 1), dtype = torch.long)
    
    for i in range(num):
        
        curr_class = Y[indices[i], 0]
        
        curr_coeff = mean_list[curr_class]/mean_list[(curr_class + 1)%(num_class)]        
         
        curr_coeff = curr_coeff[curr_coeff != np.inf]
        
        curr_coeff = torch.sum(curr_coeff[curr_coeff == curr_coeff])
         
#         curr_coeff = torch.sum(curr_coeff[curr_coeff != np.inf and np.isnan(curr_coeff.numpy())])
        
#         print(curr_coeff)
        
        selected_point = (X[indices[i]].clone())*curr_coeff*5
        
        
        if predict_labels[indices[i]] == curr_class:
            selected_label = (curr_class + 1)%num_class
        else:
            selected_label = curr_class
        
        noise_data_X[i,:] = selected_point
        
        noise_data_Y[i] = selected_label
        
        
    X = torch.cat([X, noise_data_X], 0)
        
    Y = torch.cat([Y, noise_data_Y], 0)    
        
        
        
#     class_list = []
#     
#     for j in range(num_class):
# #         if j == expected_selected_label:
#             for i in range(indices.shape[0]):
#                 
#                 if Y[indices[i], 0].numpy() == j and predict_labels[indices[i]].numpy() == j:
#                     selected_point = X[indices[i]].clone()
#                     selected_points.append(selected_point)
#                     class_list.append(j)
#                     
#                     selected_id = indices[i]
#                     selected_label = updated_selected_label
#                     break
#     
#         
#         
#     selected_num = int(num/len(selected_points))
#     
#     for i in range(len(selected_points)):
#         selected_point = selected_points[i]
# #     for selected_point in selected_points:
#     
# 
#         print(torch.mm(selected_point.view(1,-1), res))        
#                 
#         curr_coeff = mean_list[class_list[i]]/mean_list[class_list[(i+1)%(len(class_list))]]        
#          
#         curr_coeff = torch.mean(curr_coeff[curr_coeff != np.inf])
#            
#         
# #         selected_point = selected_point - 5*(mean_list[class_list[i]] - mean_list[updated_selected_label])# + torch.rand(selected_point.shape, dtype = torch.double)
#         selected_point = selected_point*curr_coeff
#         
#         print('distance::', torch.mm(selected_point.view(1,-1), res))       
#         
#         dist_range = torch.rand(selected_point.view(-1).shape, dtype = torch.double)
#         
#         
#         dist = torch.distributions.Normal(selected_point.view(-1), dist_range)
#     
#     
# #     noise_X = []
# #     
# #     for i in range(num):
# #         
# #         noise_X.append(dist.sample())
# #     
# #     
# #     noise_X = torch.cat(noise_X, 0)
# 
#         noise_X = dist.sample((selected_num,))
#         
#         noise_Y = torch.zeros([selected_num, 1], dtype = torch.long)
#         
#         
#         noise_Y[:,0] = class_list[(i+1)%(len(class_list))]
#         
#         X = torch.cat([X, noise_X], 0)
#         
#         Y = torch.cat([Y, noise_Y], 0)
    
    
    
    
    
    
    
    
    
    
    
#     uniqe_Y_values = torch.unique(Y)
#     
#     
#     new_X = torch.zeros([num, X.shape[1]], dtype= torch.double)
#     
#     new_Y = torch.zeros([num, Y.shape[1]], dtype= torch.double)
#     
#     for i in range(num):
#         curr_X = torch.rand(X.shape[1], dtype = torch.double)
#         
# #         curr_Y = uniqe_Y_values[torch.randint(low = 0, high = uniqe_Y_values.shape[0], size = 1)]
# 
#         curr_Y = uniqe_Y_values[torch.LongTensor(1).random_(0, uniqe_Y_values.shape[0])]
#         
#         new_X[i] = curr_X
#         
#         new_Y[i] = curr_Y
#         
# #         X = torch.cat((X, curr_X.view(1,-1)), 0)
# #         
# #         Y = torch.cat((Y, curr_Y.view(1,-1)), 0)
#         
#     X = torch.cat((X, new_X), 0)
#     
#     Y = torch.cat((Y, new_Y), 0)    
    
    return X, Y

def compute_taylor_expansion(dim, loss_1, gradient_1, para_1, para_2, hessian_1):
    
    del_para = get_all_vectorized_parameters(para_2) - get_all_vectorized_parameters(para_1)
    
    
    approx_loss_2 = loss_1 + torch.mm(del_para.view(1,-1), get_all_vectorized_parameters(gradient_1).view(-1,1)) + 0.5*torch.mm(torch.mm(del_para.view(1,-1), hessian_1/dim[0]), del_para.view(-1,1)) 
    
    return approx_loss_2, del_para
    
    
def compute_taylor_expansion_gradient(dim, gradient_1, para_1, para_2, hessian_1, epsilon):
    
    del_para = get_all_vectorized_parameters(para_2) - get_all_vectorized_parameters(para_1)
    
    approx_gradient_2 = get_all_vectorized_parameters(gradient_1) + torch.mm(del_para.view(1,-1), hessian_1/dim[0]) + epsilon*del_para.view(1,-1)
    
    return approx_gradient_2
    
    
    
def random_generate_subset_ids(num, delta_size):
    
    delta_data_ids = set()
    
    while len(delta_data_ids) < delta_size:
        id = random.randint(0, num-1)
        delta_data_ids.add(id)
    
    return torch.tensor(list(delta_data_ids))


def random_generate_subset_ids2(delta_size, all_ids_list):
    
    num = len(all_ids_list)
    
    delta_data_ids = set()
    
    while len(delta_data_ids) < delta_size:
        id = random.randint(0, num-1)
        delta_data_ids.add(all_ids_list[id])
    
    return torch.tensor(list(delta_data_ids))

def random_deletion(num, delta_num):
    delta_data_ids = random_generate_subset_ids(num, delta_num)     
#     torch.save(delta_data_ids, git_ignore_folder+'delta_data_ids')
    return delta_data_ids



# def random_deletion(X, Y, delta_num, num_class, model):
#     
#     multi_res = model(X)#softmax_layer(torch.mm(X, res))
#     
#     prob, predict_labels = torch.max(multi_res, 1)
#     
#     changed_values, changed_label = torch.max(-multi_res, 1)
#     
#     
#     print(prob)
#     
# #     predict_labels = torch.argmax(multi_res, 1)
#     
#     
#     sorted_prob, indices = torch.sort(prob.view(-1), descending = True)
#     
#     delta_id_array = []
#     
# #     delta_data_ids = torch.zeros(delta_num, dtype = torch.long)
#     
# #     for i in range(delta_num):
# 
#     expected_selected_label =0
#     
#      
# #     if torch.sum(Y == 1) > torch.sum(Y == -1):
# #         expected_selected_label = 1
# #         
# #     else:
# #         expected_selected_label = -1
# 
# 
#     i = 0
# 
#     while len(delta_id_array) < delta_num and i < X.shape[0]:
#         if Y[indices[i]] == predict_labels[indices[i]]:
# #             Y[indices[i]] = (Y[indices[i]] + 1)%num_class
# #             Y[indices[i]] = changed_label[indices[i]]
#             delta_id_array.append(indices[i])
#             
#             X[indices[i]] = X[indices[i]] * (-2)
#     
#     
#         i = i + 1
#     
#     delta_data_ids = torch.tensor(delta_id_array, dtype = torch.long)
#     
# #     print(delta_data_ids[:100])
# #     delta_data_ids = random_generate_subset_ids(X.shape, delta_num)     
# #     torch.save(delta_data_ids, git_ignore_folder+'delta_data_ids')
#     return X, Y, delta_data_ids    

# def verify_incremental_hessian(X, Y, dim, para_list_all_epochs, gradient_list_all_epochs, model, num, beta):
#     
#     
#     last_para_list_all_epochs = None
#     
#     last_gradient_list_all_epochs = None
#     
#     
# #     last_real_hessian_matrix = None
# 
#     last_loss = None
# 
#     
#     last_exp_hessian_matrix = None
#     
#     first_real_hessian_matrix = None
#     
#     
#     for i in range(num):
#         id = -(num-i)
#     
#     
#         init_model(model, para_list_all_epochs[id])
#             
# #         pred_2 = model(X)
#         
#         loss_2 = compute_loss(model, error, X, Y, beta)#error(pred_2, Y.view(-1).type(torch.LongTensor)) 
#         
#         curr_vec_gradient_2,_ = compute_gradient_iteration(model, input_dim, hidden_dim, output_dim, X, Y, gradient_list_all_epochs[-2])
#         
#         exp_vec_gradient_2 = get_all_vectorized_parameters(gradient_list_all_epochs[id]) 
#         
#         
#         print(torch.norm(get_all_vectorized_parameters(curr_vec_gradient_2)/dim[0] - exp_vec_gradient_2))
#         
#         
#         hessian_matrix2 = compute_hessian_matrix(model, curr_vec_gradient_2, input_dim, hidden_dim, output_dim)
#     
#     
#         if last_para_list_all_epochs is not None and last_gradient_list_all_epochs is not None and last_exp_hessian_matrix is not None:
#     
#     
# #             pred_1 = model(X)
# #         
# #             loss_1 = error(pred_1, Y.view(-1).type(torch.LongTensor))
#     
#     
#             approx_loss_1,del_para = compute_taylor_expansion(dim, last_loss, last_gradient_list_all_epochs, last_para_list_all_epochs, para_list_all_epochs[id], last_exp_hessian_matrix)
#             
#             
#             print(loss_2 - approx_loss_1)
#             
#             epsilon = (loss_2 - approx_loss_1)/(torch.pow(torch.norm(del_para),2))
#             
#             print('epsilon::', epsilon)
#             
#             approx_gradient_1 = compute_taylor_expansion_gradient(dim, last_gradient_list_all_epochs, last_para_list_all_epochs, para_list_all_epochs[id], last_exp_hessian_matrix, epsilon)
#     
#     
#             print('gradient_diff::', torch.norm(approx_gradient_1 - exp_vec_gradient_2))
#     
#     
#     
#     
#             approx_gradient_1_2 = compute_taylor_expansion_gradient(dim, last_gradient_list_all_epochs, last_para_list_all_epochs, para_list_all_epochs[id], first_real_hessian_matrix, epsilon)
#     
#     
#             print('gradient_diff_2::', torch.norm(approx_gradient_1_2 - exp_vec_gradient_2))
#     
#     
#     
#     
#             
#             last_exp_hessian_matrix = last_exp_hessian_matrix + epsilon*torch.eye(last_exp_hessian_matrix.shape[1], dtype = torch.double)*dim[0]
#             
#             print('hessian_matrix_diff::', torch.norm(last_exp_hessian_matrix - hessian_matrix2))
#         
#         
#         else:
#             last_exp_hessian_matrix = hessian_matrix2   
#         
# #         last_real_hessian_matrix = hessian_matrix2
#         
#         
#         if first_real_hessian_matrix is None:
#             first_real_hessian_matrix = hessian_matrix2
#         
#         
#         
#         last_para_list_all_epochs = para_list_all_epochs[id]
#         
#         last_gradient_list_all_epochs = gradient_list_all_epochs[id]
#         
# #         last_real_hessian_matrix = hessian_matrix2
#         
#         last_loss = loss_2
    

    
            
#         print(torch.norm(exp_vec_gradient_2 - last_gradient_list_all_epochs))




def store_bfgs_values(para_list_all_epochs, gradient_list_all_epochs):


    para_num = get_all_vectorized_parameters(para_list_all_epochs[0]).shape[1]


    print('para_dim::', para_num)

    S_k_list = torch.zeros([para_num, len(para_list_all_epochs) - 1], dtype = torch.double)
    
    Y_k_list = torch.zeros([para_num, len(para_list_all_epochs) - 1], dtype = torch.double)
    
    for i in range(len(para_list_all_epochs)-1):
        s_k = get_all_vectorized_parameters(para_list_all_epochs[i + 1]) - get_all_vectorized_parameters(para_list_all_epochs[i])
        
        y_k = get_all_vectorized_parameters(gradient_list_all_epochs[i + 1]) - get_all_vectorized_parameters(gradient_list_all_epochs[i])
        
        
        S_k_list[:,i] = s_k
        
        Y_k_list[:,i] = y_k 
        
        
    torch.save(S_k_list, git_ignore_folder + 'S_k_list')
    
    torch.save(Y_k_list, git_ignore_folder + 'Y_k_list')


   
            
# def verify_incremental_BFGS_hessian(X, Y, dim, para_list_all_epochs, gradient_list_all_epochs, model, num, m):
# 
# 
#     para_num = get_all_vectorized_parameters(para_list_all_epochs[0]).shape[1]
# 
# 
#     print('para_dim::', para_num)
# 
#     S_k_list = torch.zeros([para_num, len(para_list_all_epochs) - 1], dtype = torch.double)
#     
#     Y_k_list = torch.zeros([para_num, len(para_list_all_epochs) - 1], dtype = torch.double)
#     
#     for i in range(len(para_list_all_epochs)-1):
#         s_k = get_all_vectorized_parameters(para_list_all_epochs[i + 1]) - get_all_vectorized_parameters(para_list_all_epochs[i])
#         
#         y_k = get_all_vectorized_parameters(gradient_list_all_epochs[i + 1]) - get_all_vectorized_parameters(gradient_list_all_epochs[i])
#         
#         
#         S_k_list[:,i] = s_k
#         
#         Y_k_list[:,i] = y_k 
#     
#         
#     
#     
#     
# 
# 
#     for i in range(num):
#         
#         id = -(num-i)
#         
#         curr_S_k = S_k_list[:,id-m:id]
#         
#         curr_Y_k = Y_k_list[:,id-m:id]
#         
#         S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
#         
#         
#         S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)
#         
#         
#         R_k = torch.triu(S_k_time_Y_k)
#         
#         L_k = S_k_time_Y_k - R_k
#         
#         D_k_diag = torch.diag(S_k_time_Y_k)
#         
#         
#         sigma_k = torch.dot(Y_k_list[:,id-1],S_k_list[:,id-1])/(torch.dot(S_k_list[:,id-1], S_k_list[:,id-1]))
#         
#         
#         interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
#         
#         J_k = torch.from_numpy(np.linalg.cholesky(interm.numpy())).type(torch.DoubleTensor)
#         
#         
#         v_vec = S_k_list[:,id-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
#         
# #         v_vec = torch.rand([para_num, 1], dtype = torch.double)
#         
#         
#         p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
#         
#         
#         D_k_sqr_root = torch.pow(D_k_diag, 0.5)
#         
#         D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
#         
#         upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
#         
#         lower_mat_1 = torch.cat([torch.zeros([m, m], dtype = torch.double), torch.t(J_k)], dim = 1)
#         
#         
#         mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
#         
#         
#         upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([m, m], dtype = torch.double)], dim = 1)
#         
#         lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
#         
#         mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
#         
#         
#         p_mat = torch.mm(torch.inverse(mat_1), torch.mm(torch.inverse(mat_2), p_mat))
#         
#         
#         approx_prod = sigma_k*v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
#         
#         
#         
#         init_model(model, para_list_all_epochs[id])
#         
#         curr_vec_gradient,_ = compute_gradient_iteration(model, input_dim, hidden_dim, output_dim, X, Y, gradient_list_all_epochs[id])
#         
#         hessian_matrix = compute_hessian_matrix(model, curr_vec_gradient, input_dim, hidden_dim, output_dim)/dim[0]
# 
#         
#         exp_prod = torch.mm(hessian_matrix, v_vec)
#         
#         print(torch.norm(approx_prod - exp_prod))
        
        
        
        
    
def decompose_model_paras(para_list1, para_list2, gradient_list, alpha):
    
    for i in range(len(para_list1)):
        
        para1 = para_list1[i]
        
        para2 = para_list2[i]
        
        grad = gradient_list[i]
        
        
        if len(para1.shape) <= 1:
            continue
        
        u1,s1,v1 = torch.svd(para1)
        
        u2,s2,v2 = torch.svd(para2)
        
        u0,s0,v0 = torch.svd(grad)
        
#         v = torch.mm(torch.inverse(torch.mm(torch.t(u1), u1) + torch.mm(torch.t(u2), u2)), torch.mm(torch.t(u1), para1) + torch.mm(torch.t(u2), para2))
        
        
        diff0 = torch.norm(u2*s2 - (u1*s1 - alpha*u0*s0))
        
        diff1 = torch.norm(s2*v2 - (s1*v1 - alpha*s0*v0))
        
        
        print(diff0, diff1)
        


def decompose_model_paras2(para_list1, para_list2, gradient_list1, gradient_list2):
    
    print("here")
    
    for i in range(len(para_list1)):
        
        para1 = para_list1[i]
        
        para2 = para_list2[i]
        
        grad1 = gradient_list1[i]
        
        grad2 = gradient_list2[i]
        
        
        if len(para1.shape) <= 1:
            continue
        
        u1,s1,v1 = torch.svd(para1)
         
        u2,s2,v2 = torch.svd(para2)
         
        u3,s3,v3 = torch.svd(grad1)
         
        u4,s4,v4 = torch.svd(grad2) 
        
        
        uu1, ss1, vv1 = torch.svd(para1 - para2)
        
        uu2, ss2, vv2 = torch.svd(grad1 - grad2)
        
        print("layer::", i)
        print("para_diff::")
        print(torch.norm(para1 - para2))
        
        print(torch.norm(para1 - para2, p=float("inf")))
        
        print(torch.norm(para1, p=float("inf")))
        
        
        print("grad_diff::")
        
        print(torch.norm(grad1 - grad2))
        
        
        print("difference::")
        
#         print(torch.norm(torch.mm(uu1, torch.t(uu2)) - torch.eye(uu1.shape[0], dtype = torch.double)))
#         
#         print(torch.norm(torch.mm(torch.t(vv2), vv1) - torch.eye(vv1.shape[1], dtype = torch.double)))
        
        print(torch.norm(torch.mm(u1, torch.t(u4)) - torch.eye(u1.shape[0], dtype = torch.double)))
         
        print(torch.norm(torch.mm(u2, torch.t(u4)) - torch.eye(u1.shape[0], dtype = torch.double)))
         
         
        print(torch.norm(torch.mm(v1, torch.t(v2)) - torch.eye(v1.shape[0], dtype = torch.double)))
         
        print(torch.norm(torch.mm(v1, torch.t(v3)) - torch.eye(v1.shape[0], dtype = torch.double)))
         
        print(torch.norm(torch.mm(v1, torch.t(v4)) - torch.eye(v1.shape[0], dtype = torch.double)))
         
        print(torch.norm(torch.mm(v2, torch.t(v4)) - torch.eye(v1.shape[0], dtype = torch.double)))
        
#         v = torch.mm(torch.inverse(torch.mm(torch.t(u1), u1) + torch.mm(torch.t(u2), u2)), torch.mm(torch.t(u1), para1) + torch.mm(torch.t(u2), para2))
        
        
#         diff0 = torch.norm(u2*s2 - (u1*s1 - alpha*u0*s0))
#         
#         diff1 = torch.norm(s2*v2 - (s1*v1 - alpha*s0*v0))
        
        
#         print("here")



        
                
        
def save_random_id_orders(git_ignore_folder, random_ids_multi_super_iterations):
    sorted_ids_multi_super_iterations = []
    
    
    for i in range(len(random_ids_multi_super_iterations)):
        sorted_ids_multi_super_iterations.append(random_ids_multi_super_iterations[i].numpy().argsort())
        
        
    torch.save(sorted_ids_multi_super_iterations, git_ignore_folder + 'sorted_ids_multi_super_iterations')

    



def get_train_test_data_loader(Model, training_data, test_data, specified_batch_size):
    
    
    dataset_train = Model.MyDataset(training_data)
    dataset_test = Model.MyDataset(test_data)
    
    data_train_loader = DataLoader(dataset_train, batch_size=specified_batch_size, shuffle=True, num_workers=0)
    data_test_loader = DataLoader(dataset_test, batch_size=specified_batch_size, num_workers=0)

    return dataset_train, dataset_test, data_train_loader, data_test_loader




def get_train_test_data_loader_by_name(data_preparer, Model, name, specified_batch_size):
    
    
    function=getattr(Data_preparer, "prepare_" + name)
    
    training_data, test_data = function(data_preparer)
    
    
    dataset_train, dataset_test, data_train_loader, data_test_loader = get_train_test_data_loader(Model, training_data, test_data, specified_batch_size)
    
    return dataset_train, dataset_test, data_train_loader, data_test_loader







def get_train_test_data_loader_lr(Model, train_X, train_Y, test_X, test_Y, specified_batch_size):
    
    
    dataset_train = Model.MyDataset(train_X, train_Y)
    dataset_test = Model.MyDataset(test_X, test_Y)
    
    data_train_loader = DataLoader(dataset_train, batch_size=specified_batch_size, shuffle=True, num_workers=0)
    data_test_loader = DataLoader(dataset_test, batch_size=specified_batch_size, num_workers=0)

    return dataset_train, dataset_test, data_train_loader, data_test_loader




def get_train_test_data_loader_by_name_lr(git_ignore_folder, data_preparer, Model, name, specified_batch_size):
    
    
    function=getattr(Data_preparer, "prepare_" + name)
    
    train_X, train_Y, test_X, test_Y = function(data_preparer)
    
#     if not start:
    delta_data_ids = torch.load(git_ignore_folder + 'delta_data_ids')
    
    
    selected_rows = get_subset_training_data0(train_X.shape[0], delta_data_ids)
    
    torch.save(train_X[delta_data_ids], git_ignore_folder + 'X_to_add')
    
    torch.save(train_Y[delta_data_ids], git_ignore_folder + 'Y_to_add')
    
    dataset_train, dataset_test, data_train_loader, data_test_loader = get_train_test_data_loader_lr(Model, train_X[selected_rows], train_Y[selected_rows], test_X, test_Y, specified_batch_size)
    
    return dataset_train, dataset_test, data_train_loader, data_test_loader

def get_data_class_num_by_name(data_preparer, name):
    
    
    function=getattr(Data_preparer, "get_num_class_" + name)
    
    num_class = function(data_preparer)
    
    
#     dataset_train, dataset_test, data_train_loader, data_test_loader = get_train_test_data_loader(Model, training_data, test_data, specified_batch_size)
    
    return num_class


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
    
    lrs = ast.literal_eval(input)
    
    model_name = sys_argv[5]

    dataset_name = sys_argv[6]
    
    learning_rate = 0.1
    
    repetition = int(sys_argv[7])
    
    regularization_coeff = float(sys_argv[8])
    
#     start = bool(int(sys_argv[9]))
    
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
    
    
    delta_data_ids = torch.load(git_ignore_folder + 'delta_data_ids')
    
    selected_rows = get_subset_training_data0(dataset_train.data.shape[0], delta_data_ids)
    
    torch.save(dataset_train.data[delta_data_ids], git_ignore_folder + 'X_to_add')
    
    torch.save(dataset_train.labels[delta_data_ids], git_ignore_folder + 'Y_to_add')
    
    
    
    
    
#     dataset_train, dataset_test, data_train_loader, data_test_loader = get_train_test_data_loader_by_name_lr(git_ignore_folder, data_preparer, model_class, dataset_name, batch_size)
    
    hyper_para_function=getattr(Data_preparer, "get_hyperparameters_" + dataset_name)
    
    model = model_class()
    
    if is_GPU:
        model.to(device)
    
    init_model_params = list(model.parameters())
    
    
    criterion, optimizer, lr_scheduler = hyper_para_function(data_preparer, model.parameters(), learning_rate, regularization_coeff)
    
    hyper_params = [criterion, optimizer, lr_scheduler]
    
    random_ids_all_epochs = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations_' + str(repetition))
    
    
#     model, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs, random_ids_multi_super_iterations = model_training_test(random_ids_all_epochs, num_epochs, model, data_train_loader, data_test_loader, len(dataset_train), len(dataset_test), optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs)
    model, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs = model_training_test(random_ids_all_epochs, num_epochs, model, dataset_train, dataset_test, len(dataset_train), len(dataset_test), optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs)


#     capture_provenance(X, Y, dim, epoch, num_class, batch_size, mini_epochs_per_super_iteration, random_ids_multi_super_iterations_tensors)


    
    torch.save(dataset_train, git_ignore_folder + "dataset_train")
    
    torch.save(dataset_test, git_ignore_folder + "test_data")
    
#     torch.save(delta_data_ids, git_ignore_folder + "delta_data_ids")
    
    
    torch.save(gradient_list_all_epochs, git_ignore_folder + 'gradient_list_all_epochs')
    
    torch.save(para_list_all_epochs, git_ignore_folder + 'para_list_all_epochs')
    
    torch.save(learning_rate_all_epochs, git_ignore_folder + 'learning_rate_all_epochs')


    torch.save(random_ids_all_epochs, git_ignore_folder + 'random_ids_multi_super_iterations')
                  
    torch.save(num_epochs, git_ignore_folder+'epoch')    
    
    torch.save(hyper_params, git_ignore_folder + 'hyper_params')
    
    save_random_id_orders(git_ignore_folder, random_ids_all_epochs)
    
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



    test(model, dataset_test, batch_size, criterion, len(dataset_test), is_GPU, device)

