'''
Created on Feb 5, 2019


'''
import os
import sys
from torch.autograd import Variable

from torch import nn, optim
import torch
import time
# from main.watcher import Watcher
import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.utils.extmath import randomized_svd


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_IO.Load_data import *
    from Interpolation.piecewise_linear_interpolation_multi_dimension import *
    from sensitivity_analysis_SGD.multi_nomial_logistic_regression.incremental_updates_logistic_regression_multi_dim import *
    from sensitivity_analysis_SGD.multi_nomial_logistic_regression.evaluating_test_samples import *
except ImportError:
    from Load_data import *
    from piecewise_linear_interpolation_multi_dimension import *
    from incremental_updates_logistic_regression_multi_dim import *
    from evaluating_test_samples import *

import gc 
import sys


# from sensitivity_analysis.logistic_regression.incremental_updates_logistic_regression import X_product


# max_epoch = 200


'''shuttle_dataset: para'''
alpha = 1e-6
   
beta = 0.001

svd_ratio = 12
# alpha = 0.002
#   
# beta = 0.02

max_epoch = 1000

threshold = 1e-5

batch_size = 1000


cut_off_threshold = 0.001


theta_record_threshold = 0.01

res_prod_seq = torch.zeros(0, dtype = torch.double)

epoch_record_epoch_seq = []

random_ids_multi_super_iterations = []

X_theta_prod_seq = []

X_theta_prod_softmax_seq = []





inter_result1 = []


inter_result2 = []

prod_time1 = 0

prod_time2 = 0




# sample_level = True
# 
# if sample_level:
#     from main.matrix_prov_sample_level import M_prov
# else:
#     from main.matrix_prov_entry_level import M_prov
# from main.add_prov import add_prov_token_per_row
# torch.set_printoptions(precision=10)
# 
# # x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
# #                     [9.779], [6.182], [7.59], [2.167], [7.042],
# #                     [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
# 
# x_train = np.array([[0, 1], [4.4, 0], [5.5, 3]], dtype=np.float32)
# 
# # y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
# #                     [3.366], [2.596], [2.53], [1.221], [2.827],
# #                     [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
# 
# y_train = np.array([[1.7], [2.06], [2.59]], dtype=np.float32)
# 
# 
# x_train = torch.from_numpy(x_train)
# 
# y_train = torch.from_numpy(y_train)
# 
# X = Variable(x_train)
# 
# 
# 
# Y = Variable(y_train)
# 
# shape = list(X.size())
# 
# X = torch.cat((X, torch.ones([shape[0], 1])), 1)


# x_train[0][0].a[0] = 1

log_sigmoid_layer = torch.nn.LogSigmoid()

sig_layer = torch.nn.Sigmoid()


softmax_layer = torch.nn.Softmax(dim = 1)

log_softmax_layer = torch.nn.LogSoftmax(dim = 1)

class logistic_regressor_parameter:
    def __init__(self, theta):
        self.theta = theta

def sigmoid_function(x):
    return 1/(1 + torch.exp(-x))

def sigmoid(x):
    return 1-1 / (1 +np.exp(-x))

def non_linear_function(x):
    return 1-1/(1 + torch.exp(-x))

def non_linear_function_nump(x):
    return 1-1/(1 + np.exp(-x))
        

def compute_sigmoid_function(x_i, theta):
    return sigmoid_function(torch.dot(x_i, theta))
        
def binary_cross_entropy(x_i, y_i, theta):
    
    return y_i*compute_sigmoid_function(x_i, theta) + (1-y_i)*(1 - compute_sigmoid_function(x_i, theta)) 

def sigmoid_function2(x_i, y_i, theta):
    return 1/(1+torch.exp(-y_i*torch.dot(x_i, theta)))

def non_linear_terms(x_i, y_i, theta):
#     return 1-binary_cross_entropy(x_i, y_i, theta)
    return (compute_sigmoid_function(x_i, theta) - y_i)

def non_linear_terms2(x_i, y_i, theta):
    return -(1-sigmoid_function2(x_i, y_i, theta))
#     return (sigmoid_function(x_i, theta) - y_i)

def loss_function(X, Y, theta, dim, beta):
    
    res = 0
    
    for i in range(dim[0]):
#         res += torch.log(1 + torch.exp(-Y[i]*torch.dot(X[i,:].view(dim[1]), theta.view(dim[1]))))
        res = res - (Y[i]*torch.log(compute_sigmoid_function(X[i,:].view(dim[1]), theta.view(dim[1]))) + (1 - Y[i])*torch.log(1 - compute_sigmoid_function(X[i,:].view(dim[1]), theta.view(dim[1])))) 
        
    res = res/dim[0]
    
    return res + beta*torch.pow(torch.norm(theta, p =2), 2)

def bia_function(x):
    return -log_sigmoid_layer(x)

def second_derivative_loss_function(x):
    return torch.exp(x)/torch.pow((1 + torch.exp(x)), 2)

def compute_hessian_matrix(X, Y, theta, dim, X_product):
    X_Y_theta_prod = torch.mm(X, theta)*Y
    
    res = torch.zeros([dim[1], dim[1]], dtype = torch.float64)
    
    for i in range(dim[0]):
        res += second_derivative_loss_function(-X_Y_theta_prod[i])*X_product[i]
        
    res = -res/dim[0] + beta*torch.eye(dim[1], dtype = torch.float64)
    
    return res
    
    

def loss_function2(X, Y, theta, dim, beta, tracking_prov):
    
#     res = 0
    
    
#     sigmoid_res = torch.stack(list(map(bia_function, Y*torch.mm(X, theta))))

#     sigmoid_res = Y*torch.mm(X, theta)
#     data_trans = sigmoid_res.apply(lambda x :  ())

#     sigmoid_res = -log_sigmoid_layer(Y*torch.mm(X, theta))
#     theta = theta.view(dim[1], num_class)




    X_theta_prod = torch.mm(X, theta)
    
    
    X_theta_prod_softmax = softmax_layer(X_theta_prod)
    
    
    
    if tracking_prov:
        X_theta_prod_softmax_seq.append(X_theta_prod_softmax)
        X_theta_prod_seq.append(X_theta_prod)
    
#     global X_theta_prod_softmax_seq
#          
#     if list(X_theta_prod_softmax_seq.shape) == [0]:
#         X_theta_prod_softmax_seq = (X_theta_prod_softmax.clone()).view(1, dim[0], theta.shape[1])
# #             res_prod_seq.append(lr.theta.clone())
#     else:
#         X_theta_prod_softmax_seq = torch.cat((X_theta_prod_softmax_seq, X_theta_prod_softmax.clone()), 0)
    
    
    res = -torch.sum(torch.log(torch.gather(X_theta_prod_softmax, 1, Y.view(-1,1))))/dim[0]

#     output = torch.log(torch.sum(torch.exp(X_theta_prod_softmax), dim = 1))
#      
# #     print(output.shape)
# #     
# #     print(X_theta_prod_softmax.shape)
#      
#     res = torch.sum((output.view(-1, 1) - torch.gather(X_theta_prod_softmax, 1, Y.view(-1, 1))),dim=0)/dim[0]
    
    
#     output = softmax_layer(torch.mm(X, theta))
#     
#     print(theta)
#     
#     inter_result1.append(output)
#     
#     output = torch.log(output)
#     
#     res = -torch.sum(torch.gather(output, 1, Y.view(dim[0], 1)))/dim[0]
    
    
#     res = torch.sum(-log_sigmoid_layer(Y*torch.mm(X, theta)))/dim[0]
    
    
#     for i in range(dim[0]):
# #         print(X[i,:])
# #         print(theta)
# #         print(X[i,:].view(dim[1]))
# #         print(theta.view(dim[1]))
#         res += torch.log(1 + torch.exp(-Y[i]*torch.dot(X[i,:].view(dim[1]), theta.view(dim[1]))))
#         res = res - (Y[i]*torch.log(sigmoid_function(X[i,:].view(dim[1]), theta.view(dim[1]))) + (1 - Y[i])*torch.log(1 - sigmoid_function(X[i,:].view(dim[1]), theta.view(dim[1])))) 
        
#     res = res
    
    return res + beta/2*torch.sum(theta.view(-1,1)*theta.view(-1,1))
    
#     return res + beta/2*torch.pow(torch.norm(theta, p =2), 2)

def gradient(X, Y, dim, theta):
    
#     res = torch.zeros(theta.shape, dtype = torch.float64)
    
#     print('res!!!', res)
    
    res = torch.stack(list(map(non_linear_function, torch.mul(torch.mm(X, theta), Y))))
    
    res = torch.mul(res, Y)
    
    res = torch.mm(torch.t(X), res)
#     for i in range(dim[0]):
# #         print(X[i,:].view(dim[1]))
#         non_linear_value = non_linear_terms2(X[i,:].view(dim[1]), Y[i], theta.view(dim[1]))
# #         print(non_linear_value)
# #         print(Y[i]*X[i,:]*non_linear_value)
# #         print(res)
#         
#         res = res + Y[i]*non_linear_value*(X[i,:].view(theta.shape))
    
#     print('res', res)
#     
#     print('res_size::', res.shape)
    
    return res/dim[0]
    

def logistic_regression(origin_X, origin_Y, lr, dim, num_class, tracking_prov):

#     dim = X.shape
#     lr.theta.requires_grad = False
# 
#     vectorized_theta = lr.theta.view(-1,1).clone()
#     
#     vectorized_theta.requires_grad = True
    
#     print(vectorized_theta)
#     print('init_theta', lr.theta)
    origin_Y = origin_Y.type(torch.LongTensor)
    
    epoch = 0
    
    last_theta = None
    
    
    last_recorded_theta = None
    
#     for epoch in range(max_epoch):

    mini_batch_epoch = 0

    while epoch < max_epoch:
        
        end = False
        
#         X = origin_X
# 
#         Y = origin_Y
#         
#         random_ids_multi_super_iterations.append(torch.tensor(list(range(X.shape[0]))))
        random_ids = torch.randperm(dim[0])
         
        X = origin_X[random_ids]
         
        Y = origin_Y[random_ids]
#         
        random_ids_multi_super_iterations.append(random_ids)
                
#         gap_to_be_averaged = []

        for i in range(0, dim[0], batch_size):
            
            
            end_id = i + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            
            
            batch_x = X[i: end_id]
            
            batch_y = Y[i: end_id]
        
        
        
            if tracking_prov:
             
                global res_prod_seq
                 
                 
#                 if last_recorded_theta is None:
#                     last_recorded_theta = lr.theta.clone()
#                     res_prod_seq = lr.theta.clone()
#                     epoch_record_epoch_seq.append(epoch)
#                     tracking_prov = True
#     #                 print('here')
#                 else:
#                     if torch.norm(last_recorded_theta - lr.theta) > theta_record_threshold:
#                         last_recorded_theta = lr.theta.clone()
#                         res_prod_seq = torch.cat((res_prod_seq, lr.theta.clone()), 1)
#                         epoch_record_epoch_seq.append(epoch)
#                         tracking_prov = True
#     #                     print('here')
#                     else:
#                         tracking_prov = False
#             if res_prod_seq.shape == 0:
#                 
#     #             res_prod_seq.append(lr.theta.clone())
#             else:
#                 res_prod_seq = torch.cat((res_prod_seq, lr.theta.clone()), 1)
        
        
        
        
            loss = loss_function2(batch_x, batch_y, lr.theta, batch_x.shape, beta, tracking_prov)
       
            loss.backward()
           
            with torch.no_grad():
    #             print('gradient::', lr.theta.grad)
                lr.theta -= alpha * lr.theta.grad
    #             print(lr.theta.grad[:, 1] - lr.theta.grad[:, 0])
    #             print(lr.theta.grad[:,1] - lr.theta.grad[:,0])
    #             print(epoch, lr.theta)
    
                gap = torch.norm(lr.theta.grad)
    #             
    #             
    #             if gap < threshold:
    #                 break
    #             
    #             if gap < theta_record_threshold:
    #                 tracking_prov = False
    #             
    #             
    #             print(gap)
                
                lr.theta.grad.zero_()
        
            if last_theta is not None:
                
#                 gap = torch.norm(last_theta - lr.theta)
                
                print(epoch, gap)
                
                if epoch == 36:
                    y = 0
                    
                    y = y + 1
                
                
                
#                 if len(gap_to_be_averaged) >= (X.shape[0] - 1)/batch_size + 1:
#                          
#                         average_gap = np.sum(gap_to_be_averaged)/len(gap_to_be_averaged)
#                          
#                          
#                         print('avg_gap::', average_gap, len(gap_to_be_averaged))
#                          
#                         if average_gap < theta_record_threshold:
#                             tracking_prov = False    
#      
#                          
#                          
#                         gap_to_be_averaged.pop(0)
                     
#                 if tracking_prov:
#                      
#                     gap_to_be_averaged.append(gap.item())  
#                 
#                 
#                 if len(gap_to_be_averaged) >= (X.shape[0] - 1)/batch_size:
#                          
#                         average_gap = np.sum(gap_to_be_averaged)/len(gap_to_be_averaged)
#                          
#                          
#                         print('avg_gap::', average_gap, len(gap_to_be_averaged))
#                          
#                         if average_gap < theta_record_threshold:
#                             tracking_prov = False    
     
                         
                         
#                         gap_to_be_averaged.pop(0)
                
                
                
            
    #         
            if last_theta is not None and gap < threshold:
                
                end = True
                
                break
            
            mini_batch_epoch += 1
            
            epoch = epoch + 1
                
            last_theta = lr.theta.clone()
        
        if end:
            break
        
        

#         print('epoch', epoch)
#         print('start', lr.theta)
#         print('step 0', (torch.mm(X, lr.theta)))
#         print('step 1', (torch.mm(X, lr.theta) - Y))
#         print('step 2', alpha*torch.mm(torch.transpose(X, 0, 1),(torch.mm(X, lr.theta) - Y)))
        
#         lr.theta = lr.theta - 2*alpha*(torch.mm(torch.transpose(X, 0, 1),(torch.mm(X, lr.theta) - Y)) + beta*lr.theta)



#         lr.theta = lr.theta + alpha*gradient(X, Y, dim, lr.theta)- alpha*beta*lr.theta 
        
#         print('gradient::', - alpha*gradient(X, Y, dim, lr.theta))
        
#         print('theta!!!!', lr.theta)
#         global res_prod_seq
#          
#         if res_prod_seq.shape == 0:
#             res_prod_seq = lr.theta.clone()
# #             res_prod_seq.append(lr.theta.clone())
#         else:
#             res_prod_seq = torch.cat((res_prod_seq, lr.theta.clone()), 1)
        
        
         
#         print('loss:', loss)
        
#         print('theta!!!!', lr.theta)
#         err = Y - torch.mm(X, lr.theta)
#         error = torch.mm(torch.transpose(err, 0, 1), err)# + beta*torch.matmul(torch.transpose(theta, 0, 1), theta)
        
#         print('error', error)
      
#     lr.theta = vectorized_theta.view(dim[1], num_class)  
    
    return lr.theta, epoch

def get_id_mappings_per_batch(X, Y, selected_data_ids, batch_size):
    id_mappings = {}
    
    batch_X_list = []
    
    batch_Y_list = []
    
#     curr_selected_data_ids_list = []
    
    for i in range(0, X.shape[0], batch_size):
        end_id = i + batch_size
            
        if end_id > X.shape[0]:
            end_id = X.shape[0]
            
        
        if i not in id_mappings:
            
            curr_selected_data_ids = selected_data_ids[(torch.nonzero((selected_data_ids >= i)*(selected_data_ids < end_id))).view(-1)]

            id_mappings[i] = curr_selected_data_ids
        
        
        curr_selected_data_ids = id_mappings[i]

        if curr_selected_data_ids.shape[0] <= 0:
            continue
#         print(curr_selected_data_ids.shape[0])
        
        batch_X, batch_Y = X[curr_selected_data_ids], Y[curr_selected_data_ids]

        batch_X_list.append(batch_X)
        
        batch_Y_list.append(batch_Y)
        
#         curr_selected_data_ids_list.append(curr_selected_data_ids)
        
    return batch_X_list, batch_Y_list
        
        

def logistic_regression_by_standard_library(random_ids_multi_super_iterations, selected_rows, X, Y, lr, dim, max_epoch, alpha, beta, batch_size):

#     dim = X.shape
    Y = Y.type(torch.LongTensor)
    

    
    
    selected_rows_set = set(selected_rows.view(-1).tolist())

    theta_list = []
    
    grad_list = []
    
#     batch_X_list = []
    
#     for epoch in range(max_epoch):
    for k in range(len(random_ids_multi_super_iterations)):
        random_ids = random_ids_multi_super_iterations[k]
        
#         res_prod_seq.append(lr.theta.clone())

        for i in range(0, X.shape[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            
            curr_rand_ids = random_ids[i:end_id]
            
            curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
            
            curr_matched_ids,_ = torch.sort(curr_matched_ids)
            
#             print(curr_matched_ids)
            
            batch_X = X[curr_matched_ids]
            
            batch_Y = Y[curr_matched_ids]
        
#             batch_X_list.append(batch_X)
        
            loss = loss_function2(batch_X, batch_Y, lr.theta, batch_X.shape, beta, False)
       
            loss.backward()
           
            with torch.no_grad():
                lr.theta -= alpha * lr.theta.grad
                
#                 grad_list.append(lr.theta.grad.clone())
                
                lr.theta.grad.zero_()
                
#                 theta_list.append(lr.theta.clone())
#         print('epoch', epoch)
#         print('start', lr.theta)
#         print('step 0', (torch.mm(X, lr.theta)))
#         print('step 1', (torch.mm(X, lr.theta) - Y))
#         print('step 2', alpha*torch.mm(torch.transpose(X, 0, 1),(torch.mm(X, lr.theta) - Y)))
        
#         lr.theta = lr.theta - 2*alpha*(torch.mm(torch.transpose(X, 0, 1),(torch.mm(X, lr.theta) - Y)) + beta*lr.theta)



#         lr.theta = lr.theta + alpha*gradient(X, Y, dim, lr.theta)- alpha*beta*lr.theta 
        
#         print('gradient::', - alpha*gradient(X, Y, dim, lr.theta))
        
#         print('theta!!!!', lr.theta)
#         global res_prod_seq
#          
#         if res_prod_seq.shape == 0:
#             res_prod_seq = lr.theta.clone()
# #             res_prod_seq.append(lr.theta.clone())
#         else:
#             res_prod_seq = torch.cat((res_prod_seq, lr.theta.clone()), 1)
        
        
         
#         print('loss:', loss)
        
#         print('theta!!!!', lr.theta)
#         err = Y - torch.mm(X, lr.theta)
#         error = torch.mm(torch.transpose(err, 0, 1), err)# + beta*torch.matmul(torch.transpose(theta, 0, 1), theta)
        
#         print('error', error)
      
    return lr.theta, theta_list, grad_list


def compute_parameters(X, Y, lr, dim, num_class, tracking_prov):
    
    
    lr.theta, epoch = logistic_regression(X, Y, lr, dim, num_class, tracking_prov)
    
    print('res_real:::', lr.theta)
    
    return lr.theta, epoch
    
    
def compute_single_coeff(X, Y, w_seq, dim, epoch, X_products):
    
#     print(dim)
    
    res = (1 - beta*alpha)*torch.eye(dim[1])
    
#     for i in range(dim[0]):
        
#     b_seq_tensor = torch.tensor(w_seq[epoch], dtype = torch.double)
#     
#     b_seq_tensor = torch.t(b_seq_tensor.repeat(dim[1],1))
    
#     print('b_seq_size::', b_seq_tensor.shape)
#     
#     print('b_seq::', b_seq_tensor)

#     print(b_seq_tensor.shape)
    global prod_time1


    t1  =time.time()

#     prod_res = X_products.mul(w_seq[epoch].view([dim[0], 1, 1]))
#     prod_res = torch.mm(torch.t(X), torch.diag(w_seq[epoch]))
    
    prod_res = torch.t(X.mul(w_seq[epoch].view([dim[0], 1])))
    
    prod_res = torch.mm(prod_res, X)
    
    t2  = time.time()
    
    prod_time1 += (t2 -t1)
    
#     res = res + alpha * torch.mm(torch.t(X), torch.mul(X, b_seq_tensor))/dim[0]
#     res = res + alpha * (torch.sum(prod_res, dim = 0).view([dim[1], dim[1]]))/dim[0]
    res = res + (alpha/dim[0]) * prod_res
        
    return res
        
#         res = res + alpha * w_seq[epoch], b_seq[epoch]

def compute_x_y_terms(X, Y, b_seq, dim, epoch, X_Y_products):
    
#     b_seq_tensor = torch.tensor(b_seq[epoch], dtype = torch.double)
#     
#     b_seq_tensor = torch.t(b_seq_tensor.view(Y.shape) * Y)
    
#     b_seq_tensor = torch.t(b_seq_tensor.repeat(dim[1], 1))
    
#     print('b_seq_size::', b_seq_tensor.shape)
#      
#     print('b_seq::', b_seq_tensor)
    global prod_time2
    t1 = time.time()

    res = X_Y_products.mul(b_seq[epoch].view([dim[0], 1]))
    
    t2 = time.time()
    
    prod_time2 += (t2 - t1)
    
    res = torch.sum(res, dim = 0).view(dim[1], 1)/dim[0]
    
#     res = torch.t(torch.mm(b_seq_tensor, X))/dim[0]
    
#     print(res)
    
    
    return res


def compute_curr_linear_paras(X, Pi, total_time):
    
#     res = torch.tensor(list(map(Pi.piecewise_linear_interpolate_coeff, X)))
#     
# #     print(res.shape)
# #     
# #     print(res[:,0].shape)
# #     
# #     print(res[:,1].shape)
#     
#     return res
    
    
    return Pi.piecewise_linear_interpolate_coeff_batch(X, total_time)

def get_tensor_size(a):
    return a.element_size() * a.nelement()/np.power(2,20)


def compute_first_derivative_single_data(X_Y_mult, ids, theta, dim):
    
    print('X_Y_shape::', X_Y_mult.shape)
    
    curr_X_Y_mult = torch.index_select(X_Y_mult, 0, ids)
    
    non_linear_term = curr_X_Y_mult*(1 - sig_layer(torch.mm(curr_X_Y_mult, theta)))
    
#     print(ids, non_linear_term, theta)
    
    non_linear_term = torch.sum(non_linear_term, dim=0).view(theta.shape)
    
    res = -non_linear_term + beta*theta
    
    return res


def compute_model_parameter_by_iteration(dim, theta,  X, Y, x_sum_by_class_list, num_class, max_epoch, alpha, beta, batch_X_list, batch_Y_list):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    for j in range(max_epoch):
        
#         multi_res = torch.mm(X_Y_mult, theta)
        
#         w_seq, b_seq, total_time = compute_curr_linear_paras(multi_res, Pi, total_time)
#         t1 = time.time()
#         lin_res = 1 - sig_layer(torch.mm(X_Y_mult, theta))
#         t2 = time.time()
#         
#         total_time += t2 - t1
#         multi_res *= w_seq
#         
#         multi_res += b_seq

        for i in range(len(batch_X_list)):
            batch_X = batch_X_list[i]
            
            batch_Y = batch_Y_list[i]
            


            output = softmax_layer(torch.mm(batch_X, theta))
            
            
            output = torch.mm(torch.t(batch_X), output)
            
            
            output = torch.reshape(torch.t(output), [-1,1])
            
    #         print(i, theta)
            
    #         inter_result2.append(output)
    #         
    #         res = torch.mm(torch.t(torch.gather(output, 1, Y.view(dim[0], 1))), X)
            
            reshape_theta = torch.reshape(torch.t(theta), (-1, 1))
            
            res = (output - x_sum_by_class_list[i])/batch_X.shape[0] + beta*reshape_theta
            
    #         print('output::', output/dim[0])
    # #         
    #         print('x_sum_by_class::', X_sum_by_class/dim[0])
    #         
    #         print('gradient::', res.view(num_class, dim[1]))
    #         
    #         print(res[1] - res[0])
            
    #         print(torch.t(res.view(num_class, dim[1]))[:,1] - torch.t(res.view(num_class, dim[1]))[:,0])
            
            res = reshape_theta - alpha * res
            
            
            theta = torch.t(res.view(num_class, dim[1]))
        
#         print(alpha*(X_sum_by_class.view(-1, dim[1])[1]-X_sum_by_class.view(-1, dim[1])[0])/dim[0])
        
#         print((i+1)*alpha*(X_sum_by_class.view(-1, dim[1])[1]-X_sum_by_class.view(-1, dim[1])[0])/dim[0])
        
#             print(i, theta)
#         
#         print('delta::', theta[:,1] - theta[:,0])
        
#         non_linear_term = X_Y_mult*(1 - sig_layer(torch.mm(X_Y_mult, theta)))
        
#         
#         if i == max_epoch - 1:
#             for j in range(non_linear_term.shape[0]):
#                 print(j, non_linear_term[j], theta)
        
        
#         w_b_seq = compute_curr_linear_paras(multi_res, Pi)
#          
#         non_linear_term = X.mul((multi_res*(w_b_seq[:,0].view([dim[0],1])) + w_b_seq[:,1].view([dim[0],1]))*Y)
#         sum_non_linear_term = torch.sum(non_linear_term, dim=0)
#         sum_non_linear_term = np.sum(non_linear_term, axis=0)
        
#         print(sum_non_linear_term.shape)
        
#         sum_non_linear_term_diff_dim = np.reshape(sum_non_linear_term, (theta.shape))
#         sum_non_linear_term_diff_dim = torch.sum(non_linear_term, dim=0).view(theta.shape)
#         sum_non_linear_term_diff_dim *=  
        
#         theta = (1-alpha*beta)*theta + (alpha/dim[0])*torch.sum(X_Y_mult.mul(multi_res), dim=0).view(theta.shape)
#         theta = (1-alpha*beta)*theta + (alpha/dim[0])*torch.sum(non_linear_term, dim=0).view(theta.shape)
        
#         sizes = [get_tensor_size(multi_res), get_tensor_size(w_seq), get_tensor_size(b_seq)
#                  , get_tensor_size(non_linear_term), get_tensor_size(sum_non_linear_term), get_tensor_size(sum_non_linear_term_diff_dim)]
        
        
#         cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
#         add_mem = cur_mem - prev_mem
#         prev_mem = cur_mem
#         print("added mem: %sM"%(add_mem))
        
#         del multi_res, non_linear_term, sum_non_linear_term, sum_non_linear_term_diff_dim
        
#         cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
#         add_mem = cur_mem - prev_mem
#         prev_mem = cur_mem
#         print("added mem: %sM"%(add_mem))
# #         print('size::', sys.getsizeof(theta)/np.power(2, 10), sys.getsizeof(X_Y_mult)/np.power(2, 10))
#         
#         print('size::', sizes)
        
    
        
#     print('total_time::', total_time)
    
    return theta, total_time





def compute_model_parameter_by_iteration2(batch_size, theta_list, grad_list, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, dim, theta,  X, Y, selected_rows, num_class, max_epoch, alpha, beta):
    
    total_time = 0.0
    
#     selected_rows_set = set(selected_rows.view(-1).tolist())
    
#     theta_list = []
    
#     for j in range(max_epoch):

    theta_list = []
    
    grad_list = []
    
    output_list = []
    
    x_sum_by_class_list = []

    end = False
    epoch = 0
    
    overhead = 0
    
    t_time = 0
    
    t_time2 = 0
    
    for k in range(len(random_ids_multi_super_iterations)):
        
        random_ids = random_ids_multi_super_iterations[k]
        
#         for i in range(len(batch_X_list)):

        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
#         all_indexes = np.sort(np.searchsorted(random_ids.numpy(), delta_ids.numpy()))
        
        all_indexes = np.sort(sort_idx[np.searchsorted(random_ids.numpy(),selected_rows.numpy(),sorter = sort_idx)])

        id_start = 0
        
        id_end = 0


        for i in range(0, X.shape[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            
#             curr_rand_ids = random_ids[i:end_id]
            
#             curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)


            while 1:
                if id_end >= all_indexes.shape[0] or all_indexes[id_end] >= end_id:
                    break
                
                id_end = id_end + 1
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
            
            curr_matched_ids_size = curr_matched_ids.shape[0]



#             curr_matched_ids,_ = torch.sort(curr_matched_ids)

#             print(curr_matched_ids)
            
            
            batch_X = X[curr_matched_ids]
            
            batch_Y = Y[curr_matched_ids]
            
            t3 = time.time()
            
            X_times_theta = torch.mm(batch_X, theta)
            
            t1 = time.time()

            output = softmax_layer(X_times_theta)
            
            t2 = time.time()
        
            overhead += (t2 - t1)
            
            t1 = time.time()
            
            output = torch.mm(torch.t(batch_X), output)
            
            
            
            
            
            
            output = torch.reshape(torch.t(output), [-1,1])
            
            reshape_theta = torch.reshape(torch.t(theta), (-1, 1))
            
            x_sum_by_class = compute_x_sum_by_class(batch_X, batch_Y, num_class, batch_X.shape)
            
            
            output -= x_sum_by_class
            
            output /= curr_matched_ids_size
#             output_list.append(output)
            
#             x_sum_by_class_list.append(x_sum_by_class)
            
            output += beta*reshape_theta
            
            reshape_theta -= alpha * output
            
            
            theta = reshape_theta.view(num_class, dim[1]).T
            
#             theta_list.append(theta)
            
#             grad_list.append(grad)
            
#             print('theta_diff:', torch.norm(theta - theta_list[epoch]))
#             
#             print('grad_diff:', torch.norm(grad.view(theta.shape) - grad_list[epoch]))
            
            t2 = time.time()
            
            t_time2 += (t2 - t1)
            
            t4 = time.time()
            
            t_time += (t4 - t3)
            
            epoch = epoch + 1
            
            id_start = id_end
            
            if epoch >= max_epoch:
                end = True
                break
        
        if end == True:
            break
        
    
#             theta_list.append(theta)
    print('overhead::', overhead)
    
    print('t_time::', t_time)
    
    print('t_time2::', t_time2)
    
    return theta, total_time,theta_list, grad_list, output_list, x_sum_by_class_list

# def compute_model_parameter_by_iteration2(dim, theta,  X_Y_mult, max_epoch):
#     
#     total_time = 0.0
#     
# #     pid = os.getpid()
#     
# #     prev_mem=0
# #     
# #     print('pid::', pid)
#     
#     for i in range(max_epoch):
#         
# #         multi_res = torch.mm(X_Y_mult, theta)
#         
# #         w_seq, b_seq, total_time = compute_curr_linear_paras(multi_res, Pi, total_time)
# #         t1 = time.time()
# #         lin_res = 1 - sig_layer(torch.mm(X_Y_mult, theta))
# #         t2 = time.time()
# #         
# #         total_time += t2 - t1
# #         multi_res *= w_seq
# #         
# #         multi_res += b_seq
#         
#         
#         non_linear_term = X_Y_mult*(1 - sig_layer(torch.mm(X_Y_mult, theta)))
#         
#         
#         if i == max_epoch - 1:
#             for j in range(non_linear_term.shape[0]):
#                 print(j, non_linear_term[j], theta)
#         
#         
# #         w_b_seq = compute_curr_linear_paras(multi_res, Pi)
# #          
# #         non_linear_term = X.mul((multi_res*(w_b_seq[:,0].view([dim[0],1])) + w_b_seq[:,1].view([dim[0],1]))*Y)
# #         sum_non_linear_term = torch.sum(non_linear_term, dim=0)
# #         sum_non_linear_term = np.sum(non_linear_term, axis=0)
#         
# #         print(sum_non_linear_term.shape)
#         
# #         sum_non_linear_term_diff_dim = np.reshape(sum_non_linear_term, (theta.shape))
# #         sum_non_linear_term_diff_dim = torch.sum(non_linear_term, dim=0).view(theta.shape)
# #         sum_non_linear_term_diff_dim *=  
#         
# #         theta = (1-alpha*beta)*theta + (alpha/dim[0])*torch.sum(X_Y_mult.mul(multi_res), dim=0).view(theta.shape)
#         theta = (1-alpha*beta)*theta + (alpha/dim[0])*torch.sum(non_linear_term, dim=0).view(theta.shape)
#         
# #         sizes = [get_tensor_size(multi_res), get_tensor_size(w_seq), get_tensor_size(b_seq)
# #                  , get_tensor_size(non_linear_term), get_tensor_size(sum_non_linear_term), get_tensor_size(sum_non_linear_term_diff_dim)]
#         
#         
# #         cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
# #         add_mem = cur_mem - prev_mem
# #         prev_mem = cur_mem
# #         print("added mem: %sM"%(add_mem))
#         
# #         del multi_res, non_linear_term, sum_non_linear_term, sum_non_linear_term_diff_dim
#         
# #         cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
# #         add_mem = cur_mem - prev_mem
# #         prev_mem = cur_mem
# #         print("added mem: %sM"%(add_mem))
# # #         print('size::', sys.getsizeof(theta)/np.power(2, 10), sys.getsizeof(X_Y_mult)/np.power(2, 10))
# #         
# #         print('size::', sizes)
#         
#     
#         
# #     print('total_time::', total_time)
#     
#     return theta, total_time


def compute_model_parameter_by_approx2(dim, theta, Pi, X_Y_mult, max_epoch):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    for i in range(max_epoch):
        
        multi_res = np.matmul(X_Y_mult, theta)
#         multi_res = torch.mm(X_Y_mult, theta)
        
        lin_res, total_time = compute_curr_linear_paras(multi_res, Pi, total_time)
        
#         multi_res *= w_seq
#         
#         multi_res += b_seq
        
        
        non_linear_term = X_Y_mult*(lin_res)
#         w_b_seq = compute_curr_linear_paras(multi_res, Pi)
#          
#         non_linear_term = X.mul((multi_res*(w_b_seq[:,0].view([dim[0],1])) + w_b_seq[:,1].view([dim[0],1]))*Y)
#         sum_non_linear_term = torch.sum(non_linear_term, dim=0)
        sum_non_linear_term = np.sum(non_linear_term, axis=0)
        
#         print(sum_non_linear_term.shape)
        
        sum_non_linear_term_diff_dim = np.reshape(sum_non_linear_term, (theta.shape))
#         sum_non_linear_term_diff_dim = sum_non_linear_term.view( (theta.shape))
        
#         sum_non_linear_term_diff_dim *=  
        
#         theta = (1-alpha*beta)*theta + (alpha/dim[0])*torch.sum(X_Y_mult.mul(multi_res), dim=0).view(theta.shape)
        theta = (1-alpha*beta)*theta + (alpha/dim[0])*sum_non_linear_term_diff_dim
        
#         sizes = [get_tensor_size(multi_res), get_tensor_size(w_seq), get_tensor_size(b_seq)
#                  , get_tensor_size(non_linear_term), get_tensor_size(sum_non_linear_term), get_tensor_size(sum_non_linear_term_diff_dim)]
        
        
#         cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
#         add_mem = cur_mem - prev_mem
#         prev_mem = cur_mem
#         print("added mem: %sM"%(add_mem))
        
#         del multi_res, non_linear_term, sum_non_linear_term, sum_non_linear_term_diff_dim
        
#         cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
#         add_mem = cur_mem - prev_mem
#         prev_mem = cur_mem
#         print("added mem: %sM"%(add_mem))
# #         print('size::', sys.getsizeof(theta)/np.power(2, 10), sys.getsizeof(X_Y_mult)/np.power(2, 10))
#         
#         print('size::', sizes)
        
    
        
    print('total_time::', total_time)
    
    return theta


def compute_model_parameter_by_approx_incremental_2(term1, term2, x_sum_by_class, dim, theta, num_class, max_epoch):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    vectorized_theta = theta.view(-1,1)
    
    for i in range(max_epoch):
        
#         multi_res = np.matmul(X_Y_mult, theta)
#         multi_res = torch.mm(X_Y_mult, theta)

#         sum_sub_term1 = torch.sum(sub_term1[i], dim = 0)
#         
#         sum_sub_term2 = torch.sum(sub_term2[i], dim = 0)
        
#         print(term1[i].shape)
#         
#         print(term2[i].shape)
#         
#         print(sub_term1[i].shape)
#         
#         print(sub_term2[i].shape)

        output = torch.mm(term1[i], vectorized_theta) + (term2[i].view(-1,1))
        
        
#         print('gradient::', gradient)
#         
#         print('approx_output::', output/dim[0])
        
        
        gradient = (output- x_sum_by_class)/dim[0] + beta*vectorized_theta
#         
#         print('x_sum_by_class::', x_sum_by_class/dim[0])
        
        
        
        
        vectorized_theta = vectorized_theta - alpha*gradient
        
        
        
        
#         theta = (1-alpha*beta)*theta + (alpha*torch.mm(term1[i], theta) + alpha*(term2[i]).view(theta.shape))/dim[0]

#         print(multi_res.shape, sub_w_seq[:,i].shape, sub_b_seq[:,i].shape)
        
#         lin_res = multi_res * (sub_w_seq[:, i].view(multi_res.shape)) + sub_b_seq[:, i].view(multi_res.shape)
#         
# #         print(lin_res.shape)
# #         print(lin_res.shape, X_Y_mult.shape, sub_w_seq.shape, sub_b_seq.shape)
#         
# #         lin_res, total_time = compute_curr_linear_paras(multi_res, Pi, total_time)
#         
# #         multi_res *= w_seq
# #         
# #         multi_res += b_seq
#         
#         
#         non_linear_term = X_Y_mult*(lin_res)
# #         w_b_seq = compute_curr_linear_paras(multi_res, Pi)
# #          
# #         non_linear_term = X.mul((multi_res*(w_b_seq[:,0].view([dim[0],1])) + w_b_seq[:,1].view([dim[0],1]))*Y)
#         sum_non_linear_term = torch.sum(non_linear_term, dim=0)
# #         sum_non_linear_term = np.sum(non_linear_term, axis=0)
#         
# #         print(sum_non_linear_term.shape)
#         
# #         sum_non_linear_term_diff_dim = np.reshape(sum_non_linear_term, (theta.shape))
#         sum_non_linear_term_diff_dim = sum_non_linear_term.view( (theta.shape))
#         
# #         sum_non_linear_term_diff_dim *=  
#         
# #         theta = (1-alpha*beta)*theta + (alpha/dim[0])*torch.sum(X_Y_mult.mul(multi_res), dim=0).view(theta.shape)
#         theta = (1-alpha*beta)*theta + (alpha/dim[0])*sum_non_linear_term_diff_dim
        
#         sizes = [get_tensor_size(multi_res), get_tensor_size(w_seq), get_tensor_size(b_seq)
#                  , get_tensor_size(non_linear_term), get_tensor_size(sum_non_linear_term), get_tensor_size(sum_non_linear_term_diff_dim)]
        
        
#         cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
#         add_mem = cur_mem - prev_mem
#         prev_mem = cur_mem
#         print("added mem: %sM"%(add_mem))
        
#         del multi_res, non_linear_term, sum_non_linear_term, sum_non_linear_term_diff_dim
        
#         cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
#         add_mem = cur_mem - prev_mem
#         prev_mem = cur_mem
#         print("added mem: %sM"%(add_mem))
# #         print('size::', sys.getsizeof(theta)/np.power(2, 10), sys.getsizeof(X_Y_mult)/np.power(2, 10))
#         
#         print('size::', sizes)
        
    
        
    print('total_time::', total_time)
    
    theta = torch.t(vectorized_theta.view(num_class, dim[1]))
    
    return theta

def compute_model_parameter_by_approx_incremental_4(term1, term2, x_sum_by_class, dim, theta, num_class, max_epoch, cut_off_epoch, epoch_record_epoch_seq, alpha, beta):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    vectorized_theta = theta.view(-1,1)
    
#     for i in range(max_epoch):
    for i in range(cut_off_epoch):
        
#         multi_res = np.matmul(X_Y_mult, theta)
#         multi_res = torch.mm(X_Y_mult, theta)

#         sum_sub_term1 = torch.sum(sub_term1[i], dim = 0)
#         
#         sum_sub_term2 = torch.sum(sub_term2[i], dim = 0)
        
#         print(term1[i].shape)
#         
#         print(term2[i].shape)
#         
#         print(sub_term1[i].shape)
#         
#         print(sub_term2[i].shape)



        if i < cut_off_epoch:

            output = torch.mm(term1[i], vectorized_theta) + (term2[i].view(-1,1))
            
            
    #         print('gradient::', gradient)
    #         
    #         print('approx_output::', output/dim[0])
            
            
            gradient = (output- x_sum_by_class)/dim[0] + beta*vectorized_theta
    #         
    #         print('x_sum_by_class::', x_sum_by_class/dim[0])
            
            
            
            
            vectorized_theta = vectorized_theta - alpha*gradient
        
        else:
            
            output = torch.mm(term1[cut_off_epoch - 1], vectorized_theta) + (term2[cut_off_epoch - 1].view(-1,1))
            
            
    #         print('gradient::', gradient)
    #         
    #         print('approx_output::', output/dim[0])
            
            
            gradient = (output- x_sum_by_class)/dim[0] + beta*vectorized_theta
    #         
    #         print('x_sum_by_class::', x_sum_by_class/dim[0])
            
            
            
            
            vectorized_theta = vectorized_theta - alpha*gradient
        
        
#         theta = (1-alpha*beta)*theta + (alpha*torch.mm(term1[i], theta) + alpha*(term2[i]).view(theta.shape))/dim[0]

#         print(multi_res.shape, sub_w_seq[:,i].shape, sub_b_seq[:,i].shape)
        
#         lin_res = multi_res * (sub_w_seq[:, i].view(multi_res.shape)) + sub_b_seq[:, i].view(multi_res.shape)
#         
# #         print(lin_res.shape)
# #         print(lin_res.shape, X_Y_mult.shape, sub_w_seq.shape, sub_b_seq.shape)
#         
# #         lin_res, total_time = compute_curr_linear_paras(multi_res, Pi, total_time)
#         
# #         multi_res *= w_seq
# #         
# #         multi_res += b_seq
#         
#         
#         non_linear_term = X_Y_mult*(lin_res)
# #         w_b_seq = compute_curr_linear_paras(multi_res, Pi)
# #          
# #         non_linear_term = X.mul((multi_res*(w_b_seq[:,0].view([dim[0],1])) + w_b_seq[:,1].view([dim[0],1]))*Y)
#         sum_non_linear_term = torch.sum(non_linear_term, dim=0)
# #         sum_non_linear_term = np.sum(non_linear_term, axis=0)
#         
# #         print(sum_non_linear_term.shape)
#         
# #         sum_non_linear_term_diff_dim = np.reshape(sum_non_linear_term, (theta.shape))
#         sum_non_linear_term_diff_dim = sum_non_linear_term.view( (theta.shape))
#         
# #         sum_non_linear_term_diff_dim *=  
#         
# #         theta = (1-alpha*beta)*theta + (alpha/dim[0])*torch.sum(X_Y_mult.mul(multi_res), dim=0).view(theta.shape)
#         theta = (1-alpha*beta)*theta + (alpha/dim[0])*sum_non_linear_term_diff_dim
        
#         sizes = [get_tensor_size(multi_res), get_tensor_size(w_seq), get_tensor_size(b_seq)
#                  , get_tensor_size(non_linear_term), get_tensor_size(sum_non_linear_term), get_tensor_size(sum_non_linear_term_diff_dim)]
        
        
#         cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
#         add_mem = cur_mem - prev_mem
#         prev_mem = cur_mem
#         print("added mem: %sM"%(add_mem))
        
#         del multi_res, non_linear_term, sum_non_linear_term, sum_non_linear_term_diff_dim
        
#         cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
#         add_mem = cur_mem - prev_mem
#         prev_mem = cur_mem
#         print("added mem: %sM"%(add_mem))
# #         print('size::', sys.getsizeof(theta)/np.power(2, 10), sys.getsizeof(X_Y_mult)/np.power(2, 10))
#         
#         print('size::', sizes)

    t1 = time.time()

    A = (1- beta*alpha)*torch.eye(dim[1]*num_class, dtype = torch.double) - alpha*term1[cut_off_epoch - 1]/dim[0]
    
    B = -alpha/dim[0]*(term2[cut_off_epoch - 1].view(-1,1) - x_sum_by_class)
    
    
    s, M = torch.eig(A, True)
    
    s = s[:,0]
    
    s_power = torch.pow(s, float(max_epoch - cut_off_epoch))
    
    res1 = M.mul(s_power.view(1,-1))

    res1 = torch.mm(res1, torch.inverse(M))
    
    
#     temp = torch.eye(dim[1], dtype = torch.double)
#     
#     sum_temp = torch.zeros((dim[1], dim[1]), dtype = torch.double)
#     
#     for i in range(max_epoch):
#         sum_temp += temp
#         temp = torch.mm(temp, A)
        
    
    
#     print('temp_gap::', temp - res1)
    
    sub_sum = (1-s_power)/(1-s)
    
    res2 = M.mul(sub_sum.view(1, -1))
    
    res2 = torch.mm(res2, torch.inverse(M))
    
    
    
    
    
    
    
    
    
    
    
    
    
#     u,s,v = torch.svd(A)
#     
#     
#     s_power = torch.pow(s, float(max_epoch - cut_off_epoch))
#     
#     res1 = u.mul(s_power.view(1,-1))
# 
#     res1 = torch.mm(res1, torch.t(v))
#     
#     res2 = u.mul((1-s_power)/(1-s))
#     
#     res2 = torch.mm(res2, torch.t(v))
    
    vectorized_theta = torch.mm(res1, vectorized_theta) + torch.mm(res2, B)
    
    t2 = time.time()
    
    print('total_time::', t2 - t1) 
        
#     print('total_time::', total_time)
    
    theta = torch.t(vectorized_theta.view(num_class, dim[1]))
    
    return theta



def compute_model_parameter_by_approx_incremental_1(A, B, term1, term2, dim, theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    
    num = 0
     
    theta = theta.view(-1,1) 
    
    min_batch_num_per_epoch = int((dim[0] - 1)/batch_size) + 1


    
    
    for i in range(A.shape[0]):
        
#         curr_theta = theta.clone()
#             
#         for j in range(0, dim[0], batch_size):
#             
#             curr_min_epoch = int(j/batch_size)
#             
#             curr_A = (1 - alpha*beta)*torch.eye(dim[1]*num_class, dtype = torch.double) - alpha*term1[i*min_batch_num_per_epoch + curr_min_epoch]
#                 
#             curr_B = -alpha*term2[i*min_batch_num_per_epoch + curr_min_epoch]
#          
#             curr_theta = torch.mm(curr_A, curr_theta) + curr_B.view(curr_theta.shape)
#          
#             if i > 0:
#                 print(j, torch.t((curr_theta).view(num_class, dim[1])))
        
        theta = torch.mm(A[i], theta) + B[i]
        
#         print(torch.t((theta).view(num_class, dim[1])))
        
    if A.shape[0] >= max_epoch:
        return torch.t((theta).view(num_class, dim[1]))    
        
    
         
    num = A.shape[0]*min_batch_num_per_epoch
     
     
    this_A = torch.eye(dim[1], dtype = torch.double)
     
    this_B = torch.zeros([dim[1], 1], dtype = torch.double)
     
     
    if cut_off_epoch > min_batch_num_per_epoch: 
        avg_term1 = torch.sum(term1[-min_batch_num_per_epoch:-1], 0)/min_batch_num_per_epoch
        avg_term2 = torch.sum(term2[-min_batch_num_per_epoch:-1], 0)/min_batch_num_per_epoch
    else:
        avg_term1 = torch.sum(term1, 0)/cut_off_epoch
        avg_term2 = torch.sum(term2, 0)/cut_off_epoch
     
#     avg_term2 = torch.zeros([dim[1], 1], dtype = torch.double)
     
     
     
#     for j in range(0, dim[0],batch_size):
#             
#         end_id = j + batch_size
#         
#         if end_id > dim[0]:
#             end_id = dim[0]
# 
# 
# 
#         if num < cut_off_epoch:
#             gradient = -(torch.mm(term1[num], theta) + (term2[num]).view(theta.shape)) + beta*theta
#             
#             avg_term1 += term1[num]
#         
#             avg_term2 += term2[num] 
#             
#         else:
#             gradient = -(torch.mm(term1[cut_off_epoch - 1], theta) + (term2[cut_off_epoch - 1]).view(theta.shape)) + beta*theta
#         
#             avg_term1 += term1[num]
#         
#             avg_term2 += term2[num] 
#         
# #             if num < cut_off_epoch:
#         theta = theta - alpha * gradient
     
    last_A = torch.eye(dim[1]*num_class, dtype = torch.double)
    
    
    last_B = torch.zeros([dim[1]*num_class, 1], dtype = torch.double) 
    
    for j in range(0, dim[0],batch_size):
        
        end_id = j + batch_size
        
        if end_id > dim[0]:
            end_id = dim[0]



        if num < cut_off_epoch:
            gradient = (torch.mm(term1[num], theta) + (term2[num]).view(theta.shape)) + beta*theta
            
            
            
            curr_A = (1-alpha*beta)*torch.eye(dim[1]*num_class, dtype = torch.double) - alpha*term1[num]
            
            
            curr_B = -term2[num]*alpha
            
            
        else:
            gradient = (torch.mm(avg_term1, theta) + (avg_term2).view(theta.shape)) + beta*theta
            
            curr_A = (1-alpha*beta)*torch.eye(dim[1]*num_class, dtype = torch.double) - alpha*avg_term1
            
            curr_B = -avg_term2*alpha
        
#             if num < cut_off_epoch:

        last_A = torch.mm(last_A, curr_A)
                
        last_B = torch.mm(curr_A, last_B) + curr_B.view(dim[1]*num_class, 1)

        theta = theta - alpha * gradient


#             print('gradient::', gradient)
         
#             print('theta::', theta)
        
        num += 1

         
     
     
     
    for i in range(max_epoch - A.shape[0] - 1):
        theta = torch.mm(last_A, theta) + last_B
#         for j in range(0, dim[0],batch_size):
#              
#             end_id = j + batch_size
#              
#             if end_id > dim[0]:
#                 end_id = dim[0]
#      
#      
#      
#             if num < cut_off_epoch:
#                 gradient = -(torch.mm(term1[num], theta) + (term2[num]).view(theta.shape)) + beta*theta
#                  
#             else:
#                 gradient = -(torch.mm(avg_term1, theta) + (avg_term2).view(theta.shape)) + beta*theta
#              
# #             if num < cut_off_epoch:
#             theta = theta - alpha * gradient
#  
#  
# #             print('gradient::', gradient)
#               
# #             print('theta::', theta)
#              
#             num += 1
            
    
    
    
#     for i in range(max_epoch):
#          
# #         print('epoch::', i)
#         computed_theta = torch.mm(A[i], theta) + B[i]
#          
#         for j in range(0, dim[0],batch_size):
#              
#             print('batch::', j)
#              
#             end_id = j + batch_size
#              
#             if end_id > dim[0]:
#                 end_id = dim[0]
#      
#      
#      
#             if num < cut_off_epoch:
#                 gradient = -(torch.mm(term1[num], theta) + (term2[num]).view(theta.shape)) + beta*theta
#                  
#             else:
#                 gradient = -(torch.mm(term1[cut_off_epoch - 1], theta) + (term2[cut_off_epoch - 1]).view(theta.shape)) + beta*theta
#              
# #             if num < cut_off_epoch:
#             theta = theta - alpha * gradient
#  
#  
#             print('gradient::', gradient)
#                 
#             print('theta::', theta)
#              
#             num += 1
#             
#         
#         y  = 0
#         
#         y += 1
        
            
    
#     for i in range(cut_off_epoch):
#         
# #         multi_res = np.matmul(X_Y_mult, theta)
# #         multi_res = torch.mm(X_Y_mult, theta)
# 
# #         sum_sub_term1 = torch.sum(sub_term1[i], dim = 0)
# #         
# #         sum_sub_term2 = torch.sum(sub_term2[i], dim = 0)
#         
# #         print(term1[i].shape)
# #         
# #         print(term2[i].shape)
# #         
# #         print(sub_term1[i].shape)
# #         
# #         print(sub_term2[i].shape)
# 
#         if i < cut_off_epoch:
#             gradient = -(torch.mm(term1[i], theta) + (term2[i]).view(theta.shape))/dim[0] + beta*theta
#         
#         else:
#             gradient = -(torch.mm(term1[cut_off_epoch - 1], theta) + (term2[cut_off_epoch - 1]).view(theta.shape))/dim[0] + beta*theta
# 
# #         print('approx_gradient::', gradient)
# 
#         theta = theta - alpha * gradient
# 
# 
# 
#     A = (1- beta*alpha)*torch.eye(dim[1], dtype = torch.double) + alpha*term1[cut_off_epoch - 1]/dim[0]
#     
#     B = alpha/dim[0]*(term2[cut_off_epoch - 1].view(-1,1))
#     
#     
#     s, M = torch.eig(A, True)
#     
#     s = s[:,0]
#     
#     s_power = torch.pow(s, float(max_epoch - cut_off_epoch))
#     
#     res1 = M.mul(s_power.view(1,-1))
# 
#     res1 = torch.mm(res1, torch.t(M))
#     
#     
# #     temp = torch.eye(dim[1], dtype = torch.double)
# #     
# #     sum_temp = torch.zeros((dim[1], dim[1]), dtype = torch.double)
# #     
# #     for i in range(max_epoch):
# #         sum_temp += temp
# #         temp = torch.mm(temp, A)
#         
#     
#     
# #     print('temp_gap::', temp - res1)
#     
#     sub_sum = (1-s_power)/(1-s)
#     
#     res2 = M.mul(sub_sum.view(1, -1))
#     
#     res2 = torch.mm(res2, torch.t(M))
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
#     
#     
# #     u,s,v = torch.svd(A)
# #     
# #     
# #     s_power = torch.pow(s, float(max_epoch - cut_off_epoch))
# #     
# #     res1 = u.mul(s_power.view(1,-1))
# # 
# #     res1 = torch.mm(res1, torch.t(v))
# #     
# #     res2 = u.mul((1-s_power)/(1-s))
# #     
# #     res2 = torch.mm(res2, torch.t(v))
#     
#     theta = torch.mm(res1, theta) + torch.mm(res2, B)


    theta = torch.t(theta.view(num_class, dim[1]))
    
    return theta
    
        
#     print('total_time::', total_time)
#     
#     return theta



def compute_sub_term_1(X_times_theta, X, weights, dim, num_class):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    '''X_times_theta: n*q'''
    '''dim[0]*dim[1]*(num_class)'''
    
    res1 = torch.bmm(X_times_theta.view(dim[0], 1, dim[1]), weights.view(dim[0], num_class, num_class)).view(dim[0], num_class)
    
    '''dim[1],num_class, num_class*num_class'''
    res2 = Variable(torch.mm(torch.t(res1), X).view(num_class, X.shape[1]).view(-1,1))
    
    del res1    
    
    return res2


def prepare_sub_term_2(X, offsets, dim, num_class):
    
    
    '''offsets:: dim[0], num_class, 1'''
    '''x:: dim[0], dim[1]'''
#     res = torch.t(torch.mm(torch.t(X), offsets.view(dim[0], num_class))).view(1,-1)
    
    res = Variable(torch.mm(torch.t(offsets.view(dim[0], num_class)), X).view(1,-1))
    
#     res = torch.reshape(res, [1, num_class*dim[1]])
    
    return res


def get_subset_data_per_epoch(curr_rand_ids, full_id_set):
    
    
#     ids = torch.nonzero(curr_rand_ids.view(-1,1) == full_id_set.view(1,-1))
#     
#     return curr_rand_ids[ids[:,0]]
    
    
    
    curr_rand_id_set = set(curr_rand_ids.tolist())
            
    curr_matched_ids = torch.tensor(list(curr_rand_id_set.intersection(full_id_set)))
    
    return curr_matched_ids

def get_subset_data_per_epoch2(curr_rand_ids, full_id_set):
    curr_rand_id_set = set(curr_rand_ids.view(-1).tolist())
            
    intersected_ids = full_id_set.intersection(curr_rand_id_set)        
    
    curr_matched_ids = np.array(list(intersected_ids))
    
    curr_non_matched_ids = np.array(list(curr_rand_id_set - intersected_ids))
    
    return curr_matched_ids, curr_non_matched_ids

'''res1 = torch.bmm(batch_x.view(curr_batch_size, dim[1], 1), curr_weights.view(curr_batch_size, 1, curr_weights.shape[1]*num_class*num_class))
    
        res2 = torch.mm(torch.t(batch_x), res1.view(curr_batch_size, dim[1]*curr_weights.shape[1]*num_class*num_class)).view(dim[1]*dim[1], curr_weights.shape[1], num_class*num_class)
        
        del res1
        
        res3 = torch.transpose(torch.transpose(res2, 0, 1), 2, 1).view(curr_weights.shape[1], num_class, num_class, dim[1], dim[1])
    
        del res2
    
        res4 = torch.reshape(torch.transpose(res3, 2, 3), [curr_weights.shape[1], num_class*dim[1], dim[1]*num_class])'''





def prepare_sub_term_1(X, weights, dim, num_class):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    
    
    
    w_dim = weights.shape
    
#     print(w_dim)
#     
#     print(dim)    
    '''dim[0]*dim[1]*(max_epoch*num_class*num_class)'''
    
#     t1 = time.time()
    
    res1 = torch.bmm(X.view(dim[0], dim[1], 1), weights.view(dim[0], 1, num_class*num_class))
    
    '''dim[1],dim[1]*t*num_class*num_class'''
    res2 = torch.mm(torch.t(X), res1.view(dim[0], dim[1]*num_class*num_class)).view(dim[1]*dim[1], num_class*num_class)
    
    del res1
    
    res3 = torch.transpose(res2, 0, 1).view(num_class, num_class, dim[1], dim[1])
    
    del res2
    
    res4 = torch.reshape(torch.transpose(res3, 1, 2), [num_class*dim[1], dim[1]*num_class])
    
#     res4 = torch.transpose(res3, 2, 3).view(w_dim[1], dim[1]*num_class, dim[1]*num_class)
    
    del res3
    
#     t2 = time.time()
#     
#     print('time::', t2 - t1)    
    
    return res4

def compute_model_parameter_by_approx_incremental_1_3(weights, offsets, batch_size, theta_list, grad_list, random_ids_multi_super_iterations, dim, theta,  X, Y, selected_rows, num_class, max_epoch, alpha, beta, x_sum_by_class_list):
    
    total_time = 0.0
    
    selected_rows_set = set(selected_rows.view(-1).tolist())
    
#     theta_list = []
    
#     for j in range(max_epoch):

    theta_list = []
    
    grad_list = []
    
    output_list = []
    
#     x_sum_by_class_list = []
    cut_off_super_iteration = int((weights.shape[0]-1)/dim[0])

    end = False
    epoch = 0
    
    overhead = 0
    t_time = 0
    t_time2 = 0
    
    for j in range(len(random_ids_multi_super_iterations)):
        
        random_ids = random_ids_multi_super_iterations[j]
        
        super_iter_id = j
        
        if j > cut_off_super_iteration:
            super_iter_id = cut_off_super_iteration
        
#         for i in range(len(batch_X_list)):
        for i in range(0, X.shape[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            
            curr_rand_ids = random_ids[i:end_id]
            
            curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
#             curr_matched_ids,_ = torch.sort(curr_matched_ids)

#             print(curr_matched_ids)
            
            batch_X = X[curr_matched_ids]
            
#             delta_Y = Y[curr_non_matched_ids]
            
            batch_Y = Y[curr_matched_ids]
            t3 = time.time()
            matched_ids = curr_matched_ids + super_iter_id*dim[0]
            
            batch_weights = weights[matched_ids]
            
            batch_offsets = offsets[matched_ids]
            
            t4 = time.time()
            
            t_time += (t4 - t3)
            
            
            X_times_theta = torch.mm(batch_X, theta)
            
#             res1 = torch.bmm(X_times_theta.view(batch_X.shape[0], 1, num_class), batch_weights.view(batch_X.shape[0], num_class, num_class))
    
            t1 = time.time()
            res1 = torch.baddbmm(batch_offsets.view(batch_X.shape[0], 1, num_class), X_times_theta.view(batch_X.shape[0], 1, num_class), batch_weights.view(batch_X.shape[0], num_class, num_class))
            '''dim[1],num_class, num_class*num_class'''
            t2 = time.time()
            
            overhead += (t2 - t1)
#             res1 = 
            
            t1 = time.time()
            res2 = torch.mm(torch.t(batch_X), res1.view(batch_X.shape[0], num_class))

            
            
            output = torch.reshape(torch.t(res2), [-1,1])

#             output = softmax_layer(torch.mm(batch_X, theta))
#             
#             
#             output = torch.mm(torch.t(batch_X), output)
#             
#             
#             output = torch.reshape(torch.t(output), [-1,1])
            
            reshape_theta = torch.reshape(torch.t(theta), (-1, 1))
            
            x_sum_by_class = compute_x_sum_by_class(batch_X, batch_Y, num_class, batch_X.shape)
            
#             output_list.append(output)
            
#             x_sum_by_class_list.append(x_sum_by_class)
            
            grad = (output - x_sum_by_class)/batch_X.shape[0] + beta*reshape_theta
            
            res = reshape_theta - alpha * grad
            
            
            theta = torch.t(res.view(num_class, dim[1]))
            
            t2 = time.time()
            
            t_time2 += (t2 - t1)
            
            

#             theta_list.append(theta)
            
#             grad_list.append(grad)
            
#             print('theta_diff:', torch.norm(theta - theta_list[epoch]))
#             
#             print('grad_diff:', torch.norm(grad.view(theta.shape) - grad_list[epoch]))
            
            epoch = epoch + 1
            
            if epoch >= max_epoch:
                end = True
                break
        
        if end == True:
            break
        
    
#             theta_list.append(theta)
    
    print('overhead::', overhead)
    
    print('t_time::', t_time)
    
    print('t_time2::', t_time2)
    
    return theta

    

def compute_model_parameter_by_approx_incremental_1_2(output_list, exp_x_sum_by_class_list, theta_list, grad_list, origin_X, origin_Y, weights, offsets, delta_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, term1, term2, dim, theta, max_epoch, alpha, beta, batch_size, num_class, x_sum_by_class_list, avg_term2, u_list, v_s_list):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    
#     num = 0
     
#     theta = theta.view(-1,1) 
    
#     batch_num_per_epoch = int((X.shape[0] - 1)/batch_size) + 1
    min_batch_num_per_epoch = int((origin_X.shape[0] - 1)/batch_size) + 1
    cut_off_super_iteration = int((weights.shape[0]-1)/dim[0])
#     
#     cut_off_random_ids_multi_super_iterations = random_ids_multi_super_iterations[0:cut_off_super_iteration]
# 
# 
# 
#     matched_ids = (cut_off_random_ids_multi_super_iterations.view(-1,1) == delta_ids.view(1,-1))
#     
#     '''T, n, |delta_X|'''
#     
#     matched_ids = matched_ids.view(cut_off_super_iteration, dim[0], delta_ids.shape[0])
#         
#         
#     '''n, T, |delta_X|'''
# #     matched_ids = torch.transpose(matched_ids, 1, 0)
#     
#     '''ids of [n, T, delta_X]'''
#     total_time = 0
#     
#     t1 = time.time()
#     
#     
#     all_noise_data_ids = delta_ids.view(1,-1) + (torch.tensor(list(range(cut_off_super_iteration)))*dim[0]).view(-1, 1)
#     
#     '''delta_X * T''' 
#     
#     all_noise_data_ids = all_noise_data_ids.view(-1)

    '''T, |delta_X|, q^2'''

#     curr_weights = weights[all_noise_data_ids].view(-1, delta_data_ids.view(-1).shape[0], num_class*num_class)
#     
#     '''T, |delta_X|, q'''    
#     curr_offsets = offsets[all_noise_data_ids].view(-1, delta_data_ids.view(-1).shape[0], num_class)

#     delta_X = origin_X[delta_ids]


    epoch = 0

    overhead = 0

    overhead2 = 0
    
    overhead3 = 0
#     
        
    delta_ids_set = set(delta_ids.view(-1).tolist())    
    
    X = origin_X
        
        
    Y = origin_Y
    
    vectorized_theta = Variable(torch.reshape(torch.t(theta), [-1,1]))
    
    theta = Variable(theta)
    
    
    
#     sub_term_2_list = []
    
    avg_A = 0
    
    avg_B = 0
    
    
    end = False
    
#     for k in range(max_epoch):
    for k in range(random_ids_multi_super_iterations.shape[0]):
    
#     for k in range(5):
        
        
        random_ids = random_ids_multi_super_iterations[k]
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
#         all_indexes = np.sort(np.searchsorted(random_ids.numpy(), delta_ids.numpy()))
        
        all_indexes = np.sort(sort_idx[np.searchsorted(random_ids.numpy(),delta_ids.numpy(),sorter = sort_idx)])
        
        
        super_iter_id = k
        
        if k > cut_off_super_iteration:
            super_iter_id = cut_off_super_iteration
        
#         else:
#             
#             matched_ids = (random_ids.view(-1,1) == delta_ids.view(1,-1))
#     
#             '''n, |delta_X|'''
#         
#             matched_ids = matched_ids.view(dim[0], delta_ids.shape[0])
#             
#             
#             '''|delta_X|*2'''
#             
#             nonzero_ids = torch.nonzero(matched_ids)
        
        
        end_id_super_iteration = (super_iter_id + 1)*dim[0]
        
        id_start = 0
    
        id_end = 0
        
        if end_id_super_iteration > weights.shape[0]:
            end_id_super_iteration = weights.shape[0]
            
        
        
#         weights_this_super_iteration = weights[super_iter_id*dim[0]:end_id_super_iteration]
        
#         delta_ids = delta_ids[delta_ids < weights_this_super_iteration.shape[0]]
        
        
#         weights_this_super_iteration = weights_this_super_iteration[delta_ids]
        
        
#         offsets_this_super_iteration = offsets[super_iter_id*dim[0]:end_id_super_iteration]
        
        
#         offsets_this_super_iteration = offsets_this_super_iteration[delta_ids]
        
        for i in range(0, dim[0], batch_size):
        
            end_id = i + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            
#             curr_rand_ids = random_ids[i:end_id]
            
            
#             curr_matched_ids = (get_subset_data_per_epoch(curr_rand_ids, delta_ids_set))
            
            
            
            
            while 1:
                if id_end >= all_indexes.shape[0] or all_indexes[id_end] >= end_id:
                    break
                
                id_end = id_end + 1
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
            
            curr_matched_ids_size = curr_matched_ids.shape[0]
            
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

                
#             if epoch < cut_off_epoch:
            
            
#             if epoch < cut_off_epoch:
            
#                 curr_weights = weights[k*dim[0] + i: k*dim[0] + end_id]
            
#             print(weights_this_super_iteration.shape)
#             
#             print(curr_matched_ids)
    
            sub_term2 = term2[epoch].view(-1,1)
            
            if curr_matched_ids_size > 0:
#                     batch_weights = weights_this_super_iteration[curr_matched_ids]
                t1 = time.time()
                coeff_rand_ids = curr_matched_ids + super_iter_id*dim[0]
                
                batch_weights = weights[coeff_rand_ids]
            
                batch_offsets = offsets[coeff_rand_ids]
#                     batch_offsets = offsets_this_super_iteration[curr_matched_ids]
            
#             else:
# #                 curr_weights = weights[(cut_off_super_iteration - 1)*dim[0] + i: (cut_off_super_iteration - 1)*dim[0] + end_id]
#                 
#                 batch_weights = weights_this_super_iteration[curr_nonzero_ids_this_batch]
#                 
#                 
#                 batch_offsets = offsets_this_super_iteration[curr_nonzero_ids_this_batch]
            
            
                
                batch_X_multi_theta = torch.mm(batch_delta_X, theta)
        
                t2 = time.time()
                overhead3 += (t2 - t1)
#                 vectorized_theta = Variable(torch.reshape(torch.t(theta), [-1,1]))
            
#             t1 = time.time()
            

#                 vectorized_sub_term_1 = (torch.reshape(sub_term_1_without_weights, [-1,1]))
#                 print(batch_offsets.shape, batch_delta_X.shape)
                t3 = time.time()
                sub_term2 -= torch.mm(torch.t(batch_offsets.view(curr_matched_ids_size, num_class)), batch_delta_X).view(-1,1)#prepare_sub_term_2(batch_delta_X, batch_offsets, batch_delta_X.shape, num_class).view(-1,1)
#                 sub_term2 -= prepare_sub_term_2(batch_delta_X, batch_offsets, batch_delta_X.shape, num_class).view(-1,1)
                
                t4 = time.time()
                overhead2 += (t4 - t3)
                
            t1 = time.time()
#             if epoch < cut_off_epoch:
#                 full_term1 = term1[epoch]
            
#             full_term2 = 
            
#             else:
#                 full_term1 = avg_term1
                
#                 full_term2 = avg_term2    
            
            delta_x_sum_by_class = x_sum_by_class_list[epoch]
            if curr_matched_ids_size > 0:
                delta_x_sum_by_class = delta_x_sum_by_class - compute_x_sum_by_class(batch_delta_X, batch_delta_Y, num_class, batch_delta_X.shape)
#                 delta_x_sum_by_class = x_sum_by_class_list[epoch] - delta_x_sum_by_class
                
#                 delta_x_sum_by_class = -delta_x_sum_by_class
#             curr_x_sum_by_class =  
            
#                 delta_x_sum_by_class.sub_()
            
            
#                 if epoch >= cut_off_epoch - min_batch_num_per_epoch:
#                     
#                     sub_term_1_without_weights = 0
#                     
#                     if curr_matched_ids.shape[0] > 0:
#                         sub_term_1_without_weights = prepare_sub_term_1(batch_delta_X, batch_weights, batch_delta_X.shape, num_class)
# 
#                     curr_A = Variable((1-alpha*beta)*torch.eye(vectorized_theta.shape[0], dtype = torch.double) - alpha*(term1[epoch] - sub_term_1_without_weights)/(end_id - i- curr_matched_ids.shape[0]))  
#                     curr_B = -alpha*(term2[epoch].view(-1,1) - x_sum_by_class)/(end_id - i- curr_matched_ids.shape[0])    
# 
#                     
#                     avg_A += (curr_A)
#                     avg_B += (curr_B)
#                     
#                     output = torch.mm((full_term1 - sub_term_1_without_weights), vectorized_theta) + (full_term2.view(-1,1) - sub_term2)
#                     
#                     del sub_term_1_without_weights
#                 
#                 else:
            
#             sub_term_1 = 0
            vectorized_sub_term_1 = 0
            if curr_matched_ids_size > 0:
#                 t5 = time.time()
                
                res1 = torch.bmm(batch_X_multi_theta.view(curr_matched_ids_size, 1, num_class), batch_weights).view(curr_matched_ids_size, num_class)
    
                '''dim[1],num_class, num_class*num_class'''
                vectorized_sub_term_1 = torch.mm(torch.t(res1), batch_delta_X).view(num_class, dim[1]).view(-1,1)
                
                del res1
                
#                 vectorized_sub_term_1 = (compute_sub_term_1(batch_X_multi_theta, batch_delta_X, batch_weights, batch_X_multi_theta.shape, num_class))
            
#                 vectorized_sub_term_1 = Variable(torch.reshape(sub_term_1, [-1,1]))
                
#                 t6 = time.time()
#                 
#                 overhead3 += (t6 - t5)
                
            
#             if epoch < cut_off_epoch:
                
#                 print(u_list[epoch].shape)
#                 print(v_s_list[epoch].shape)
#                 print(vectorized_theta.shape)
#             if batch_size < num_class*dim[1]:
            output = torch.mm(u_list[epoch], torch.mm(v_s_list[epoch], vectorized_theta))
            output -= vectorized_sub_term_1
            output += sub_term2
            output -= delta_x_sum_by_class
            output /= (end_id - i - curr_matched_ids_size)
#             else:
#                 output = torch.mm(term1[epoch], vectorized_theta) - vectorized_sub_term_1 + sub_term2
#                 output -= delta_x_sum_by_class
#                 output /= (end_id - i - curr_matched_ids_size)
#             else:
#                 output = torch.mm(avg_term1, vectorized_theta) - vectorized_sub_term_1 + (full_term2.view(-1,1) - sub_term2)
#                 output = torch.mm(avg_u, torch.mm(avg_s, vectorized_theta)) - vectorized_sub_term_1 + (full_term2.view(-1,1) - sub_term2)
#             del sub_term_1

            
            
            
            
#                 exp_x_sum_by_class = exp_x_sum_by_class_list[epoch]
#                 
#                 exp_full_x_sum_by_class = compute_x_sum_by_class(X[random_ids[i:end_id]], Y[random_ids[i:end_id]], num_class, X[random_ids[i:end_id]].shape)
#                 
#                 print(torch.norm(exp_full_x_sum_by_class - x_sum_by_class_list[epoch]))
#                 
#                 print('x_sum_diff::', torch.norm(exp_x_sum_by_class - x_sum_by_class))
#                 
#                 print('output_difference::', torch.norm(output - output_list[epoch]))
            
#             t2 = time.time()
#             
#             overhead += t2 - t1
            
            output += vectorized_theta*beta     
            t2 = time.time()
             
            overhead += (t2 - t1)
#             del output
            
#             del x_sum_by_class
            
            del delta_x_sum_by_class
            
            del sub_term2
            
            if curr_matched_ids_size > 0:
                del batch_delta_X
                del batch_delta_Y
                del batch_weights
                del batch_offsets
            
            vectorized_theta -= output*alpha  
            
            theta = vectorized_theta.view(num_class, dim[1]).T
            
#                 print(epoch)
            
#             exp_theta = theta_list[epoch]
#               
# #                 print(epoch, theta)
#             if epoch < cut_off_epoch:
#                 print(torch.norm(theta - exp_theta))
# #                 
#                 print(torch.norm(torch.t(gradient).view(theta.shape) - grad_list[epoch].view(theta.shape)))
#             else:
#                 print('after cut_off_epoch')
#                  
#                 print(torch.norm(theta - exp_theta))
# #                 
#                 print(torch.norm(torch.t(gradient).view(theta.shape) - grad_list[epoch].view(theta.shape)))
            
            del output
            
#                 epoch = epoch + 1
                    
#             full_term1 = None
#             
#             full_term2 = None
#             
#             if epoch < cut_off_epoch:
#                 
#             else:
#                 
#                 if epoch == cut_off_epoch:
#                     avg_A = avg_A/min_batch_num_per_epoch
#                     
#                     avg_B = avg_B/min_batch_num_per_epoch
#                 
# #                     full_term1 = Variable(avg_term1 - avg_A)
# #                 
# #                     full_term2 = Variable(avg_term2.view(-1,1) - avg_B)
#                 
#                 
#                 
#                 vectorized_theta = torch.mm(avg_A, vectorized_theta) + avg_B
#     
#                 theta = Variable(torch.t(vectorized_theta.view(num_class, dim[1])))
            
            
            
            
#                 output = torch.mm((full_term1), vectorized_theta) + (full_term2)
#                 
#                 delta_x_sum_by_class = 0
#                 
#                 if curr_matched_ids.shape[0] > 0:
#                     delta_x_sum_by_class = compute_x_sum_by_class(batch_delta_X, batch_delta_Y, num_class, batch_delta_X.shape)
#                 
#                 x_sum_by_class = x_sum_by_class_list[epoch] - delta_x_sum_by_class
#                 
#     #             t2 = time.time()
#     #             
#     #             overhead += t2 - t1
#                 del delta_x_sum_by_class
#     
#                 
#                 gradient = (output- x_sum_by_class)/(end_id - i - curr_matched_ids.shape[0]) + beta*vectorized_theta     
#     
#                 
#                 del output      
#                 
#                 del x_sum_by_class
#                 
#                 vectorized_theta = (vectorized_theta - alpha*gradient)  
#                 
#                 theta = torch.t(vectorized_theta.view(num_class, dim[1]))
#                 print(torch.norm(theta - exp_theta))
#                 
#                 print('angle::', torch.dot(torch.reshape(theta, [-1]), torch.reshape(exp_theta, [-1]))/(torch.norm(torch.reshape(theta, [-1]))*torch.norm(torch.reshape(exp_theta, [-1]))))

            
#                 print(torch.norm(torch.t(gradient).view(theta.shape) - grad_list[epoch].view(theta.shape)))
#                 
#                 del gradient

#                 print(epoch)
            
            epoch = epoch + 1
            
            id_start = id_end
            
            if epoch >= max_epoch:
                end = True
                
                break
            
                
        if end == True:
            break
                
#             print(epoch)
            
        
#     for i in range(A.shape[0]):
#         
#         theta = torch.mm(A[i], theta) + B[i]
#         
#     if A.shape[0] >= max_epoch:
#         return torch.t((theta).view(num_class, dim[1]))    
#         
#     
#          
#     num = A.shape[0]*min_batch_num_per_epoch
#      
#      
#     this_A = torch.eye(dim[1], dtype = torch.double)
#      
#     this_B = torch.zeros([dim[1], 1], dtype = torch.double)
#      
#      
#     if cut_off_epoch > min_batch_num_per_epoch: 
#         avg_term1 = torch.sum(term1[-min_batch_num_per_epoch:-1], 0)/min_batch_num_per_epoch
#         avg_term2 = torch.sum(term2[-min_batch_num_per_epoch:-1], 0)/min_batch_num_per_epoch
#     else:
#         avg_term1 = torch.sum(term1, 0)/cut_off_epoch
#         avg_term2 = torch.sum(term2, 0)/cut_off_epoch
#      
#      
#     last_A = torch.eye(dim[1]*num_class, dtype = torch.double)
#     
#     
#     last_B = torch.zeros([dim[1]*num_class, 1], dtype = torch.double) 
#     
#     for j in range(0, dim[0],batch_size):
#         
#         end_id = j + batch_size
#         
#         if end_id > dim[0]:
#             end_id = dim[0]
# 
# 
# 
#         if num < cut_off_epoch:
#             gradient = (torch.mm(term1[num], theta) + (term2[num]).view(theta.shape)) + beta*theta
#             
#             curr_A = (1-alpha*beta)*torch.eye(dim[1]*num_class, dtype = torch.double) - alpha*term1[num]
#             
#             curr_B = -term2[num]*alpha
#             
#             
#         else:
#             gradient = (torch.mm(avg_term1, theta) + (avg_term2).view(theta.shape)) + beta*theta
#             
#             curr_A = (1-alpha*beta)*torch.eye(dim[1]*num_class, dtype = torch.double) - alpha*avg_term1
#             
#             curr_B = -avg_term2*alpha
#         
# 
#         last_A = torch.mm(last_A, curr_A)
#                 
#         last_B = torch.mm(curr_A, last_B) + curr_B.view(dim[1]*num_class, 1)
# 
#         theta = theta - alpha * gradient
# 
#         num += 1
# 
#          
#      
#      
#      
#     for i in range(max_epoch - A.shape[0] - 1):
#         theta = torch.mm(last_A, theta) + last_B
# 
# 
#     theta = torch.t(theta.view(num_class, dim[1]))
    
    print('overhead::', overhead)
    
    print('overhead2::', overhead2)
    
    print('overhead3::', overhead3)
    
    return theta
    
        
#     print('total_time::', total_time)
#     
#     return theta


def compute_sub_term1_3(X_theta_prod_softmax_seq_tensor, epoch, delta_data_ids, X, num_class, M, M_inverse):
    
    delta_X = X[delta_data_ids]
    '''n,q'''
    curr_X_theta_prod_softmax_seq_tensor = X_theta_prod_softmax_seq_tensor[epoch][delta_data_ids]
    
    '''n,q*m'''
    
    X_times_weight = torch.bmm(curr_X_theta_prod_softmax_seq_tensor.view(delta_data_ids.shape[0], num_class, 1), delta_X.view(delta_data_ids.shape[0], 1, delta_X.shape[1])).view(delta_data_ids.shape[0], -1)
    
    component_2 = torch.mm(torch.mm(M_inverse, torch.t(X_times_weight)), torch.mm(X_times_weight, M))
    
    
    torch.bmm(torch.transpose(X_times_weight.view(delta_data_ids.shape[0], num_class, delta_X.shape[1]), 1,2).view(delta_data_ids.shape[0], delta_X.shape[1]*num_class), delta_X)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    torch.bmm(X_theta_prod_softmax_seq_tensor[epoch][delta_data_ids].view())
    

def compute_delta_s(M, M_inverse, X, weights, dim, num_class):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    
    '''dim[0]*dim[1]*(max_epoch*num_class*num_class)'''
        
        
    '''to be figured out'''    
            
    '''(|delta_X|, dim[1]) * (dim[1], dim[1]) -> (|delta_X|, dim[1]), overhead::|delta_X| dim[1]^2, km^2'''    
    
    '''res1: n*mq^2'''
    res1 = torch.bmm(X.view(dim[0], dim[1], 1), weights.view(dim[0], 1, num_class*num_class))
    
    '''n(mq)^2'''
    res2 = torch.mm(torch.t(X), res1.view(dim[0], dim[1]*num_class*num_class)).view(dim[1]*dim[1], num_class*num_class)
    
    del res1
    
    res3 = torch.reshape(torch.t(res2), [num_class, num_class, dim[1], dim[1]])
    
    del res2
    
    res4 = torch.reshape(torch.transpose(res3, 1, 2), [num_class*dim[1], dim[1]*num_class])
        
    del res3

    res5 = torch.diag(torch.mm(torch.mm(M_inverse, res4), M))


    return res5


def compute_model_parameter_by_approx_incremental_4_3(cut_off_epoch, full_term2_list, M, M_inverse, s, weights, offsets, batch_size, theta_list, grad_list, random_ids_multi_super_iterations, dim, theta,  X, Y, selected_rows, num_class, max_epoch, alpha, beta):
    
    total_time = 0.0
    
    selected_rows_set = set(selected_rows.view(-1).tolist())
    
#     theta_list = []
    
#     for j in range(max_epoch):
    min_batch_num_per_epoch = int((X.shape[0] - 1)/batch_size) + 1

    theta_list = []
    
    grad_list = []
    
    output_list = []
    
    x_sum_by_class_list = []
    cut_off_super_iteration = int((weights.shape[0]-1)/dim[0])

    end = False
    epoch = 0
    
    overhead = 0
    
    avg_s = 0
    
    avg_B = 0
    
    for j in range(len(random_ids_multi_super_iterations)):
        
        random_ids = random_ids_multi_super_iterations[j]
        
        super_iter_id = j
        
        if j > cut_off_super_iteration:
            super_iter_id = cut_off_super_iteration
        
#         for i in range(len(batch_X_list)):
        for i in range(0, X.shape[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            
            curr_rand_ids = random_ids[i:end_id]
            
            curr_matched_ids, curr_non_matched_ids = get_subset_data_per_epoch2(curr_rand_ids, selected_rows_set)
#             curr_matched_ids,_ = torch.sort(curr_matched_ids)

#             print(curr_matched_ids)
            
            batch_X = X[curr_matched_ids]
            
            batch_Y = Y[curr_matched_ids]
            batch_weights = weights[curr_matched_ids + super_iter_id*dim[0]]
            
            batch_offsets = offsets[curr_matched_ids + super_iter_id*dim[0]]
            t1 = time.time()
            
            X_times_theta = torch.mm(batch_X, theta)
            
            res1 = torch.bmm(X_times_theta.view(batch_X.shape[0], 1, num_class), batch_weights.view(batch_X.shape[0], num_class, num_class))
    
            '''dim[1],num_class, num_class*num_class'''
            
#             res1 = 
            
            
            res2 = torch.mm(torch.t(batch_X), res1.view(batch_X.shape[0], num_class) + batch_offsets)

            t2 = time.time()
            
            overhead += (t2 - t1)
            
            x_sum_by_class = compute_x_sum_by_class(batch_X, batch_Y, num_class, batch_X.shape)
            
            if epoch >= cut_off_epoch - min_batch_num_per_epoch:
                    
#                     curr_M = M_list[epoch - (cut_off_epoch - min_batch_num_per_epoch)]
#                     
#                     curr_M_inverse = M_inverse_list[epoch - (cut_off_epoch - min_batch_num_per_epoch)]
#                     
#                     curr_s = s_list[epoch - (cut_off_epoch - min_batch_num_per_epoch)]
                    
                    curr_delta_s = 0
                    
                    sub_term2 = 0
                    
                    if curr_non_matched_ids.shape[0] > 0:
                        
                        batch_delta_X = X[curr_non_matched_ids]
                        
                        sub_term2 = (prepare_sub_term_2(batch_delta_X, offsets[curr_non_matched_ids + super_iter_id*dim[0]], batch_delta_X.shape, num_class)).view(-1,1)
                    
                        curr_delta_s =  compute_delta_s(M, M_inverse, batch_delta_X, weights[curr_non_matched_ids + super_iter_id*dim[0]], batch_delta_X.shape, num_class)
#                         compute_delta_s(M, M_inverse, delta_X, sub_weights[:,cut_off_epoch - 1], delta_X.shape, num_class)
                    
#                     avg_s = avg_s + curr_delta_s
                    
                    curr_s = (1-alpha*beta) - alpha*(s - curr_delta_s)/(curr_matched_ids.shape[0])
                    
                    avg_s = avg_s + curr_s
                    
                    
                    
                    
#                     print(epoch, avg_s)
                    
#                     avg_s = avg_s + prepate_term_1_batch_by_epoch_with_eigen_matrix(batch_delta_X, batch_weights, end_id - i - batch_delta_X.shape[0], M, M_inverse, s)

#                     avg_B = avg_B - (full_term2.view(-1,1) - sub_term2.view(-1,1) - x_sum_by_class)

                    avg_B = avg_B - alpha/(curr_matched_ids.shape[0])*(full_term2_list[epoch].view(-1,1) - sub_term2 - x_sum_by_class)

            
            output = Variable(torch.reshape(torch.t(res2), [-1,1]))

#             output = softmax_layer(torch.mm(batch_X, theta))
#             
#             
#             output = torch.mm(torch.t(batch_X), output)
#             
#             
#             output = torch.reshape(torch.t(output), [-1,1])
            
            reshape_theta = torch.reshape(torch.t(theta), (-1, 1))
            
            
            
#             output_list.append(output)
            
#             x_sum_by_class_list.append(x_sum_by_class)
            
            grad = (output - x_sum_by_class)/batch_X.shape[0] + beta*reshape_theta
            
            reshape_theta -=  alpha * grad
            
            
            theta = torch.t(reshape_theta.view(num_class, dim[1]))
            
            
            
#             theta_list.append(theta)
            
#             grad_list.append(grad)
            
#             print('theta_diff:', torch.norm(theta - theta_list[epoch]))
#             
#             print('grad_diff:', torch.norm(grad.view(theta.shape) - grad_list[epoch]))
            
            epoch = epoch + 1
            
            if epoch >= cut_off_epoch:
                end = True
                break
        
        if end == True:
            break
        
    
    
    
    avg_s = avg_s/min_batch_num_per_epoch
    
    avg_B = avg_B/min_batch_num_per_epoch
    updated_s = avg_s
    
    
    updated_B = avg_B
#     s = s[:,0].view(-1)
#     updated_s = (1-alpha*beta) - (s - avg_s)/(dim[0] - delta_ids.shape[0])
    
#     updated_B = - alpha/(dim[0] - delta_ids.shape[0])*avg_B

#     avg_s = avg_s/min_batch_num_per_epoch
#     
#     avg_B = avg_B/min_batch_num_per_epoch
#     
#     avg_M = torch.mean(torch.stack(M, 0), 0)
#      
#     avg_M_inverse = torch.mean(torch.stack(M_inverse, 0), 0)
                 
#     updated_s = avg_s#(1-alpha*beta) - alpha*(s - (delta_s))/dim[0]
    updated_s[updated_s > 1] = 1-1e-6
    
#     updated_s = torch.abs(updated_s)
    
    s_power = torch.pow(updated_s, float(max_epoch - cut_off_epoch))
#      
    res1 = M.mul(s_power.view(1,-1))
    
    res1_2 = torch.mm(M_inverse, reshape_theta)
    
    sub_sum = (1-s_power)/(1-updated_s)
    
    res2 = torch.mm(M.mul(sub_sum), torch.mm(M_inverse, updated_B))
    
    vectorized_theta = torch.mm(res1, res1_2) + res2

    theta = torch.t(vectorized_theta.view(num_class, dim[1]))
    
    
#             theta_list.append(theta)
    
    print('overhead::', overhead)
    
    return theta


def compute_model_parameter_by_approx_incremental_4_2(s, M, M_inverse, theta_list, origin_X, origin_Y, weights, offsets, delta_ids, random_ids_multi_super_iterations, term1, term2, dim, theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, x_sum_by_class_list, avg_term2, u_list, v_s_list, avg_u, avg_s):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    
#     num = 0
     
#     theta = theta.view(-1,1) 
    
#     batch_num_per_epoch = int((X.shape[0] - 1)/batch_size) + 1
    min_batch_num_per_epoch = int((origin_X.shape[0] - 1)/batch_size) + 1
    cut_off_super_iteration = int((weights.shape[0]-1)/dim[0])
#     
#     cut_off_random_ids_multi_super_iterations = random_ids_multi_super_iterations[0:cut_off_super_iteration]
# 
# 
# 
#     matched_ids = (cut_off_random_ids_multi_super_iterations.view(-1,1) == delta_ids.view(1,-1))
#     
#     '''T, n, |delta_X|'''
#     
#     matched_ids = matched_ids.view(cut_off_super_iteration, dim[0], delta_ids.shape[0])
#         
#         
#     '''n, T, |delta_X|'''
# #     matched_ids = torch.transpose(matched_ids, 1, 0)
#     
#     '''ids of [n, T, delta_X]'''
#     total_time = 0
#     
#     t1 = time.time()
#     
#     
#     all_noise_data_ids = delta_ids.view(1,-1) + (torch.tensor(list(range(cut_off_super_iteration)))*dim[0]).view(-1, 1)
#     
#     '''delta_X * T''' 
#     
#     all_noise_data_ids = all_noise_data_ids.view(-1)

    '''T, |delta_X|, q^2'''

#     curr_weights = weights[all_noise_data_ids].view(-1, delta_data_ids.view(-1).shape[0], num_class*num_class)
#     
#     '''T, |delta_X|, q'''    
#     curr_offsets = offsets[all_noise_data_ids].view(-1, delta_data_ids.view(-1).shape[0], num_class)

#     delta_X = origin_X[delta_ids]


    epoch = 0

    overhead = 0

#     
        
    delta_ids_set = set(delta_ids.view(-1).tolist())    
    
    X = origin_X
        
        
    Y = origin_Y
    
    vectorized_theta = Variable(torch.reshape(torch.t(theta), [-1,1]))
    
    theta = Variable(theta)
    
    
    
#     sub_term_2_list = []
    
#     avg_sub_term_1 = 0
#     
#     avg_sub_term_2 = 0
    
    
    avg_B = 0
    
    avg_s = 0
    
    
#     for k in range(max_epoch):
#     while epoch < cut_off_epoch:

    end = False
    
    num = 0
    
    
    overhead2 = 0
    
    for k in range(random_ids_multi_super_iterations.shape[0]):
    
#     for k in range(5):
        
        
        random_ids = random_ids_multi_super_iterations[k]
        
        
        super_iter_id = k
        
        if k > cut_off_super_iteration:
            super_iter_id = cut_off_super_iteration
        
#         else:
#             
#             matched_ids = (random_ids.view(-1,1) == delta_ids.view(1,-1))
#     
#             '''n, |delta_X|'''
#         
#             matched_ids = matched_ids.view(dim[0], delta_ids.shape[0])
#             
#             
#             '''|delta_X|*2'''
#             
#             nonzero_ids = torch.nonzero(matched_ids)
        
        
        end_id_super_iteration = (super_iter_id + 1)*dim[0]
        
        
        if end_id_super_iteration > weights.shape[0]:
            end_id_super_iteration = weights.shape[0]
            
        
        
        weights_this_super_iteration = weights[super_iter_id*dim[0]:end_id_super_iteration]
        
#         delta_ids = delta_ids[delta_ids < weights_this_super_iteration.shape[0]]
        
        
#         weights_this_super_iteration = weights_this_super_iteration[delta_ids]
        
        
        offsets_this_super_iteration = offsets[super_iter_id*dim[0]:end_id_super_iteration]
        
        
#         offsets_this_super_iteration = offsets_this_super_iteration[delta_ids]
        
        for i in range(0, X.shape[0], batch_size):
        
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            
            
            curr_rand_ids = Variable(random_ids[i:end_id])
            
            
            curr_matched_ids = (get_subset_data_per_epoch(curr_rand_ids, delta_ids_set))
            
            
            
            
#             curr_rand_id_set = set(curr_rand_ids.view(-1).tolist())
            
#             curr_matched_ids = (curr_rand_ids.view(-1,1) == delta_ids.view(1,-1))
#             curr_matched_ids = torch.tensor(list(delta_ids_set.intersection(curr_rand_id_set)))
            
            
#             curr_nonzero_ids = torch.nonzero(((nonzero_ids[:, 0] >= i)*(nonzero_ids[:, 0] < end_id))).view(-1)
#             
#             curr_nonzero_ids_this_batch0 = nonzero_ids[curr_nonzero_ids][:, 1]
            
#             curr_nonzero_ids_this_batch = torch.nonzero(curr_matched_ids)[:, 1]
            if curr_matched_ids.shape[0] > 0:
                batch_delta_X = (X[curr_matched_ids])
                
                batch_delta_Y = (Y[curr_matched_ids])

            
            if epoch < cut_off_epoch:
            
            
#             if epoch < cut_off_epoch:
            
#                 curr_weights = weights[k*dim[0] + i: k*dim[0] + end_id]
            
#             print(weights_this_super_iteration.shape)
#             
#             print(curr_matched_ids)

                sub_term2 = 0
                
                delta_x_sum_by_class = 0
                
                if curr_matched_ids.shape[0] > 0:

                    batch_weights = weights_this_super_iteration[curr_matched_ids]
                    
                    
                    batch_offsets = offsets_this_super_iteration[curr_matched_ids]
                
    #             else:
    # #                 curr_weights = weights[(cut_off_super_iteration - 1)*dim[0] + i: (cut_off_super_iteration - 1)*dim[0] + end_id]
    #                 
    #                 batch_weights = weights_this_super_iteration[curr_nonzero_ids_this_batch]
    #                 
    #                 
    #                 batch_offsets = offsets_this_super_iteration[curr_nonzero_ids_this_batch]
                
                
                    
                    batch_X_multi_theta = Variable(torch.mm(batch_delta_X, theta))
                    sub_term2 = (prepare_sub_term_2(batch_delta_X, batch_offsets, batch_delta_X.shape, num_class)).view(-1,1)
                    
                    delta_x_sum_by_class = compute_x_sum_by_class(batch_delta_X, batch_delta_Y, num_class, batch_delta_X.shape)

                    
                t1 = time.time()
#                 vectorized_theta = Variable(torch.reshape(torch.t(theta), [-1,1]))
                
    #             t1 = time.time()
                

#                 vectorized_sub_term_1 = (torch.reshape(sub_term_1_without_weights, [-1,1]))
                
                
                full_term1 = term1[epoch]
                
                full_term2 = term2[epoch]
                
                x_sum_by_class = x_sum_by_class_list[epoch] - delta_x_sum_by_class
                if epoch >= cut_off_epoch - min_batch_num_per_epoch:
                    
#                     curr_M = M_list[epoch - (cut_off_epoch - min_batch_num_per_epoch)]
#                     
#                     curr_M_inverse = M_inverse_list[epoch - (cut_off_epoch - min_batch_num_per_epoch)]
#                     
#                     curr_s = s_list[epoch - (cut_off_epoch - min_batch_num_per_epoch)]
                    t1 = time.time()
                    curr_delta_s = 0
                    
                    if curr_matched_ids.shape[0] > 0:
                    
                        curr_delta_s =  compute_delta_s(M, M_inverse, batch_delta_X, batch_weights, batch_delta_X.shape, num_class)
#                         compute_delta_s(M, M_inverse, delta_X, sub_weights[:,cut_off_epoch - 1], delta_X.shape, num_class)
                    
#                     avg_s = avg_s + curr_delta_s
                    
                    curr_s = (1-alpha*beta) - alpha*(s - curr_delta_s)/(end_id - i - curr_matched_ids.shape[0])
                    
                    avg_s = avg_s + curr_s
                    
                    
                    t2 = time.time()
                    
                    overhead2 += (t2  -t1)
                    
#                     print(epoch, avg_s)
                    
#                     avg_s = avg_s + prepate_term_1_batch_by_epoch_with_eigen_matrix(batch_delta_X, batch_weights, end_id - i - batch_delta_X.shape[0], M, M_inverse, s)

#                     avg_B = avg_B - (full_term2.view(-1,1) - sub_term2.view(-1,1) - x_sum_by_class)

                    avg_B = avg_B - alpha/(end_id - i - curr_matched_ids.shape[0])*(full_term2.view(-1,1) - sub_term2 - x_sum_by_class)

#                     B = -alpha*(term2[cut_off_epoch - 1].view(-1,1) - x_sum_by_class)/dim[0]    
#                     if curr_matched_ids.shape[0] > 0:
#                         sub_term_1_without_weights = prepare_sub_term_1(batch_delta_X, batch_weights, batch_delta_X.shape, num_class)
# 
# #                     avg_sub_term_1 += (sub_term_1_without_weights)
# #                     avg_sub_term_2 += (sub_term2)
#                      
#                     output = torch.mm((full_term1 - sub_term_1_without_weights), vectorized_theta) + (full_term2.view(-1,1) - sub_term2)
#                     
#                     del sub_term_1_without_weights
                
#                 else:

                vectorized_sub_term_1 = 0
                
                if curr_matched_ids.shape[0] > 0:
                    sub_term_1 = (torch.t(compute_sub_term_1(batch_X_multi_theta, batch_delta_X, batch_weights, batch_X_multi_theta.shape, num_class)))
                
                    vectorized_sub_term_1 = Variable(torch.reshape(sub_term_1, [-1,1]))
                    
                    del sub_term_1
                
                output = torch.mm((full_term1), vectorized_theta) - vectorized_sub_term_1 + (full_term2.view(-1,1) - sub_term2)
                
                

            
                
                
                
    #             t2 = time.time()
    #             
    #             overhead += t2 - t1
                
                gradient = (output- x_sum_by_class)/(end_id - i - curr_matched_ids.shape[0]) + beta*vectorized_theta     
    
                del output
                
                del x_sum_by_class
                
                del delta_x_sum_by_class
                
                del sub_term2
                                
                vectorized_theta = (vectorized_theta - alpha*gradient)  
                
                del gradient
                
                theta = torch.t(vectorized_theta.view(num_class, dim[1]))
                
                t2 = time.time()
                 
                overhead += (t2 - t1)
                
    #             exp_theta = theta_list[epoch]
    #             
    #             print(torch.norm(theta - exp_theta))
                
                epoch = epoch + 1
                        
#             full_term1 = None
#             
#             full_term2 = None
#             
#             if epoch < cut_off_epoch:
#                 
            else:
                end = True
                break
        
        
        if end == True:
            break
#                     avg_sub_term_1 = avg_sub_term_1/min_batch_num_per_epoch
#                     
#                     avg_sub_term_2 = avg_sub_term_2/min_batch_num_per_epoch
#                 
#                     full_term1 = Variable(avg_term1 - avg_sub_term_1)
#                 
#                     full_term2 = Variable(avg_term2 - avg_sub_term_2)

    avg_s = avg_s/min_batch_num_per_epoch
    
    avg_B = avg_B/min_batch_num_per_epoch
    updated_s = avg_s
    
    
    updated_B = avg_B
#     s = s[:,0].view(-1)
#     updated_s = (1-alpha*beta) - (s - avg_s)/(dim[0] - delta_ids.shape[0])
    
#     updated_B = - alpha/(dim[0] - delta_ids.shape[0])*avg_B

#     avg_s = avg_s/min_batch_num_per_epoch
#     
#     avg_B = avg_B/min_batch_num_per_epoch
#     
#     avg_M = torch.mean(torch.stack(M, 0), 0)
#      
#     avg_M_inverse = torch.mean(torch.stack(M_inverse, 0), 0)
                 
#     updated_s = avg_s#(1-alpha*beta) - alpha*(s - (delta_s))/dim[0]
    updated_s[updated_s > 1] = 1-1e-6
    
#     updated_s = torch.abs(updated_s)
    
    s_power = torch.pow(updated_s, float(max_epoch - cut_off_epoch))
#      
    res1 = M.mul(s_power.view(1,-1))
    
    res1_2 = torch.mm(M_inverse, vectorized_theta)
    
    sub_sum = (1-s_power)/(1-updated_s)
    
    res2 = torch.mm(M.mul(sub_sum), torch.mm(M_inverse, updated_B))
    
    vectorized_theta = torch.mm(res1, res1_2) + res2

    theta = torch.t(vectorized_theta.view(num_class, dim[1]))
                
                
                
                
                
                
                
                
                
                
                
                
                
                
#                 if batch_delta_X.shape[0] < batch_delta_X.shape[1]:
#                     delta_s = torch.mm(torch.mm(M_inverse, torch.t(torch.bmm(batch_delta_X.view(batch_delta_X.shape[0], dim[1], 1), batch_weights.view(batch_delta_X.shape[0], 1, batch_weights.shape[1])).view(batch_delta_X.shape[0], dim[1], batch_weights.shape[1]))), torch.mm(batch_delta_X, M))
#                      
#                 else:
#                     delta_s = torch.mm(torch.mm(M_inverse, torch.mm(torch.t(batch_delta_X)*batch_weights.view(-1,1)), batch_delta_X), M)
#                  
#                  
#                 updated_s = (1 - alpha*beta) + alpha*(s - torch.diag(delta_s))/dim[0]
#              
#              
#             #     A = (1- beta*alpha)*torch.eye(dim[1], dtype = torch.float) + alpha*term1[cut_off_epoch - 1]/dim[0]
#                  
#                 B = alpha/dim[0]*(term2[cut_off_epoch - 1].view(-1,1))
#                  
#             #     updated_s = torch.diag(torch.mm(M_inverse, torch.mm(A-expected_A, M)))
#                  
#             #     updated_s = s + updated_s
#                  
#                 updated_s[updated_s > 1] = 1-1e-6
#                  
#                 s_power = torch.pow(updated_s, float(max_epoch - cut_off_epoch))
#                  
#                 res1 = M.mul(s_power.view(1,-1))
#              
#                 sub_sum = (1-s_power)/(1-updated_s)
#                  
#                 res2 = M.mul(sub_sum.view(1, -1))
#                     
#                 theta = torch.mm(res1, torch.mm(M_inverse,theta)) + torch.mm(res2, torch.mm(M_inverse, B))
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
#                 
#                 
#                 
#                 
#                 output = torch.mm((full_term1), vectorized_theta) + (full_term2.view(-1,1))
#                 
#                 
#                 delta_x_sum_by_class = compute_x_sum_by_class(batch_delta_X, batch_delta_Y, num_class, batch_delta_X.shape)
#                 
#                 x_sum_by_class = x_sum_by_class_list[epoch] - delta_x_sum_by_class
#                 
#     #             t2 = time.time()
#     #             
#     #             overhead += t2 - t1
#                 del delta_x_sum_by_class
#     
#                 
#                 gradient = (output- x_sum_by_class)/(end_id - i - batch_delta_X.shape[0]) + beta*vectorized_theta     
#     
#                 
#                 del output      
#                 
#                 del x_sum_by_class
#                 
#                 vectorized_theta = (vectorized_theta - alpha*gradient)  
#                 
#                 del gradient
#                 
#                 theta = torch.t(vectorized_theta.view(num_class, dim[1]))
#                 
# #                 print(epoch)
#                 
#                 epoch = epoch + 1
    
    print('overhead::', overhead)
    
    
    
    return theta
    
        
#     print('total_time::', total_time)
#     
#     return theta

def compute_model_parameter_by_approx_incremental_4_4(s, M, M_inverse, theta_list, origin_X, origin_Y, weights, offsets, delta_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, term1, term2, dim, theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, x_sum_by_class_list, avg_term2, u_list, v_s_list):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    
#     num = 0
     
#     theta = theta.view(-1,1) 
    
#     batch_num_per_epoch = int((X.shape[0] - 1)/batch_size) + 1
    min_batch_num_per_epoch = int((origin_X.shape[0] - 1)/batch_size) + 1
    cut_off_super_iteration = int((weights.shape[0]-1)/dim[0])
#     
#     cut_off_random_ids_multi_super_iterations = random_ids_multi_super_iterations[0:cut_off_super_iteration]
# 
# 
# 
#     matched_ids = (cut_off_random_ids_multi_super_iterations.view(-1,1) == delta_ids.view(1,-1))
#     
#     '''T, n, |delta_X|'''
#     
#     matched_ids = matched_ids.view(cut_off_super_iteration, dim[0], delta_ids.shape[0])
#         
#         
#     '''n, T, |delta_X|'''
# #     matched_ids = torch.transpose(matched_ids, 1, 0)
#     
#     '''ids of [n, T, delta_X]'''
#     total_time = 0
#     
#     t1 = time.time()
#     
#     
#     all_noise_data_ids = delta_ids.view(1,-1) + (torch.tensor(list(range(cut_off_super_iteration)))*dim[0]).view(-1, 1)
#     
#     '''delta_X * T''' 
#     
#     all_noise_data_ids = all_noise_data_ids.view(-1)

    '''T, |delta_X|, q^2'''

#     curr_weights = weights[all_noise_data_ids].view(-1, delta_data_ids.view(-1).shape[0], num_class*num_class)
#     
#     '''T, |delta_X|, q'''    
#     curr_offsets = offsets[all_noise_data_ids].view(-1, delta_data_ids.view(-1).shape[0], num_class)

#     delta_X = origin_X[delta_ids]


    epoch = 0

    overhead = 0

#     
        
    delta_ids_set = set(delta_ids.view(-1).tolist())    
    
    X = origin_X
        
        
    Y = origin_Y
    
    vectorized_theta = Variable(torch.reshape(torch.t(theta), [-1,1]))
    
    theta = Variable(theta)
    
    
    
#     sub_term_2_list = []
    
#     avg_sub_term_1 = 0
#     
#     avg_sub_term_2 = 0
    
    
    avg_B = 0
    
    avg_s = 0
    
    
#     for k in range(max_epoch):
#     while epoch < cut_off_epoch:

    end = False
    
    num = 0
    
    
    overhead2 = 0
    
    all_delta_sub_term_2= 0
    
    all_delta_x_sum_by_class = 0
    
    
    for k in range(random_ids_multi_super_iterations.shape[0]):
    
#     for k in range(5):
        
        
        random_ids = random_ids_multi_super_iterations[k]
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
#         all_indexes = np.sort(np.searchsorted(random_ids.numpy(), delta_ids.numpy()))
        
        all_indexes = np.sort(sort_idx[np.searchsorted(random_ids.numpy(),delta_ids.numpy(),sorter = sort_idx)])
        
        
        super_iter_id = k
        
        if k > cut_off_super_iteration:
            super_iter_id = cut_off_super_iteration
        
#         else:
#             
#             matched_ids = (random_ids.view(-1,1) == delta_ids.view(1,-1))
#     
#             '''n, |delta_X|'''
#         
#             matched_ids = matched_ids.view(dim[0], delta_ids.shape[0])
#             
#             
#             '''|delta_X|*2'''
#             
#             nonzero_ids = torch.nonzero(matched_ids)
        
        
        end_id_super_iteration = (super_iter_id + 1)*dim[0]
        
        
        if end_id_super_iteration >= weights.shape[0]:
            end_id_super_iteration = weights.shape[0]
            
        
        
        weights_this_super_iteration = weights[super_iter_id*dim[0]:end_id_super_iteration]
        
#         delta_ids = delta_ids[delta_ids < weights_this_super_iteration.shape[0]]
        
        
#         weights_this_super_iteration = weights_this_super_iteration[delta_ids]
        
        
        offsets_this_super_iteration = offsets[super_iter_id*dim[0]:end_id_super_iteration]
        
        
#         offsets_this_super_iteration = offsets_this_super_iteration[delta_ids]
        
        id_start = 0
        
        id_end = 0
        
        for i in range(0, X.shape[0], batch_size):
        
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            
            
#             curr_rand_ids = Variable(random_ids[i:end_id])
            
            
#             curr_matched_ids = (get_subset_data_per_epoch(curr_rand_ids, delta_ids_set))
            
            while 1:
                if id_end >= all_indexes.shape[0] or all_indexes[id_end] >= end_id:
                    break
                
                id_end = id_end + 1
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
            
            curr_matched_ids_size = curr_matched_ids.shape[0]
            
            
#             curr_rand_id_set = set(curr_rand_ids.view(-1).tolist())
            
#             curr_matched_ids = (curr_rand_ids.view(-1,1) == delta_ids.view(1,-1))
#             curr_matched_ids = torch.tensor(list(delta_ids_set.intersection(curr_rand_id_set)))
            
            
#             curr_nonzero_ids = torch.nonzero(((nonzero_ids[:, 0] >= i)*(nonzero_ids[:, 0] < end_id))).view(-1)
#             
#             curr_nonzero_ids_this_batch0 = nonzero_ids[curr_nonzero_ids][:, 1]
            
#             curr_nonzero_ids_this_batch = torch.nonzero(curr_matched_ids)[:, 1]
            if curr_matched_ids_size > 0:
                batch_delta_X = (X[curr_matched_ids])
                
                batch_delta_Y = (Y[curr_matched_ids])

            
            if epoch < cut_off_epoch:
            
            
#             if epoch < cut_off_epoch:
            
#                 curr_weights = weights[k*dim[0] + i: k*dim[0] + end_id]
            
#             print(weights_this_super_iteration.shape)
#             
#             print(curr_matched_ids)

                sub_term2 = 0
                
                delta_x_sum_by_class = 0
                
                if curr_matched_ids.shape[0] > 0:

                    batch_weights = weights_this_super_iteration[curr_matched_ids]
                    
                    
                    batch_offsets = offsets_this_super_iteration[curr_matched_ids]
                
    #             else:
    # #                 curr_weights = weights[(cut_off_super_iteration - 1)*dim[0] + i: (cut_off_super_iteration - 1)*dim[0] + end_id]
    #                 
    #                 batch_weights = weights_this_super_iteration[curr_nonzero_ids_this_batch]
    #                 
    #                 
    #                 batch_offsets = offsets_this_super_iteration[curr_nonzero_ids_this_batch]
                
                
                    
                    batch_X_multi_theta = Variable(torch.mm(batch_delta_X, theta))
                    sub_term2 = (prepare_sub_term_2(batch_delta_X, batch_offsets, batch_delta_X.shape, num_class)).view(-1,1)
                    
                    delta_x_sum_by_class = compute_x_sum_by_class(batch_delta_X, batch_delta_Y, num_class, batch_delta_X.shape)

                    
                t1 = time.time()
#                 vectorized_theta = Variable(torch.reshape(torch.t(theta), [-1,1]))
                
    #             t1 = time.time()
                

#                 vectorized_sub_term_1 = (torch.reshape(sub_term_1_without_weights, [-1,1]))
                
                
#                 full_term1 = term1[epoch]
                
                full_term2 = term2[epoch]
                
                x_sum_by_class = x_sum_by_class_list[epoch] - delta_x_sum_by_class
                if epoch >= cut_off_epoch - min_batch_num_per_epoch:
                    
#                     curr_M = M_list[epoch - (cut_off_epoch - min_batch_num_per_epoch)]
#                     
#                     curr_M_inverse = M_inverse_list[epoch - (cut_off_epoch - min_batch_num_per_epoch)]
#                     
#                     curr_s = s_list[epoch - (cut_off_epoch - min_batch_num_per_epoch)]


                    all_delta_x_sum_by_class += x_sum_by_class
                    
                    all_delta_sub_term_2 += (full_term2.view(-1,1) - sub_term2)
                    
 
#                     B = -alpha*(term2[cut_off_epoch - 1].view(-1,1) - x_sum_by_class)/dim[0]    
#                     if curr_matched_ids.shape[0] > 0:
#                         sub_term_1_without_weights = prepare_sub_term_1(batch_delta_X, batch_weights, batch_delta_X.shape, num_class)
# 
# #                     avg_sub_term_1 += (sub_term_1_without_weights)
# #                     avg_sub_term_2 += (sub_term2)
#                      
#                     output = torch.mm((full_term1 - sub_term_1_without_weights), vectorized_theta) + (full_term2.view(-1,1) - sub_term2)
#                     
#                     del sub_term_1_without_weights
                
#                 else:

                vectorized_sub_term_1 = 0
                
                if curr_matched_ids.shape[0] > 0:
                    sub_term_1 = (torch.t(compute_sub_term_1(batch_X_multi_theta, batch_delta_X, batch_weights, batch_X_multi_theta.shape, num_class)))
                
                    vectorized_sub_term_1 = Variable(torch.reshape(sub_term_1, [-1,1]))
                    
                    del sub_term_1
                
#                 output = torch.mm((full_term1), vectorized_theta) - vectorized_sub_term_1 + (full_term2.view(-1,1) - sub_term2)
                
#                 if  batch_size < num_class*X.shape[1]:
                output = torch.mm(u_list[epoch], torch.mm(v_s_list[epoch], vectorized_theta)) - vectorized_sub_term_1 + (full_term2.view(-1,1) - sub_term2)
#                 else:
#                     output = torch.mm(term1[epoch], vectorized_theta) - vectorized_sub_term_1 + (full_term2.view(-1,1) - sub_term2)
                
                

            
                
                
                
    #             t2 = time.time()
    #             
    #             overhead += t2 - t1
                
                gradient = (output- x_sum_by_class)/(end_id - i - curr_matched_ids.shape[0]) + beta*vectorized_theta     
    
                del output
                
                del x_sum_by_class
                
                del delta_x_sum_by_class
                
                del sub_term2
                                
                vectorized_theta = (vectorized_theta - alpha*gradient)  
                
                del gradient
                
                theta = torch.t(vectorized_theta.view(num_class, dim[1]))
                
                t2 = time.time()
                 
                overhead += (t2 - t1)
                
    #             exp_theta = theta_list[epoch]
    #             
    #             print(torch.norm(theta - exp_theta))
                
                epoch = epoch + 1
                      
                id_start = id_end      
                
#             full_term1 = None
#             
#             full_term2 = None
#             
#             if epoch < cut_off_epoch:
#                 
            else:
                end = True
                break
        
        
        if end == True:
            break
#                     avg_sub_term_1 = avg_sub_term_1/min_batch_num_per_epoch
#                     
#                     avg_sub_term_2 = avg_sub_term_2/min_batch_num_per_epoch
#                 
#                     full_term1 = Variable(avg_term1 - avg_sub_term_1)
#                 
#                     full_term2 = Variable(avg_term2 - avg_sub_term_2)



    t1 = time.time()
#     if curr_matched_ids.shape[0] > 0:
    
    delta_X = X[delta_ids]
    
    curr_delta_s =  compute_delta_s(M, M_inverse, delta_X, weights[super_iter_id*dim[0] + delta_ids], delta_X.shape, num_class)
#                         compute_delta_s(M, M_inverse, delta_X, sub_weights[:,cut_off_epoch - 1], delta_X.shape, num_class)
    
#                     avg_s = avg_s + curr_delta_s
    
    curr_s = (1-alpha*beta) - alpha*(s - curr_delta_s)/(dim[0] - delta_ids.shape[0])
    
    avg_s = avg_s + curr_s
    
    
    t2 = time.time()
    
    overhead2 += (t2  -t1)
    
#                     print(epoch, avg_s)
    
#                     avg_s = avg_s + prepate_term_1_batch_by_epoch_with_eigen_matrix(batch_delta_X, batch_weights, end_id - i - batch_delta_X.shape[0], M, M_inverse, s)

#                     avg_B = avg_B - (full_term2.view(-1,1) - sub_term2.view(-1,1) - x_sum_by_class)

    avg_B = avg_B - alpha/(dim[0] - delta_ids.shape[0])*(all_delta_sub_term_2 - all_delta_x_sum_by_class)




#     avg_s = avg_s/min_batch_num_per_epoch
#     
#     avg_B = avg_B/min_batch_num_per_epoch
    updated_s = avg_s
    
    
    updated_B = avg_B
#     s = s[:,0].view(-1)
#     updated_s = (1-alpha*beta) - (s - avg_s)/(dim[0] - delta_ids.shape[0])
    
#     updated_B = - alpha/(dim[0] - delta_ids.shape[0])*avg_B

#     avg_s = avg_s/min_batch_num_per_epoch
#     
#     avg_B = avg_B/min_batch_num_per_epoch
#     
#     avg_M = torch.mean(torch.stack(M, 0), 0)
#      
#     avg_M_inverse = torch.mean(torch.stack(M_inverse, 0), 0)
                 
#     updated_s = avg_s#(1-alpha*beta) - alpha*(s - (delta_s))/dim[0]
    updated_s[updated_s > 1] = 1-1e-6
    
#     updated_s = torch.abs(updated_s)
    
    s_power = torch.pow(updated_s, float(max_epoch - cut_off_epoch))
#      
    res1 = M.mul(s_power.view(1,-1))
    
    res1_2 = torch.mm(M_inverse, vectorized_theta)
    
    sub_sum = (1-s_power)/(1-updated_s)
    
    res2 = torch.mm(M.mul(sub_sum), torch.mm(M_inverse, updated_B))
    
    vectorized_theta = torch.mm(res1, res1_2) + res2

    theta = torch.t(vectorized_theta.view(num_class, dim[1]))
                
                
                
                
                
                
                
                
                
                
                
                
                
                
#                 if batch_delta_X.shape[0] < batch_delta_X.shape[1]:
#                     delta_s = torch.mm(torch.mm(M_inverse, torch.t(torch.bmm(batch_delta_X.view(batch_delta_X.shape[0], dim[1], 1), batch_weights.view(batch_delta_X.shape[0], 1, batch_weights.shape[1])).view(batch_delta_X.shape[0], dim[1], batch_weights.shape[1]))), torch.mm(batch_delta_X, M))
#                      
#                 else:
#                     delta_s = torch.mm(torch.mm(M_inverse, torch.mm(torch.t(batch_delta_X)*batch_weights.view(-1,1)), batch_delta_X), M)
#                  
#                  
#                 updated_s = (1 - alpha*beta) + alpha*(s - torch.diag(delta_s))/dim[0]
#              
#              
#             #     A = (1- beta*alpha)*torch.eye(dim[1], dtype = torch.float) + alpha*term1[cut_off_epoch - 1]/dim[0]
#                  
#                 B = alpha/dim[0]*(term2[cut_off_epoch - 1].view(-1,1))
#                  
#             #     updated_s = torch.diag(torch.mm(M_inverse, torch.mm(A-expected_A, M)))
#                  
#             #     updated_s = s + updated_s
#                  
#                 updated_s[updated_s > 1] = 1-1e-6
#                  
#                 s_power = torch.pow(updated_s, float(max_epoch - cut_off_epoch))
#                  
#                 res1 = M.mul(s_power.view(1,-1))
#              
#                 sub_sum = (1-s_power)/(1-updated_s)
#                  
#                 res2 = M.mul(sub_sum.view(1, -1))
#                     
#                 theta = torch.mm(res1, torch.mm(M_inverse,theta)) + torch.mm(res2, torch.mm(M_inverse, B))
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
#                 
#                 
#                 
#                 
#                 output = torch.mm((full_term1), vectorized_theta) + (full_term2.view(-1,1))
#                 
#                 
#                 delta_x_sum_by_class = compute_x_sum_by_class(batch_delta_X, batch_delta_Y, num_class, batch_delta_X.shape)
#                 
#                 x_sum_by_class = x_sum_by_class_list[epoch] - delta_x_sum_by_class
#                 
#     #             t2 = time.time()
#     #             
#     #             overhead += t2 - t1
#                 del delta_x_sum_by_class
#     
#                 
#                 gradient = (output- x_sum_by_class)/(end_id - i - batch_delta_X.shape[0]) + beta*vectorized_theta     
#     
#                 
#                 del output      
#                 
#                 del x_sum_by_class
#                 
#                 vectorized_theta = (vectorized_theta - alpha*gradient)  
#                 
#                 del gradient
#                 
#                 theta = torch.t(vectorized_theta.view(num_class, dim[1]))
#                 
# #                 print(epoch)
#                 
#                 epoch = epoch + 1
    
    print('overhead::', overhead)
    
    
    
    return theta
    
        
#     print('total_time::', total_time)
#     
#     return theta


def compute_model_parameter_by_approx_incremental_3(term1, term2, x_sum_by_class, dim, theta, num_class, max_epoch, cut_off_epoch, epoch_record_epoch_seq, alpha, beta):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    vectorized_theta = theta.view(-1,1)
    
#     for i in range(max_epoch):
    for i in range(max_epoch):
        
#         multi_res = np.matmul(X_Y_mult, theta)
#         multi_res = torch.mm(X_Y_mult, theta)

#         sum_sub_term1 = torch.sum(sub_term1[i], dim = 0)
#         
#         sum_sub_term2 = torch.sum(sub_term2[i], dim = 0)
        
#         print(term1[i].shape)
#         
#         print(term2[i].shape)
#         
#         print(sub_term1[i].shape)
#         
#         print(sub_term2[i].shape)



        if i < cut_off_epoch:

            output = torch.mm(term1[i], vectorized_theta) + (term2[i].view(-1,1))
            
            
    #         print('gradient::', gradient)
    #         
    #         print('approx_output::', output/dim[0])
            
            
            gradient = (output- x_sum_by_class)/dim[0] + beta*vectorized_theta
    #         
    #         print('x_sum_by_class::', x_sum_by_class/dim[0])
            
            
            
            
            vectorized_theta = vectorized_theta - alpha*gradient
        
        else:
            
            output = torch.mm(term1[cut_off_epoch - 1], vectorized_theta) + (term2[cut_off_epoch - 1].view(-1,1))
            
            
    #         print('gradient::', gradient)
    #         
    #         print('approx_output::', output/dim[0])
            
            
            gradient = (output- x_sum_by_class)/dim[0] + beta*vectorized_theta
    #         
    #         print('x_sum_by_class::', x_sum_by_class/dim[0])
            
            
            
            
            vectorized_theta = vectorized_theta - alpha*gradient
        
        
#         theta = (1-alpha*beta)*theta + (alpha*torch.mm(term1[i], theta) + alpha*(term2[i]).view(theta.shape))/dim[0]

#         print(multi_res.shape, sub_w_seq[:,i].shape, sub_b_seq[:,i].shape)
        
#         lin_res = multi_res * (sub_w_seq[:, i].view(multi_res.shape)) + sub_b_seq[:, i].view(multi_res.shape)
#         
# #         print(lin_res.shape)
# #         print(lin_res.shape, X_Y_mult.shape, sub_w_seq.shape, sub_b_seq.shape)
#         
# #         lin_res, total_time = compute_curr_linear_paras(multi_res, Pi, total_time)
#         
# #         multi_res *= w_seq
# #         
# #         multi_res += b_seq
#         
#         
#         non_linear_term = X_Y_mult*(lin_res)
# #         w_b_seq = compute_curr_linear_paras(multi_res, Pi)
# #          
# #         non_linear_term = X.mul((multi_res*(w_b_seq[:,0].view([dim[0],1])) + w_b_seq[:,1].view([dim[0],1]))*Y)
#         sum_non_linear_term = torch.sum(non_linear_term, dim=0)
# #         sum_non_linear_term = np.sum(non_linear_term, axis=0)
#         
# #         print(sum_non_linear_term.shape)
#         
# #         sum_non_linear_term_diff_dim = np.reshape(sum_non_linear_term, (theta.shape))
#         sum_non_linear_term_diff_dim = sum_non_linear_term.view( (theta.shape))
#         
# #         sum_non_linear_term_diff_dim *=  
#         
# #         theta = (1-alpha*beta)*theta + (alpha/dim[0])*torch.sum(X_Y_mult.mul(multi_res), dim=0).view(theta.shape)
#         theta = (1-alpha*beta)*theta + (alpha/dim[0])*sum_non_linear_term_diff_dim
        
#         sizes = [get_tensor_size(multi_res), get_tensor_size(w_seq), get_tensor_size(b_seq)
#                  , get_tensor_size(non_linear_term), get_tensor_size(sum_non_linear_term), get_tensor_size(sum_non_linear_term_diff_dim)]
        
        
#         cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
#         add_mem = cur_mem - prev_mem
#         prev_mem = cur_mem
#         print("added mem: %sM"%(add_mem))
        
#         del multi_res, non_linear_term, sum_non_linear_term, sum_non_linear_term_diff_dim
        
#         cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
#         add_mem = cur_mem - prev_mem
#         prev_mem = cur_mem
#         print("added mem: %sM"%(add_mem))
# #         print('size::', sys.getsizeof(theta)/np.power(2, 10), sys.getsizeof(X_Y_mult)/np.power(2, 10))
#         
#         print('size::', sizes)

#     t1 = time.time()
# 
#     A = (1- beta*alpha)*torch.eye(dim[1]*num_class, dtype = torch.double) - alpha*term1[cut_off_epoch - 1]/dim[0]
#     
#     B = -alpha/dim[0]*(term2[cut_off_epoch - 1].view(-1,1) - x_sum_by_class)
#     
#     
#     s, M = torch.eig(A, True)
#     
#     s = s[:,0]
#     
#     s_power = torch.pow(s, float(max_epoch - cut_off_epoch))
#     
#     res1 = M.mul(s_power.view(1,-1))
# 
#     res1 = torch.mm(res1, torch.t(M))
#     
#     
# #     temp = torch.eye(dim[1], dtype = torch.double)
# #     
# #     sum_temp = torch.zeros((dim[1], dim[1]), dtype = torch.double)
# #     
# #     for i in range(max_epoch):
# #         sum_temp += temp
# #         temp = torch.mm(temp, A)
#         
#     
#     
# #     print('temp_gap::', temp - res1)
#     
#     sub_sum = (1-s_power)/(1-s)
#     
#     res2 = M.mul(sub_sum.view(1, -1))
#     
#     res2 = torch.mm(res2, torch.t(M))
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
#     
#     
# #     u,s,v = torch.svd(A)
# #     
# #     
# #     s_power = torch.pow(s, float(max_epoch - cut_off_epoch))
# #     
# #     res1 = u.mul(s_power.view(1,-1))
# # 
# #     res1 = torch.mm(res1, torch.t(v))
# #     
# #     res2 = u.mul((1-s_power)/(1-s))
# #     
# #     res2 = torch.mm(res2, torch.t(v))
#     
#     vectorized_theta = torch.mm(res1, vectorized_theta) + torch.mm(res2, B)
#     
#     t2 = time.time()
#     
#     print('total_time::', t2 - t1) 
        
#     print('total_time::', total_time)
    
    theta = torch.t(vectorized_theta.view(num_class, dim[1]))
    
    return theta



# def compute_model_parameter_by_approx_incremental_3(term1, term2, x_sum_by_class, dim, theta, num_class, max_epoch, sub_X):
#     
#     total_time = 0.0
#     
# #     pid = os.getpid()
#     
# #     prev_mem=0
# #     
# #     print('pid::', pid)
#     
#     vectorized_theta = theta.view(-1,1)
#     
#     for i in range(max_epoch):
#         
# #         multi_res = np.matmul(X_Y_mult, theta)
# #         multi_res = torch.mm(X_Y_mult, theta)
# 
# #         sum_sub_term1 = torch.sum(sub_term1[i], dim = 0)
# #         
# #         sum_sub_term2 = torch.sum(sub_term2[i], dim = 0)
#         
# #         print(term1[i].shape)
# #         
# #         print(term2[i].shape)
# #         
# #         print(sub_term1[i].shape)
# #         
# #         print(sub_term2[i].shape)
# 
#         output = torch.mm(term1[i], (torch.mm(torch.t(theta), sub_X)).view(-1,1))
# 
#         output = output + (term2[i].view(-1,1))
#         
#         
# #         print('gradient::', gradient)
# #         
# #         print('approx_output::', output/dim[0])
#         
#         
#         gradient = (output- x_sum_by_class)/dim[0] + beta*vectorized_theta
# #         
# #         print('x_sum_by_class::', x_sum_by_class/dim[0])
#         
#         
#         
#         
#         vectorized_theta = vectorized_theta - alpha*gradient
#         
#         
#         theta = torch.t(vectorized_theta.view(num_class, dim[1]))
# 
#         
# #         theta = (1-alpha*beta)*theta + (alpha*torch.mm(term1[i], theta) + alpha*(term2[i]).view(theta.shape))/dim[0]
# 
# #         print(multi_res.shape, sub_w_seq[:,i].shape, sub_b_seq[:,i].shape)
#         
# #         lin_res = multi_res * (sub_w_seq[:, i].view(multi_res.shape)) + sub_b_seq[:, i].view(multi_res.shape)
# #         
# # #         print(lin_res.shape)
# # #         print(lin_res.shape, X_Y_mult.shape, sub_w_seq.shape, sub_b_seq.shape)
# #         
# # #         lin_res, total_time = compute_curr_linear_paras(multi_res, Pi, total_time)
# #         
# # #         multi_res *= w_seq
# # #         
# # #         multi_res += b_seq
# #         
# #         
# #         non_linear_term = X_Y_mult*(lin_res)
# # #         w_b_seq = compute_curr_linear_paras(multi_res, Pi)
# # #          
# # #         non_linear_term = X.mul((multi_res*(w_b_seq[:,0].view([dim[0],1])) + w_b_seq[:,1].view([dim[0],1]))*Y)
# #         sum_non_linear_term = torch.sum(non_linear_term, dim=0)
# # #         sum_non_linear_term = np.sum(non_linear_term, axis=0)
# #         
# # #         print(sum_non_linear_term.shape)
# #         
# # #         sum_non_linear_term_diff_dim = np.reshape(sum_non_linear_term, (theta.shape))
# #         sum_non_linear_term_diff_dim = sum_non_linear_term.view( (theta.shape))
# #         
# # #         sum_non_linear_term_diff_dim *=  
# #         
# # #         theta = (1-alpha*beta)*theta + (alpha/dim[0])*torch.sum(X_Y_mult.mul(multi_res), dim=0).view(theta.shape)
# #         theta = (1-alpha*beta)*theta + (alpha/dim[0])*sum_non_linear_term_diff_dim
#         
# #         sizes = [get_tensor_size(multi_res), get_tensor_size(w_seq), get_tensor_size(b_seq)
# #                  , get_tensor_size(non_linear_term), get_tensor_size(sum_non_linear_term), get_tensor_size(sum_non_linear_term_diff_dim)]
#         
#         
# #         cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
# #         add_mem = cur_mem - prev_mem
# #         prev_mem = cur_mem
# #         print("added mem: %sM"%(add_mem))
#         
# #         del multi_res, non_linear_term, sum_non_linear_term, sum_non_linear_term_diff_dim
#         
# #         cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
# #         add_mem = cur_mem - prev_mem
# #         prev_mem = cur_mem
# #         print("added mem: %sM"%(add_mem))
# # #         print('size::', sys.getsizeof(theta)/np.power(2, 10), sys.getsizeof(X_Y_mult)/np.power(2, 10))
# #         
# #         print('size::', sizes)
#         
#     
#         
#     print('total_time::', total_time)
#     
#     return theta
#  
    

def compute_model_parameter_by_approx(w_seq, b_seq, X, Y, dim, theta, X_products, X_Y_products, max_epoch):    
    
    single_coeff = compute_single_coeff(X, Y, w_seq, dim, max_epoch - 1, X_products)
    
#     print('single_coeff::', single_coeff.shape)
    
    
#     print('x_y_term::', compute_x_y_terms(X, Y, b_seq, dim, max_epoch - 1))
    if max_epoch - 2 >= 0:
    
        total_coeff = alpha*torch.mm(single_coeff, compute_x_y_terms(X, Y, b_seq, dim, max_epoch - 2, X_Y_products))

#     print(total_coeff)
    
    
        '''0 to t - 2'''
        for i in range(max_epoch-1):
            
            '''t-1 to t - i - 2'''
            single_coeff = torch.mm(single_coeff, compute_single_coeff(X, Y, w_seq, dim, max_epoch - i - 2, X_products))
            
            if i < max_epoch - 2:
                total_coeff = torch.add(total_coeff, alpha * torch.mm(single_coeff, compute_x_y_terms(X, Y, b_seq, dim, max_epoch - i -3, X_Y_products)))
        
        
        first_term = torch.mm(single_coeff, theta)
        
        second_term = total_coeff
        
        third_term = alpha* compute_x_y_terms(X, Y, b_seq, dim, max_epoch-1, X_Y_products)
        
#         print('first_term1::', first_term)
#         
#         print('second_term1::', second_term)
#         
#         print('third_term1::', third_term)
        
        res = first_term + second_term + third_term
        
        print('res1::', res)
        
        return res
    else:
        first_term = torch.mm(single_coeff, theta)
        
        third_term = alpha* compute_x_y_terms(X, Y, b_seq, dim, max_epoch-1)
        
#         print('first_term1::', first_term)
#         
#         print('third_term1::', third_term)
        
        res = first_term + third_term
        
        print('res1::', res)
        
        return res

def initialize(X, class_num):
    shape = list(X.size())
    theta = Variable(torch.zeros([shape[1],class_num], dtype = torch.double))
#     theta[0][0] = -1
    
    theta.requires_grad = True
#     lr.theta = Variable(lr.theta)

    print(theta.requires_grad)
    
    lr = logistic_regressor_parameter(theta)
    
    return lr



# def compute_naively(X, Y, dim, theta, w_seq, b_seq):
#     
# #     single_coeff1 = (1-beta*alpha)*torch.eye(dim[1], dtype = torch.double)
# #     
# # #     print(len(w_seq))
# # #     
# # #     print(len(w_seq[1]))
# #     
# #     for i in range(dim[0]):
# #         
# #         
# #         single_coeff1 += alpha*w_seq[1][i]*torch.mm(X[i,:].view(dim[1],1), X[i,:].view(1,dim[1]))
#         
#     single_coeff2 = (1-beta*alpha)*torch.eye(dim[1], dtype = torch.double)
#     
#     
#     first_first_term = torch.mm(single_coeff2, theta)
#     
#     first_second_term = torch.zeros([dim[1], dim[1]], dtype = torch.double)
#     
#     for i in range(dim[0]):
#         single_coeff2 += alpha*w_seq[0][i]*torch.mm(X[i,:].view(dim[1],1), X[i,:].view(1,dim[1]))/dim[0]
#         first_second_term += alpha*w_seq[0][i]*torch.mm(X[i,:].view(dim[1],1), X[i,:].view(1,dim[1]))/dim[0]
# #     cross_term1 = torch.zeros(theta.shape, dtype = torch.double)
#     
#     first_second_term = torch.mm(first_second_term, theta)
#     
#     
#     cross_term2 = torch.zeros(theta.shape, dtype = torch.double)
#     
#     for i in range(dim[0]):
# #         cross_term1 += alpha * b_seq[1][i]* Y[i] * X[i, :].view(cross_term1.shape)
#         
#         cross_term2 += alpha * b_seq[0][i]* Y[i] * X[i, :].view(cross_term2.shape)/dim[0]
# 
# 
#     
#     
#     first_term = torch.mm(single_coeff2, theta)
#     
# #     print(single_coeff2.shape)
# #     
# #     print(cross_term2.shape)
#     
# #     second_term = torch.mm(single_coeff1, cross_term2)
#     
#     third_term = cross_term2
#     
# #     res = first_term + second_term + third_term
#     res = first_term + third_term
#     
#     print('first_term2::', first_term)
#     
# #     print('second_term2::', second_term)
#     
#     print('third_term2::', third_term)
#     
#     print('res2::', res)
#     
#     
#     print('first_term2_1::', first_first_term)
#     
#     print('second_term2_1::', first_second_term + third_term)
#     
#     
#     return res



def average_parameter(w_array):
    arr =  np.array(w_array)
    
    avg_value = np.average(arr)
    
    
    for i in range(len(w_array)):
        for j in range(len(w_array[0])):
            w_array[i][j] = avg_value
    
    
#     arr = avg_value
#     
#     return arr

def generate_functions(dim):
    
    function_lists= []
    
    for i in range(dim):
        def curr_func(x): return softmax_layer(x)[i]
        function_lists.append(curr_func)
    
    
    return function_lists


def multi_interpolate_function(x):
    exp_res = torch.exp(x)
    
    exp_sum = torch.sum(exp_res, dim = 2).view(x.shape[0], x.shape[1], 1)
    
    return exp_res/exp_sum


def create_piecewise_linea_class(dim):
#     x_l = torch.tensor(-10, dtype=torch.double)
#     x_u = torch.tensor(10, dtype=torch.double)
    curr_softmax_layer = torch.nn.Softmax(dim = 2) 


    x_l = torch.zeros(dim, dtype = torch.double)
    
    x_l -= 20
    
    x_u = torch.zeros(dim, dtype = torch.double)
    
    x_u += 20

#     function_lists = generate_functions(dim)


#     x_l = -20.0
#     x_u =20.0

    gap = 1e-6
    
    Pis = piecewise_linear_interpolation_multi_dimension(x_l, x_u, curr_softmax_layer, gap)
    
    return Pis

def compute_linear_approx_parameters(X, Y, dim, num_class, max_epoch):
    
    Pis = create_piecewise_linea_class(num_class)
    
#     w_seq = []
#     
#     b_seq = []
    
    t1 = time.time()
    
#     print(X.shape, res_prod_seq.shape)
#     
#     print(torch.mm(X, res_prod_seq))
    '''res_prod_seq:::m*tq'''
    
    res = (torch.mm(X, res_prod_seq)).view(dim[0], max_epoch, num_class)
    
#     print('curr_res::', res)
    
#     res = torch.transpose(res, 1, 2)
    
    
    
    
    
    t2 = time.time()
    
    print('multi_time::', (t2 - t1))
    
    
#     weights = torch.zeros([dim[0], max_epoch, num_class, num_class], dtype = torch.double)
#       
#     offsets = torch.zeros([dim[0], max_epoch, num_class], dtype = torch.double)
    
    '''res:: n*t*q'''
    weights, offsets = Pis.compute_approximate_value_batch(res)
    
#     for i in range(dim[0]):
#         for j in range(max_epoch):
# #             for k in range(num_class):
# #                 print(i, j)
#                 weights[i, j], offsets[i,j] = Pis.compute_approximate_value(res[i][j])
                
#                 x_unit_space = Pis.map_from_original_space(res[i][j], Pis.get_cube_lower_vec(res[i][j]))
#     
#     
# #                 print(x_unit_space)
#                 
#                 exact_value = Pis.function(res[i][j])
#                 approx_value = torch.t(torch.mm(weights[i, j], x_unit_space.view(-1, 1))) + offsets[i,j].view(1,-1)
#                 print('gap::', exact_value - approx_value)
                
    
    
    #     
#     print('expected_sum::',torch.sum(X, dim = 0)/(2*dim[0]))
    return weights, offsets
    

def compute_linear_approx_parameters1(X, Y, dim, num_class, max_epoch):
    
    Pis = create_piecewise_linea_class(num_class)
    
#     w_seq = []
#     
#     b_seq = []
    
    t1 = time.time()
    
#     print(X.shape, res_prod_seq.shape)
#     
#     print(torch.mm(X, res_prod_seq))
    '''res_prod_seq:::m*tq'''
    
    res = (torch.mm(X, res_prod_seq)).view(dim[0], max_epoch, num_class)
    
#     print('curr_res::', res)
    
#     res = torch.transpose(res, 1, 2)
    
    
    
    
    
    t2 = time.time()
    
    print('multi_time::', (t2 - t1))
    
    
#     weights = torch.zeros([dim[0], max_epoch, num_class, num_class], dtype = torch.double)
#       
#     offsets = torch.zeros([dim[0], max_epoch, num_class], dtype = torch.double)
    
    '''res:: n*t*q'''
    offsets = Pis.compute_approximate_value_batch2(res)
    
#     for i in range(dim[0]):
#         for j in range(max_epoch):
# #             for k in range(num_class):
# #                 print(i, j)
#                 weights[i, j], offsets[i,j] = Pis.compute_approximate_value(res[i][j])
                
#                 x_unit_space = Pis.map_from_original_space(res[i][j], Pis.get_cube_lower_vec(res[i][j]))
#     
#     
# #                 print(x_unit_space)
#                 
#                 exact_value = Pis.function(res[i][j])
#                 approx_value = torch.t(torch.mm(weights[i, j], x_unit_space.view(-1, 1))) + offsets[i,j].view(1,-1)
#                 print('gap::', exact_value - approx_value)
                
    
    
    #     
#     print('expected_sum::',torch.sum(X, dim = 0)/(2*dim[0]))
    return offsets
    

   
def compute_linear_approx_parameters2(X, Y, dim, num_class, res_prod_seq, max_epoch):
    
    Pis = create_piecewise_linea_class(num_class)
    
#     w_seq = []
#     
#     b_seq = []
    
    t1 = time.time()
    
#     print(X.shape, res_prod_seq.shape)
#     
#     print(torch.mm(X, res_prod_seq))
    '''res_prod_seq:::m*tq'''
    
    res = (torch.mm(X, res_prod_seq)).view(dim[0], max_epoch, num_class)
    
#     print('curr_res::', res)
    
#     res = torch.transpose(res, 1, 2)
    
    
    
    
    
    t2 = time.time()
    
    print('multi_time::', (t2 - t1))
    
    
#     weights = torch.zeros([dim[0], max_epoch, num_class, num_class], dtype = torch.double)
#       
#     offsets = torch.zeros([dim[0], max_epoch, num_class], dtype = torch.double)
    
    '''res:: n*t*q'''
    weights, offsets = Pis.compute_approximate_value_batch(res)
    
#     for i in range(dim[0]):
#         for j in range(max_epoch):
# #             for k in range(num_class):
# #                 print(i, j)
#                 weights[i, j], offsets[i,j] = Pis.compute_approximate_value(res[i][j])
                
#                 x_unit_space = Pis.map_from_original_space(res[i][j], Pis.get_cube_lower_vec(res[i][j]))
#     
#     
# #                 print(x_unit_space)
#                 
#                 exact_value = Pis.function(res[i][j])
#                 approx_value = torch.t(torch.mm(weights[i, j], x_unit_space.view(-1, 1))) + offsets[i,j].view(1,-1)
#                 print('gap::', exact_value - approx_value)
                
    
    
    #     
#     print('expected_sum::',torch.sum(X, dim = 0)/(2*dim[0]))
    return weights, offsets
#     print(res)
    
#     w_res, b_res = Pi.piecewise_linear_interpolate_coeff_batch2(res)
    
#     print('w_res::', w_res)
#     
#     print('w_size::', w_res.shape)
#     
#     print('b_res::', b_res)
#     
#     print('b_res_size::', b_res.shape)
    
#     return torch.t(w_res), torch.t(b_res)
    
#     interpolation_paras = list(map(Pi.piecewise_linear_interpolate_coeff,res))
    
    
#     for i in range(max_epoch):
#         curr_w_seq = []
#         
#         curr_b_seq = []
#         
#         interpolation_paras = list(map(Pi.piecewise_linear_interpolate_coeff,  torch.mm(X, res_prod_seq[i])*Y))
#         
#         print(interpolation_paras)
# #         torch.dot(res_prod_seq[i].view(theta.shape[0]), X[j].view(theta.shape[0]))
#         
#         
#         for j in range(dim[0]):
#             
# #             print(res_prod_seq[i].view(theta.shape[0]).shape)
# #              
# #             print(X[j].view(theta.shape[0]).shape)
#             
#             x = torch.dot(res_prod_seq[i].view(theta.shape[0]), X[j].view(theta.shape[0]))
#             
#             x = x*Y[j]
#             
#             curr_w, curr_b = Pi.piecewise_linear_interpolate_coeff(x)
#             
# #             curr_w = torch.tensor(0.25, dtype = torch.double)
#             
#             curr_w_seq.append(curr_w)
#             
#             curr_b_seq.append(curr_b)
#             
# #             diff = non_linear_function(x) - curr_w * x - curr_b
#             
#         w_seq.append(curr_w_seq)
#         
#         b_seq.append(curr_b_seq)
    
    
#     return w_seq, b_seq

#     print(interpolation_paras)
#     return interpolation_paras
        
def compute_sample_products(X, dim):    
    X_1 = X.view([dim[0], dim[1], 1])
    
    X_2 = X.view([dim[0], 1, dim[1]])
    
    
    res = torch.bmm(X_1, X_2)
    
    return res

def compute_sample_label_products(X, Y):
    x_y_cross_term = X.mul(Y)
    
    return x_y_cross_term

def get_subset_training_data(X, Y, subset_ids):
    selected_rows = torch.tensor(subset_ids)
    print(selected_rows)
    update_X = torch.index_select(X, 0, selected_rows)
    update_Y = torch.index_select(Y, 0, selected_rows)
    return update_X, update_Y



def compute_x_sum_by_class(X, Y, num_class, dim):
    
#     x_sum_by_class = torch.zeros([num_class, dim[1]], dtype = torch.double)
    
    
    y_onehot = torch.DoubleTensor(dim[0], num_class)

    Y = Y.type(torch.LongTensor)

# In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, Y.view(-1, 1), 1)
    
    
    x_sum_by_class = torch.mm(torch.t(y_onehot), X)
    
#     for i in range(num_class):
#         Y_mask = (Y == i)
#         
#         Y_mask = Y_mask.type(torch.DoubleTensor)
#         
#         x_sum_by_class[i] = torch.mm(torch.t(Y_mask), X) 
        
    return x_sum_by_class.view(-1, 1)

def compute_x_sum_by_class_by_batch(X, Y, batch_size, num_class, max_epoch):
    
#     x_sum_by_class = torch.zeros([num_class, dim[1]], dtype = torch.double)
    
    
    x_sum_by_class_list = []
    
    
    for j in range(len(random_ids_multi_super_iterations)):
        random_ids = random_ids_multi_super_iterations[j]
    
    
#         curr_X = X[random_ids]
#         
#         curr_Y = Y[random_ids]
    
    
        for i in range(0, X.shape[0], batch_size):
            end_id = i + batch_size
                
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            
            
            curr_rand_ids = random_ids[i:end_id]
            
            batch_X, batch_Y = X[curr_rand_ids], Y[curr_rand_ids]
        
            y_onehot = torch.DoubleTensor(batch_X.shape[0], num_class)
        
        
            batch_Y = batch_Y.type(torch.LongTensor)
        
        # In your for loop
            y_onehot.zero_()
            y_onehot.scatter_(1, batch_Y.view(-1, 1), 1)
            
            
            x_sum_by_class = torch.mm(torch.t(y_onehot), batch_X)
            
            x_sum_by_class_list.append(x_sum_by_class.view(-1,1))
    
#     for i in range(num_class):
#         Y_mask = (Y == i)
#         
#         Y_mask = Y_mask.type(torch.DoubleTensor)
#         
#         x_sum_by_class[i] = torch.mm(torch.t(Y_mask), X) 
        
    return torch.stack(x_sum_by_class_list, dim = 0)


def compute_x_sum_by_class_by_batch1(batch_X_list, batch_Y_list, num_class):
    
#     x_sum_by_class = torch.zeros([num_class, dim[1]], dtype = torch.double)
    
    
    x_sum_by_class_list = []
    
#     for i in range(0, X.shape[0], batch_size):
#         end_id = i + batch_size
#             
#         if end_id > X.shape[0]:
#             end_id = X.shape[0]
#             
#         
#         curr_selected_data_ids = delta_data_ids[(torch.nonzero((delta_data_ids >= i)*(delta_data_ids < end_id))).view(-1)]
# 
#         if curr_selected_data_ids.shape[0] <= 0:
#             continue
    
    for i in range(len(batch_X_list)):
    
        batch_X, batch_Y = batch_X_list[i], batch_Y_list[i]
    
        y_onehot = torch.DoubleTensor(batch_X.shape[0], num_class)
    
    
        batch_Y = batch_Y.type(torch.LongTensor)
    
    # In your for loop
        y_onehot.zero_()
        y_onehot.scatter_(1, batch_Y.view(-1, 1), 1)
        
        
        x_sum_by_class = torch.mm(torch.t(y_onehot), batch_X)
        
        x_sum_by_class_list.append(x_sum_by_class.view(-1,1))
    
    
    
    
    
    
    
#     for i in range(len(batch_X_list)):
#         
#         batch_X = batch_X_list[i]
#         
#         batch_Y = batch_Y_list[i]
#         
#     
#         y_onehot = torch.DoubleTensor(batch_X.shape[0], num_class)
    
        
    
#     for i in range(num_class):
#         Y_mask = (Y == i)
#         
#         Y_mask = Y_mask.type(torch.DoubleTensor)
#         
#         x_sum_by_class[i] = torch.mm(torch.t(Y_mask), X) 
        
    return x_sum_by_class_list

def compute_x_sum_by_class_by_batch2(X, Y, delta_data_ids, num_class, batch_size):
    
#     x_sum_by_class = torch.zeros([num_class, dim[1]], dtype = torch.double)
    
    
    x_sum_by_class_list = {}
    
    for i in range(0, X.shape[0], batch_size):
        end_id = i + batch_size
            
        if end_id > X.shape[0]:
            end_id = X.shape[0]
            
        
        curr_selected_data_ids = delta_data_ids[(torch.nonzero((delta_data_ids >= i)*(delta_data_ids < end_id))).view(-1)]

        if curr_selected_data_ids.shape[0] <= 0:
            continue
    
    
    
        batch_X, batch_Y = X[curr_selected_data_ids], Y[curr_selected_data_ids]
    
        y_onehot = torch.DoubleTensor(batch_X.shape[0], num_class)
    
    
        batch_Y = batch_Y.type(torch.LongTensor)
    
    # In your for loop
        y_onehot.zero_()
        y_onehot.scatter_(1, batch_Y.view(-1, 1), 1)
        
        
        x_sum_by_class = torch.mm(torch.t(y_onehot), batch_X)
        
        x_sum_by_class_list[int(i/batch_size)] = x_sum_by_class.view(-1,1)
    
    
    
    
    
    
    
#     for i in range(len(batch_X_list)):
#         
#         batch_X = batch_X_list[i]
#         
#         batch_Y = batch_Y_list[i]
#         
#     
#         y_onehot = torch.DoubleTensor(batch_X.shape[0], num_class)
    
        
    
#     for i in range(num_class):
#         Y_mask = (Y == i)
#         
#         Y_mask = Y_mask.type(torch.DoubleTensor)
#         
#         x_sum_by_class[i] = torch.mm(torch.t(Y_mask), X) 
        
    return x_sum_by_class_list


def eigen_decomposition(term1, dim, num_class, batch_size, X, weights, cut_off_super_iteration):
    
    
#     A = (1-alpha*beta)*torch.eye(dim[1]*num_class, dtype = torch.double) - alpha*(term1[cut_off_epoch - 1])/dim[0]
    min_batch_num_per_epoch = int((dim[0] - 1)/batch_size) + 1

    M_list = []
    
    M_inverse_list = []
    
    s_list = []
    
    
    curr_rand_ids = random_ids_multi_super_iterations[cut_off_super_iteration - 1]

    j = 0
    weights_this_super_iteration = weights[(cut_off_super_iteration - 1)*dim[0]:(cut_off_super_iteration)*dim[0]]

    end_id = 0
    for i in range(min_batch_num_per_epoch):
        
        
        end_id = j+batch_size
            
        if j >= X.shape[0]:
            end_id = X.shape[0]    
            
            
        
        curr_term_1 = term1[-min_batch_num_per_epoch + i]
    
        s, M = torch.eig(curr_term_1, True)
        
        s = s[:,0]
        
        s_list.append(s)
        
        M_list.append(M)
        
#         print('eigen_values::', s)
            
        M_inverse = torch.tensor(np.linalg.inv(M.numpy()), dtype = torch.double)
        
        M_inverse_list.append(M_inverse)
        
        computed_s = compute_delta_s(M, M_inverse, X[curr_rand_ids[j:end_id]], weights_this_super_iteration[curr_rand_ids[j:end_id]], X[curr_rand_ids[j:end_id]].shape, num_class)
            
        j= end_id    
        
        print(computed_s - s)
        
    #     M_inverse = torch.inverse(M)
        
        
        print('inverse_gap::', torch.norm(torch.mm(M, M_inverse) - torch.eye(dim[1]*num_class, dtype = torch.double)))
#     avg_term1 = torch.mean(term1[-min_batch_num_per_epoch:], 0)
    
    torch.save(M_list, git_ignore_folder + 'eigen_vectors')
#     A = A.type(torch.FloatTensor)
    
    
    
    torch.save(M_inverse_list, git_ignore_folder + 'eigen_vectors_inverse')
    
    torch.save(s_list, git_ignore_folder + 'eigen_values')
    
# def compute_single_svd(term1):
#     curr_term1 = term1.numpy()
#         
#     u,s,vt = np.linalg.svd(curr_term1)
#     
#     non_zero_ids = (s >= 20)
#     
#     
#     sub_u = u[:,non_zero_ids]
#      
#     sub_s = s[non_zero_ids]
#      
#     sub_v = vt[non_zero_ids]
#     
#     res = np.dot(sub_u*sub_s, sub_v)
#     
#     print(np.linalg.norm(res - curr_term1))
#     
#     print(sub_s.shape)
#     
#     return torch.from_numpy(sub_u*sub_s), torch.from_numpy(sub_v)


def compute_approx_dimension(s):
    
    s_square = np.power(s, 2)
    
    s_square_sum = np.sum(s_square)
    
    curr_s_square_sum = 0
    
    id = 0
    
    for i in range(s.shape[0]):
        curr_s_square_sum += s_square[i]
        ratio = curr_s_square_sum*1.0/s_square_sum
        
        if ratio >= .99:
            id= i
            
            break
    
    
    return id
    
# def compute_approx_dimension2(u, s, vt):
#     
#     s_square = np.power(s, 2)
#     
#     s_square_sum = np.sum(s_square)
#     
#     curr_s_square_sum = 0
#     
#     id = 0
#     
#     for i in range(s.shape[0]):
#         curr_s_square_sum += s_square[i]
#         ratio = curr_s_square_sum*1.0/s_square_sum
#         
#         if ratio >= .99:
#             id= i
#             
#             break
#     
#     
#     return id    


def compute_single_svd(i, term1, batch_size):
    
    if batch_size < term1.shape[1]:
        upper_bound = int(batch_size/svd_ratio)
    else:
        upper_bound = int(term1.shape[1]/svd_ratio)
    
    curr_term1 = term1.numpy()
        
#     u,s,vt = np.linalg.svd(curr_term1)
    
    
    
    u, s, vt = randomized_svd(curr_term1, n_components=upper_bound, random_state=None)

    
#         upper_bound = compute_approx_dimension(s)
#         non_zero_ids = (s >= 1)
    
    sub_s = s[0:upper_bound]
    
    if sub_s.shape[0] <= 0:
#             non_zero_ids = np.array([0,1])
        upper_bound = 1
        
        sub_s = s[0:upper_bound]
        
    
    sub_u = u[:,0:upper_bound]
     
    
     
    sub_v = vt[0:upper_bound]
    
    res = np.dot(sub_u*sub_s, sub_v)
    
    print(i, upper_bound, np.linalg.norm(res - curr_term1), np.linalg.norm(res - curr_term1)/np.linalg.norm(curr_term1))
    
    return torch.from_numpy(sub_u*sub_s), torch.from_numpy(sub_v)




    
def compute_svd(term1, dim, num_class, batch_size):


#     u_list = []

#     s_list = []
    
#     v_s_list = []
    
    
    upper_bound = int(batch_size/10)
    
    
    directory = git_ignore_folder + 'svd_folder'
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
#     if upper_bound <= 0:
#         upper_bound = 1

    for i in range(len(term1)):
        curr_term1 = term1[i].numpy()
        
        u,s,vt = np.linalg.svd(curr_term1)
        
        
#         upper_bound = compute_approx_dimension(s)
#         non_zero_ids = (s >= 1)
        
        sub_s = s[0:upper_bound]
        
        if sub_s.shape[0] <= 0:
#             non_zero_ids = np.array([0,1])
            upper_bound = 1
            
            sub_s = s[0:upper_bound]
            
        
        sub_u = u[:,0:upper_bound]
         
        
         
        sub_v = vt[0:upper_bound]
        
        res = np.dot(sub_u*sub_s, sub_v)
        
        print(i, upper_bound, np.linalg.norm(res - curr_term1), np.linalg.norm(res - curr_term1)/np.linalg.norm(curr_term1))
#         
#         print(sub_s.shape)

        del vt, u, s
        
#         u_list.append(torch.from_numpy(sub_u*sub_s))
#         
#         v_s_list.append(torch.from_numpy(sub_v))
    
        np.save(directory + '/u_' + str(i), sub_u*sub_s)
    
        np.save(directory + '/v_' + str(i), sub_v)



    torch.save(torch.tensor([len(term1)]), directory + '/len')

#     for i in range(term1.shape[0]):
#         curr_term1 = term1[i].numpy()
#         
#         u,s,vt = np.linalg.svd(curr_term1)
#         
#         non_zero_ids = (s >= 20)
#         
#         sub_s = s[non_zero_ids]
#         
#         if sub_s.shape[0] <= 0:
#             non_zero_ids = np.array([0,1])
#             sub_s = s[non_zero_ids]
#             
#         
#         sub_u = u[:,non_zero_ids]
#          
#         
#          
#         sub_v = vt[non_zero_ids]
#         
#         res = np.dot(sub_u*sub_s, sub_v)
#         
# #         print(np.linalg.norm(res - curr_term1))
# #         
# #         print(sub_s.shape)
#         
#         u_list.append(torch.from_numpy(sub_u*sub_s))
#         
#         v_s_list.append(torch.from_numpy(sub_v))
        
#         v_list.append(torch.from_numpy(sub_v))
        
#     torch.save(u_list, git_ignore_folder + 'u_list')
    
#     torch.save(s_list, git_ignore_folder + 's_list')
    
#     torch.save(v_s_list, git_ignore_folder + 'v_s_list')

    
def eigen_decomposition2(term1, dim, num_class, batch_size):
    
    
#     A = (1-alpha*beta)*torch.eye(dim[1]*num_class, dtype = torch.double) - alpha*(term1[cut_off_epoch - 1])/dim[0]
    min_batch_num_per_epoch = int((dim[0] - 1)/batch_size) + 1

    sum_term1 = torch.mean(term1[-min_batch_num_per_epoch:], 0)


    s, M = torch.eig(sum_term1, True)
    s = s[:,0]
    M_inverse = torch.tensor(np.linalg.inv(M.numpy()), dtype = torch.double)
    
    torch.save(M, git_ignore_folder + 'eigen_vectors')
#     A = A.type(torch.FloatTensor)
    
    
    
    torch.save(M_inverse, git_ignore_folder + 'eigen_vectors_inverse')
    
    torch.save(s, git_ignore_folder + 'eigen_values')
    
#     torch.save(A, git_ignore_folder + 'expected_A')



def eigen_decomposition3(avg_term1):
    
    
#     A = (1-alpha*beta)*torch.eye(dim[1]*num_class, dtype = torch.double) - alpha*(term1[cut_off_epoch - 1])/dim[0]
#     min_batch_num_per_epoch = int((dim[0] - 1)/batch_size) + 1

#     sum_term1 = torch.mean(term1[-min_batch_num_per_epoch:], 0)


    s, M = torch.eig(avg_term1, True)
    s = s[:,0]
    M_inverse = torch.tensor(np.linalg.inv(M.numpy()), dtype = torch.double)
    
    torch.save(M, git_ignore_folder + 'eigen_vectors')
#     A = A.type(torch.FloatTensor)
    
    
    
    torch.save(M_inverse, git_ignore_folder + 'eigen_vectors_inverse')
    
    torch.save(s, git_ignore_folder + 'eigen_values')
    
#     torch.save(A, git_ignore_folder + 'expected_A')



def prepare_term_1_2_large_feature_space(X, weights, offsets, dim, max_epoch, num_class, cut_off_epoch, batch_size, curr_rand_ids_multi_super_iterations, mini_epochs_per_super_iteration):

#     term1 = torch.zeros([cut_off_epoch, num_class*dim[1], dim[1]*num_class], dtype = torch.double)
    
    term2 = torch.zeros([cut_off_epoch, num_class*dim[1]], dtype = torch.double)
    
    epoch = 0
    
    end = False
    
#     u_list = []
#     
#     v_list = []
    
    cut_off_super_iteration = (int((cut_off_epoch - 1)/mini_epochs_per_super_iteration) + 1)
    
    avg_term1  = 0
    
    directory = git_ignore_folder + 'svd_folder'
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for k in range(curr_rand_ids_multi_super_iterations.shape[0]):
    
        curr_rand_ids = curr_rand_ids_multi_super_iterations[k]
        
        
        weights_this_super_iter = weights[k*X.shape[0]:(k+1)*X.shape[0]]
        
        offsets_this_super_iter = offsets[k*X.shape[0]:(k+1)*X.shape[0]]
    
        for i in range(0, X.shape[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
    
            batch_X = X[curr_rand_ids[i:end_id]]
            
            batch_weights = weights_this_super_iter[curr_rand_ids[i:end_id]]
            
            batch_offsets = offsets_this_super_iter[curr_rand_ids[i:end_id]]
            
            batch_term1 = prepare_sub_term_1(batch_X, batch_weights, batch_X.shape, num_class)
            
            
            if epoch >= cut_off_epoch - mini_epochs_per_super_iteration:
                avg_term1 = avg_term1 + batch_term1
            
            
            sub_u, sub_v = compute_single_svd(epoch, batch_term1, batch_size)
            
            
            
            
            
            np.save(directory + '/u_' + str(epoch), sub_u)
    
            np.save(directory + '/v_' + str(epoch), sub_v)
    
            del sub_u, sub_v
    
#             u_list.append(sub_u)
#             
#             v_list.append(sub_v)
#             term1[epoch] = batch_term1
            
            batch_term2 = prepare_sub_term_2(batch_X, batch_offsets, batch_X.shape, num_class)
            
            term2[epoch] = batch_term2
            
            
            epoch = epoch + 1
            
            if epoch >= cut_off_epoch:
                end = True
                break
        if end == True:
            break
        
        
    
    torch.save(torch.tensor([cut_off_epoch]), directory + '/len')

        
    
#     torch.save(u_list, git_ignore_folder + 'u_list')
    
#     torch.save(s_list, git_ignore_folder + 's_list')
    
#     torch.save(v_list, git_ignore_folder + 'v_s_list')
    
    return avg_term1, term2




def save_svd(term1, name):
    
    
#     try:
#         torch.save(term1, git_ignore_folder + 'term1')
#     except:
    directory = git_ignore_folder + name + '_folder'
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i in range(len(term1)):
        
        
        np.save(directory + '/' + str(i), term1[i])
            
            
    torch.save(torch.tensor([len(term1)]), directory + '/' + name + '_len')




def save_random_id_orders(random_ids_multi_super_iterations):
    sorted_ids_multi_super_iterations = []
    
    
    for i in range(random_ids_multi_super_iterations.shape[0]):
        sorted_ids_multi_super_iterations.append(random_ids_multi_super_iterations[i].numpy().argsort())
        
        
    torch.save(sorted_ids_multi_super_iterations, git_ignore_folder + 'sorted_ids_multi_super_iterations')


def capture_provenance(X, Y, dim, epoch, num_class, batch_size, mini_epochs_per_super_iteration, random_ids_multi_super_iterations):
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
 
 
    # term1, term1_inter_result = prepare_term_1_serial(X, w_seq, dim)
    # term1 = prepare_term_1_batch2(X_product, weights, dim, max_epoch, num_class)
     
#     X_theta_prod_softmax_seq_tensor = torch.stack(X_theta_prod_softmax_seq, dim = 0)
#      
#     X_theta_prod_seq_tensor = torch.stack(X_theta_prod_seq, dim = 0)
    
    
#     torch.save(X_theta_prod_seq_tensor, git_ignore_folder + 'X_theta_prod_seq_tensor')
#     
#     torch.save(X_theta_prod_softmax_seq_tensor, git_ignore_folder + 'X_theta_prod_softmax_seq_tensor')
    
    
    save_random_id_orders(random_ids_multi_super_iterations)
    
    
    global X_theta_prod_softmax_seq, X_theta_prod_seq
    
    
    super_iteration = (int((len(X_theta_prod_softmax_seq) - 1)/mini_epochs_per_super_iteration) + 1)
    
    
#     cut_off_super_iteration = int(super_iteration*theta_record_threshold)#(int((cut_off_epoch - 1)/mini_epochs_per_super_iteration) + 1)
    
    
#     cut_off_epoch = cut_off_super_iteration*mini_epochs_per_super_iteration
    
    
    cut_off_epoch= len(X_theta_prod_softmax_seq)
    
    
#     cut_off_epoch = len(X_theta_prod_softmax_seq)
    
    print('super_iteration::', super_iteration)
#     print('cut_off_super_iteration::', cut_off_super_iteration)

    print('cut_off_epoch::', cut_off_epoch)
#   weights, offsets = prepare_term_1_batch3_1(random_ids_multi_super_iterations, theta_list, X, Y, dim, epoch, num_class, cut_off_epoch, batch_size)
    weights, offsets = prepare_term_1_batch3_0(X_theta_prod_softmax_seq, X_theta_prod_seq, X, dim, epoch, num_class, cut_off_epoch, batch_size) 

    cut_off_super_iteration = (int((cut_off_epoch - 1)/mini_epochs_per_super_iteration) + 1)
    
    
    curr_rand_ids_multi_super_iterations = random_ids_multi_super_iterations[0:(int((cut_off_epoch - 1)/mini_epochs_per_super_iteration) + 1)*dim[0]]
    
    '''T*dim[0]'''
    
    curr_rand_ids_multi_super_iterations = curr_rand_ids_multi_super_iterations.view(-1, dim[0])
    
    _, sorted_ids_multi_super_iterations = torch.sort(curr_rand_ids_multi_super_iterations)


    weights_copy = torch.zeros([dim[0]*cut_off_super_iteration, num_class, num_class], dtype = torch.double)
    
    offsets_copy = torch.zeros([dim[0]*cut_off_super_iteration, num_class], dtype = torch.double)
    
    weights_copy[0:weights.shape[0]] = weights
    
    offsets_copy[0:offsets.shape[0]] = offsets
    
    weights_copy = weights_copy.view(cut_off_super_iteration, dim[0], num_class*num_class)
    
    offsets_copy = offsets_copy.view(cut_off_super_iteration, dim[0], num_class)

    for i in range(cut_off_super_iteration):
        weights_copy[i, :, :] = weights_copy[i, sorted_ids_multi_super_iterations[i], :]
        offsets_copy[i, :, :] = offsets_copy[i, sorted_ids_multi_super_iterations[i], :]
    
    
    weights_copy = weights_copy.view(dim[0]*cut_off_super_iteration, num_class*num_class)
    
    offsets_copy = offsets_copy.view(dim[0]*cut_off_super_iteration, num_class)


    print('compute_weights_offsets_done!!')

    print(weights_copy.shape)

#     '''for small feature space'''
#     if num_class*dim[1] < batch_size:
#         term1, term2 = prepare_term_1_batch2_0_1(X, weights_copy, offsets_copy, dim, epoch, num_class, cut_off_epoch, batch_size, curr_rand_ids_multi_super_iterations)
#      
#      
# #         if batch_size < num_class*X.shape[1]:
# #             compute_svd(term1, dim, num_class, batch_size)
# #             torch.save(term1, git_ignore_folder + 'term1')
# #         else:
#              
#         print('store term1!!!!')
#         torch.save(term1, git_ignore_folder + 'term1')
#         term1 = term1[:cut_off_epoch]
#      
#         eigen_decomposition2(term1, dim, num_class, batch_size)
# 
# 
# 
#         '''for large feature space'''
#     else:

    avg_term1, term2 = prepare_term_1_2_large_feature_space(X, weights_copy, offsets_copy, dim, epoch, num_class, cut_off_epoch, batch_size, curr_rand_ids_multi_super_iterations, mini_epochs_per_super_iteration)

    eigen_decomposition3(avg_term1)
#     min_batch_num_per_epoch = int((X.shape[0] - 1)/batch_size) + 1
#     avg_term1 = torch.mean(term1[-min_batch_num_per_epoch:], 0)
#     avg_u, avg_v_s = compute_single_svd(avg_term1)
#     
#     torch.save(avg_u, git_ignore_folder + 'avg_u')
#     
#     torch.save(avg_v_s, git_ignore_folder + 'avg_v_s')

    cut_off_super_iteration = int(super_iteration*theta_record_threshold)#(int((cut_off_epoch - 1)/mini_epochs_per_super_iteration) + 1)
    
    
    cut_off_epoch = cut_off_super_iteration*mini_epochs_per_super_iteration


    

#     torch.save(avg_term1, git_ignore_folder + 'avg_term1')

#     term1 = prepare_term_1_batch2(X_product, weights, dim, epoch, num_class)
    
#     term1 = prepare_term_1_mini_batch(X, weights, dim, num_class, batch_size)
    
    torch.save(weights_copy, git_ignore_folder+'weights')
    
#     torch.save(term1, git_ignore_folder+'term1')
    
#     torch.save(X_product, git_ignore_folder + 'X_product')
    
    torch.save(cut_off_epoch, git_ignore_folder + 'cut_off_epoch')
    
    del weights
    
#     del term1
    
    print('save weights and term 1 done!!!')
    
    
    
#     term2 = prepare_term_2_batch2(X, offsets, dim, epoch, num_class)
     
    torch.save(offsets_copy, git_ignore_folder+'offsets')
    
    torch.save(term2, git_ignore_folder+'term2') 
    
#     torch.save(X_theta_prod_softmax_seq_tensor, git_ignore_folder + 'X_theta_prod_softmax_seq_tensor')
     
    del term2
    
    del offsets
    
    print('save offsets and term 2 done!!!')

def precomptation_influence_function(X, Y, res, dim):
    
    t5 = time.time()
    
    X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     X_Y_mult = X.mul(Y)
    
#     Hessin_matrix = compute_hessian_matrix(X, X_Y_mult, res, dim, X_product)


    Hessin_matrix = compute_hessian_matrix_2(res, X, dim, num_class, X_product)
    
#     Hessin_matrix2 = compute_hessian_matrix_3(X, X_Y_mult, res, dim)
    
#     print(Hessin_matrix)
#     
#     print(Hessin_matrix - Hessin_matrix2)
    

    Hessian_inverse = torch.inverse(Hessin_matrix)
    
    torch.save(Hessian_inverse, git_ignore_folder + 'Hessian_inverse')
    
#     torch.save(X, git_ignore_folder + 'X')
#     
#     torch.save(Y, git_ignore_folder + 'Y')
    
#     torch.save(X_Y_mult, git_ignore_folder + 'X_Y_mult')
    
    torch.save(res, git_ignore_folder + 'model_origin')
    
    t6 = time.time()
    
    
    print('preparing_time_2::', t6 - t5)
def compute_hessian_matrix_3(theta, X, dim, num_class, batch_size):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    
    '''n*q'''
    
    X_theta_prod_seq_tensor = torch.mm(X, theta)
    
    X_theta_prod_softmax_seq_tensor = softmax_layer(X_theta_prod_seq_tensor)
    
    
    
    
#     X_theta_prod_softmax_seq_tensor = torch.transpose(X_theta_prod_softmax_seq_tensor, 0 ,1)
    
#     w_dim = weights.shape
    
#     print(w_dim)
    
#     print(dim)
    
#     res = Variable(torch.zeros([max_epoch, dim[1]*num_class, dim[1]*num_class], dtype = torch.double))
    
#     last_weight = None
#     
#     last_offsets = None
    
#     for i in range(max_epoch):
    '''X_theta_prod_softmax_seq_tensor[i]: n*q'''
    
    curr_weight = Variable(torch.bmm(X_theta_prod_softmax_seq_tensor.view(dim[0],num_class ,1), X_theta_prod_softmax_seq_tensor.view(dim[0],1,num_class)))
    
    
    '''n*q*q'''
    curr_weight_sum = torch.sum(curr_weight, dim = 1)
    
    curr_weight3 = torch.diag_embed(curr_weight_sum) - curr_weight
    
#     X_product1 = torch.t(X_product.view(dim[0], dim[1]*dim[1]))
    
    res1 = torch.zeros((X.shape[1]*X.shape[1], num_class*num_class), dtype = torch.double)
    
    for i in range(0,X.size()[0], batch_size):
        
        curr_X = X[i:i+batch_size]
        
        this_weight = curr_weight3[i:i+batch_size]
        
        curr_X_prod = torch.bmm(curr_X.view(curr_X.shape[0], X.shape[1], 1), curr_X.view(curr_X.shape[0], 1, X.shape[1]))
    
        res1 += torch.mm(torch.t(curr_X_prod.view(curr_X.shape[0], dim[1]*dim[1])), this_weight.view(curr_X.shape[0], num_class*num_class))
    
        del curr_X_prod
    
#     res1 = Variable(torch.mm(X_product1, curr_weight3.view(dim[0], num_class*num_class)))
#     
#     del X_product1
    
    res2 = torch.t(res1.view(dim[1]*dim[1], num_class*num_class))

    del res1
    
    res3 = res2.view(num_class, num_class, dim[1], dim[1])

    del res2
#     res = torch.transpose(res, 1, 2)
    
    res4 = torch.transpose(res3, 1, 2)
    
    del res3
    
    print(res4.shape)
    
    res = torch.reshape(res4, [num_class*dim[1], dim[1]*num_class])
    
    del res4
    
    
    
#         curr_weight1 = torch.diag_embed(torch.diagonal(curr_weight, dim1=1, dim2=2)) - curr_weight
#         
#         
#         
#         curr_weight2 = torch.diag_embed(-torch.sum(curr_weight1, dim = 1)) + curr_weight1
#         
#         print(curr_weight3 - curr_weight2)
    
#     del curr_weight
#     
#     del curr_weight_sum
#     
#     weights[:,i,:,:] = curr_weight3
#     
#     curr_offsets = X_theta_prod_softmax_seq_tensor - (torch.bmm(X_theta_prod_seq_tensor.view(dim[0], 1, num_class), curr_weight3.view(dim[0], num_class, num_class))).view(dim[0], num_class)
#     
#         if i >= 1:
#             print('weight_changed::', i, torch.norm(curr_weight-last_weight))
#             print('offsets_changed::', i, torch.norm(offsets[:,i,:] - last_offsets))
#         last_weight = curr_weight
#         
#         last_offsets = offsets[:,i,:]
        
#         print(weights[:,i,:,:] - curr_weight)
        
#         curr_res = Variable(torch.bmm(X.view(dim[0], dim[1], 1), curr_weight.view(dim[0], 1, num_class*num_class)))
#         
#         
#         curr_res = torch.transpose(curr_res, 1, 2)
#         
#         curr_res = torch.mm(torch.t(X), curr_res.contiguous().view(dim[0], dim[1]*num_class*num_class))
#         
#         curr_res = torch.transpose(curr_res.view(dim[1], num_class, num_class, dim[1]), 0,1)
#         
#         curr_res = curr_res.contiguous().view(dim[1]*num_class, dim[1]*num_class)
#         
#         res[i] = curr_res
        
        
    return res/dim[0] + beta*torch.eye(num_class*dim[1], dtype = torch.double)


def precomptation_influence_function2(X, Y, res, dim):
    
    t5 = time.time()
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     X_Y_mult = X.mul(Y)
    
#     Hessin_matrix = compute_hessian_matrix(X, X_Y_mult, res, dim, X_product)


    Hessin_matrix = compute_hessian_matrix_3(res, X, dim, num_class, 1000)
    
#     Hessin_matrix2 = compute_hessian_matrix_3(X, X_Y_mult, res, dim)
    
#     print(Hessin_matrix)
#     
#     print(Hessin_matrix - Hessin_matrix2)
    

    Hessian_inverse = torch.inverse(Hessin_matrix)
    
    torch.save(Hessian_inverse, git_ignore_folder + 'Hessian_inverse')
    
#     torch.save(X, git_ignore_folder + 'X')
#     
#     torch.save(Y, git_ignore_folder + 'Y')
    
#     torch.save(X_Y_mult, git_ignore_folder + 'X_Y_mult')
    
    torch.save(res, git_ignore_folder + 'model_origin')
    
    t6 = time.time()
    
    
    print('preparing_time_2::', t6 - t5)
    
    return Hessian_inverse


def change_data_labels(X, Y, ratio, num_class, res):
    
    expected_selected_label =0
    
    updated_selected_label = 0
    
    max_count = -1
    
    min_count = Y.shape[0] + 1
    
    for i in range(num_class):
        label_count = torch.sum((Y == i)) 
        if label_count > max_count:
            max_count = label_count
            expected_selected_label = i
        
        if label_count < min_count:
            min_count = label_count
            updated_selected_label = i
    
    
    
    
    
    
    delta_data_ids = set()

    multi_res = softmax_layer(torch.mm(X, res))
    
    prob, predict_labels = torch.max(multi_res, 1)
    
    least_prob, least_labels = torch.min(multi_res, 1)
    
    print(prob)
    
#     predict_labels = torch.argmax(multi_res, 1)
    selected_num = int(X.shape[0]*ratio)

    
    sorted_prob, indices = torch.sort(prob.view(-1), descending = True)
    
    for i in range(selected_num):
        if Y[indices[i],0].numpy() != updated_selected_label:# and multi_res[indices[i]]*Y[indices[i]] > 0: 
#             X[indices[i]] = X[indices[i]]*torch.rand(X[indices[i]].shape, dtype = torch.double)

#         new_label = torch.randint(low=0, high = num_class, size = (1,))
#         
#         while new_label[0] == Y[indices[i]]:
#             new_label = torch.randint(low=0, high = num_class, size = (1,))

            Y[indices[i]] = updated_selected_label
            delta_data_ids.add(indices[i])
    
    
    
    
    
#     selected_ids = torch.tensor(np.random.choice(list(range(X.shape[0])), size = selected_num, replace=False))
#     
#     for i in selected_ids:
#         new_label = torch.randint(low=0, high = num_class, size = (1,))
#         
#         while new_label[0] == Y[i]:
#             new_label = torch.randint(low=0, high = num_class, size = (1,))
#             
#         Y[i] = new_label
             
    return X, Y, torch.tensor(list(delta_data_ids))

def rescale_data(X, Y, ratio, res):
    
    expected_selected_label =0
    
    updated_selected_label = 0
    
    max_count = -1
    
    min_count = Y.shape[0] + 1
    
    for i in range(num_class):
        label_count = torch.sum((Y == i)) 
        if label_count > max_count:
            max_count = label_count
            expected_selected_label = i
        
        if label_count < min_count:
            min_count = label_count
            updated_selected_label = i
    
    
    
    
    
    delta_data_ids = set()

    multi_res = softmax_layer(torch.mm(X, res))
    
    prob, predict_labels = torch.max(multi_res, 1)
    
    least_prob, least_labels = torch.min(multi_res, 1)
    
    print(prob)
    
#     predict_labels = torch.argmax(multi_res, 1)
    selected_num = int(X.shape[0]*ratio)

    
    sorted_prob, indices = torch.sort(prob.view(-1), descending = True)
    
    for i in range(selected_num):
#         if Y[indices[i],0].numpy() == expected_selected_label:# and multi_res[indices[i]]*Y[indices[i]] > 0: 
#             X[indices[i]] = X[indices[i]]*torch.rand(X[indices[i]].shape, dtype = torch.double)

#         new_label = torch.randint(low=0, high = num_class, size = (1,))
#         
#         while new_label[0] == Y[indices[i]]:
#             new_label = torch.randint(low=0, high = num_class, size = (1,))

            X[indices[i]] = X[indices[i]] + 50*torch.rand(X[indices[i]].shape, dtype = torch.double)
            delta_data_ids.add(indices[i])
    
    
    
    
    
#     selected_ids = torch.tensor(np.random.choice(list(range(X.shape[0])), size = selected_num, replace=False))
#     
#     for i in selected_ids:
#         new_label = torch.randint(low=0, high = num_class, size = (1,))
#         
#         while new_label[0] == Y[i]:
#             new_label = torch.randint(low=0, high = num_class, size = (1,))
#             
#         Y[i] = new_label
             
    return X, Y, torch.tensor(list(delta_data_ids))

def add_noise_data(X, Y, num, res, num_class):
    
    
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
    multi_res = softmax_layer(torch.mm(X, res))
    
    prob, predict_labels = torch.max(multi_res, 1)
    
    print(prob)
    
#     predict_labels = torch.argmax(multi_res, 1)
    
    
    sorted_prob, indices = torch.sort(prob.view(-1), descending = True)
#     sorted, indices = torch.sort(torch.abs(multi_res), dim = 0, descending = True)
    selected_point = None
    
    selected_label = None
    
    selected_id = 0
    
    
    selected_points = []
    
    
    noise_data_X = torch.zeros((num, X.shape[1]), dtype = torch.double)

    noise_data_Y = torch.zeros((num, 1), dtype = torch.long)
    
    for i in range(num):
        
        curr_class = Y[indices[i], 0]
        
        curr_coeff = mean_list[curr_class]/mean_list[(curr_class + 1)%(num_class)]        
         
        curr_coeff = curr_coeff[curr_coeff != np.inf]
        
        curr_coeff = torch.sum(curr_coeff[curr_coeff == curr_coeff])
         
#         curr_coeff = torch.sum(curr_coeff[curr_coeff != np.inf and np.isnan(curr_coeff.numpy())])
        
#         print(curr_coeff)
        
        selected_point = (X[indices[i]].clone())*curr_coeff
        
        
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

def random_deletion(X, Y, delta_num, res, num_class):
    
    multi_res = softmax_layer(torch.mm(X, res))
    
    prob, predict_labels = torch.max(multi_res, 1)
    
    print(prob)
    
#     predict_labels = torch.argmax(multi_res, 1)
    
    
    sorted_prob, indices = torch.sort(prob.view(-1), descending = True)
    
    delta_id_array = []
    
#     delta_data_ids = torch.zeros(delta_num, dtype = torch.long)
    
#     for i in range(delta_num):

    expected_selected_label =0
    
     
#     if torch.sum(Y == 1) > torch.sum(Y == -1):
#         expected_selected_label = 1
#         
#     else:
#         expected_selected_label = -1


    i = 0

    while len(delta_id_array) < delta_num and i < X.shape[0]:
        if Y[indices[i]] == predict_labels[indices[i]]:
            Y[indices[i]] = (Y[indices[i]] + 1)%num_class
            delta_id_array.append(indices[i])
    
    
        i = i + 1
    
    delta_data_ids = torch.tensor(delta_id_array, dtype = torch.long)
    
#     print(delta_data_ids[:100])
#     delta_data_ids = random_generate_subset_ids(X.shape, delta_num)     
#     torch.save(delta_data_ids, git_ignore_folder+'delta_data_ids')
    return X, Y, delta_data_ids    

def change_data_values(X, Y, num, res, unique_Ys):
    
#     X_mean_list = []
#     
#     for i in unique_Ys:
#         X_mean_list.append(torch.mean(X[Y.view(-1)==i], 0))
    
    multi_res = softmax_layer(torch.mm(X, res))
    
    prob, predict_labels = torch.max(multi_res, 1)
    
    print(prob)
    
#     predict_labels = torch.argmax(multi_res, 1)
    
    
    sorted_prob, indices = torch.sort(prob.view(-1), descending = True)
    
#     positive_X_mean = torch.mean(X[Y.view(-1)==1], 0)
#     
#     negative_X_mean = torch.mean(X[Y.view(-1)==-1], 0)
    
    mean_list = [] 
    
    for i in range(num_class):
        
        curr_mean = torch.mean(X[Y.view(-1) == i], 0)
        
        mean_list.append(curr_mean)
    
    delta_data_ids = set()

    for i in range(num):
        curr_class = Y[indices[i], 0]
        
        curr_coeff = mean_list[curr_class]/mean_list[(curr_class + 1)%(num_class)]        
        
        curr_coeff = curr_coeff[curr_coeff != np.inf]
        
        curr_coeff = torch.sum(curr_coeff[curr_coeff == curr_coeff])
         
#         print(curr_coeff) 
        
#         curr_coeff = torch.sum(curr_coeff[curr_coeff != np.inf])
        
        X[indices[i]] = (X[indices[i]])*curr_coeff
        
        Y[indices[i]] = (curr_class + 1)%num_class
        
        delta_data_ids.add(indices[i])
        
#         X = torch.cat([X, selected_point.view(1,-1)], 0)
#         
#         Y = torch.cat([Y, selected_label.view(1,-1)], 0)
    
     
#     p1 = None
#      
#     p2 = None
#     
#     
#     coeff = 1#positive_X_mean/negative_X_mean
#     
#     
#     coeff = torch.mean(coeff[coeff != np.inf])
#     
#      
#     for i in range(num):
#         if Y[indices[i],0].numpy()[0] == 1:
#             X[indices[i]] = coeff*X[indices[i]]
#          
#         if Y[indices[i],0].numpy()[0] == -1:
#             X[indices[i]] = X[indices[i]]/coeff
#         
#         delta_data_ids.add(indices[i])
    
#         if p1 is not None and p2 is not None:
#             break
    
#     middle_point = (positive_X_mean + negative_X_mean)/2
#     
#     X[Y.view(-1) == 1] = X[Y.view(-1)==1]*torch.mean(positive_X_mean/negative_X_mean)
    
    return X, Y, torch.tensor(list(delta_data_ids), dtype = torch.long)

def extended_by_multi_copies(X, Y, num):
    
    
    X_list = []
    
    Y_list = []
    
    for i in range(num):
        X_list.append(X)
        
        Y_list.append(Y)
        
        
    X_tensor = torch.cat(X_list, 0)
    
    Y_tensor = torch.cat(Y_list, 0)
    
    return X_tensor, Y_tensor
    
    
def random_pick_subsets(size_per_set, num, dim):
    
    
    delta_id_list = []
    
    selected_id_list = []
    
    
    for i in range(num):
        ids = random.sample(range(dim[0]), size_per_set)
        delta_id_list.append(torch.tensor(ids, dtype = torch.long))
        
        selected_rows = torch.tensor(list(set(range(dim[0])) - set(ids)))
        
        selected_id_list.append(selected_rows)
    
    torch.save(delta_id_list, git_ignore_folder + 'delta_id_list')
    
    torch.save(selected_id_list, git_ignore_folder + 'selected_id_list')
    
    









if __name__ == '__main__':
    
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    sys_args = sys.argv
    
    file_name = sys_args[1]
    
    alpha = float(sys_args[2])
    
    beta = float(sys_args[3])
    
    threshold = float(sys_args[4])
    
    batch_size =int(sys_args[5])
    
    max_epoch = int(sys_args[6])

    noise_rate = float(sys_args[7])
    
    delta_id_list_num = int(sys_args[8])
    
    num_cp = int(sys_args[9])
    
    theta_record_threshold = float(sys_args[10])
    
    repetition = 1
    
    
    [X, Y, test_X, test_Y] = load_data_multi_classes(True, file_name)

#         [X, Y, test_X, test_Y] = clean_sensor_data(file_name)
    
    
    Y = Y.type(torch.LongTensor)
    
    X = extended_by_constant_terms(X, False)
    
    test_X = extended_by_constant_terms(test_X, False)
    

    X, Y  = extended_by_multi_copies(X, Y, num_cp)
    
    test_X, test_Y  = extended_by_multi_copies(test_X, test_Y, num_cp)
    
    
    dim = X.shape
    
    print('X_shape::', dim)
    
    print('unique_Y::', torch.unique(Y))
    
    num_class = torch.unique(Y).shape[0]
    
    print('num_class::', num_class)


    random_pick_subsets(int(X.shape[0]*noise_rate), delta_id_list_num, X.shape)
        
#     torch.save(delta_id_list, git_ignore_folder + 'delta_id_list')
    t1  =time.time()
    lr = initialize(X, num_class)
    res1, epoch = compute_parameters(X, Y, lr, dim, num_class, True)
    
    t2  =time.time()


    x_sum_by_class_by_batch = compute_x_sum_by_class_by_batch(X, Y, batch_size, num_class, epoch)
        
    torch.save(X, git_ignore_folder + 'X')
    
    torch.save(Y, git_ignore_folder + 'Y')
    
    
    torch.save(alpha, git_ignore_folder + 'alpha')
    
    torch.save(beta, git_ignore_folder + 'beta')
    
    torch.save(batch_size, git_ignore_folder + 'batch_size')
    
    torch.save(x_sum_by_class_by_batch, git_ignore_folder+'x_sum_by_class')
    
    torch.save(torch.tensor(epoch), git_ignore_folder+'epoch')
    
#     torch.save(res2, git_ignore_folder+'model_origin')
    
    torch.save(torch.tensor(epoch_record_epoch_seq), git_ignore_folder + 'epoch_record_epoch_seq')
    
    print('epoch::', epoch)

    print('recorded_seq_size::', epoch_record_epoch_seq)
    
#     print('gap::', (res1 - res2))
    
    t3 = time.time()
    
    mini_epochs_per_super_iteration = int((dim[0] - 1)/batch_size) + 1
    
    random_ids_multi_super_iterations_tensors = torch.stack(random_ids_multi_super_iterations)
    
    torch.save(random_ids_multi_super_iterations_tensors, git_ignore_folder + 'random_ids_multi_super_iterations')
    
    capture_provenance(X, Y, dim, epoch, num_class, batch_size, mini_epochs_per_super_iteration, random_ids_multi_super_iterations_tensors)
    
    
#     precomptation_influence_function2(X, Y, res2, dim)
    
#     torch.save(X.mul(Y.type(torch.DoubleTensor)), git_ignore_folder + 'X_Y_mult')
    
    t4 = time.time()
    
    
    print('training_time::', t2 - t1)
    
    print('preparing_time::', t4 - t3)
    
    print('train_accuracy::', compute_accuracy2(X, Y.type(torch.DoubleTensor), res1))
    
    print('test_accuracy::', compute_accuracy2(test_X, test_Y, res1))


        
     
#     torch.save(res1, git_ignore_folder + 'model_without_noise')
#     
#     torch.save(X, git_ignore_folder+'X')
#     
#     torch.save(Y, git_ignore_folder+'Y')
    
    torch.save(test_X, git_ignore_folder+'test_X')
    
    torch.save(test_Y, git_ignore_folder+'test_Y')        
#     res1 = torch.load(git_ignore_folder + 'model_without_noise')
    
#     print('train_accuracy::', compute_accuracy2(X, Y.type(torch.DoubleTensor), res1))
#     
#     print('test_accuracy::', compute_accuracy2(test_X, test_Y, res1))
    
    
        
        
        
    #     X, Y, selected_ids = change_data_labels(X, Y, 0.6, num_class)
    #     torch.save(selected_ids, git_ignore_folder + 'noise_data_ids')
    
#     else:
#         
# #         interpretability = bool(int(sys_args[12]))
#         
#         
#         res1 = torch.load(git_ignore_folder + 'model_without_noise')
#         
#         X = torch.load(git_ignore_folder + 'X')
#         
#         Y = torch.load(git_ignore_folder + 'Y')
#         
#         test_X = torch.load(git_ignore_folder + 'test_X')
#         
#         test_Y = torch.load(git_ignore_folder + 'test_Y')
#         
#         dim = X.shape
#         
#         num_class = torch.unique(Y).shape[0]
        
#         if interpretability:
        
        
#         else:
#             if random_deletion_or_not:
#                 X, Y, noise_data_ids = random_deletion(X, Y, int(X.shape[0]*noise_rate), res1, num_class)
#             else:
#                 if add_noise_or_not:
#                     X, Y = add_noise_data(X, Y, int(X.shape[0]*noise_rate), res1, num_class)
#                     noise_data_ids = torch.tensor(list(set(range(X.shape[0])) - set(range(dim[0]))), dtype = torch.long)
#         #         X, Y, noise_data_ids = change_data_values(X, Y, int(X.shape[0]*0.0005), res1, torch.unique(Y))
#         #         X, Y, noise_data_ids = rescale_data(X, Y, 0.3, res1)
#         #         X, Y, noise_data_ids = change_data_labels(X, Y, 0.3, num_class, res1)
#                 
#                 else:
#                     X, Y, noise_data_ids = change_data_values(X, Y, int(X.shape[0]*noise_rate), res1, torch.unique(Y))        
#         print('noise_data_num::', noise_data_ids.shape)
#         
#         random_ids = torch.randperm(dim[0])
#          
#         X = X[random_ids]
#            
#            
#         Y = Y[random_ids]
# 
#         shuffled_noise_data_ids = torch.argsort(random_ids)[noise_data_ids]#random_ids[noise_data_ids]

          
#         shuffled_noise_data_ids = torch.zeros(noise_data_ids.shape)
#           
#         for i in range(noise_data_ids.shape[0]):
#               
#             shuffled_id = torch.nonzero(random_ids == noise_data_ids[i])
#               
# #             print(shuffled_id)
#               
#             shuffled_noise_data_ids[i] = shuffled_id 
#           
#           
#           
# #         print(shuffled_noise_data_ids[:100])
#           
#           
#           
#         torch.save(shuffled_noise_data_ids, git_ignore_folder + 'noise_data_ids')
        
        
#         torch.save(torch.zeros(0, dtype = torch.long), git_ignore_folder + 'noise_data_ids')
        
        
        
#         print(noise_data_ids)
#         
#         print('noise_data_id_size::', noise_data_ids.shape)
#         
#         torch.save(noise_data_ids, git_ignore_folder + 'noise_data_ids')
        
        
    #     curr_theta_sum = torch.zeros([X.shape[1], 1], dtype = torch.double) 
    #     
    #     num = 10
    #     
    #     for i in range(num):
    #     
    #         subset_X_1, subset_Y_1 = get_subset_training_data(X, Y, [2*i, 2*i+1])
    #         lr = initialize(X)
    #         curr_theta_sum += compute_parameters(subset_X_1, subset_Y_1, lr)
    # 
    # #     subset_X_2, subset_Y_2 = get_subset_training_data(X, Y, [6,7,8,9,10])
    #     
    #     subset_X_2, subset_Y_2 = get_subset_training_data(X, Y, range(num))
    #     lr = initialize(X)
    #     theta_1 = compute_parameters(subset_X_2, subset_Y_2, lr)
    #     average_theta = curr_theta_sum/num
    #     
    #     
    #     print(theta_1)
    #     
    #     print(average_theta)
    #     
    #     print(theta_1 - average_theta)
    #     
    #     print(torch.norm(theta_1 - average_theta))
        
        
        
        
    #     initialized_theta = lr.theta
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#         torch.save(shuffled_noise_data_ids, git_ignore_folder + 'noise_data_ids')
        
#         t1  = time.time()
#         
#         epoch = 0
#         
#     #     for i in range(repetition):
#         lr = initialize(X, num_class)
#         res2, epoch = compute_parameters(X, Y, lr, dim, num_class, True)
#         
#         t2 = time.time()
    
    
#         x_sum_by_class = compute_x_sum_by_class(X, Y, num_class, dim)
        
        
    
    
#     X_products = compute_sample_products(X, dim)
#     
#     X_Y_products = compute_sample_label_products(X, Y)
    
#     print('X::', X)
#     
#     print('X_products::', X_products)
    
#     print(theta)
#     X_Y_mult = X.mul(Y)
    
    
    
#     res2 = None
#     
#     t01 = time.time()
#     
#     for i in range(repetition):
#         initialized_theta = Variable(initialize(X, num_class).theta)
#         
#         res2 = compute_model_parameter_by_iteration(dim, initialized_theta, X, Y, x_sum_by_class, num_class)
#         
# #         res2 = compute_model_parameter_by_iteration(dim, initialized_theta, X_Y_mult)
#     
#     
#     t02 = time.time()
    
#     X_Y_mult = X.mul(Y).numpy()
#     
# #     print('initialized_theta::', initialized_theta)
#     
#     t5 = time.time()
#      
# #     w_seq, b_seq = compute_linear_approx_parameters(X, Y, dim, theta)
#      
# #     average_parameter(w_seq)
#      
#     t6 = time.time()
#      
#     print('dimension::', dim)
#     
#     
#     
#     Pi = create_piecewise_linea_class()
#      
#     t3 = time.time()
#      
# #     res_approx = compute_model_parameter_by_approx(w_seq, b_seq, X, Y, dim, initialized_theta, X_products, X_Y_products)
#     
#     res_approx = None
#     
#     for i in range(repetition):
# #         gc.collect()
#         initialized_theta = initialize(X).theta.detach().numpy()
# #         initialized_theta = initialize(X).theta
#         res_approx = compute_model_parameter_by_approx2(dim, initialized_theta, Pi, X_Y_mult)
#      
#     t4 = time.time()
#      
#     gap = res_approx - theta.detach().numpy()
#      
#     print(res_approx)
#      
#     print(gap)
#      
#     gap2 = theta - res2
#     
#     print(gap2) 
    
     
#     time1 = (t2 - t1)/repetition
#      
# #     time2 = (t4 - t3)/repetition
#      
#     time3 = (t02  - t01)/repetition 
#      
#     
#     print(theta)
#     
#     print(res2)
#     
#     print('time1::', time1)
#      
# #     print('time2::', time2)
#      
#      
#     print('time3::', time3)
#      
#     for i in range(len(inter_result1)):
#         print(torch.max(torch.abs(inter_result1[i] - inter_result2[i])))
     
#     linear_time = t6 - t5
     
#     print('linear_time', linear_time)
#     
#     print('prod_time1::', prod_time1)
#     
#     print('prod_time2::', prod_time2)
    
#     res_approx2 = compute_naively(X, Y, dim, initialized_theta, w_seq, b_seq)    
#     
#     gap = res_approx - res_approx2
    
#     print(initialized_theta)
    
#     print(gap)

#     print(res_approx)
