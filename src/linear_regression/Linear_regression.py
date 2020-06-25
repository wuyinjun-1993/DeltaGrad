'''
Created on Feb 4, 2019


'''
import os
import sys
from torch.autograd import Variable

from torch import nn, optim
import torch
import csv
import time

from sklearn.utils.extmath import randomized_svd

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from sensitivity_analysis.linear_regression.evaluating_test_samples import *
    from data_IO.Load_data import *
    
except ImportError:
    from evaluating_test_samples import *
    from Load_data import *


# from main.watcher import Watcher
# import matplotlib.pyplot as plt
import numpy as np


alpha = 0.001
  
beta = 0.01

max_epoch = 200

svd_ratio = 10

batch_size = 10

threshold = 1e-4

min_feature_num = 20

res_prod_seq = torch.zeros(0, dtype = torch.double)

# sample_level = True
# 
# if sample_level:
#     from main.matrix_prov_sample_level import M_prov
# else:
#     from main.matrix_prov_entry_level import M_prov
# from main.add_prov import add_prov_token_per_row



# torch.set_printoptions(precision=10)

# x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
#                     [9.779], [6.182], [7.59], [2.167], [7.042],
#                     [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)


# x_train[0][0].a[0] = 1



class linear_regressor:
    def __init__(self, theta):
        self.theta = theta

# def initialize(X):
#     shape = list(X.size())
#     theta = Variable(torch.zeros([shape[1],1], dtype = torch.float64))
# #     theta[0][0] = -1
#     
#     theta.requires_grad = True
# #     lr.theta = Variable(lr.theta)
# 
#     print(theta.requires_grad)
#     
#     lr = linear_regressor(theta)
#     
#     return lr






def linear_regression_linview(exp_X_prod_inverse, delta_X, X_prod, X_prod_inverse, updated_X_Y_mult):
    
    for i in range(delta_X.shape[0]):
        delta_X_prod_inverse = torch.mm(torch.mm(X_prod_inverse, delta_X[i].view(-1,1)), torch.mm(delta_X[i].view(1,-1), X_prod_inverse))
        delta_X_prod_inverse = delta_X_prod_inverse/(1 + torch.mm(torch.mm(delta_X[i].view(1,-1), X_prod_inverse), delta_X[i].view(-1,1)))
        X_prod_inverse = X_prod_inverse - delta_X_prod_inverse
        
    
    
#     expected_X_prod_inverse = torch.inverse(X_prod - torch.mm(torch.t(delta_X), delta_X))
#         
#     print(torch.norm(expected_X_prod_inverse - X_prod_inverse))   
#     
#     print(torch.norm(updated_X_Y_mult - exp_X_prod_inverse)) 
    
    res = torch.mm(X_prod_inverse, updated_X_Y_mult)
    return res

def linear_regression_closed_form(X_prod, X_Y_mult, dim):
    X_prod_inverse = torch.inverse(X_prod)
    
    res = torch.mm(X_prod_inverse, X_Y_mult)
    
    return res


def linear_regression_iteration(X_prod, X_Y_mult, dim, lr, max_epoch, alpha, beta):


#     theta_list = []

    for epoch in range(max_epoch):
#         print('epoch', epoch)
#         print('start', lr.theta)
#         print('step 0', (torch.mm(X, lr.theta)))
#         print('step 1', (torch.mm(X, lr.theta) - Y))
#         print('step 2', alpha*torch.mm(torch.transpose(X, 0, 1),(torch.mm(X, lr.theta) - Y)))
#         print('theta!!!!', lr.theta)
#         lr.theta = lr.theta - 2*alpha*(torch.mm(torch.transpose(X, 0, 1),(torch.mm(X, lr.theta) - Y)) + beta*lr.theta)
        
        lr.theta = lr.theta - 2*alpha*((torch.mm(X_prod, lr.theta) - X_Y_mult)/dim[0] + 0.5*beta*lr.theta)
        
        
#         theta_list.append(lr.theta.clone())
        
#         print('theta!!!!', lr.theta)
#         err = Y - torch.mm(X, lr.theta)
#         error = torch.mm(torch.transpose(err, 0, 1), err) + beta*torch.matmul(torch.transpose(theta, 0, 1), theta)
        
#         print('error', error)
      
    return lr.theta

def linear_regression_provenance(X_prod, X_Y_mult, dim, lr, max_epoch, alpha, beta):

    A = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.double) - 2*alpha*X_prod/dim[0]
    
    B = 2*alpha*X_Y_mult/dim[0]
    
#     max_epoch = 2
    
    s, M = torch.eig(A, True)
    
    s = s[:,0]
    
    s_power = torch.pow(s, float(max_epoch))
    
    res1 = M.mul(s_power.view(1,-1))

    res1 = torch.mm(res1, torch.t(M))
    
    
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
    
    res2 = torch.mm(res2, torch.t(M))
    
    
#     print('temp_sum_gap::', sum_temp - res2)
    
    lr.theta = torch.mm(res1, lr.theta) + torch.mm(res2, B)
    
    
    return lr.theta

def get_subset_data_per_epoch(curr_rand_ids, full_id_set):
    curr_rand_id_set = set(curr_rand_ids.view(-1).tolist())
            
    curr_matched_ids = np.sort(np.array(list(full_id_set.intersection(curr_rand_id_set))))
    
    return curr_matched_ids


def get_subset_data_per_epoch2(curr_rand_ids, full_id_set):
    curr_rand_id_set = set(curr_rand_ids.view(-1).tolist())
            
    intersected_ids = full_id_set.intersection(curr_rand_id_set)        
    
    curr_matched_ids = np.array(list(intersected_ids))
    
    curr_non_matched_ids = np.array(list(curr_rand_id_set - intersected_ids))
    
    return curr_matched_ids, curr_non_matched_ids

def compute_model_parameter_by_iteration2(theta_list, selected_data_ids, X, Y, dim, theta,  max_epoch, alpha, beta, batch_size, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    epoch = 0
    
    
#     id_mappings = {}
    
#     selected_rows_set = set(selected_data_ids.view(-1).tolist())
    
#    while epoch < mini_batch_epoch:

    end = False
    
    for k in range(len(random_ids_multi_super_iterations)):
        
        random_ids = random_ids_multi_super_iterations[k]
        
        id_start = 0
    
        id_end = 0
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
        all_indexes = np.sort(sort_idx[selected_data_ids])
        
        
        for i in range(0, dim[0], batch_size):
            

            end_id = i + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            
            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
            
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
            
            curr_matched_ids_size = curr_matched_ids.shape[0]
            
#             curr_rand_ids = random_ids[i:end_id]
#             
#             curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
        
        
        
            if curr_matched_ids_size <= 0:
                continue
        
#     while epoch < mini_batch_epoch:
        
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
        
#         print('epoch::',i)


#         for i in range(0,dim[0], batch_size):
#         for j in range(len(batch_X_list)):

            curr_X = X[curr_matched_ids]
            
            curr_Y = Y[curr_matched_ids]
            
#             curr_X_prod = batch_X_prod_list[j]
#             
#             
#             curr_X_Y_mult = batch_X_Y_mult_list[j] 
            
            theta = theta - 2*alpha*((torch.mm(torch.t(curr_X), torch.mm(curr_X, theta)) - torch.mm(torch.t(curr_X), curr_Y))/curr_X.shape[0] + 0.5*beta*theta)
            
#             print('theta_diff::', torch.norm(theta - theta_list[epoch]))
            
            epoch = epoch + 1
            id_start = id_end
            if epoch >= max_epoch:
                
                end = True
                
                break
            
            del curr_X, curr_Y
            
        if end:
            break
    
    return theta, total_time

def compute_model_parameter_by_iteration(lr, selected_data_ids, X, Y, dim, max_epoch, alpha, beta, batch_size, random_ids_multi_super_iterations):
    
    epoch = 0
    
    selected_rows_set = set(selected_data_ids.view(-1).tolist())
    
    end = False
    
    theta_list = []
    
    for k in range(len(random_ids_multi_super_iterations)):
        
        random_ids = random_ids_multi_super_iterations[k]
        
        
        for i in range(0, dim[0], batch_size):
            

            end_id = i + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            
            
            curr_rand_ids = random_ids[i:end_id]
            
            curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
        
        
        
            if curr_matched_ids.shape[0] <= 0:
                continue
        
            curr_X = X[curr_matched_ids]
            
            curr_Y = Y[curr_matched_ids]
            
#             print(curr_X.shape)
#             
#             print(curr_Y.shape)
#             
            loss = loss_function(curr_X, curr_Y, lr, beta, curr_X.shape)
           
            loss.backward()
            
            
            with torch.no_grad():
                lr.theta -= alpha * lr.theta.grad
                
                
#                 exp_gradient = 2*alpha*((torch.mm(torch.t(curr_X), torch.mm(curr_X, lr.theta)) - torch.mm(torch.t(curr_X), curr_Y))/curr_X.shape[0] + 0.5*beta*lr.theta)
                
            
                theta_list.append(lr.theta.clone())
            
                lr.theta.grad.zero_()
            
#             theta = theta - 2*alpha*((torch.mm(torch.t(curr_X), torch.mm(curr_X, theta)) - torch.mm(torch.t(curr_X), curr_Y))/curr_X.shape[0] + 0.5*beta*theta)
            
            epoch = epoch + 1
            
            if epoch >= max_epoch:
                
                end = True
                
                break
            
        if end:
            break
    
    return lr.theta, theta_list


def compute_model_parameter_by_iteration3(theta_list, delta_data_ids, X_prod_list, X_Y_prod_list, X, Y, dim, theta,  max_epoch, alpha, beta, batch_size, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, u_list, v_s_list):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    epoch = 0
        
#     delta_rows_set = set(delta_data_ids.view(-1).tolist())
    
#    while epoch < mini_batch_epoch:

    end = False
    
    for k in range(len(random_ids_multi_super_iterations)):
        
        random_ids = random_ids_multi_super_iterations[k]
        
        
        id_start = 0
    
        id_end = 0
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
        all_indexes = np.sort(sort_idx[delta_data_ids])
        
        for i in range(0, dim[0], batch_size):
            

            end_id = i + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            
            
#             curr_rand_ids = random_ids[i:end_id]
#             
#             curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, delta_rows_set)
        
            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
        
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
            
            curr_matched_ids_size = curr_matched_ids.shape[0]
        
            if (end_id - i - curr_matched_ids_size) <= 0:
                
                    epoch += 1
                
                    continue
#     while epoch < mini_batch_epoch:
        
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
        
#         print('epoch::',i)


#         for i in range(0,dim[0], batch_size):
#         for j in range(len(batch_X_list)):
            sub_term1 = 0
            
            sub_term2 = 0
            
            if curr_matched_ids_size > 0:
                curr_X = X[curr_matched_ids]
            
                curr_Y = Y[curr_matched_ids]
                
                
                sub_term1 = torch.mm(torch.t(curr_X), torch.mm(curr_X, theta))
                
                sub_term2 = torch.mm(torch.t(curr_X), curr_Y)
                
            
            
#             curr_X_prod = batch_X_prod_list[j]
#             
#             
#             curr_X_Y_mult = batch_X_Y_mult_list[j] 
            
            if X.shape[1] < min_feature_num:
                theta = theta - 2*alpha*(((torch.mm(X_prod_list[epoch], theta) - sub_term1) - (X_Y_prod_list[epoch] - sub_term2))/(end_id - i - curr_matched_ids.shape[0]) + 0.5*beta*theta)
                 
            else:
                
#             print(epoch)
                theta = theta - 2*alpha*(((torch.mm(u_list[epoch], torch.mm(v_s_list[epoch], theta)) - sub_term1) - (X_Y_prod_list[epoch] - sub_term2))/(end_id - i - curr_matched_ids.shape[0]) + 0.5*beta*theta)
            
#             print(epoch, torch.norm(theta-theta_list[epoch]))
#             
#             print(torch.norm(torch.mm(torch.t(X[curr_rand_ids]), X[curr_rand_ids]) - X_prod_list[epoch]))
#             
#             print(torch.norm(torch.mm(torch.t(X[curr_rand_ids]), X[curr_rand_ids]) - torch.mm(u_list[epoch], v_s_list[epoch])))
            
            epoch = epoch + 1
            
            id_start = id_end
            
            
            if epoch >= max_epoch:
                
                end = True
                
                break
            
        if end:
            break
    
    return theta, total_time



def linear_regression_provenance_opt2(origin_s, M, M_inverse, full_X_prod, X, delta_X, X_Y_mult, dim, lr, max_epoch, alpha, beta, M_inverse_times_theta):

#     print(delta_X.shape)
        
    if(delta_X.shape[0] < M_inverse.shape[0]):
        
        '''m * |delta|'''
        prod1 = torch.mm(M_inverse, torch.t(delta_X))
        '''|delta| * m'''
        prod2 = torch.mm(delta_X, M)
            
        diag_elems = torch.bmm(prod1.view(prod1.shape[0], 1, prod1.shape[1]), torch.t(prod2).view(prod1.shape[0], prod1.shape[1], 1)).view(origin_s.shape)
        
        
        diag_elems = diag_elems/torch.diag(torch.mm(torch.t(M), M))
        
        s = (1 - alpha*beta) - 2*alpha*(origin_s - diag_elems)/dim[0]
        
    else:
        
    
    
        prod1 = torch.mm(M_inverse, torch.mm(torch.t(delta_X), delta_X))
        
#         prod2 = torch.mm(prod1, M)
        
        diag_elems = torch.bmm(prod1.view(prod1.shape[0], 1, prod1.shape[1]), torch.t(M).view(prod1.shape[0], prod1.shape[1], 1)).view(origin_s.shape)
        
        diag_elems = diag_elems/torch.diag(torch.mm(torch.t(M), M))

        s = (1 - alpha*beta) - 2*alpha*(origin_s - diag_elems)/dim[0]
    
        
#     print(torch.norm(expected_s - s))
    
#     print(s.shape)
    
#     full_A = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.double) - 2*alpha*full_X_prod/X.shape[0]
# 
#     
# #     print(A.shape)
#     
#     s = torch.diag(torch.mm(M, torch.mm(A, M_inverse)))
#     
#     
#     s = s + expected_s
    
    s[s>1] = 1 - 1e-6
    
    B = 2*alpha*X_Y_mult/dim[0]
    
    
#     s, M = torch.eig(A, True)
#      
#     M_inverse = torch.inverse(M)
#     
#     s = s[:,0]
    
#     max_epoch = 2
    
#     s, M = torch.eig(A, True)
    
#     s = s[:,0]
    
    s_power = torch.pow(s, float(max_epoch))
    
    res1 = M.mul(s_power.view(1,-1))

    res1 = torch.mm(res1, M_inverse_times_theta)
    
    
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
    
    res2 = torch.mm(res2, M_inverse)
    
    
#     print('temp_sum_gap::', sum_temp - res2)
    
    lr.theta = res1 + torch.mm(res2, B)
    
#     print(torch.norm(res1 - exp_res1))
#     
#     print(torch.norm(torch.mm(res2, B) - exp_res2))
    
#     print('time1::', t2 - t1)
#     
#     
#     print('time2::', t3 - t2)
#     
#     print('time3::', t4 - t3)
    
    return lr.theta


def linear_regression_provenance_opt(expected_s, M, M_inverse, full_X_prod, X, X_prod, X_Y_mult, dim, lr, max_epoch, alpha, beta):

    A = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.double) - 2*alpha*X_prod/dim[0]
    
#     full_A = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.float) - 2*alpha*full_X_prod/X.shape[0]

    
#     print(A.shape)
    
    s = torch.diag(torch.mm(M_inverse, torch.mm(A, M)))
    
    
#     s = s + expected_s
    
    s[s>1] = 1 - 1e-6
    
    B = 2*alpha*X_Y_mult/dim[0]
    
    
#     s, M = torch.eig(A, True)
#      
#     M_inverse = torch.inverse(M)
#     
#     s = s[:,0]
    
#     max_epoch = 2
    
#     s, M = torch.eig(A, True)
    
#     s = s[:,0]
    
    s_power = torch.pow(s, float(max_epoch))
    
    res1 = M.mul(s_power.view(1,-1))

    res1 = torch.mm(res1, M_inverse)
    
    
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
    
    res2 = torch.mm(res2, M_inverse)
    
    
#     print('temp_sum_gap::', sum_temp - res2)
    
    lr.theta = torch.mm(res1, lr.theta) + torch.mm(res2, B)
    
    
    return lr.theta, s, torch.mm(res1, lr.theta), torch.mm(res2, B)
  



def linear_regression(origin_X, origin_Y, dim, lr, tracking_or_not):

    last_theta = None
    
    epoch = 0
    
    random_ids_multi_super_iterations = []
    
    end = False
    
    while epoch < max_epoch:
        
#         if tracking_or_not:
#             global res_prod_seq
#              
#             if res_prod_seq.shape == 0:
#                 res_prod_seq = lr.theta.clone()
#     #             res_prod_seq.append(lr.theta.clone())
#             else:
#                 res_prod_seq = torch.cat((res_prod_seq, lr.theta.clone()), 1)
        
        
        random_ids = torch.randperm(dim[0])
#         random_ids = torch.tensor(list(range(dim[0])))
        
#         print('rand_ids::', random_ids)
        
        X = origin_X[random_ids]
        
        Y = origin_Y[random_ids]
        
        random_ids_multi_super_iterations.append(random_ids)
        
        
        for i in range(0,X.shape[0], batch_size):
            
            
#             optimizer.zero_grad()
    
            end_id = i + batch_size
            
            if end_id >= X.shape[0]:
                end_id = X.shape[0]
    
    
#             indices = permutation[i:i+batch_size]
            batch_x, batch_y = X[i:end_id], Y[i:end_id]
        
        
            loss = loss_function(batch_x, batch_y, lr, beta, dim)
           
            loss.backward()
            
            
            with torch.no_grad():
                lr.theta -= alpha * lr.theta.grad
                
    #             if last_theta is not None:
                gap = torch.norm(lr.theta.grad)
                
                print(epoch, gap)
                
                if gap < threshold:
                    end = True
                    
                    break
                
                
                lr.theta.grad.zero_()
        
        
            epoch = epoch + 1
            
            if epoch >= max_epoch:
                end = True
                break
        
        del X
        
        del Y
        
        if end == True:
            break
    
            
    return lr.theta, epoch, random_ids_multi_super_iterations


def loss_function(X, Y, lr, beta, dim):
    
    residual = Y - torch.mm(X, lr.theta)
    
    return torch.sum(residual*residual)/dim[0] + beta/2*torch.sum(lr.theta*lr.theta)


def linear_regression_standard_library(X, Y, lr, dim, epoch, alpha, beta):
    
    
    for i in range(epoch):
        
    
        loss = loss_function(X, Y, lr, beta, dim)
       
        loss.backward()
        
        
        with torch.no_grad():
            lr.theta -= alpha * lr.theta.grad
            
            lr.theta.grad.zero_()
    
    
    
    
    return lr.theta





def initialize(dim, num_of_output):
    theta = Variable(torch.zeros([dim[1],num_of_output], dtype = torch.double))
    # theta[0][0] = 0
    lr = linear_regressor(theta)
    
    lr.theta.requires_grad = True
    
    return lr

def compute_parameters(X, Y, dim, lr, tracking_or_not):
    
#     start_time = time.time()
    
    return linear_regression(X, Y, dim, lr, tracking_or_not)
    
    
#     end_time = time.time()
#     
#     print(lr.theta)
#     
#     print('time::', (end_time -start_time))
#     
#     
#     result = torch.mm(torch.inverse(torch.mm(torch.transpose(X, 0 , 1), X)), torch.mm(torch.transpose(X,0,1), Y))
#     
#     print('exact_result::')
#     print(result)


def compute_hessian_matrix(X_prod, theta, dim):
    return 2*X_prod/dim[0] + beta*torch.eye(dim[1], dtype = torch.double)

def compute_first_derivative(X, X_Y_mult, theta, dim):
    return 2*(torch.mm(torch.t(X), torch.mm(X, theta)) - X_Y_mult)/dim[0]



def compute_single_svd(i, term1, batch_size):
    
    if batch_size < term1.shape[1]:
        upper_bound = int(batch_size/svd_ratio)
    else:
        upper_bound = int(term1.shape[1]/svd_ratio)
    
    if upper_bound <= 1:
        upper_bound = 2
    
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

def record_x_prod_list(X, Y, random_ids_multi_super_iterations, max_epoch):
    
    epoch = 0
    
    end = False
    
    
    X_prod_list = []
    
    X_Y_prod_list = []
    
    
    directory = git_ignore_folder + 'svd_folder'
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for k in range(len(random_ids_multi_super_iterations)):
    
        random_ids = random_ids_multi_super_iterations[k]
    
#         X = origin_X[random_ids]
#         
#         Y = origin_Y[random_ids]
        
#         random_ids_multi_super_iterations.append(random_ids)
        
        
        for i in range(0,X.shape[0], batch_size):
            
            
#             optimizer.zero_grad()
    
            end_id = i + batch_size
            
            if end_id >= X.shape[0]:
                end_id = X.shape[0]
    
    
#             indices = permutation[i:i+batch_size]
            if i == 0 and k ==0:
                print(random_ids[i:end_id])

            batch_x, batch_y = X[random_ids[i:end_id]], Y[random_ids[i:end_id]]
            if batch_x.shape[1] < min_feature_num:
                X_prod_list.append(torch.mm(torch.t(batch_x), batch_x))
            else:
                sub_u, sub_v = compute_single_svd(epoch, torch.mm(torch.t(batch_x), batch_x), batch_size)
            
            
            
            
            
                np.save(directory + '/u_' + str(epoch), sub_u)
        
                np.save(directory + '/v_' + str(epoch), sub_v)
        
                del sub_u, sub_v
            
            X_Y_prod_list.append(torch.mm(torch.t(batch_x), batch_y))
    
            epoch += 1
            
            if epoch >= max_epoch:
                
                end = True
                
                break
        if end == True:
            break
    if batch_x.shape[1] >= min_feature_num:   
        torch.save(torch.tensor([max_epoch]), directory + '/len')
    else:
        torch.save(X_prod_list, git_ignore_folder + 'X_prod_list')
    return X_Y_prod_list
                
                
    
def compute_svd(term1, batch_size):


    u_list = []

#     s_list = []
    
    v_s_list = []
    
    if batch_size < term1.shape[1]:
        upper_bound = int(batch_size/svd_ratio)
    else:
        upper_bound = int(term1.shape[1]/svd_ratio)

    for i in range(len(term1)):
        curr_term1 = term1[i].numpy()
        
        u,s,vt = randomized_svd(curr_term1, n_components=upper_bound, random_state=None)
        
        
        
#         non_zero_ids = (s >= 1)
        
        sub_s = s[0:upper_bound]
        
        if sub_s.shape[0] <= 0:
#             non_zero_ids = np.array([0,1])
            upper_bound = 1
            
            sub_s = s[0:upper_bound]
            
        
        sub_u = u[:,0:upper_bound]
         
        
         
        sub_v = vt[0:upper_bound]
        
        res = np.dot(sub_u*sub_s, sub_v)
        
#         print(np.linalg.norm(res - curr_term1))
#         
#         print(sub_s.shape)
        
        u_list.append(torch.from_numpy(sub_u*sub_s))
        
        v_s_list.append(torch.from_numpy(sub_v))
        
#         v_list.append(torch.from_numpy(sub_v))
        
    torch.save(u_list, git_ignore_folder + 'u_list')
    
#     torch.save(s_list, git_ignore_folder + 's_list')
    
    torch.save(v_s_list, git_ignore_folder + 'v_s_list')


def save_random_id_orders(random_ids_multi_super_iterations):
    sorted_ids_multi_super_iterations = []
    
    
    for i in range(len(random_ids_multi_super_iterations)):
        sorted_ids_multi_super_iterations.append(random_ids_multi_super_iterations[i].numpy().argsort())
        
        
    torch.save(sorted_ids_multi_super_iterations, git_ignore_folder + 'sorted_ids_multi_super_iterations')


def capture_provenance(X, Y, alpha, beta, random_ids_multi_super_iterations, max_epoch):
    save_random_id_orders(random_ids_multi_super_iterations)
    
    X_prod = torch.mm(torch.t(X), X)
    
    X_Y_mult = torch.mm(torch.t(X), Y)
    
#     A = (1-alpha*beta)*torch.eye(X.shape[1], dtype = torch.float) - 2*alpha*torch.mm(torch.t(X), X)/X.shape[0]
    
    A = torch.mm(torch.t(X), X)
    
    s, M = torch.eig(A, True)
    
    s = s[:,0]
    
    M_inverse = torch.inverse(M)
    
    torch.save(M, git_ignore_folder + 'eigen_vectors')
    
    torch.save(M_inverse, git_ignore_folder + 'eigen_vectors_inverse')
    
    torch.save(s, git_ignore_folder + 'eigen_values')
    
    X_Y_prod_list = record_x_prod_list(X, Y, random_ids_multi_super_iterations, max_epoch)
    
#     if X.shape[1] < batch_size:
        
#     torch.save(X_prod_list, git_ignore_folder + 'X_prod_list')
    
#     if X.shape[1] > batch_size:
        
#     compute_svd(X_prod_list, batch_size)
    torch.save(X_Y_prod_list, git_ignore_folder + 'X_Y_prod_list')

    
    torch.save(X_prod, git_ignore_folder + 'X_prod')
    
    torch.save(X_Y_mult, git_ignore_folder + 'X_Y_mult')

def precomptation_influence_function(X, Y, res, dim):
    
    t5 = time.time()
    
    X_prod = torch.mm(torch.t(X), X)
    
#     X_Y_mult = X.mul(Y)
    
#     Hessin_matrix = compute_hessian_matrix(X, X_Y_mult, res, dim, X_product)


    Hessin_matrix = compute_hessian_matrix(X_prod, res, dim)#(res, X, dim, num_class, X_product)
    
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



def add_noise_data(X, Y, dim, num_of_output, res, num):
    
    
    gap = torch.mm(X, res) - Y
    
    
    distance_list = torch.sum(gap*gap, 1)
    
    sorted, indices = torch.sort(distance_list.view(-1), dim = 0, descending = True)
    
    noise_X = torch.zeros((num, dim[1]), dtype = torch.double)
    
    noise_Y = torch.zeros((num, num_of_output), dtype = torch.double)
    
    
    
    
    for i in range(num):
        noise_X[i] = X[indices[i]].clone()
       
        noise_Y[i] = Y[indices[i]].clone() + 2*torch.abs(gap[indices[i]])
       
#         if gap[indices[i]] > 0:
#             noise_Y[i] = Y[indices[i]] + 100*gap[indices[i]]
#         else:
#             noise_Y[i] = Y[indices[i]] - 100*gap[indices[i]]
        
        
    X = torch.cat([X, noise_X], 0)
        
    Y = torch.cat([Y, noise_Y], 0)    
        
#     Y.requires_grad = False
        
    
    
    
    
    
    return X, Y





def change_data_values(X, Y, res, num):
    
    gap = torch.mm(X, res) - Y
    
    
    distance_list = torch.sum(gap*gap, 1)
    
    sorted, indices = torch.sort(distance_list.view(-1), dim = 0, descending = True)
    
    delta_data_ids = set()
    
    for i in range(num):
        Y[indices[i]] = Y[indices[i]].clone() + 2*torch.abs(gap[indices[i]])
        delta_data_ids.add(indices[i])
        
    
    
    return X, Y, torch.tensor(list(delta_data_ids))


def add_features(X, Y, feature_num, dim):
    
    
    torch.rand((dim[0], feature_num), dtype = torch.double)
        
        


if __name__ == '__main__':
    
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    sys_args = sys.argv
    
    file_name= sys_args[1]
    
    start = bool(int(sys_args[2]))
    
    input_alpha = float(sys_args[3])
    
    input_beta = float(sys_args[4])
    
    input_threshold = float(sys_args[5])
    
    max_epoch = int(sys_args[6])
    
    batch_size = int(sys_args[7])
    
#     global alpha, beta, threshold
    noise_rate = float(sys_args[8])
    
#     add_feature = bool(int(sys_args[8]))
    
    add_noise_or_not = bool(int(sys_args[9]))
    
    
    extend_dimesions = bool(int(sys_args[10]))

    
    alpha = input_alpha
    
    beta = input_beta
    
    threshold = input_threshold
    
    if start:
    
        [X, Y, test_X, test_Y] = load_data(False, file_name)
        
        
        X = extended_by_constant_terms(X, extend_dimesions)
        
        test_X = extended_by_constant_terms(test_X, extend_dimesions)
        
        print("X_dim::", X.shape)
    
        dim = X.shape
    
        num_of_output = Y.shape[1]
    
        lr = initialize(dim, num_of_output)
    
        res1, epoch, _ = compute_parameters(X, Y, dim, lr, False)
        
        
        torch.save(X, git_ignore_folder + 'X')
        
        torch.save(Y, git_ignore_folder + 'Y')
        
        torch.save(test_X, git_ignore_folder + 'test_X')
        
        torch.save(test_Y, git_ignore_folder + 'test_Y')
        
        torch.save(res1, git_ignore_folder + 'model_without_noise')
        
        torch.save(epoch, git_ignore_folder + 'epoch')
        
        training_accuracy = compute_accuracy2(X, Y, res1)
        
        test_accuracy = compute_accuracy2(test_X, test_Y, res1)
        
        torch.save(batch_size, git_ignore_folder + 'batch_size')
        
        print('training_accuracy::', training_accuracy)
        
        print('test_accuracy::', test_accuracy)
        
        print('epoch::', epoch)
        
        print(res1)
        
    else:
        
        res1 = torch.load(git_ignore_folder + 'model_without_noise').detach()
        
        X = torch.load(git_ignore_folder + 'X')
        
        Y = torch.load(git_ignore_folder + 'Y')
        
        test_X = torch.load(git_ignore_folder + 'test_X')
        
        test_Y = torch.load(git_ignore_folder + 'test_Y')
        
        num_of_output = Y.shape[1]
        
        print('X_shape::', X.shape)
        
        print('model_without_noise::', res1)
        
        dim = X.shape
        
        if add_noise_or_not:
            X, Y = add_noise_data(X, Y, dim, num_of_output, res1, int(X.shape[0]*noise_rate))
            noise_data_ids = torch.tensor(list(set(range(X.shape[0])) - set(range(dim[0]))))
        else:
            X, Y, noise_data_ids = change_data_values(X, Y, res1, int(X.shape[0]*noise_rate))
        
        
        

        dim = X.shape
        
        random_ids = torch.randperm(dim[0])
          
        X = X[random_ids]
          
          
        Y = Y[random_ids]
         
         
         
        shuffled_noise_data_ids = torch.argsort(torch.tensor(random_ids))[noise_data_ids]#random_ids[noise_data_ids]
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
#         print(shuffled_noise_data_ids[:100])
         
         
         
        torch.save(shuffled_noise_data_ids, git_ignore_folder + 'noise_data_ids')
        
        
        
        
        
        
        
        
        
        
        
        
#         torch.save(noise_data_ids, git_ignore_folder + 'noise_data_ids')

#     X, Y = add_noise_data2(X, Y, added_x, added_y, 1000)
#         X, Y, noise_data_ids = change_instance_labels(X, Y, int(X.shape[0]*0.01), dim, res1)
#         X, Y = add_noise_data(X, Y, int(X.shape[0]*0.3), res1)
    
        
         
#         
         
         
        print(dim)
        
    #     X, Y, noise_data_ids = change_data_labels2(X, Y, 0.8, res) 
    #                    
    #     torch.save(noise_data_ids, git_ignore_folder + 'noise_data_ids')
    
        t1 = time.time()  
        lr = initialize(X.shape, num_of_output)
        res2, epoch, random_ids_multi_super_iterations = compute_parameters(X, Y, dim, lr, True)
        
        t2 = time.time()
        
        print('epoch::', epoch)
        
        print(res2 - res1)
        
    #     positive_ids = (Y.view(-1) == 1).nonzero()
          
          
    #     _, ids = torch.sort(torch.abs(torch.mm(X, res)).view(-1), descending = True)
    #       
    #       
    #     noise_data_ids = torch.tensor(ids[0:int(ids.shape[0]*0.5)].numpy())
    #       
    #     torch.save(noise_data_ids.view(-1), git_ignore_folder + 'noise_data_ids')
    #       
    #     torch.save(Y, git_ignore_folder + 'Y')
    #       
    #     Y[noise_data_ids] = 1
    #       
    #     print(noise_data_ids)
    #       
    #     Y = Y.view(-1,1)
    #      
         
         
         
    #     X_Y_mult = X.mul(Y)
        
        
        
        torch.save(X, git_ignore_folder + 'noise_X')
        
        torch.save(Y, git_ignore_folder + 'noise_Y')
        
        torch.save(batch_size, git_ignore_folder + 'batch_size')
        
        
        torch.save(torch.tensor(epoch), git_ignore_folder + 'epoch')
        
        torch.save(res2, git_ignore_folder + 'model_origin')
    
        torch.save(alpha, git_ignore_folder + 'alpha')
        
        torch.save(beta, git_ignore_folder + 'beta')
    
        torch.save(random_ids_multi_super_iterations, git_ignore_folder + 'random_ids_multi_super_iterations')
    
        torch.save(torch.inverse(torch.mm(torch.t(X), X)), git_ignore_folder + 'exp_X_prod_inverse')
    
        
        capture_provenance(X, Y, alpha, beta, random_ids_multi_super_iterations, epoch)
        
        
        precomptation_influence_function(X, Y, res2, dim)
        
        print('training_time::', t2 - t1)
        
        training_accuracy = compute_accuracy2(X, Y, res1)
        
        test_accuracy = compute_accuracy2(test_X, test_Y, res1)
        
        print('training_accuracy::', training_accuracy)
        
        print('test_accuracy::', test_accuracy)
        
        
        training_accuracy = compute_accuracy2(X, Y, res2)
        
        test_accuracy = compute_accuracy2(test_X, test_Y, res2)
        
        print('training_accuracy::', training_accuracy)
        
        print('test_accuracy::', test_accuracy)
        
        
        
#     else:
#         
#         X = torch.load(git_ignore_folder + 'X')
#         
#         Y = torch.load(git_ignore_folder + 'Y')
#         
#         
#         test_X = torch.load(git_ignore_folder + 'test_X')
#         
#         test_Y = torch.load(git_ignore_folder + 'test_Y')
#         
#         
#         if add_noise_or_not:
#             X, Y = add_noise_data(X, Y)
#     
#     X_prod = torch.mm(torch.t(X), X)
#     
#     X_Y_mult = torch.mm(torch.t(X), Y)
# 
#     lr = initilize(dim, num_of_output)
# 
# 
#     capture_provenance(X, Y)
# 
#     
#     res2 = linear_regression_iteration(X_prod, X_Y_mult, dim, lr, epoch, alpha, beta)
#     
#     
#     
#     
#     print('res1::', res1)
#     
#     print('res2::', res2)
#     
#     print(torch.norm(res1 - res2))
    
    
    # x_train = np.array([[0, 1], [4.4, 0], [5.5, 3]], dtype=np.float32)
    # 
    # # y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
    # #                     [3.366], [2.596], [2.53], [1.221], [2.827],
    # #                     [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
    # 
    # y_train = np.array([[1.7], [2.06], [2.59]], dtype=np.float32)
    
    
    # x_train = torch.from_numpy(x_train)
    # 
    # y_train = torch.from_numpy(y_train)
    # 
    # X = Variable(x_train)
    # 
    # Y = Variable(y_train)
    
    
