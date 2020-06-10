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


threshold = 1e-4

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


def linear_regression_iteration(A, X_prod, X_Y_mult, dim, lr, max_epoch, alpha, beta, all_parameters_all_epochs, all_A_vector_prod_all_epochs):


#     theta_list = []

    for epoch in range(max_epoch):
        print('epoch', epoch)
#         print('start', lr.theta)
#         print('step 0', (torch.mm(X, lr.theta)))
#         print('step 1', (torch.mm(X, lr.theta) - Y))
#         print('step 2', alpha*torch.mm(torch.transpose(X, 0, 1),(torch.mm(X, lr.theta) - Y)))
#         print('theta!!!!', lr.theta)
#         lr.theta = lr.theta - 2*alpha*(torch.mm(torch.transpose(X, 0, 1),(torch.mm(X, lr.theta) - Y)) + beta*lr.theta)
        all_parameters_all_epochs.append(lr.theta.clone())
        all_A_vector_prod_all_epochs.append(torch.mm(A, lr.theta))
        
        
        
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





def linear_regression_provenance_opt2(origin_s, M, M_inverse, full_X_prod, X, delta_X, X_Y_mult, dim, lr, max_epoch, alpha, beta, M_inverse_times_theta):

#     print(delta_X.shape)
        
    if(delta_X.shape[0] < M_inverse.shape[0]):
        
        '''m * |delta|'''
        prod1 = torch.mm(M_inverse, torch.t(delta_X))
        '''|delta| * m'''
        prod2 = torch.mm(delta_X, M)
            
        diag_elems = torch.bmm(prod1.view(prod1.shape[0], 1, prod1.shape[1]), torch.t(prod2).view(prod1.shape[0], prod1.shape[1], 1)).view(origin_s.shape)
        
        s = (1 - alpha*beta) - 2*alpha*(origin_s - diag_elems)/dim[0]
        
    else:
        
    
    
        prod1 = torch.mm(M_inverse, torch.mm(torch.t(delta_X), delta_X))
        
#         prod2 = torch.mm(prod1, M)
        
        diag_elems = torch.bmm(prod1.view(prod1.shape[0], 1, prod1.shape[1]), torch.t(M).view(prod1.shape[0], prod1.shape[1], 1)).view(origin_s.shape)
        
            
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
  



def linear_regression(X, Y, dim, lr, tracking_or_not, all_parameters_all_epochs):

    last_theta = None
    
    epoch = 0
    
    while epoch < max_epoch:
        
#         if tracking_or_not:
#             global res_prod_seq
#              
#             if res_prod_seq.shape == 0:
#                 res_prod_seq = lr.theta.clone()
#     #             res_prod_seq.append(lr.theta.clone())
#             else:
#                 res_prod_seq = torch.cat((res_prod_seq, lr.theta.clone()), 1)
        
        
        
        
        
        
        
        
        loss = loss_function(X, Y, lr, beta, dim)
       
        loss.backward()
        
        
        with torch.no_grad():
            
            all_parameters_all_epochs.append(lr.theta)
            
            
            lr.theta -= alpha * lr.theta.grad
            
#             if last_theta is not None:
            gap = torch.norm(lr.theta.grad)
            
            print(gap)
            
            if gap < threshold:
                break
            
            
            lr.theta.grad.zero_()
    
    
        epoch = epoch + 1
    
        last_theta = lr.theta.clone()
            
    return lr.theta, epoch


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
    theta = Variable(torch.rand([dim[1],num_of_output], dtype = torch.double))
    # theta[0][0] = 0
    lr = linear_regressor(theta)
    
    lr.theta.requires_grad = True
    
    return lr

def compute_parameters(X, Y, dim, lr, tracking_or_not, all_parameters_all_epochs):
    
#     start_time = time.time()
    
    return linear_regression(X, Y, dim, lr, tracking_or_not, all_parameters_all_epochs)
    
    
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

def compute_first_derivative(X_prod, X_Y_mult, theta, dim):
    return 2*(torch.mm(X_prod, theta) - X_Y_mult)/dim[0]


def capture_provenance(X, Y, alpha, beta):
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
       
        noise_Y[i] = Y[indices[i]].clone() + 10*torch.abs(gap[indices[i]])
       
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
        Y[indices[i]] = Y[indices[i]].clone() + 10*torch.abs(gap[indices[i]])
        delta_data_ids.add(indices[i])
        
    
    
    return X, Y, torch.tensor(list(delta_data_ids))


def add_features(X, Y, feature_num, dim):
    
    
    torch.rand((dim[0], feature_num), dtype = torch.double)
        
        


if __name__ == '__main__':
    
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    sys_args = sys.argv
    
    file_name= sys_args[1]
    
    input_alpha = float(sys_args[2])
    
    input_beta = float(sys_args[3])
    
#     input_threshold = float(sys_args[4])
    
    max_epoch = int(sys_args[4])
    
#     global alpha, beta, threshold
#     noise_rate = float(sys_args[6])
#     
# #     add_feature = bool(int(sys_args[8]))
#     
#     add_noise_or_not = bool(int(sys_args[7]))
#     
#     
    extend_dimesions = bool(int(sys_args[5]))

    
    alpha = input_alpha
    
    beta = input_beta
    
#     threshold = input_threshold
    
#     if start:
        
    all_parameters_all_epochs = []
    

    [X, Y, test_X, test_Y] = load_data(False, file_name)
    
    
    X = extended_by_constant_terms(X, extend_dimesions)
    
    test_X = extended_by_constant_terms(test_X, extend_dimesions)
    
    print("X_dim::", X.shape)

    dim = X.shape

#     num_of_output = Y.shape[1]

    Y = Y[:,0:1]
    
    all_A_vector_prod_all_epochs = []
    
    num_of_output = 1
    lr = initialize(dim, num_of_output)
    
    A = torch.rand([X.shape[1], X.shape[1]], dtype = torch.double)#(1-input_alpha*input_beta)*torch.eye(X.shape[1], dtype = torch.double) - 2*input_alpha/X.shape[0]*torch.mm(torch.t(X), X)
    
    B = torch.mm(torch.t(X), Y)*input_alpha/X.shape[0]

    eig_vec, eig_value = torch.eig(A, True)

    res1 = linear_regression_iteration(A, torch.mm(torch.t(X), X), torch.mm(torch.t(X), Y), X.shape, lr, max_epoch, input_alpha, input_beta, all_parameters_all_epochs, all_A_vector_prod_all_epochs)

#     res1, epoch = compute_parameters(X, Y, dim, lr, False, all_parameters_all_epochs)
    
    
    torch.save(X, git_ignore_folder + 'X')
    
    torch.save(Y, git_ignore_folder + 'Y')
    
    torch.save(test_X, git_ignore_folder + 'test_X')
    
    torch.save(test_Y, git_ignore_folder + 'test_Y')
    
    torch.save(res1, git_ignore_folder + 'model_without_noise')
    
    torch.save(epoch, git_ignore_folder + 'epoch')
    
    training_accuracy = compute_accuracy2(X, Y, res1)
    
    test_accuracy = compute_accuracy2(test_X, test_Y, res1)
    
    print('training_accuracy::', training_accuracy)
    
    print('test_accuracy::', test_accuracy)
    
    print('epoch::', epoch)
        
    print(res1)