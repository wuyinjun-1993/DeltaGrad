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
from scipy.sparse import coo_matrix
import scipy.sparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))




try:
    from Interpolation.piecewise_linear_interpolation_2D import *
    from logistic_regression.incremental_updates_logistic_regression import *
    from data_IO.Load_data import *
    from logistic_regression.evaluating_test_samples import *
except ImportError:
    from piecewise_linear_interpolation_2D import *
    from incremental_updates_logistic_regression import *
    from Load_data import *    
    from evaluating_test_samples import *
    
# try:
#     from sensitivity_analysis.logistic_regression.incremental_updates_logistic_regression import *
# except ImportError:
#     from incremental_updates_logistic_regression import *
#     
#     
# try:
#     from sensitivity_analysis.Load_data import *
# except ImportError:
#     from Load_data import *    
# 
# 
# try:
#     from sensitivity_analysis.logistic_regression.evaluating_test_samples import *
# except ImportError:
#     from evaluating_test_samples import *


# from sensitivity_analysis.logistic_regression.incremental_updates_logistic_regression import X_product


max_epoch = 1000

'''cov_bin'''
# alpha = 0.0001
#   
# beta = 0.05


  
alpha = 0.0001
  
beta = 0.05


threshold = 1e-4

prov_record_rate = 0.1


res_prod_seq = torch.zeros(0, dtype = torch.double)


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

class logistic_regressor_parameter:
    def __init__(self, theta):
        self.theta = theta

def sigmoid_function(x):
    return 1/(1 + torch.exp(-x))

def sigmoid(x):
    return 1-1 / (1 +np.exp(-x))

def sigmoid_np(x):
    return 1 / (1 +np.exp(-x))

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
    
#     res = 0
    
    X_theta_prod = torch.mm(X, theta)
    
    
    res = torch.sum(-Y*log_sigmoid_layer(X_theta_prod))/dim[0] - torch.sum((1- Y)*(log_sigmoid_layer(-X_theta_prod)))/dim[0] 
    
    
#     for i in range(dim[0]):
# #         res += torch.log(1 + torch.exp(-Y[i]*torch.dot(X[i,:].view(dim[1]), theta.view(dim[1]))))
#         res = res - (Y[i]*torch.log(compute_sigmoid_function(X[i,:].view(dim[1]), theta.view(dim[1]))) + (1 - Y[i])*torch.log(1 - compute_sigmoid_function(X[i,:].view(dim[1]), theta.view(dim[1])))) 
#         
#     res = res/dim[0]
    
    return res + beta/2*torch.pow(torch.norm(theta, p =2), 2)

def bia_function(x):
    return -log_sigmoid_layer(x)

def second_derivative_loss_function(x):
    return torch.exp(x)/torch.pow((1 + torch.exp(x)), 2)

def compute_hessian_matrix_2(X, X_Y_mult, theta, dim, X_product):
    X_Y_theta_prod = torch.mm(X_Y_mult, theta)
    
    second_derivative = torch.pow(sig_layer(X_Y_theta_prod),2)*torch.exp(-X_Y_theta_prod)
    
    
    second_derivative = torch.mm(second_derivative.view(1,X.shape[0]), X_product.view(X.shape[0], X.shape[1]*X.shape[1])).view(X.shape[1], X.shape[1])


    second_derivative = second_derivative/dim[0] + beta*torch.eye(dim[1], dtype = torch.double)
    
    return second_derivative

def compute_hessian_matrix_4(X, X_Y_mult, theta, dim):
    X_Y_theta_prod = torch.mm(X_Y_mult, theta)
    
    second_derivative = torch.pow(sig_layer(X_Y_theta_prod),2)*torch.exp(-X_Y_theta_prod)
    
    
    second_deriv = X*second_derivative.view(X.shape[0], 1)
    
    second_derivative = torch.mm(torch.t(X), second_deriv)
    
#     second_derivative = torch.mm(second_derivative.view(1,X.shape[0]), X_product.view(X.shape[0], X.shape[1]*X.shape[1])).view(X.shape[1], X.shape[1])


    second_derivative = second_derivative/dim[0]
    
    for i in range(dim[1]):
        second_derivative[i,i] += beta
    
    return second_derivative


def compute_hessian_matrix_3(X, X_Y_mult, theta, dim): 
    theta.requires_grad = True
    
    first_derivative = compute_first_derivative(X_Y_mult, theta, dim)
    
    second_derivative = torch.zeros([dim[1], dim[1]], dtype = torch.double)
    
    for i in range(dim[1]):
        
        mask = torch.zeros(first_derivative.shape, dtype = torch.double)
#         
        mask[i] = 1
         
        first_derivative.backward(mask, retain_graph=True)
        
        second_derivative[i] = theta.grad.data.view(1,-1)
        
        
        theta.grad.zero_()
        
        
    return second_derivative
       
    

def compute_hessian_matrix(X, X_Y_mult, theta, dim, X_product):
    X_Y_theta_prod = torch.mm(X_Y_mult, theta)
    
    print(X.shape, X_Y_mult.shape, X_product.shape)
    
    res = torch.zeros([dim[1], dim[1]], dtype = torch.double)
    
    for i in range(X.shape[0]):
        res += second_derivative_loss_function(-X_Y_theta_prod[i])*X_product[i]
        
    res = res/dim[0] + beta*torch.eye(dim[1], dtype = torch.double)
    
#     theta.requires_grad = True
#     
#     z = compute_first_derivative_single_data2(X_Y_mult, theta, dim)
#     
#     for i in range(theta.shape[0]):
#         
#         mask = torch.zeros(z.shape, dtype = torch.double)
#         
#         mask[i] = 1
#         
#         z.backward(mask, retain_graph=True)
#         
#         curr_grad = theta.grad.data/dim[0]
#         
#         theta.grad.data.zero_()
    
    
    
    
    
    return res
    
    

def loss_function2(X, Y, theta, dim, beta):
    
#     res = 0
    
    
#     sigmoid_res = torch.stack(list(map(bia_function, Y*torch.mm(X, theta))))

#     sigmoid_res = Y*torch.mm(X, theta)
#     data_trans = sigmoid_res.apply(lambda x :  ())

#     sigmoid_res = -log_sigmoid_layer(Y*torch.mm(X, theta))
    
    res = torch.sum(-log_sigmoid_layer(Y*torch.mm(X, theta)))/dim[0]
    
    
#     for i in range(dim[0]):
# #         print(X[i,:])
# #         print(theta)
# #         print(X[i,:].view(dim[1]))
# #         print(theta.view(dim[1]))
#         res += torch.log(1 + torch.exp(-Y[i]*torch.dot(X[i,:].view(dim[1]), theta.view(dim[1]))))
#         res = res - (Y[i]*torch.log(sigmoid_function(X[i,:].view(dim[1]), theta.view(dim[1]))) + (1 - Y[i])*torch.log(1 - sigmoid_function(X[i,:].view(dim[1]), theta.view(dim[1])))) 
        
#     res = res
    
    return res + beta/2*torch.sum(theta*theta)
    
#     return res + beta/2*torch.pow(torch.norm(theta, p =2), 2)
def loss_function3(X_Y_mult, theta, dim, beta):
    
#     res = 0
    
    
#     sigmoid_res = torch.stack(list(map(bia_function, Y*torch.mm(X, theta))))

#     sigmoid_res = Y*torch.mm(X, theta)
#     data_trans = sigmoid_res.apply(lambda x :  ())

#     sigmoid_res = -log_sigmoid_layer(Y*torch.mm(X, theta))
    
    res = torch.sum(-log_sigmoid_layer(torch.mm(X_Y_mult, theta)))/dim[0]
    
    
#     for i in range(dim[0]):
# #         print(X[i,:])
# #         print(theta)
# #         print(X[i,:].view(dim[1]))
# #         print(theta.view(dim[1]))
#         res += torch.log(1 + torch.exp(-Y[i]*torch.dot(X[i,:].view(dim[1]), theta.view(dim[1]))))
#         res = res - (Y[i]*torch.log(sigmoid_function(X[i,:].view(dim[1]), theta.view(dim[1]))) + (1 - Y[i])*torch.log(1 - sigmoid_function(X[i,:].view(dim[1]), theta.view(dim[1])))) 
        
#     res = res
    
    return res + beta/2*torch.sum(theta*theta)*X_Y_mult.shape[0]/dim[0]



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
    

def logistic_regression(X, Y, lr, dim, tracking_or_not):

#     dim = X.shape
    
    epoch = 0
    
    last_theta = None
    
    last_recorded_theta = None
    
    while epoch < max_epoch:
        
        if tracking_or_not:
         
            global res_prod_seq
        
        
            if last_recorded_theta is None:
                last_recorded_theta = lr.theta.clone()
                res_prod_seq = lr.theta.clone()
    #             epoch_record_epoch_seq.append(epoch)
                tracking_or_not = True
    #                 print('here')
            else:
#                 print(torch.norm(last_recorded_theta - lr.theta))
                
                if torch.norm(last_recorded_theta - lr.theta) > prov_record_rate:
                    last_recorded_theta = lr.theta.clone()
                    res_prod_seq = torch.cat((res_prod_seq, lr.theta.clone()), 1)
    #                 epoch_record_epoch_seq.append(epoch)
                    tracking_or_not = True
    #                     print('here')
                else:
                    tracking_or_not = False
        
        
        
        
#         if tracking_or_not:
#             global res_prod_seq
#              
#             if res_prod_seq.shape == 0:
#                 res_prod_seq = lr.theta.clone()
#     #             res_prod_seq.append(lr.theta.clone())
#             else:
#                 res_prod_seq = torch.cat((res_prod_seq, lr.theta.clone()), 1)
        
        loss = loss_function2(X, Y, lr.theta, X.shape, beta)
   
        loss.backward()
       
        with torch.no_grad():
            lr.theta -= alpha * lr.theta.grad
            
#             gap = torch.norm(lr.theta.grad)
#             
#             if gap < threshold:
#                 break
#             if gap < prov_record_rate:
#                 tracking_or_not = False
#             
#             print(gap)
            
            lr.theta.grad.zero_()
            
        epoch = epoch + 1

            
        if last_theta is not None:
            print(torch.norm(last_theta - lr.theta))
         
        if last_theta is not None:
             
            gap = torch.norm(last_theta - lr.theta)
             
            if gap < threshold:
                break
#             if gap < prov_record_rate:
#                 tracking_or_not = False
        
        
        
            
        last_theta = lr.theta.clone()
            
            
#         print('epoch', epoch)
#         print('start', lr.theta)
#         print('step 0', (torch.mm(X, lr.theta)))
#         print('step 1', (torch.mm(X, lr.theta) - Y))
#         print('step 2', alpha*torch.mm(torch.transpose(X, 0, 1),(torch.mm(X, lr.theta) - Y)))
        
#         lr.theta = lr.theta - 2*alpha*(torch.mm(torch.transpose(X, 0, 1),(torch.mm(X, lr.theta) - Y)) + beta*lr.theta)



#         lr.theta = lr.theta + alpha*gradient(X, Y, dim, lr.theta)- alpha*beta*lr.theta 
        
#         print('gradient::', - alpha*gradient(X, Y, dim, lr.theta))
        
#         print('theta!!!!', lr.theta)
        
        
         
#         print('loss:', loss)
        
#         print('theta!!!!', lr.theta)
#         err = Y - torch.mm(X, lr.theta)
#         error = torch.mm(torch.transpose(err, 0, 1), err)# + beta*torch.matmul(torch.transpose(theta, 0, 1), theta)
        
#         print('error', error)
      
    return lr.theta, epoch

def logistic_regression_by_standard_library(X, Y, lr, dim, max_epoch, alpha, beta):

#     dim = X.shape
    
    for epoch in range(max_epoch):
        
        loss = loss_function2(X, Y, lr.theta, dim, beta)
   
        loss.backward()
       
        with torch.no_grad():
            lr.theta -= alpha * lr.theta.grad
            
            lr.theta.grad.zero_()
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
      
    return lr.theta


def compute_parameters(X, Y, lr, dim, tracking_or_not):
    
    
    lr.theta, epoch = logistic_regression(X, Y, lr, dim, tracking_or_not)
    
    print('res_real:::', lr.theta)
    
    return lr.theta, epoch
    
    
def compute_single_coeff(X, Y, w_seq, dim, epoch, X_products):
    
#     print(dim)
    
    res = (1 - beta*alpha)*torch.eye(dim[1], dtype = torch.double)
    
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


def compute_first_derivative_single_data(X_Y_mult, ids, theta):
    
    print('X_Y_shape::', X_Y_mult.shape)
    
    curr_X_Y_mult = torch.index_select(X_Y_mult, 0, ids)
    
    non_linear_term = curr_X_Y_mult*(1 - sig_layer(torch.mm(curr_X_Y_mult, theta)))
    
#     print(ids, non_linear_term, theta)
    
    non_linear_term = torch.sum(non_linear_term, dim=0).view(theta.shape)
    
    res = -non_linear_term# + beta*theta*curr_X_Y_mult.shape[0]
    
    return res


def compute_first_derivative_single_data1(curr_X_Y_mult, theta):
    
#     print('X_Y_shape::', X_Y_mult.shape)
    
#     curr_X_Y_mult = torch.index_select(X_Y_mult, 0, ids)
    
    non_linear_term = curr_X_Y_mult*(1 - sig_layer(torch.mm(curr_X_Y_mult, theta)))
    
#     print(ids, non_linear_term, theta)
    
    non_linear_term = torch.sum(non_linear_term, dim=0).view(theta.shape)
    
    res = -non_linear_term + beta*theta*curr_X_Y_mult.shape[0]
    
    return res

def compute_first_derivative(X_Y_mult, theta, dim):
    
    non_linear_term = X_Y_mult*(1 - sig_layer(torch.mm(X_Y_mult, theta)))
    
#     print(ids, non_linear_term, theta)
    
    non_linear_term = torch.sum(non_linear_term, dim=0).view(theta.shape)
    
    res = -non_linear_term/dim[0] + beta*theta
    
    return res



def compute_first_derivative_single_data2(X_Y_mult, theta, dim):
    
    print('X_Y_shape::', X_Y_mult.shape)
    
    curr_X_Y_mult = X_Y_mult#torch.index_select(X_Y_mult, 0, ids)
    
    non_linear_term = curr_X_Y_mult*(1 - sig_layer(torch.mm(curr_X_Y_mult, theta)))
    
#     print(ids, non_linear_term, theta)
    
    non_linear_term = torch.sum(non_linear_term, dim=0).view(theta.shape)
    
    res = -non_linear_term + beta*theta*X_Y_mult.shape[0]
    
    theta.requires_grad = True
    
    loss = loss_function3(X_Y_mult, theta, dim, beta)
    
    loss.backward()
    
    
    print(theta.grad.data)
    
    return res


def compute_model_parameter_by_iteration(dim, theta,  X_Y_mult, max_epoch, alpha, beta):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    for i in range(max_epoch):
        
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
        
        
        non_linear_term = X_Y_mult*(1 - sig_layer(torch.mm(X_Y_mult, theta)))
        
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

        gradient = -torch.sum(non_linear_term, dim=0).view(theta.shape)/dim[0] + beta*theta

#         print('iteration_gradient::', gradient)

        theta = theta - alpha*gradient#(1-alpha*beta)*theta + (alpha/dim[0])*torch.sum(non_linear_term, dim=0).view(theta.shape)
        
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



def compute_model_parameter_by_iteration_sparse(dim, theta,  X_Y_mult, max_epoch, alpha, beta):
    
    total_time = 0.0
    
    theta = theta.numpy()
    
    for i in range(max_epoch):
        
        non_linear_term = X_Y_mult.multiply(1 - sigmoid_np(X_Y_mult.dot(theta)))
        
        gradient = -np.reshape(np.sum(non_linear_term, axis=0), (theta.shape))/dim[0] + beta*theta

        theta = theta - alpha*gradient#(1-alpha*beta)*theta + (alpha/dim[0])*torch.sum(non_linear_term, dim=0).view(theta.shape)
            
    return torch.from_numpy(theta).type(torch.DoubleTensor), total_time


def compute_model_parameter_by_iteration2(dim, theta,  X_Y_mult):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    for i in range(max_epoch):
        
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
        
        
        non_linear_term = X_Y_mult*(1 - sig_layer(torch.mm(X_Y_mult, theta)))
        
        
        if i == max_epoch - 1:
            for j in range(non_linear_term.shape[0]):
                print(j, non_linear_term[j], theta)
        
        
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
        theta = (1-alpha*beta)*theta + (alpha/dim[0])*torch.sum(non_linear_term, dim=0).view(theta.shape)
        
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


def compute_model_parameter_by_approx2(dim, theta, Pi, X_Y_mult):
    
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


def compute_model_parameter_by_approx_incremental_2(term1, term2, dim, theta, max_epoch):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
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


        gradient = -(torch.mm(term1[i], theta) + (term2[i]).view(theta.shape))/dim[0] + beta*theta

#         print('approx_gradient::', gradient)

        theta = theta - alpha * gradient
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
    
    return theta


def compute_model_parameter_by_approx_incremental_3(term1, term2, dim, theta, max_epoch, cut_off_epoch, alpha, beta):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
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
            gradient = -(torch.mm(term1[i], theta) + (term2[i]).view(theta.shape))/dim[0] + beta*theta
        
        else:
            gradient = -(torch.mm(term1[cut_off_epoch - 1], theta) + (term2[cut_off_epoch - 1]).view(theta.shape))/dim[0] + beta*theta

#         print('approx_gradient::', gradient)

        theta = theta - alpha * gradient
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
    
    
#     term1 = 
    
    
#     for i in range(max_epoch - cut_off_epoch):
        
    
    
        
    print('total_time::', total_time)
    
    return theta


def prepare_term_1_batch2_theta(X, X_mult_theta, w_seq_this_epoch):
    
#     w_dim = w_seq.shape
    
    res = torch.mm(torch.t(X), X_mult_theta.view(-1,1)*w_seq_this_epoch.view(-1,1))
    
#     X_product = torch.t(X_product.view(dim[0], dim[1]*dim[1]))
#     
#     res = torch.t(torch.mm(X_product, w_seq)).view(w_dim[1], dim[1], dim[1])
    
    return res

def prepare_term_1_batch2_theta_sparse(X, X_mult_theta, w_seq_this_epoch):
    
#     w_dim = w_seq.shape
    
#     res = torch.mm(torch.t(X), X_mult_theta.view(-1,1)*w_seq_this_epoch.view(-1,1))
#     print(X_mult_theta.shape)
#     print(w_seq_this_epoch.view(-1,1).numpy().shape)
#     print(X.shape)
    
    res = X.transpose().dot(np.multiply(X_mult_theta, w_seq_this_epoch.view(-1,1).numpy()))
    
#     X_product = torch.t(X_product.view(dim[0], dim[1]*dim[1]))
#     
#     res = torch.t(torch.mm(X_product, w_seq)).view(w_dim[1], dim[1], dim[1])
#     print(res.shape)
    return res


def prepate_term_1_batch_by_epoch(X, w_seq):
    
    res1 = X*w_seq.view(w_seq.shape[0], -1)

    res = torch.mm(torch.t(X), res1)
    
    
    return res



def prepate_term_1_batch_by_epoch_sparse(X, w_seq):
    
    res1 = X.multiply(w_seq.view(w_seq.shape[0], -1).numpy())

    res = X.transpose().dot(res1)
    
    
    return scipy.sparse.csr_matrix(res)

def compute_model_parameter_by_approx_incremental_3_2(term1, delta_X, term2, dim, theta, max_epoch, cut_off_epoch, alpha, beta, w_seq):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    for i in range(max_epoch):
        
        if i < cut_off_epoch:
            X_mult_theta = torch.mm(delta_X, theta)
            sub_term1 = prepare_term_1_batch2_theta(delta_X, X_mult_theta, w_seq[:,i])
            gradient = -(torch.mm(term1[i], theta) - sub_term1 + (term2[i]).view(theta.shape))/dim[0] + beta*theta
        
        else:
            
            if i == cut_off_epoch:
                sub_term1_without_theta = prepate_term_1_batch_by_epoch(delta_X, w_seq[:,cut_off_epoch - 1])
            
#             sub_term1 = prepare_term_1_batch2_theta(delta_X, X_mult_theta, w_seq[:,cut_off_epoch - 1])
            gradient = -(torch.mm(term1[cut_off_epoch - 1] - sub_term1_without_theta, theta) + (term2[cut_off_epoch - 1]).view(theta.shape))/dim[0] + beta*theta


        theta = theta - alpha * gradient   
    
        
    print('total_time::', total_time)
    
    return theta


def convert_coo_matrix2tensor(Y_coo):
    
    indices = np.vstack((Y_coo.row, Y_coo.col))
    values = Y_coo.data
    
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    shape = Y_coo.shape
    
#     print(Y_coo)
    
    Y = torch.sparse.DoubleTensor(i, v, torch.Size(shape))
    
    return Y

def convert_coo_matrix2_dense_tensor(Y_coo):
    
#     indices = np.vstack((Y_coo.row, Y_coo.col))
#     values = Y_coo.data
#     
#     i = torch.LongTensor(indices)
#     v = torch.DoubleTensor(values)
#     shape = Y_coo.shape
    
    
    res = Y_coo.todense()
    
    Y = torch.from_numpy(res).type(torch.DoubleTensor)
    
#     print(Y_coo)
    
#     Y = torch.sparse.DoubleTensor(i, v, torch.Size(shape)).to_dense()
    
    return Y

def compute_model_parameter_by_approx_incremental_3_2_sparse(term1, delta_X, term2, dim, theta, max_epoch, cut_off_epoch, alpha, beta, w_seq):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    overhead1 = 0
    
    overhead2 = 0
    
    overhead3 = 0
    
    theta = theta.numpy()
    
    for i in range(max_epoch):
        
#         X_mult_theta = torch.mm(delta_X, theta)
        X_mult_theta = delta_X.dot(theta)
        
        if i < cut_off_epoch:
            t1 = time.time()

            sub_term1 = prepare_term_1_batch2_theta_sparse(delta_X, X_mult_theta, w_seq[:,i])
            t2 = time.time()
            
            overhead1 += (t2 - t1)
            
#             curr_term_1 = convert_coo_matrix2tensor(term1[i])

            print('term1 shape', term1[i].shape)
            print('term2 shape', term2[i].shape)
            print('theta shape', theta.shape)
            
            t3 = time.time()
            
            res = term1[i].dot(theta)
            
            t4 = time.time()
            
            overhead2 += (t4 - t3)
            
            print('intermediate result', res.shape)
            
            t5 = time.time()
            gradient = -( res - sub_term1 + (np.reshape(term2[i], theta.shape)))/dim[0] + beta*theta
            t6 = time.time()
            
            overhead3 += (t6 - t5)
            
#             print('theta shape', theta.shape)
#             
#             print('time1::', t2 - t1)
#             
#             print('time2::', t6 - t5)
        
        else:
            
            if i == cut_off_epoch:
                curr_term_1 = (term1[cut_off_epoch - 1])
#                 sub_term1_without_theta = prepate_term_1_batch_by_epoch_sparse(delta_X, w_seq[:,cut_off_epoch - 1])

            t1 = time.time()

            sub_term1 = prepare_term_1_batch2_theta_sparse(delta_X, X_mult_theta, w_seq[:,cut_off_epoch - 1])
            
            t2 = time.time()
            
            overhead1 += (t2 - t1)

                
#             t3 = time.time()

#             curr_term_1 = convert_coo_matrix2tensor(term1[cut_off_epoch - 1])
            
#             t4 = time.time()
            
#             overhead2 += (t4 - t3)
#             sub_term1 = prepare_term_1_batch2_theta(delta_X, X_mult_theta, w_seq[:,cut_off_epoch - 1])
#             t7 = time.time()
#             itermediate_res = torch.sparse.mm(curr_term_1, theta)
            
#             t8 = time.time()

            t3 = time.time()
            
            res = curr_term_1.dot(theta)
            
            t4 = time.time()

            overhead2 += (t4 - t3)

            print('intermediate result', res.shape)

            t5 = time.time()

            gradient = -(res - sub_term1 + (np.reshape(term2[cut_off_epoch - 1], theta.shape)))/dim[0] + beta*theta
#             gradient = -((curr_term_1 - sub_term1_without_theta).dot(theta) + (term2[cut_off_epoch - 1].transpose()))/dim[0] + beta*theta
            t6 = time.time()
            
            overhead3 += (t6 - t5)
#             print('time1::', t2 - t1)
#             
#             print('time2::', t6 - t5)

#             overhead3 += (t8 - t7)

        theta = theta - alpha * gradient   
    
    print('overhead1::', overhead1)
     
    print('overhead2::', overhead2)
     
    print('overhead3::', overhead3)
        
        
    print('total_time::', total_time)
    
    return torch.from_numpy(theta).type(torch.DoubleTensor)



def compute_model_parameter_by_approx_incremental_4(s, M, M_inverse, expected_A, term1, term2, dim, theta, max_epoch, cut_off_epoch, alpha, beta):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
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
            gradient = -(torch.mm(term1[i], theta) + (term2[i]).view(theta.shape))/dim[0] + beta*theta
        
        else:
            gradient = -(torch.mm(term1[cut_off_epoch - 1], theta) + (term2[cut_off_epoch - 1]).view(theta.shape))/dim[0] + beta*theta

#         print('approx_gradient::', gradient)

        theta = theta - alpha * gradient



    A = (1- beta*alpha)*torch.eye(dim[1], dtype = torch.double) + alpha*term1[cut_off_epoch - 1]/dim[0]
    
    B = alpha/dim[0]*(term2[cut_off_epoch - 1].view(-1,1))
    
    updated_s = torch.diag(torch.mm(M_inverse, torch.mm(A-expected_A, M)))
    
    updated_s = s + updated_s
    
#     pow_updated_s = updated_s.clone()
    
    updated_s[updated_s > 1] = 1-1e-6
    
    s_power = torch.pow(updated_s, float(max_epoch - cut_off_epoch))
    
    res1 = M.mul(s_power.view(1,-1))

#     res1 = torch.mm(res1, M_inverse)
    
    
#     temp = torch.eye(dim[1], dtype = torch.double)
#     
#     sum_temp = torch.zeros((dim[1], dim[1]), dtype = torch.double)
#     
#     for i in range(max_epoch):
#         sum_temp += temp
#         temp = torch.mm(temp, A)
        
    
    
#     print('temp_gap::', temp - res1)
    
    sub_sum = (1-s_power)/(1-updated_s)
    
    res2 = M.mul(sub_sum.view(1, -1))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
    
    theta = torch.mm(res1, torch.mm(M_inverse,theta)) + torch.mm(res2, torch.mm(M_inverse, B))


    
    
        
    print('total_time::', total_time)
    
    return theta

def compute_model_parameter_by_approx_incremental_4_2(s, M, M_inverse, term1, delta_X, term2, dim, theta, max_epoch, cut_off_epoch, alpha, beta, w_seq):
    
    total_time = 0.0
    
    for i in range(cut_off_epoch):

        X_mult_theta = torch.mm(delta_X, theta)
        
        sub_term1 = prepare_term_1_batch2_theta(delta_X, X_mult_theta, w_seq[:,i])


        if i < cut_off_epoch:
            gradient = -(torch.mm(term1[i], theta) - sub_term1 + (term2[i]).view(theta.shape))/dim[0] + beta*theta
        
        else:
            gradient = -(torch.mm(term1[cut_off_epoch - 1], theta) - sub_term1 + (term2[cut_off_epoch - 1]).view(theta.shape))/dim[0] + beta*theta

        theta = theta - alpha * gradient




    if delta_X.shape[0] < delta_X.shape[1]:
        delta_s = torch.mm(torch.mm(M_inverse, torch.t(delta_X*w_seq[:,cut_off_epoch - 1].view(-1,1))), torch.mm(delta_X, M))
        
    else:
        delta_s = torch.mm(torch.mm(M_inverse, torch.mm(torch.t(delta_X*w_seq[:,cut_off_epoch - 1].view(-1,1)), delta_X)), M)
    
    
    updated_s = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*(s - torch.diag(delta_s))/dim[0]


#     A = (1- beta*alpha)*torch.eye(dim[1], dtype = torch.float) + alpha*term1[cut_off_epoch - 1]/dim[0]
    
    B = alpha/dim[0]*(term2[cut_off_epoch - 1].view(-1,1))
    
#     updated_s = torch.diag(torch.mm(M_inverse, torch.mm(A-expected_A, M)))
    
#     updated_s = s + updated_s
    
    updated_s[updated_s > 1] = 1-1e-6
    
    s_power = torch.pow(updated_s, float(max_epoch - cut_off_epoch))
    
    res1 = M.mul(s_power.view(1,-1))

    sub_sum = (1-s_power)/(1-updated_s)
    
    res2 = M.mul(sub_sum.view(1, -1))
       
    theta = torch.mm(res1, torch.mm(M_inverse,theta)) + torch.mm(res2, torch.mm(M_inverse, B))


    
    
        
    print('total_time::', total_time)
    
    return theta




def compute_model_parameter_by_approx(w_seq, b_seq, X, Y, dim, theta, X_products, X_Y_products):    
    
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

def initialize(X):
    shape = list(X.size())
    theta = Variable(torch.zeros([shape[1],1], dtype = torch.double))
#     theta[0][0] = -1
    
    theta.requires_grad = True
#     lr.theta = Variable(lr.theta)

    print(theta.requires_grad)
    
    lr = logistic_regressor_parameter(theta)
    
    return lr


def initialize_by_size(dim):
#     shape = list(X.size())
    theta = Variable(torch.zeros([dim[1],1], dtype = torch.double))
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

def linearized_Function(x):
    return sigmoid(x)

def create_piecewise_linea_class():
#     x_l = torch.tensor(-10, dtype=torch.double)
#     x_u = torch.tensor(10, dtype=torch.double)
    x_l = -20.0
    x_u =20.0
    num = 1000000
    Pi = piecewise_linear_interpolication(x_l, x_u, linearized_Function, num)
    
    return Pi


def compute_linear_approx_parameters_2(X, Y, X_Y_mult):
    res = torch.mm(X_Y_mult, res_prod_seq)
    torch.pow(sig_layer(res),2)*torch.exp(-res)
    
    
    
    

def compute_linear_approx_parameters(X, Y, X_Y_mult, max_epoch):
    
    Pi = create_piecewise_linea_class()
    
#     w_seq = []
#     
#     b_seq = []
    
    t1 = time.time()
    
#     print('res_prod_seq::', res_prod_seq)
    
    
    print(X.shape, res_prod_seq.shape)
    
#     res = torch.mm(X, res_prod_seq)*(Y.repeat(1, epoch))
    
    
    '''n*t'''
    
    res = torch.mm(X_Y_mult, res_prod_seq)
    
#     print(torch.norm(res2 - res))
    
    t2 = time.time()
    
    print('multi_time::', (t2 - t1))
    
    print(res_prod_seq.shape)
    
    print(res)
    
    w_res, b_res = Pi.piecewise_linear_interpolate_coeff_batch2(res)
    
    '''n*t'''
    
    
    
#     cut_off_epoch = max_epoch
    cut_off_epoch = w_res.shape[1]
     
#     for i in range(w_res.shape[1]-1):
#         curr_gap = torch.norm(w_res[:, i + 1] - w_res[:, i])
#          
# #         print(i, curr_gap)
#          
#         if curr_gap < prov_record_rate:
#             cut_off_epoch = i+1
#             break
#      
#      
#     curr_w_res = w_res[:,0:cut_off_epoch]
#        
#     b_res = b_res[:,0:cut_off_epoch]
     
    torch.save(cut_off_epoch, git_ignore_folder + 'cut_off_epoch')
    
    
    print('w_res_shape::', w_res.shape)
    
    print('b_res_shape::', b_res.shape)
    
    
    
    
#     for i in range(dim[0]):
#         for j in range(max_epoch):
#             curr_arg_value = torch.mm(res_prod_seq[:,j].view(1, dim[1]), X[i].view(dim[1], 1))*Y[i]
#              
#             expect_value = 1- sig_layer(curr_arg_value)
#              
#             approx_value = w_res[i,j]*curr_arg_value + b_res[i,j]
#             
#             print('gap::', expect_value - approx_value)
            
            
#     print('w_res::', w_res)
#     
#     print('w_size::', w_res.shape)
#     
#     print('b_res::', b_res)
#     
#     print('b_res_size::', b_res.shape)
    
    return w_res, b_res
    
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

def get_subset_training_data2(X, Y, subset_ids):
    selected_rows = torch.tensor(subset_ids)
    print(selected_rows)
    update_X = torch.index_select(X, 0, selected_rows)
    update_Y = torch.index_select(Y, 0, selected_rows)
    return update_X, update_Y

def random_deletion(X, Y, delta_num, res):
    
    multi_res = torch.mm(X, res)
    
    sorted, indices = torch.sort(torch.abs(multi_res), dim = 0, descending = True)
    
    delta_id_array = []
    
#     delta_data_ids = torch.zeros(delta_num, dtype = torch.long)
    
#     for i in range(delta_num):

    expected_selected_label =0
    
     
    if torch.sum(Y == 1) > torch.sum(Y == -1):
        expected_selected_label = 1
        
    else:
        expected_selected_label = -1


    i = 0

    while len(delta_id_array) < delta_num and i < X.shape[0]:
        if Y[indices[i]]*multi_res[indices[i]] >= 0:
            Y[indices[i]] = -Y[indices[i]]
            delta_id_array.append(indices[i])
    
    
        i = i + 1
    
    delta_data_ids = torch.tensor(delta_id_array, dtype = torch.long)
    
#     print(delta_data_ids[:100])
#     delta_data_ids = random_generate_subset_ids(X.shape, delta_num)     
#     torch.save(delta_data_ids, git_ignore_folder+'delta_data_ids')
    return X, Y, delta_data_ids    
    
    
    


def add_noise_data(X, Y, num, res):
    
    
#     X_distance = torch.sqrt(torch.bmm(X.view(dim[0], 1, dim[1]), X.view(dim[0],dim[1], 1))).view(-1,1)
    
    positive_X_mean = torch.mean(X[Y.view(-1)==1], 0)
    
    negative_X_mean = torch.mean(X[Y.view(-1)==-1], 0)
    
    coeff = positive_X_mean/negative_X_mean
    
    coeff = coeff[coeff != np.inf]
        
    coeff = torch.sum(coeff[coeff == coeff])
    
    
    print('coeff::', coeff)
    
#     coeff = torch.sum(coeff[coeff != np.inf])
    
    
    
    expected_selected_label =0
    
     
    if torch.sum(Y == 1) > torch.sum(Y == -1):
        expected_selected_label = 1
        
    else:
        expected_selected_label = -1
        coeff = 1/coeff 
    
    
    multi_res = torch.mm(X, res)
    
    sorted, indices = torch.sort(torch.abs(multi_res), dim = 0, descending = True)
    
    
    selected_point = None
    
    selected_label = None
    
    
    selected_id = 0
    
    noise_data_X = torch.zeros((num, X.shape[1]), dtype = torch.double)

    noise_data_Y = torch.zeros((num, 1), dtype = torch.double)
    
    for i in range(num):
        
        curr_class = Y[indices[i], 0]
        
        if curr_class == 1:
            curr_coeff = positive_X_mean/negative_X_mean#mean_list[curr_class]/mean_list[(curr_class + 1)%(num_class)]        
        
        
        if curr_class == -1:
            curr_coeff = negative_X_mean/positive_X_mean
         
        curr_coeff = curr_coeff[curr_coeff != np.inf]
        
        curr_coeff = torch.sum(curr_coeff[curr_coeff == curr_coeff])*5
         
#         curr_coeff = torch.sum(curr_coeff[curr_coeff != np.inf and np.isnan(curr_coeff.numpy())])
        
#         print(curr_coeff)
        
        selected_point = (X[indices[i]].clone())*curr_coeff
        
        
        if multi_res[indices[i],0]*Y[indices[i], 0] > 0:
            selected_label = -curr_class
        else:
            selected_label = curr_class
        
        noise_data_X[i,:] = selected_point
        
        noise_data_Y[i] = selected_label
        
        
    X = torch.cat([X, noise_data_X], 0)
        
    Y = torch.cat([Y, noise_data_Y], 0)    
    
#     for i in range(indices.shape[0]):
#         if Y[indices[i],0].numpy()[0] == expected_selected_label and multi_res[indices[i]]*Y[indices[i]] > 0: 
#             selected_point = X[indices[i]].clone()
#             selected_id = indices[i]
#             selected_label = -Y[indices[i]]
#             break
#     
#     
#     
#     print(torch.mm(selected_point, res))        
#             
#     selected_point = selected_point/coeff# + torch.rand(selected_point.shape, dtype = torch.double)
#     
#     print('distance::', torch.mm(selected_point, res))       
#     
#     dist_range = torch.rand(selected_point.view(-1).shape, dtype = torch.double)
#     
#     
#     dist = torch.distributions.Normal(selected_point.view(-1), dist_range)
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
#     noise_X = dist.sample((num,))
#     
#     noise_Y = torch.zeros([num, 1], dtype = torch.double)
#     
#     
#     noise_Y[:,0] = selected_label
#     
#     X = torch.cat([X, noise_X], 0)
#     
#     Y = torch.cat([Y, noise_Y], 0)
    
    
    
    
    
    
    
    
    
    
    
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

def add_noise_data2(X, Y, added_x, added_y, num):
    
    added_x_tensor = torch.tensor(added_x, dtype = torch.double)

#     added_x_tensor = extended_by_constant_terms(added_x_tensor.view(-1,X.shape[1]))
    
    added_y_tensor = torch.tensor(added_y, dtype = torch.double) 
    
#     dist_range = 0.01*torch.rand(added_x_tensor.view(-1).shape, dtype = torch.double)
#     
#     dist = torch.distributions.Normal(added_x_tensor.view(-1), dist_range)
#     
#     noise_X = dist.sample((num,))
    
    noise_Y = torch.zeros((num,1), dtype = torch.double)
    
    noise_X = added_x_tensor.repeat(int(num/added_x_tensor.shape[0]),1)
    
    
    
    noise_Y[:] = added_y_tensor[0]
    
    X = torch.cat([X, noise_X], 0)
    
    Y = torch.cat([Y, noise_Y], 0)
    
    return X, Y
    
    
    
    
    
    
def change_data_labels(X, Y, ratio, res):
    
    positive_ids = (Y.view(-1) == 1).nonzero().view(-1).numpy()
    
    negative_ids = (Y.view(-1) == -1).nonzero().view(-1).numpy()
    
    mult_res = torch.mm(X, res)
    
    positive_prediction_ids = (mult_res.view(-1) > 0).nonzero().view(-1).numpy()
    
    negative_prediction_ids = (mult_res.view(-1) < 0).nonzero().view(-1).numpy()
    
    positive_ids = torch.tensor(list(set(positive_ids) & set(positive_prediction_ids)))
    
    negative_ids = torch.tensor(list(set(negative_ids) & set(negative_prediction_ids)))
    
    
    
    positive_num = int(positive_ids.shape[0]*ratio)
    
    negative_num = int(negative_ids.shape[0]*ratio)
    
#     positive_id_ids = torch.tensor(np.random.choice(list(range(positive_ids.shape[0])), size = positive_num, replace=False))
    rid_seq = []
    
    if positive_num != 0:
        _, ids = torch.sort(torch.mm(X[positive_ids].view(positive_ids.shape[0], X.shape[1]), res).view(-1), descending = False)
      
      
        positive_rids = torch.tensor(positive_ids[ids[0:positive_num]].numpy())
        
        rid_seq.append(positive_rids)
        
        
    if negative_num != 0:

        _, ids = torch.sort(torch.abs(torch.mm(X[negative_ids].view(negative_ids.shape[0], X.shape[1]), res)).view(-1), descending = False)
          
          
        negative_rids = torch.tensor(negative_ids[ids[0:negative_num]].numpy())


        rid_seq.append(negative_rids)
#     negative_id_ids = torch.tensor(np.random.choice(list(range(negative_ids.shape[0])), size = negative_num, replace=False))
    
    
#     selected_positive_ids = positive_ids[positive_id_ids]
#     
# #     selected_negative_ids = negative_ids[negative_id_ids]
#     
#     rids = selected_positive_ids
    
    rids = torch.cat(rid_seq, 0)
    
    for id in rids:
        Y[id] = -Y[id]
        
    return X, Y, torch.tensor(list(rids))
    
def change_data_labels2(X, Y, ratio, res):
    
    positive_ids = (Y.view(-1) == 1).nonzero().view(-1).numpy()
    
    negative_ids = (Y.view(-1) == -1).nonzero().view(-1).numpy()
    
    mult_res = torch.mm(X.mul(Y), res).view(-1)
    
    _, ids = torch.sort(torch.abs(mult_res), descending = True)
    
    
    
    
    
    
#     positive_prediction_ids = (mult_res.view(-1) > 0).nonzero().view(-1).numpy()
#     
#     negative_prediction_ids = (mult_res.view(-1) < 0).nonzero().view(-1).numpy()
#     
#     positive_ids = torch.tensor(list(set(positive_ids) & set(positive_prediction_ids)))
#     
#     negative_ids = torch.tensor(list(set(negative_ids) & set(negative_prediction_ids)))
    
    
    
    num = int(ids.shape[0]*ratio)
    
    rids = ids[0:num]
    
#     rid_seq = []
#     
#     if positive_num != 0:
#         _, ids = torch.sort(torch.mm(X[positive_ids].view(positive_ids.shape[0], X.shape[1]), res).view(-1), descending = False)
#       
#       
#         positive_rids = torch.tensor(positive_ids[ids[0:positive_num]].numpy())
#         
#         rid_seq.append(positive_rids)
#         
#         
#     if negative_num != 0:
# 
#         _, ids = torch.sort(torch.abs(torch.mm(X[negative_ids].view(negative_ids.shape[0], X.shape[1]), res)).view(-1), descending = False)
#           
#           
#         negative_rids = torch.tensor(negative_ids[ids[0:negative_num]].numpy())
# 
# 
#         rid_seq.append(negative_rids)
#     negative_id_ids = torch.tensor(np.random.choice(list(range(negative_ids.shape[0])), size = negative_num, replace=False))
    
    
#     selected_positive_ids = positive_ids[positive_id_ids]
#     
# #     selected_negative_ids = negative_ids[negative_id_ids]
#     
#     rids = selected_positive_ids
    
#     rids = torch.cat(rid_seq, 0)
    
    for id in rids:
        Y[id] = -Y[id]
        
    return X, Y, torch.tensor(list(rids))






def eigen_decomposition(term1, cut_off_epoch, dim):
    
    
#     A = (1-alpha*beta)*torch.eye(dim[1], dtype = torch.double) - alpha*(term1[cut_off_epoch - 1])/dim[0]
    
    
#     A = A.type(torch.FloatTensor)
    curr_term1 = None

    try:
        curr_term1 = convert_coo_matrix2tensor(term1[cut_off_epoch - 1])
    
    except:
        curr_term1 = term1[cut_off_epoch - 1]

    del term1
    

    s, M = torch.eig(curr_term1, True)
        
    s = s[:,0]
    
    print('eigen_values::', s)
        
    torch.save(M, git_ignore_folder + 'eigen_vectors')
    
    M_inverse = torch.tensor(np.linalg.inv(M.numpy()), dtype = torch.double)
    
#     M_inverse = torch.inverse(M)
    
    
    print('inverse_gap::', torch.norm(torch.mm(M, M_inverse) - torch.eye(dim[1], dtype = torch.double)))
    
    torch.save(M_inverse, git_ignore_folder + 'eigen_vectors_inverse')
    
    torch.save(s, git_ignore_folder + 'eigen_values')

#     torch.save(A, git_ignore_folder + 'expected_A')


def save_term1(term1, run_rc1):
    
    
    if not run_rc1:
        torch.save(term1, git_ignore_folder + 'term1')
    else:
        directory = git_ignore_folder + 'term1_folder'
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for i in range(len(term1)):
            
            
                scipy.sparse.save_npz(directory + '/' + str(i), term1[i])
                
                
        torch.save(torch.tensor([len(term1)]), directory + '/term1_len')
    
def load_term1(directory):
    
#     try:
#         return torch.load(git_ignore_folder + 'term1')
#     
#     except:
    
    
    term1_len = torch.load(directory + '/term1_len')[0]
    
    term1 = []
    
    for i in range(term1_len):
#         try:
#             term1.append(torch.load(directory + '/' + str(i)))
#         except:
        term1.append(scipy.sparse.load_npz(directory + '/' + str(i) + '.npz'))
            
    return term1
    



def capture_provenance(X, Y, dim, epoch, run_rc1):
    
    cut_off_epoch = res_prod_seq.shape[1]
    
    print('cut_off_epoch::', cut_off_epoch)
    
    t3 = time.time()
    
    X_Y_mult = X.mul(Y)
    
    memory_usage = 0
    
    
    t3_1 = time.time()
    
    w_seq, b_seq = compute_linear_approx_parameters(X, Y, X_Y_mult, cut_off_epoch)

    memory_usage += w_seq.element_size() * w_seq.nelement()
    
    memory_usage += b_seq.element_size() * b_seq.nelement()

    t3_2 = time.time()

#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
    
#     term1 = prepare_term_1_serial(X, w_seq, dim)
#     term1 = prepare_term_1_batch2(X_product, w_seq, dim)

    term1 = prepate_term_1_batch_epoch(X, w_seq, dim, run_rc1)
#     term1 = prepare_term_1_batch(X_product, w_seq, dim, 10000)
    
#     memory_usage += term1.element_size() * term1.nelement()
    
#     term2 = prepare_term_2_serial(X_Y_mult, b_seq, dim)
#     term2 = prepare_term_2_batch2(X_Y_mult, b_seq, dim)
    term2 = prepare_term_2_batch(X_Y_mult, b_seq, dim, 10000)

    if run_rc1:
        X_Y_mult = check_and_convert_to_sparse_tensor(X_Y_mult)
        scipy.sparse.save_npz(git_ignore_folder + 'X_Y_mult' , X_Y_mult)

    else:
        torch.save(X_Y_mult, git_ignore_folder + 'X_Y_mult')
    
    del X_Y_mult

    
    save_term1(term1, run_rc1)
    
    torch.save(term2, git_ignore_folder + 'term2')
    
    torch.save(w_seq, git_ignore_folder + 'w_seq')
    
    torch.save(b_seq, git_ignore_folder + 'b_seq')
    
    del term2, w_seq, b_seq
    
    if not run_rc1:
        eigen_decomposition(term1, cut_off_epoch, dim)
    
#     memory_usage += term2.element_size() * term2.nelement()
    
    
#     torch.save(term1, git_ignore_folder + 'term1')
    
    
#     torch.save(X_product, git_ignore_folder + 'X_product')
    
    t4 = time.time()
    
    print('preparing_time::', t4 - t3)
    
    print('para_time::', t3_2 - t3_1)
    

def precomptation_influence_function(X, Y, res, dim):
    
    t5 = time.time()
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
    X_Y_mult = X.mul(Y)
    
#     Hessin_matrix = compute_hessian_matrix(X, X_Y_mult, res, dim, X_product)


#     Hessin_matrix = compute_hessian_matrix_2(X, X_Y_mult, res, dim, X_product)
    
    Hessin_matrix = compute_hessian_matrix_4(X, X_Y_mult, res, dim)
    
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


def change_instance_labels(X, Y, num, dim, res):
    
    expected_selected_label =0
    
     
    if torch.sum(Y == 1) > torch.sum(Y == -1):
        expected_selected_label = 1
        
    else:
        expected_selected_label = -1 
    
    
    multi_res = torch.mm(X, res)
    
    sorted, indices = torch.sort(torch.abs(multi_res), dim = 0, descending = False)
    
    
    selected_point = None
    
    selected_label = None
    
    
    selected_id = 0
    
    delta_data_ids = set()
    
    for i in range(num):
#         if Y[indices[i],0].numpy()[0] == expected_selected_label and multi_res[indices[i]]*Y[indices[i]] > 0: 
#             X[indices[i]] = X[indices[i]]*torch.rand(X[indices[i]].shape, dtype = torch.double)
            Y[indices[i]] = -Y[indices[i]]
            delta_data_ids.add(indices[i])
#             selected_id = indices[i]
#             selected_label = -Y[indices[i]]
#             break
    
    
    
#     print(torch.mm(selected_point, res))        
#             
#     selected_point = 10*selected_point# + torch.rand(selected_point.shape, dtype = torch.double)
#     
#     print('distance::', torch.mm(selected_point, res))       
#     
#     dist_range = torch.rand(selected_point.view(-1).shape, dtype = torch.double)
#     
#     
#     dist = torch.distributions.Normal(selected_point.view(-1), dist_range)
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
#     noise_X = dist.sample((num,))
#     
#     noise_Y = torch.zeros([num, 1], dtype = torch.double)
#     
#     
#     noise_Y[:,0] = selected_label
#     
#     X = torch.cat([X, noise_X], 0)
#     
#     Y = torch.cat([Y, noise_Y], 0)
    
    
    
    
    
    
    
    
    
    
    
    
#     unique_Ys = torch.unique(Y)
#     
#     min_count = 0#Y.shape[0] + 1
#     
#     expected_label = 0
#     
#     for y in unique_Ys:
#         curr_count = torch.sum(Y == y)
#         
#         if curr_count > min_count:
#             min_count = curr_count
#             expected_label = y
#     
#     ids = torch.nonzero(Y.view(-1) == expected_label)
    
#     while len(delta_data_ids) < num:
#         id = random.randint(0, Y.shape[0]-1)
#         delta_data_ids.add(id)
#     
#     for id in delta_data_ids:
#         X[id] = (1-X[id])
        
    return X, Y, torch.tensor(list(delta_data_ids))
    


def change_data_values(X, Y, num, res):
    
    positive_X_mean = torch.mean(X[Y.view(-1)==1], 0)
    
    negative_X_mean = torch.mean(X[Y.view(-1)==-1], 0)
    
    multi_res = torch.mm(X, res)
     
    sorted, indices = torch.sort(torch.abs(multi_res), dim = 0, descending = True)
     
    p1 = None
     
    p2 = None
    
    delta_data_ids = set()
    
    
    coeff = positive_X_mean/negative_X_mean
    
    coeff = coeff[coeff != np.inf]
        
    coeff = torch.sum(coeff[coeff == coeff])*5
    
    print(coeff)
    
#     coeff = torch.sum(coeff[coeff != np.inf])
    
     
    for i in range(num):
        if Y[indices[i],0].numpy()[0] == 1:
            X[indices[i]] = coeff*X[indices[i]]
         
        if Y[indices[i],0].numpy()[0] == -1:
            X[indices[i]] = X[indices[i]]/coeff
        
        delta_data_ids.add(indices[i])
    
#         if p1 is not None and p2 is not None:
#             break
    
#     middle_point = (positive_X_mean + negative_X_mean)/2
#     
#     X[Y.view(-1) == 1] = X[Y.view(-1)==1]*torch.mean(positive_X_mean/negative_X_mean)
    
    return X, Y, torch.tensor(list(delta_data_ids))
     
    
    

if __name__ == '__main__':
    
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    sys_args = sys.argv
    
    file_name = sys_args[1]
    
#     print(file_name)
    
    start = bool(int(sys_args[2]))
    
    input_alpha = float(sys_args[3])
    
    input_beta = float(sys_args[4])
    
    input_threshold = float(sys_args[5])
    
    max_epoch = int(sys_args[6])
    
#     global alpha, beta, threshold
    noise_rate = float(sys_args[7])
    
#     add_features = bool(int(sys_args[]))
    
    
    
    random_deletion_or_not = bool(int(sys_args[8]))
    
    add_noise_or_not = bool(int(sys_args[9]))
    
    prov_record_rate = float(sys_args[10])
    
    run_rc1 = bool(int(sys_args[11]))
    
    alpha = input_alpha
    
    beta = input_beta
    
    threshold = input_threshold
    
#     test_file_name = sys_args[2]
    
    repetition = 1
     
     
#     X, Y = load_data(True, file_name)
    if start:
        
#         X, Y, test_X, test_Y = load_data(True, file_name)

        if run_rc1:
            [X, Y, test_X, test_Y] = load_data_multi_classes_rcv1()
        else:
            X, Y, test_X, test_Y = load_data(True, file_name)  
        
        
    #     added_x = np.load('modified_X.npy')[X.shape[0] + test_X.shape[0]:]
    #     
    #     added_y = np.load('modified_Y.npy')[X.shape[0] + test_X.shape[0]:]
        
    # 
    #     X = torch.cat([X, test_X], 0)
    #     
    #     Y = torch.cat([Y, test_Y], 0)
    
    #     X, Y, test_X, test_Y = clean_sensor_data(file_name)
    
        min_label = torch.min(Y)
                
        if min_label == 0:
            Y = 2*Y-1
    #         test_Y = 2*test_Y - 1
        Y = Y.view(-1,1)
        
        print(torch.unique(Y))
        
    #     print(torch.unique(test_Y))
    
        print(X)
        
        print(Y)
        
        print(X.shape)
        
        print(Y.shape)
    
        print(torch.sum(Y == 1))
        
        print(torch.sum(Y == -1))
    
    #     previous_size = X.shape[0]
    #     
    #     X, Y = add_noise_data(X, Y, int(X.shape[0]*0.3))
    #     curr_size = X.shape[0]
    #          
    #     new_data_ids = torch.tensor((range(previous_size, curr_size)))
    #          
    #     torch.save(new_data_ids, git_ignore_folder + 'noise_data_ids')
         
        
        X = extended_by_constant_terms(X, False)
        
        test_X = extended_by_constant_terms(test_X, False)
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        dim = X.shape
        
        print(dim)
    
    
        t1  = time.time()
        
    #     for i in range(repetition):
        lr = initialize(X)
        res2, epoch = compute_parameters(X, Y, lr, dim, False)
        
        t2 = time.time()
        
        print(res2)
        
        print('epoch::', epoch)
        
        torch.save(res2, git_ignore_folder + 'model_without_noise')
        
        
        print('training_accuracy::',compute_accuracy2(X, Y, res2))
    
        print('test_accuracy::', compute_accuracy2(test_X, test_Y, res2))
        
        if run_rc1:
            X = check_and_convert_to_sparse_tensor(X)
        
            test_X = check_and_convert_to_sparse_tensor(test_X)
            
            try:
                scipy.sparse.save_npz(git_ignore_folder + 'X', X)
                
                scipy.sparse.save_npz(git_ignore_folder + 'test_X', test_X)
            
            except:
                torch.save(X, git_ignore_folder + 'X')
    
                torch.save(test_X, git_ignore_folder + 'test_X')
        
        
        else:
            torch.save(X, git_ignore_folder + 'X')
    
            torch.save(test_X, git_ignore_folder + 'test_X')
            
        
            
        torch.save(Y, git_ignore_folder + 'Y')
        
        torch.save(test_Y, git_ignore_folder + 'test_Y')
            
        
            
        
    else:
        
        res1 = torch.load(git_ignore_folder + 'model_without_noise')
        
        
        
        Y = torch.load(git_ignore_folder + 'Y')
        test_Y = torch.load(git_ignore_folder + 'test_Y')
        
        if run_rc1:
            try:
                sparse_X = scipy.sparse.load_npz(git_ignore_folder + 'X.npz')
                X = convert_coo_matrix2_dense_tensor(sparse_X)
                test_X = convert_coo_matrix2_dense_tensor(scipy.sparse.load_npz(git_ignore_folder + 'test_X.npz'))
            except:
    #             X = torch.load(git_ignore_folder + 'X').to_dense()
    #             test_X = torch.load(git_ignore_folder + 'test_X').to_dense()
    
                X = torch.load(git_ignore_folder + 'X')
                test_X = torch.load(git_ignore_folder + 'test_X')

        else:
            
            X = torch.load(git_ignore_folder + 'X')
            test_X = torch.load(git_ignore_folder + 'test_X')
        
        
        
        
        
        print('model_without_noise::', res1)
        
        dim = X.shape
        
        
        if random_deletion_or_not:
            X, Y, noise_data_ids = random_deletion(X, Y, int(X.shape[0]*noise_rate), res1)
        else:
            if add_noise_or_not:
                X, Y = add_noise_data(X, Y, int(X.shape[0]*noise_rate), res1)
                noise_data_ids = torch.tensor(list(set(range(X.shape[0])) - set(range(dim[0]))))
            else:
                X, Y, noise_data_ids = change_data_values(X, Y, int(X.shape[0]*noise_rate), res1)
        
        
#     X, Y = add_noise_data2(X, Y, added_x, added_y, 1000)
#         X, Y, noise_data_ids = change_instance_labels(X, Y, int(X.shape[0]*0.01), dim, res1)
#         X, Y = add_noise_data(X, Y, int(X.shape[0]*0.3), res1)
    
        
         
#         
         
        torch.save(noise_data_ids, git_ignore_folder + 'noise_data_ids')
         
        dim = X.shape
         
        print(dim)
        
    #     X, Y, noise_data_ids = change_data_labels2(X, Y, 0.8, res) 
    #                    
    #     torch.save(noise_data_ids, git_ignore_folder + 'noise_data_ids')
    
        t1 = time.time()  
        lr = initialize(X)
        res2, epoch = compute_parameters(X, Y, lr, dim, True)
        
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
        
        print('training_accuracy::',compute_accuracy2(X, Y, res2))
    
        print('test_accuracy::', compute_accuracy2(test_X, test_Y, res2))
        
        capture_provenance(X, Y, dim, epoch, run_rc1)
        
        
        precomptation_influence_function(X, Y, res2, dim)
        
        if run_rc1:
            X = check_and_convert_to_sparse_tensor(X)
             
             
            try:
                scipy.sparse.save_npz(git_ignore_folder + 'noise_X', X)       
            except:
                torch.save(X, git_ignore_folder + 'noise_X')  
                
        else:
            torch.save(X, git_ignore_folder + 'noise_X')
            
            
#         torch.save(X, git_ignore_folder + 'noise_X')
        
        torch.save(Y, git_ignore_folder + 'noise_Y')
        
        
        torch.save(torch.tensor(epoch), git_ignore_folder + 'epoch')
        
        torch.save(res2, git_ignore_folder + 'model_origin')
    
        torch.save(alpha, git_ignore_folder + 'alpha')
        
        torch.save(beta, git_ignore_folder + 'beta')
    
        
        
        print('training_time::', t2 - t1)

        
         
         
        
    #     X_products = compute_sample_products(X, dim)
    #     
    #     X_Y_products = compute_sample_label_products(X, Y)
        
    #     print('X::', X)
    #     
    #     print('X_products::', X_products)
        
    #     print(theta)
    #     X_Y_mult = X.mul(Y)
    #     
    #     
    #     
    #     res2 = None
    #     
    #     t01 = time.time()
    #     
    #     for i in range(repetition):
    #         initialized_theta = Variable(initialize(X).theta)
    #         res2 = compute_model_parameter_by_iteration(dim, initialized_theta, X_Y_mult)
    #     
    #     
    #     t02 = time.time()
    #     
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
    #     
    #      
    #     time1 = (t2 - t1)/repetition
    #      
    #     time2 = (t4 - t3)/repetition
    #      
    #     time3 = (t02  - t01)/repetition 
    #      
    #     
    #     print('time1::', time1)
    #      
    #     print('time2::', time2)
    #      
    #      
    #     print('time3::', time3)
    #      
    #     linear_time = t6 - t5
    #      
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
