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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_IO.Load_data import *
    from sensitivity_analysis.multi_nomial_logistic_regression.piecewise_linear_interpolation_multi_dimension import *
    from sensitivity_analysis.multi_nomial_logistic_regression.incremental_updates_logistic_regression_multi_dim import *
    from sensitivity_analysis.multi_nomial_logistic_regression.evaluating_test_samples import *
    from sensitivity_analysis.multi_nomial_logistic_regression.incremental_updates_logistic_regression_multi_dim import *
except ImportError:
    from Load_data import *
    from piecewise_linear_interpolation_multi_dimension import *
    from incremental_updates_logistic_regression_multi_dim import *
    from evaluating_test_samples import *
    from incremental_updates_logistic_regression_multi_dim import *
import gc 
import sys
import random
from hessian import hessian



# from sensitivity_analysis.logistic_regression.incremental_updates_logistic_regression import X_product


# max_epoch = 200


'''shuttle_dataset: para'''
alpha = 1e-6
   
beta = 0.001

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



def compute_hessian_matrix2(X, Y, theta, dim, beta):
    
    theta.grad.data.zero_()
    
    loss = loss_function2(X, Y, theta, dim, beta, False)
    
    
    h = hessian(loss, theta, create_graph=True)


    return h
#     print(h)
#     
#     
#     
#     X_Y_theta_prod = torch.mm(X, theta)*Y
#     
#     res = torch.zeros([dim[1], dim[1]], dtype = torch.float64)
#     
#     for i in range(dim[0]):
#         res += second_derivative_loss_function(-X_Y_theta_prod[i])*X_product[i]
#         
#     res = -res/dim[0] + beta*torch.eye(dim[1], dtype = torch.float64)
#     
#     return res
    
    

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
    

def logistic_regression(X, Y, lr, dim, num_class, tracking_prov):

#     dim = X.shape
#     lr.theta.requires_grad = False
# 
#     vectorized_theta = lr.theta.view(-1,1).clone()
#     
#     vectorized_theta.requires_grad = True
    
#     print(vectorized_theta)
#     print('init_theta', lr.theta)
    Y = Y.type(torch.LongTensor)
    
    epoch = 0
    
    last_theta = None
    
    
    last_recorded_theta = None
    
#     for epoch in range(max_epoch):


    while epoch < max_epoch:
        
        
#         if tracking_prov:
#          
#             global res_prod_seq
#              
#              
#             if last_recorded_theta is None:
#                 last_recorded_theta = lr.theta.clone()
#                 res_prod_seq = lr.theta.clone()
#                 epoch_record_epoch_seq.append(epoch)
#                 tracking_prov = True
# #                 print('here')
#             else:
#                 if torch.norm(last_recorded_theta - lr.theta) > theta_record_threshold:
#                     last_recorded_theta = lr.theta.clone()
#                     res_prod_seq = torch.cat((res_prod_seq, lr.theta.clone()), 1)
#                     epoch_record_epoch_seq.append(epoch)
#                     tracking_prov = True
# #                     print('here')
#                 else:
#                     tracking_prov = False
#             if res_prod_seq.shape == 0:
#                 
#     #             res_prod_seq.append(lr.theta.clone())
#             else:
#                 res_prod_seq = torch.cat((res_prod_seq, lr.theta.clone()), 1)
        
        
        
        
        loss = loss_function2(X, Y, lr.theta, X.shape, beta, tracking_prov)
   
        loss.backward()
       
        with torch.no_grad():
#             print('gradient::', lr.theta.grad)
            lr.theta -= alpha * lr.theta.grad
#             print(lr.theta.grad[:, 1] - lr.theta.grad[:, 0])
#             print(lr.theta.grad[:,1] - lr.theta.grad[:,0])
#             print(epoch, lr.theta)

#             gap = torch.norm(lr.theta.grad)
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
        
        epoch = epoch + 1
        
        if last_theta is not None:
            print(torch.norm(last_theta - lr.theta))
#         
        if last_theta is not None and torch.norm(last_theta - lr.theta) < threshold:
            break
            
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

def logistic_regression_by_standard_library(X, Y, lr, dim, max_epoch, alpha, beta):

#     dim = X.shape
    Y = Y.type(torch.LongTensor)
    
    
    for epoch in range(max_epoch):
        
#         res_prod_seq.append(lr.theta.clone())
        
        loss = loss_function2(X, Y, lr.theta, dim, beta, False)
   
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


def compute_model_parameter_by_iteration(dim, theta,  X, Y, X_sum_by_class, num_class, max_epoch, alpha, beta):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    theta_list = []
    
    grad_list = []
    
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
        output = softmax_layer(torch.mm(X, theta))
        
        
        output = torch.mm(torch.t(X), output)
        
        
        output = torch.reshape(torch.t(output), [-1,1])
        
#         print(i, theta)
        
#         inter_result2.append(output)
#         
#         res = torch.mm(torch.t(torch.gather(output, 1, Y.view(dim[0], 1))), X)
        
        reshape_theta = torch.reshape(torch.t(theta), (-1, 1))
        
        res = (output - X_sum_by_class)/dim[0] + beta*reshape_theta
        
#         print('output::', output/dim[0])
# #         
#         print('x_sum_by_class::', X_sum_by_class/dim[0])
#         
#         print('gradient::', res.view(num_class, dim[1]))
#         
#         print(res[1] - res[0])
        
#         print(torch.t(res.view(num_class, dim[1]))[:,1] - torch.t(res.view(num_class, dim[1]))[:,0])
        
#         grad_list.append(res.clone())
        
        res = reshape_theta - alpha * res
        
#         theta_list.append(res)
        
        
        theta = torch.t(res.view(num_class, dim[1]))
        
#         print(alpha*(X_sum_by_class.view(-1, dim[1])[1]-X_sum_by_class.view(-1, dim[1])[0])/dim[0])
        
#         print((i+1)*alpha*(X_sum_by_class.view(-1, dim[1])[1]-X_sum_by_class.view(-1, dim[1])[0])/dim[0])
        
#         print(i, theta)
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
    
    return theta, total_time, theta_list, grad_list

def compute_model_parameter_by_iteration2(dim, theta,  X_Y_mult, max_epoch):
    
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

def compute_model_parameter_by_approx_incremental_4(s, M, M_inverse, term1, sub_term_1, term2, x_sum_by_class, dim, theta, num_class, max_epoch, cut_off_epoch, epoch_record_epoch_seq, alpha, beta):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    vectorized_theta = Variable(theta.view(-1,1))
    
    print('vec_theta_shape::', vectorized_theta.shape[0])
    
    A = Variable((1-alpha*beta)*torch.eye(vectorized_theta.shape[0], dtype = torch.double) - alpha*(term1[cut_off_epoch - 1] - sub_term_1[cut_off_epoch - 1])/dim[0])
#
#     expected_A = (1-alpha*beta)*torch.eye(vectorized_theta.shape[0], dtype = torch.double) - alpha*(term1[cut_off_epoch - 1])/X.shape[0]
# 

    delta_s = torch.diag(torch.mm(M_inverse, torch.mm(sub_term_1[cut_off_epoch - 1], M)))
    
#     updated_s = torch.diag(torch.mm(M_inverse, torch.mm(A-expected_A, M)))
    
    updated_s = (1-alpha*beta) - alpha*(s - delta_s)/dim[0]
    
#     pow_updated_s = updated_s.clone()
    
    updated_s[updated_s > 1] = 1-1e-6
    
#     pow_updated_s[pow_updated_s > 1] = 1
    
#     tmp = torch.mm(M_inverse, torch.mm(expected_A, M))
#     
#     expected_s = torch.diag(tmp)
#     
#     print(tmp)
# #     
# #     print(torch.norm(expected_A - A))
# #     
# #     print(torch.norm(A))
# #     
#     print(torch.sort(updated_s))
# #     
# #     print(torch.max(updated_s))
# #     
# #     print(torch.min(updated_s))
# #     
#     expected_s_2, M_2 = torch.eig(expected_A, True)
#     
#     expected_s_2 = expected_s_2[:,0]
#     
#     print('expected_s::', torch.sort(expected_s_2))
#     
#     print(torch.norm(M - M_2))
#     
#     print(torch.norm(expected_s_2 - expected_s))
#     
#     print(torch.norm(updated_s - expected_s_2))
# #     
# #     print(expected_s)
# #     
# 
#     print(torch.sort(expected_s))
#     print(torch.norm(expected_s - updated_s))
    

    B = -alpha*(term2[cut_off_epoch - 1].view(-1,1) - x_sum_by_class)/dim[0]
    
    
    
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

            output = torch.mm((term1[i] - sub_term_1[i]), vectorized_theta) + (term2[i].view(-1,1))
            
            
    #         print('gradient::', gradient)
    #         
    #         print('approx_output::', output/dim[0])
            
            
            gradient = (output- x_sum_by_class)/dim[0] + beta*vectorized_theta
    #         
    #         print('x_sum_by_class::', x_sum_by_class/dim[0])
            
            
            
            
            vectorized_theta = vectorized_theta - alpha*gradient
        
        else:
            
#             output = torch.mm(term1[cut_off_epoch - 1], vectorized_theta) + (term2[cut_off_epoch - 1].view(-1,1))
#             
#             
#     #         print('gradient::', gradient)
#     #         
#     #         print('approx_output::', output/dim[0])
#             
#             
#             gradient = (output- x_sum_by_class)/dim[0] + beta*vectorized_theta
#     #         
#     #         print('x_sum_by_class::', x_sum_by_class/dim[0])
#             
#             
#             
#             
#             vectorized_theta = vectorized_theta - alpha*gradient
        
        
            vectorized_theta = torch.mm(A, vectorized_theta) + B
        
        
    s_power = torch.pow(updated_s, float(max_epoch - cut_off_epoch))
#      
    res1 = M.mul(s_power.view(1,-1))
    
    res1_2 = torch.mm(M_inverse, vectorized_theta)
    
    
#     res2_1 = torch.mm(M_inverse, B)
#     
#     res2_2 = M.mul(s_power.view(1,-1))
#     
#     res2 = B - torch.mm(res2_2, res2_1)
#     
# #     A_float = A.type(torch.float)
#     
#     tmp_inverse = (torch.eye(vectorized_theta.shape[0], dtype = torch.double) - A).numpy()
#     
#     tmp_inverse = np.linalg.inv(tmp_inverse)
#     
#     tmp_inverse = torch.tensor(tmp_inverse, dtype = torch.double)
#     
#     res2 = torch.mm(tmp_inverse, res2)
    
#  
#     res1 = torch.mm(res1, M_inverse)
#     
    sub_sum = (1-s_power)/(1-updated_s)
    
    res2 = torch.mm(M.mul(sub_sum), torch.mm(M_inverse, B))
    
    
    
# # #     
#     res2 = torch.inverse(torch.eye(vectorized_theta.shape, dtype =torch.double) - A)
#     
#     res2_2 = torch.mm(torch.eye(vectorized_theta.shape, dtype =torch.double) - A, B)
#      
#     res2 = torch.mm(res2, M_inverse)
#     
#     vectorized_theta = torch.mm(res1, vectorized_theta) + torch.mm(res2, B)
    vectorized_theta = torch.mm(res1, res1_2) + res2

    theta = torch.t(vectorized_theta.view(num_class, dim[1]))
    
    return theta, updated_s



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

















# def compute_model_parameter_by_approx_incremental_4(term1, term2, x_sum_by_class, dim, theta, num_class, max_epoch, cut_off_epoch, epoch_record_epoch_seq, alpha, beta):
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
# #     for i in range(max_epoch):
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
# 
# 
#         if i < cut_off_epoch:
# 
#             output = torch.mm(term1[i], vectorized_theta) + (term2[i].view(-1,1))
#             
#             
#     #         print('gradient::', gradient)
#     #         
#     #         print('approx_output::', output/dim[0])
#             
#             
#             gradient = (output- x_sum_by_class)/dim[0] + beta*vectorized_theta
#     #         
#     #         print('x_sum_by_class::', x_sum_by_class/dim[0])
#             
#             
#             
#             
#             vectorized_theta = vectorized_theta - alpha*gradient
#         
#         else:
#             
#             output = torch.mm(term1[cut_off_epoch - 1], vectorized_theta) + (term2[cut_off_epoch - 1].view(-1,1))
#             
#             
#     #         print('gradient::', gradient)
#     #         
#     #         print('approx_output::', output/dim[0])
#             
#             
#             gradient = (output- x_sum_by_class)/dim[0] + beta*vectorized_theta
#     #         
#     #         print('x_sum_by_class::', x_sum_by_class/dim[0])
#             
#             
#             
#             
#             vectorized_theta = vectorized_theta - alpha*gradient
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
#     res1 = torch.mm(res1, torch.inverse(M))
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
#     res2 = torch.mm(res2, torch.inverse(M))
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
#         
# #     print('total_time::', total_time)
#     
#     theta = torch.t(vectorized_theta.view(num_class, dim[1]))
#     
#     return theta



def compute_model_parameter_by_approx_incremental_4_2(origin_X, delta_X, weights, theta_list, s, M, M_inverse, X, term1, term2, x_sum_by_class, dim, theta, num_class, max_epoch, cut_off_epoch, sub_weights, alpha, beta, grad_list, expected_sub_term_1):
    
    total_time = 0.0
    
    sub_term_1 = None
    
    for i in range(cut_off_epoch):

        X_multi_theta = torch.mm(X, theta)
        
        
        vectorized_theta = torch.reshape(torch.t(theta), [-1,1])


        sub_term_1 = Variable(torch.t(compute_sub_term_1(X_multi_theta, X, sub_weights[:,i], X_multi_theta.shape, num_class)))
        
        vectorized_sub_term_1 = Variable(torch.reshape(sub_term_1, [-1,1]))
        
        output = Variable(torch.mm((term1[i]), vectorized_theta) - vectorized_sub_term_1 + (term2[i].view(-1,1)))    
        
        gradient = (output- x_sum_by_class)/dim[0] + beta*vectorized_theta     
                    
        vectorized_theta = Variable(vectorized_theta - alpha*gradient)  
        
        theta = Variable(torch.t(vectorized_theta.view(num_class, dim[1])))
        
    
#     A = Variable((1-alpha*beta)*torch.eye(vectorized_theta.shape[0], dtype = torch.double) - alpha*(term1[cut_off_epoch - 1])/dim[0])   

    B = -alpha*(term2[cut_off_epoch - 1].view(-1,1) - x_sum_by_class)/dim[0]    


#     sub_term_1 = prepare_term_1(delta_X, sub_weights[:, cut_off_epoch - 1], delta_X.shape,  num_class)

    delta_s = compute_delta_s(M, M_inverse, delta_X, sub_weights[:,cut_off_epoch - 1], delta_X.shape, num_class)


    updated_s = (1-alpha*beta) - alpha*(s - (delta_s))/dim[0]
    
    s_power = torch.pow(updated_s, float(max_epoch - cut_off_epoch))
#      
    res1 = M.mul(s_power.view(1,-1))
    
    res1_2 = torch.mm(M_inverse, vectorized_theta)
    
    sub_sum = (1-s_power)/(1-updated_s)
    
    res2 = torch.mm(M.mul(sub_sum), torch.mm(M_inverse, B))
    
    vectorized_theta = torch.mm(res1, res1_2) + res2

    theta = torch.t(vectorized_theta.view(num_class, dim[1]))
    
    
    
#     for i in range(max_epoch - cut_off_epoch):
#         
#         X_multi_theta = Variable(torch.mm(X, theta))
#         
#         sub_term_1 = Variable(torch.t(compute_sub_term_1(X_multi_theta, X, sub_weights[:,cut_off_epoch - 1], X_multi_theta.shape, num_class)))
#                 
#                 
#         vectorized_sub_term_1 = Variable(torch.reshape(sub_term_1, [-1,1]))        
#         
#         vectorized_theta = Variable(torch.mm(A, vectorized_theta) + B + vectorized_sub_term_1*alpha/dim[0])
#     
#         theta = Variable(torch.t(vectorized_theta.view(num_class, dim[1])))
        
    return theta, updated_s



def multiply_matrix_power_method(A, B, pow, theta):
    
    total_pow = 0
    
    
    res_A = torch.eye(A.shape[0], dtype = torch.double)
    
    while pow > 0:
        curr_pow = int(np.floor(np.log2(pow)))
        
        print(pow, curr_pow)
        
        pow = pow - np.power(2, curr_pow)
        
        curr_A = A.clone()
        
        for i in range(curr_pow):
            curr_A = torch.mm(curr_A, curr_A)
        
        res_A = torch.mm(res_A, curr_A)
        
    
    res = torch.mm(res_A, theta) + torch.mm(torch.inverse(torch.eye(A.shape[0], dtype = torch.double) - A), torch.mm((torch.eye(A.shape[0], dtype = torch.double) - res_A), B))
    
    
    return res
    




def compute_model_parameter_by_approx_incremental_3(s, M, M_inverse, X, term1, sub_term_1, term2, x_sum_by_class, dim, theta, num_class, max_epoch, cut_off_epoch, sub_weights, alpha, beta):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    vectorized_theta = Variable(theta.view(-1,1))
    
    print('vec_theta_shape::', vectorized_theta.shape[0])
    
    A = Variable((1-alpha*beta)*torch.eye(vectorized_theta.shape[0], dtype = torch.double) - alpha*(term1[cut_off_epoch - 1] - sub_term_1[cut_off_epoch - 1])/dim[0])
#
#     expected_A = (1-alpha*beta)*torch.eye(vectorized_theta.shape[0], dtype = torch.double) - alpha*(term1[cut_off_epoch - 1])/X.shape[0]
# # 
#     updated_s = torch.diag(torch.mm(M_inverse, torch.mm(A, M)))
#     
#     pow_updated_s = updated_s.clone()
#     
#     
#     
#     pow_updated_s[pow_updated_s > 1] = 1
#     
#     tmp = torch.mm(M_inverse, torch.mm(expected_A, M))
#     
#     expected_s = torch.diag(tmp)
#     
#     print(tmp)
# #     
# #     print(torch.norm(expected_A - A))
# #     
# #     print(torch.norm(A))
# #     
#     print(torch.sort(updated_s))
# #     
# #     print(torch.max(updated_s))
# #     
# #     print(torch.min(updated_s))
# #     
#     expected_s_2, M_2 = torch.eig(expected_A, True)
#     
#     expected_s_2 = expected_s_2[:,0]
#     
#     print('expected_s::', torch.sort(expected_s_2))
#     
#     print(torch.norm(M - M_2))
#     
#     print(torch.norm(expected_s_2 - expected_s))
#     
#     print(torch.norm(updated_s - expected_s_2))
# #     
# #     print(expected_s)
# #     
# 
#     print(torch.sort(expected_s))
#     print(torch.norm(expected_s - updated_s))
#     

    B = -alpha*(term2[cut_off_epoch - 1].view(-1,1) - x_sum_by_class)/dim[0]
    
    theta_list = []
    
    gradient_list = []
    
    expected_sub_term_list = []
    
    for i in range(max_epoch):
#     for i in range(cut_off_epoch):
        
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

            output = torch.mm((term1[i] - sub_term_1[i]), vectorized_theta) + (term2[i].view(-1,1))
            
            expected_sub_term_list.append(torch.mm(sub_term_1[i], vectorized_theta))
#             output = torch.mm((term1[i]), vectorized_theta) - vectorized_sub_term_1 + (term2[i].view(-1,1))    
#         
#         gradient = (output- x_sum_by_class)/dim[0] + beta*vectorized_theta     
#                     
#         vectorized_theta = vectorized_theta - alpha*gradient  
            
            X_multi_theta = torch.mm(X, torch.t(vectorized_theta.view(num_class, dim[1])))
             
            curr_sub_term_1 = compute_sub_term_1(X_multi_theta, X, sub_weights[:,i], X_multi_theta.shape, num_class)
             
#             curr_expected_sub_term_1 = prepare_term_1(X, sub_weights[:,i], X.shape, num_class)
             
#             print(torch.norm(torch.t(curr_sub_term_1) - torch.mm(sub_term_1[i], vectorized_theta).view(num_class, dim[1])))
            
            vectorized_curr_sub_term_1 = torch.reshape(torch.t(curr_sub_term_1), [-1,1])
            
            output2 = torch.mm((term1[i]), vectorized_theta) - vectorized_curr_sub_term_1  + (term2[i].view(-1,1))
            
            gradient2 = (output- x_sum_by_class)/dim[0] + beta*vectorized_theta     
                     
            vectorized_theta2 = vectorized_theta - alpha*gradient2 
            
            
    #         print('gradient::', gradient)
    #         
    #         print('approx_output::', output/dim[0])
            
            
            gradient = (output- x_sum_by_class)/dim[0] + beta*vectorized_theta
    #         
#     
#             print(torch.norm(output - output2))
#             
#             print(torch.norm(gradient - gradient2))
    #         print('x_sum_by_class::', x_sum_by_class/dim[0])
            
            
            
            
            vectorized_theta = vectorized_theta - alpha*gradient
            
#             print(torch.norm(vectorized_theta - vectorized_theta2))
            
            theta_list.append(vectorized_theta)
            
            gradient_list.append(gradient)
        
        else:
            
#             output = torch.mm(term1[cut_off_epoch - 1], vectorized_theta) + (term2[cut_off_epoch - 1].view(-1,1))
#             
#             
#     #         print('gradient::', gradient)
#     #         
#     #         print('approx_output::', output/dim[0])
#             
#             
#             gradient = (output- x_sum_by_class)/dim[0] + beta*vectorized_theta
#     #         
#     #         print('x_sum_by_class::', x_sum_by_class/dim[0])
#             
#             
#             
#             
#             vectorized_theta = vectorized_theta - alpha*gradient
        
        
            vectorized_theta = torch.mm(A, vectorized_theta) + B
            
            theta_list.append(vectorized_theta)
        
        
#     s_power = torch.pow(updated_s, double(max_epoch - cut_off_epoch))
# #      
#     res1 = M.mul(s_power.view(1,-1))
#     
#     res1_2 = torch.mm(M_inverse, vectorized_theta)
#     
#     
#     res2_1 = torch.mm(M_inverse, B)
#     
#     res2_2 = M.mul(s_power.view(1,-1))
#     
#     res2 = B - torch.mm(res2_2, res2_1)
#     
# #     A_float = A.type(torch.double)
#     
#     tmp_inverse = (torch.eye(vectorized_theta.shape[0], dtype = torch.double) - A).numpy()
#     
#     tmp_inverse = np.linalg.inv(tmp_inverse)
#     
#     tmp_inverse = torch.tensor(tmp_inverse, dtype = torch.double)
#     
#     res2 = torch.mm(tmp_inverse, res2)
    
#  
#     res1 = torch.mm(res1, M_inverse)
#     
#     sub_sum = (1-s_power)/(1-pow_updated_s)
# # #     
#     res2 = torch.inverse(torch.eye(vectorized_theta.shape, dtype =torch.double) - A)
#     
#     res2_2 = torch.mm(torch.eye(vectorized_theta.shape, dtype =torch.double) - A, B)
#      
#     res2 = torch.mm(res2, M_inverse)
#     
#     vectorized_theta = torch.mm(res1, vectorized_theta) + torch.mm(res2, B)
#     vectorized_theta = torch.mm(res1, res1_2) + res2

    theta = torch.t(vectorized_theta.view(num_class, dim[1]))
    
    return theta, A, B, theta_list, gradient_list, expected_sub_term_list



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
def compute_sub_term_1(X_times_theta, X, weights, dim, num_class):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    '''X_times_theta: n*q'''
    
    
#     w_dim = weights.shape
    
#     print(w_dim)
#     
#     print(dim)
    
   
    '''dim[0]*dim[1]*(num_class)'''
    
#     t1 = time.time()
    
    res1 = torch.bmm(X_times_theta.view(dim[0], 1, dim[1]), weights.view(dim[0], num_class, num_class))
    
    '''dim[1],num_class, num_class*num_class'''
    res2 = Variable(torch.mm(torch.t(X), res1.view(dim[0], num_class)).view(X.shape[1], num_class))
    
    del res1
    
#     del res1
    
#     res3 = torch.transpose(torch.transpose(res2, 0, 1), 2, 1).view(w_dim[1], num_class, num_class, dim[1], dim[1])
#     res3 = Variable(torch.sum(res2, 1))
    
#     del res2
    
#     res4 = torch.reshape(torch.transpose(res3, 2, 3), [w_dim[1], num_class*dim[1], dim[1]*num_class])
#         
#     del res3
    
#     t2 = time.time()

    
#     print('time::', t2 - t1)
 
    
    
    
    return res2

def prepare_term_1(X, weights, dim, num_class):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    
    
    
#     w_dim = weights.shape
    
    '''dim[0]*dim[1]*(max_epoch*num_class*num_class)'''
        
    res1 = torch.bmm(X.view(dim[0], dim[1], 1), weights.view(dim[0], 1, num_class*num_class))
    
    '''dim[1],dim[1]*t*num_class*num_class'''
    res2 = torch.mm(torch.t(X), res1.view(dim[0], dim[1]*num_class*num_class)).view(dim[1]*dim[1], num_class*num_class)
    
    del res1
    
    res3 = torch.reshape(torch.t(res2), [num_class, num_class, dim[1], dim[1]])
    
    del res2
    
    res4 = torch.reshape(torch.transpose(res3, 1, 2), [num_class*dim[1], dim[1]*num_class])
        
    del res3
    
    return res4


def compute_delta_s(M, M_inverse, X, weights, dim, num_class):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    
    
    
#     w_dim = weights.shape
    
    '''dim[0]*dim[1]*(max_epoch*num_class*num_class)'''
        
        
    '''to be figured out'''    
#     if dim[1]*num_class > dim[0]:
# 
#         '''(m*q, m*q) * (m, k) -> (mq^2, k), overhead::k(mq)^2'''
#     
#         M_inverse_times_X = torch.mm(M_inverse.view(num_class, dim[1], num_class, dim[1]).view(num_class*dim[1]*num_class, dim[1]), torch.t(X))
#             
#         '''(|delta_X|, dim[1]) * (dim[1], dim[1]) -> (|delta_X|, dim[1]), overhead::|delta_X| dim[1]^2, km^2'''    
#         M_times_X = torch.mm(X, M.view(num_class, dim[1], num_class, dim[1]).view())
#         
#         
#         '''res1: n*mq^2'''
#         res1 = torch.bmm(M_times_X.view(dim[0], dim[1], 1), weights.view(dim[0], 1, num_class*num_class))
#         
#         '''n(mq)^2'''
#         res2 = torch.mm(M_inverse_times_X, res1.view(dim[0], dim[1]*num_class*num_class)).view(dim[1]*dim[1], num_class*num_class)
#         
#         del res1
#         
#         res3 = torch.reshape(torch.t(res2), [num_class, num_class, dim[1], dim[1]])
#         
#         del res2
#         
#         res4 = torch.reshape(torch.transpose(res3, 1, 2), [num_class*dim[1], dim[1]*num_class])
#             
#         del res3
#         
#         return res4
# 
#     else:
#         M_inverse_times_X = torch.mm(M_inverse, torch.t(X))
            
    '''(|delta_X|, dim[1]) * (dim[1], dim[1]) -> (|delta_X|, dim[1]), overhead::|delta_X| dim[1]^2, km^2'''    
#         M_times_X = torch.mm(X, M)
    
    
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


def prepare_term_1_mini_batch(X, weights, dim, num_class, batch_size):
     
    '''weights: dim[0], max_epoch, num_class, num_class''' 
    
    w_dim = weights.shape
    
    batch_num = int(X.shape[0]/batch_size)
    
#     batch_num = 1
    
    term_1 = Variable(torch.zeros([w_dim[1], num_class*dim[1], dim[1]*num_class], dtype = torch.double))
    
    print(batch_num)
    
    for i in range(batch_num):
         
         
        print(i)
         
        X_product_subset = Variable(torch.bmm(X[i*batch_size:(i+1)*batch_size].view(batch_size, dim[1], 1), X[i*batch_size:(i+1)*batch_size].view(batch_size, 1, dim[1])))
         
         
        X_product1 = torch.t(X_product_subset.view(batch_size, dim[1]*dim[1]))
     
#         del X_product_subset
         
        res1 = Variable(torch.mm(X_product1, torch.reshape(weights[i*batch_size:(i+1)*batch_size], (batch_size, w_dim[1]*num_class*num_class))))
         
        del X_product1
         
        res2 = torch.transpose(torch.transpose((res1.view(dim[1]*dim[1], w_dim[1], num_class*num_class)), 1, 0), 1, 2)
     
        del res1
         
        res3 = res2.view(w_dim[1], num_class, num_class, dim[1], dim[1])
     
        del res2
    #     res = torch.transpose(res, 1, 2)
         
        res4 = torch.transpose(res3, 2, 3)
         
        del res3
         
#         print(res4.shape)
         
         
#         X_product_subset = torch.bmm(X[0:2*batch_size].view(2*batch_size, dim[1], 1), X[0:2*batch_size].view(2*batch_size, 1, dim[1]))

#         term1_0_0 = prepare_term_1_batch2(X_product_subset, weights[i*batch_size:(i+1)*batch_size].view(batch_size, weights.shape[1] , weights.shape[2], weights.shape[3]), X_product_subset.shape, epoch, num_class)
# 
        res = torch.reshape(res4, [w_dim[1], num_class*dim[1], dim[1]*num_class])
         
         
        term_1 += res 
        
        
#         X_product_subset = Variable(torch.bmm(X[0:(i+1)*batch_size].view((i+1)*batch_size, dim[1], 1), X[0:(i+1)*batch_size].view((i+1)*batch_size, 1, dim[1])))
#      
# #     term1_0_0 = prepare_term_1_batch2(X_product_subset, weights[i*batch_size:(i+1)*batch_size].view(batch_size, weights.shape[1] , weights.shape[2], weights.shape[3]), X_product_subset.shape, epoch, num_class) 
#     
#         term1_0 = prepare_term_1_batch2(X_product_subset, weights[0:(i+1)*batch_size].view((i+1)*batch_size, weights.shape[1] , weights.shape[2], weights.shape[3]), X_product_subset.shape, epoch, num_class) 
# 
#         
#         print(torch.norm(term1_0 - term_1))
        
        
        
        del res4, res
         
#     
    if batch_num*batch_size < dim[0]:
        curr_batch_size = dim[0] - batch_num*batch_size
         
#         X_product_subset = X_product[batch_num*batch_size:dim[0], :]
         
        X_product_subset = torch.bmm(X[batch_num*batch_size:dim[0]].view(curr_batch_size, dim[1], 1), X[batch_num*batch_size:dim[0]].view(curr_batch_size, 1, dim[1])) 
        
        X_product1 = torch.t(X_product_subset.view(curr_batch_size, dim[1]*dim[1])) 
        
        res1 = Variable(torch.mm(X_product1, torch.reshape(weights[batch_num*batch_size:dim[0]], (curr_batch_size, w_dim[1]*num_class*num_class))))
         
        del X_product1, X_product_subset
         
        res2 = torch.transpose(torch.transpose((res1.view(dim[1]*dim[1], w_dim[1], num_class*num_class)), 1, 0), 1, 2)
     
        del res1
         
        res3 = res2.view(w_dim[1], num_class, num_class, dim[1], dim[1])
     
        del res2
    #     res = torch.transpose(res, 1, 2)
         
        res4 = torch.transpose(res3, 2, 3)
         
        del res3
         
#         print(res4.shape)
         
        res = torch.reshape(res4, [w_dim[1], num_class*dim[1], dim[1]*num_class])
         
        term_1 += res 
        
        del res4, res
    
    
    return term_1

def compute_model_parameter_by_approx_incremental_3_2(origin_X, delta_X, weights, theta_list, s, M, M_inverse, X, term1, term2, x_sum_by_class, dim, theta, num_class, max_epoch, cut_off_epoch, sub_weights, alpha, beta, grad_list, expected_sub_term_1):
    
    total_time = 0.0
    
#     vectorized_theta = Variable(torch.t(theta).view(-1,1))
    
#     print('vec_theta_shape::', vectorized_theta.shape[0])
    
    sub_term_1 = None
    
    for i in range(cut_off_epoch):

        X_multi_theta = torch.mm(X, theta)
        
        
        vectorized_theta = torch.reshape(torch.t(theta), [-1,1])
#         vectorized_theta = Variable(torch.reshape(theta, [-1,1]))


#         output1 = softmax_layer(torch.mm(update_X, theta))
#         
#         
#         output1 = torch.mm(torch.t(update_X), output1)
#         
#         
#         output1 = torch.reshape(torch.t(output1), [-1,1])



        sub_term_1 = Variable(torch.t(compute_sub_term_1(X_multi_theta, X, sub_weights[:,i], X_multi_theta.shape, num_class)))
        
        
#         computed_output0 = compute_sub_term_1(torch.mm(origin_X, theta), origin_X, weights[:,i], torch.mm(origin_X, theta).shape, num_class)
#         
#         
#         computed_output1 = computed_output0 + torch.t(term2[i].view(num_class, X.shape[1]))
#         
#         computed_output2 = torch.reshape(torch.t(computed_output1), [-1,1])
#         
#         
#         expected_term_1 = prepare_term_1(origin_X, weights[:,i], origin_X.shape, num_class)
        
#         expected_sub_term_1 = prepare_term_1(X, sub_weights[:,i], X.shape, num_class).view(num_class*X.shape[1], num_class*X.shape[1])
# #          
#         expected_sub_term_1 = torch.mm(expected_sub_term_1, vectorized_theta)
        
#          
#         expected_term_1_2 = prepare_term_1_mini_batch(origin_X, weights[:, i].view(weights.shape[0], 1, weights.shape[2], weights.shape[3]), origin_X.shape, num_class, origin_X.shape[0])
        
#         print(sub_term_1)
        vectorized_sub_term_1 = Variable(torch.reshape(sub_term_1, [-1,1]))
        
        output = Variable(torch.mm((term1[i]), vectorized_theta) - vectorized_sub_term_1 + (term2[i].view(-1,1)))    
        
        gradient = (output- x_sum_by_class)/dim[0] + beta*vectorized_theta     
                    
        vectorized_theta = Variable(vectorized_theta - alpha*gradient)  
        
#         expected_theta = theta_list[i]
#         
#         expected_gradient = gradient_list[i]
#         
#         expected_curr_sub_term_1 = expected_sub_term_1[i]
#         
#         print(torch.norm(expected_curr_sub_term_1 - torch.reshape(sub_term_1, [-1,1])))
#         
#         print(torch.norm(gradient - expected_gradient))
#         
#         print(torch.norm(expected_theta - vectorized_theta))

#         expected_theta = theta_list[i]
#         
#         expected_grad = grad_list[i]
#         
#         print('grad_diff::', torch.norm(gradient - expected_grad))
        
        theta = Variable(torch.t(vectorized_theta.view(num_class, dim[1])))
        
#         print(torch.norm(expected_theta - vectorized_theta))
        
        
    
#     print(sub_weights[:cut_off_epoch - 1].shape)
    
    sub_term_1 = prepare_term_1(delta_X, sub_weights[:, cut_off_epoch - 1], delta_X.shape, num_class)

    
    A = Variable((1-alpha*beta)*torch.eye(vectorized_theta.shape[0], dtype = torch.double) - alpha*(term1[cut_off_epoch - 1] - sub_term_1)/dim[0])   

    B = -alpha*(term2[cut_off_epoch - 1].view(-1,1) - x_sum_by_class)/dim[0]    



    for i in range(max_epoch - cut_off_epoch):
        
#         X_multi_theta = Variable(torch.mm(X, theta))
        
#         vectorized_theta = Variable(theta.view(-1,1))
        
#         sub_term_1 = Variable(torch.t(compute_sub_term_1(X_multi_theta, X, sub_weights[:,cut_off_epoch - 1], X_multi_theta.shape, num_class)))
                
                
#         vectorized_sub_term_1 = Variable(torch.reshape(sub_term_1, [-1,1]))        
        
        vectorized_theta = torch.mm(A, vectorized_theta) + B
    
        theta = Variable(torch.t(vectorized_theta.view(num_class, dim[1])))
        
#         expected_theta = theta_list[i + cut_off_epoch]
        
#         print(torch.norm(expected_theta - vectorized_theta))
        
#     theta = torch.t(vectorized_theta.view(num_class, dim[1]))
    
    return theta, A, B




 
def compute_model_parameter_by_approx_incremental_5(term1, term2, x_sum_by_class, dim, theta, num_class, max_epoch, cut_off_epoch, epoch_record_epoch_seq, alpha, beta):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    vectorized_theta = theta.view(-1,1)
    
    
    A = (1-alpha*beta)*torch.eye(vectorized_theta.shape[0], dtype = torch.double) - alpha*term1[cut_off_epoch - 1]/dim[0]
#     
    B = -alpha*(term2[cut_off_epoch - 1].view(-1,1) - x_sum_by_class)/dim[0]
    
    for i in range(cut_off_epoch):
        output = torch.mm(term1[i], vectorized_theta) + (term2[i].view(-1,1))
            
            
    #         print('gradient::', gradient)
    #         
    #         print('approx_output::', output/dim[0])
            
            
        gradient = (output- x_sum_by_class)/dim[0] + beta*vectorized_theta
#         
#         print('x_sum_by_class::', x_sum_by_class/dim[0])
        
        
        
        
        vectorized_theta = vectorized_theta - alpha*gradient
        
        
    vectorized_theta = multiply_matrix_power_method(A, B, max_epoch - cut_off_epoch, vectorized_theta)
    
#     for i in range(max_epoch):

    
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


def eigen_decomposition(term1, cut_off_epoch, dim, num_class):
    
    
#     A = (1-alpha*beta)*torch.eye(dim[1]*num_class, dtype = torch.double) - alpha*(term1[cut_off_epoch - 1])/dim[0]
    
    
#     A = A.type(torch.FloatTensor)
    
    s, M = torch.eig(term1[cut_off_epoch - 1], True)
        
    s = s[:,0]
    
    print('eigen_values::', s)
        
    torch.save(M, git_ignore_folder + 'eigen_vectors')
    
    M_inverse = torch.tensor(np.linalg.inv(M.numpy()), dtype = torch.double)
    
#     M_inverse = torch.inverse(M)
    
    
    print('inverse_gap::', torch.norm(torch.mm(M, M_inverse) - torch.eye(dim[1]*num_class, dtype = torch.double)))
    
    torch.save(M_inverse, git_ignore_folder + 'eigen_vectors_inverse')
    
    torch.save(s, git_ignore_folder + 'eigen_values')
    
#     torch.save(A, git_ignore_folder + 'expected_A')
        
        
    


def capture_provenance(X, Y, dim, epoch, num_class):
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
 
 
    # term1, term1_inter_result = prepare_term_1_serial(X, w_seq, dim)
    # term1 = prepare_term_1_batch2(X_product, weights, dim, max_epoch, num_class)
     
#     X_theta_prod_softmax_seq_tensor = torch.stack(X_theta_prod_softmax_seq, dim = 0)
#      
#     X_theta_prod_seq_tensor = torch.stack(X_theta_prod_seq, dim = 0)
    
    
#     torch.save(X_theta_prod_seq_tensor, git_ignore_folder + 'X_theta_prod_seq_tensor')
#     
#     torch.save(X_theta_prod_softmax_seq_tensor, git_ignore_folder + 'X_theta_prod_softmax_seq_tensor')
    mini_epochs_per_super_iteration = int((dim[0] - 1)/batch_size) + 1

    global X_theta_prod_softmax_seq, X_theta_prod_seq
    
    
    super_iteration = (int((len(X_theta_prod_softmax_seq) - 1)/mini_epochs_per_super_iteration) + 1)
    
    
    cut_off_super_iteration = int(super_iteration*theta_record_threshold)#(int((cut_off_epoch - 1)/mini_epochs_per_super_iteration) + 1)
    
    
    cut_off_epoch = cut_off_super_iteration*mini_epochs_per_super_iteration
    
#     cut_off_epoch = len(X_theta_prod_softmax_seq)
    
    print('super_iteration::', super_iteration)
    print('cut_off_super_iteration::', cut_off_super_iteration)

    print('cut_off_epoch::', cut_off_epoch)
    
    weights, offsets = prepare_term_1_batch3_0(X_theta_prod_softmax_seq, X_theta_prod_seq, X, dim, epoch, num_class, cut_off_epoch) 

    

    print('compute_weights_offsets_done!!')

    print(weights.shape)

#     term1 = prepare_term_1_batch2_0(X, weights, dim, epoch, num_class)


    
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
#     
#     
#     X_product_subset = Variable(torch.bmm(X[0:2*batch_size].view(2*batch_size, dim[1], 1), X[0:2*batch_size].view(2*batch_size, 1, dim[1])))
#      
# #     term1_0_0 = prepare_term_1_batch2(X_product_subset, weights[i*batch_size:(i+1)*batch_size].view(batch_size, weights.shape[1] , weights.shape[2], weights.shape[3]), X_product_subset.shape, epoch, num_class) 
#     
#     term1_0_2 = prepare_term_1_batch2(X_product_subset, weights[0:2*batch_size].view(2*batch_size, weights.shape[1] , weights.shape[2], weights.shape[3]), X_product_subset.shape, epoch, num_class) 
# 
#     
#     X_product_subset = Variable(torch.bmm(X[0:batch_size].view(batch_size, dim[1], 1), X[0:batch_size].view(batch_size, 1, dim[1])))
#      
# #     term1_0_0 = prepare_term_1_batch2(X_product_subset, weights[i*batch_size:(i+1)*batch_size].view(batch_size, weights.shape[1] , weights.shape[2], weights.shape[3]), X_product_subset.shape, epoch, num_class) 
#     
#     term1_0 = prepare_term_1_batch2(X_product_subset, weights[0:batch_size].view(batch_size, weights.shape[1] , weights.shape[2], weights.shape[3]), X_product_subset.shape, epoch, num_class) 
# 
#     
#     X_product_subset = Variable(torch.bmm(X[batch_size:2*batch_size].view(batch_size, dim[1], 1), X[batch_size:2*batch_size].view(batch_size, 1, dim[1])))
#      
# #     term1_0_0 = prepare_term_1_batch2(X_product_subset, weights[i*batch_size:(i+1)*batch_size].view(batch_size, weights.shape[1] , weights.shape[2], weights.shape[3]), X_product_subset.shape, epoch, num_class) 
#     
#     term1_1 = prepare_term_1_batch2(X_product_subset, weights[batch_size:2*batch_size].view(batch_size, weights.shape[1] , weights.shape[2], weights.shape[3]), X_product_subset.shape, epoch, num_class) 
# 
#     print(term1_1 + term1_0 - term1_0_2)
#     
#     
#     
#     term1_0 = prepare_term_1_batch2(X_product, weights[:,0].view(weights.shape[0], 1,weights.shape[2], weights.shape[3]), dim, epoch, num_class)
    term1 = prepare_term_1_mini_batch(X, weights, dim, num_class, batch_size)
    
    eigen_decomposition(term1, cut_off_epoch, dim, num_class)
    
    torch.save(weights, git_ignore_folder+'weights')
    
    torch.save(term1, git_ignore_folder+'term1')
    
#     torch.save(X_product, git_ignore_folder + 'X_product')
    
    torch.save(cut_off_epoch, git_ignore_folder + 'cut_off_epoch')
    
    del weights
    
    del term1
    
    print('save weights and term 1 done!!!')
    
    
    
    term2 = prepare_term_2_batch2(X, offsets, dim, epoch, num_class)
     
    torch.save(offsets, git_ignore_folder+'offsets')
    
    torch.save(term2, git_ignore_folder+'term2') 
     
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
    
    return Hessian_inverse


def precomptation_influence_function2(X, Y, res, dim):
    
    t5 = time.time()
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     X_Y_mult = X.mul(Y)
    
#     Hessin_matrix = compute_hessian_matrix(X, X_Y_mult, res, dim, X_product)


    Hessin_matrix = compute_hessian_matrix_3(res, X, dim, num_class, 1000)
    
    print("compute 1 done!!!")
    
    Hessin_matrix2 = compute_hessian_matrix2(X, Y, res, dim, beta)
    
    print('hessian diff:', torch.norm(Hessin_matrix - Hessin_matrix2))
    
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

def random_deletion(X, Y, delta_num, res, num_class):
    
    multi_res = softmax_layer(torch.mm(X, res))
    
    prob, predict_labels = torch.max(multi_res, 1)
    
    changed_values, changed_label = torch.max(-multi_res, 1)
    
    
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
#             Y[indices[i]] = (Y[indices[i]] + 1)%num_class
            Y[indices[i]] = changed_label[indices[i]]
            delta_id_array.append(indices[i])
    
    
        i = i + 1
    
    delta_data_ids = torch.tensor(delta_id_array, dtype = torch.long)
    
#     print(delta_data_ids[:100])
#     delta_data_ids = random_generate_subset_ids(X.shape, delta_num)     
#     torch.save(delta_data_ids, git_ignore_folder+'delta_data_ids')
    return X, Y, delta_data_ids    



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
        
        curr_coeff = torch.mean(curr_coeff[curr_coeff == curr_coeff])
         
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
    
    return X, Y, torch.tensor(list(delta_data_ids))

if __name__ == '__main__':
    
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    sys_args = sys.argv
    
    file_name = sys_args[1]
    
#     print(file_name)
    
    start = bool(int(sys_args[2]))
    
    alpha = float(sys_args[3])
    
    beta = float(sys_args[4])
    
    threshold = float(sys_args[5])
    
    max_epoch = int(sys_args[6])
    
#     global alpha, beta, threshold
    noise_rate = float(sys_args[7])
    
    random_deletion_or_not = bool(int(sys_args[8]))
    
    add_noise_or_not = bool(int(sys_args[9]))
    
    theta_record_threshold = float(sys_args[10])
    
    
    repetition = 1
    
    if start:
    
        [X, Y, test_X, test_Y] = load_data_multi_classes(True, file_name)

#         [X, Y, test_X, test_Y] = load_data_multi_classes_rcv1()
    
#         [X, Y, test_X, test_Y] = clean_sensor_data(file_name)
        
        
        Y = Y.type(torch.LongTensor)
        
        X = extended_by_constant_terms(X, False)
        
        test_X = extended_by_constant_terms(test_X, False)
        
        dim = X.shape
    
        print('X_shape::', dim)
        
        print(torch.unique(Y))
        
        num_class = torch.unique(Y).shape[0]
        
        print('num_class::', num_class)


    
    
        lr = initialize(X, num_class)
        res1, epoch = compute_parameters(X, Y, lr, dim, num_class, False)
         
        torch.save(res1, git_ignore_folder + 'model_without_noise')
        
        torch.save(X, git_ignore_folder+'X')
        
        torch.save(Y, git_ignore_folder+'Y')
        
        torch.save(test_X, git_ignore_folder+'test_X')
        
        torch.save(test_Y, git_ignore_folder+'test_Y')        
    #     res1 = torch.load(git_ignore_folder + 'model_without_noise')
        
        print('train_accuracy::', compute_accuracy2(X, Y.type(torch.DoubleTensor), res1))
        
        print('test_accuracy::', compute_accuracy2(test_X, test_Y, res1))
    
    
        
        
        
    #     X, Y, selected_ids = change_data_labels(X, Y, 0.6, num_class)
    #     torch.save(selected_ids, git_ignore_folder + 'noise_data_ids')
    
    else:
        res1 = torch.load(git_ignore_folder + 'model_without_noise')
        
        X = torch.load(git_ignore_folder + 'X')
        
        Y = torch.load(git_ignore_folder + 'Y')
        
        test_X = torch.load(git_ignore_folder + 'test_X')
        
        test_Y = torch.load(git_ignore_folder + 'test_Y')
        
        dim = X.shape
        
        num_class = torch.unique(Y).shape[0]
        
        if random_deletion_or_not:
            X, Y, noise_data_ids = random_deletion(X, Y, int(X.shape[0]*noise_rate), res1, num_class)
        else:
            if add_noise_or_not:
                X, Y = add_noise_data(X, Y, int(X.shape[0]*noise_rate), res1, num_class)
                noise_data_ids = torch.tensor(list(set(range(X.shape[0])) - set(range(dim[0]))))
    #         X, Y, noise_data_ids = change_data_values(X, Y, int(X.shape[0]*0.0005), res1, torch.unique(Y))
    #         X, Y, noise_data_ids = rescale_data(X, Y, 0.3, res1)
    #         X, Y, noise_data_ids = change_data_labels(X, Y, 0.3, num_class, res1)
            
            else:
                X, Y, noise_data_ids = change_data_values(X, Y, int(X.shape[0]*noise_rate), res1, torch.unique(Y))
        
        dim = X.shape
        
        print(noise_data_ids)
        
        print('noise_data_id_size::', noise_data_ids.shape)
        
        torch.save(noise_data_ids, git_ignore_folder + 'noise_data_ids')
        
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        t1  = time.time()
        
        epoch = 0
        
    #     for i in range(repetition):
        lr = initialize(X, num_class)
        res2, epoch = compute_parameters(X, Y, lr, dim, num_class, True)
        
        t2 = time.time()
    
    
        x_sum_by_class = compute_x_sum_by_class(X, Y, num_class, dim)
        
        torch.save(X, git_ignore_folder + 'noise_X')
        
        torch.save(Y, git_ignore_folder + 'noise_Y')
        
        
        torch.save(x_sum_by_class, git_ignore_folder+'x_sum_by_class')
        
        torch.save(torch.tensor(epoch), git_ignore_folder+'epoch')
        
        torch.save(res2, git_ignore_folder+'model_origin')
        
        torch.save(torch.tensor(epoch_record_epoch_seq), git_ignore_folder + 'epoch_record_epoch_seq')
        
        torch.save(alpha, git_ignore_folder + 'alpha')
        
        torch.save(beta, git_ignore_folder + 'beta')
        
        
        print('epoch::', epoch)
    
        print('recorded_seq_size::', epoch_record_epoch_seq)
        
        print('gap::', (torch.dot(res1.view(-1), res2.view(-1))/(torch.norm(res1.view(-1))*torch.norm(res2.view(-1)) )))
        
        t3 = time.time()
        
        del x_sum_by_class
        
        capture_provenance(X, Y, dim, epoch, num_class)
        
        
#         h_inverse1 = precomptation_influence_function(X, Y, res2, dim)
        
        precomptation_influence_function2(X, Y, res2, dim)
        
#         print('h_inverse_diff::', torch.norm(h_inverse1 - h_inverse2))
        
    #     torch.save(X.mul(Y.type(torch.DoubleTensor)), git_ignore_folder + 'X_Y_mult')
        
        t4 = time.time()
        
        
        print('training_time::', t2 - t1)
        
        print('preparing_time::', t4 - t3)
        
        print('training_accuracy::', compute_accuracy2(X, Y.type(torch.DoubleTensor), res2))
        
        print('test_accuracy::', compute_accuracy2(test_X, test_Y, res2))
    
    
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
