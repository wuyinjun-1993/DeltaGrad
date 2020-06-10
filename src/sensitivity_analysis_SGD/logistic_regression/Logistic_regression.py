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
import scipy
from sklearn.utils.extmath import randomized_svd


# for x in os.listdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))):
#     if os.path.isdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/' + x):
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))




try:
    from Interpolation.piecewise_linear_interpolation_2D import *
    from sensitivity_analysis_SGD.logistic_regression.incremental_updates_logistic_regression import *
    from data_IO.Load_data import *
    from sensitivity_analysis_SGD.logistic_regression.evaluating_test_samples import *
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

svd_ratio = 12
  
alpha = 0.0001
  
beta = 0.05


batch_size = 1


threshold = 1e-4

prov_record_rate = 0.0005


res_prod_seq = []


random_ids_multi_super_iterations = []

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
    return 1-1.0/ (1 +np.exp(-x))

def sigmoid_np(x):
    return 1.0/ (1 +np.exp(-x))


def non_linear_function(x):
    return 1-1.0/(1 + torch.exp(-x))

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


    second_derivative = second_derivative/dim[0] + beta*torch.eye(dim[1], dtype = torch.float64)
    
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

def compute_hessian_matrix_4_sparse(X, X_Y_mult, theta, dim):
    X_Y_theta_prod = X_Y_mult.dot(theta)
    
    second_derivative = np.power(sigmoid_np(X_Y_theta_prod),2)*np.exp(-X_Y_theta_prod)
    
    
    second_deriv = X.multiply(np.reshape(second_derivative, (X.shape[0], 1)))
    
    print(second_deriv.shape)
    
    print(X.shape)
    
    second_derivative = X.transpose().dot(second_deriv)
    
    print(second_derivative.shape)
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
    
    res = torch.zeros([dim[1], dim[1]], dtype = torch.float64)
    
    for i in range(X.shape[0]):
        res += second_derivative_loss_function(-X_Y_theta_prod[i])*X_product[i]
        
    res = res/dim[0] + beta*torch.eye(dim[1], dtype = torch.float64)
    
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
    
def loss_function1(X, Y, theta, dim, beta):
    
#     res = 0
    
    
#     sigmoid_res = torch.stack(list(map(bia_function, Y*torch.mm(X, theta))))

#     sigmoid_res = Y*torch.mm(X, theta)
#     data_trans = sigmoid_res.apply(lambda x :  ())

#     sigmoid_res = -log_sigmoid_layer(Y*torch.mm(X, theta))
    
    res = torch.sum(-log_sigmoid_layer(Y*torch.mm(X, theta)))/dim
    
    
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


def loss_function2(Y, Y_expect, dim):
    
#     Y_prod = Y.mult(Y_expect)
    
    
#     res = 0
    
    
#     sigmoid_res = torch.stack(list(map(bia_function, Y*torch.mm(X, theta))))

#     sigmoid_res = Y*torch.mm(X, theta)
#     data_trans = sigmoid_res.apply(lambda x :  ())

#     sigmoid_res = -log_sigmoid_layer(Y*torch.mm(X, theta))
    
    res = torch.sum(-log_sigmoid_layer(Y*Y_expect))/dim[0]
    
    
#     for i in range(dim[0]):
# #         print(X[i,:])
# #         print(theta)
# #         print(X[i,:].view(dim[1]))
# #         print(theta.view(dim[1]))
#         res += torch.log(1 + torch.exp(-Y[i]*torch.dot(X[i,:].view(dim[1]), theta.view(dim[1]))))
#         res = res - (Y[i]*torch.log(sigmoid_function(X[i,:].view(dim[1]), theta.view(dim[1]))) + (1 - Y[i])*torch.log(1 - sigmoid_function(X[i,:].view(dim[1]), theta.view(dim[1])))) 
        
#     res = res
    
    return res
    
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


class lr_model(torch.nn.Module):
    def __init__(self, D_in):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(lr_model, self).__init__()
        self.linear = torch.nn.Linear(D_in, 1, bias= False)
        self.linear.weight.data.fill_(0.0)
#         self.linear.bias.data.fill_(0.0)
        self.linear.to(dtype=torch.double)
#         self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        y_pred = (self.linear(x))
        
#         h_relu = self.linear1(x).clamp(min=0)
#         y_pred = self.linear2(h_relu)
        return y_pred


def format_model_parameters(paras, size):
    
    
    formatted_model_parameters = torch.zeros(size, dtype = torch.double)
    
    i = 0
    
    for para in paras:
        formatted_model_parameters[i:para.data.view(-1).shape[0] + i] = para.data.view(-1)
        i = para.data.shape[0] + i
        
    return formatted_model_parameters
    
def format_model_gradient(paras, size):
    formatted_model_parameters = torch.zeros(size, dtype = torch.double)
    
    i = 0
    
    for para in paras:
        formatted_model_parameters[i:para.grad.view(-1).shape[0] + i] = para.grad.view(-1)
        i = para.grad.shape[0] + i
        
    return formatted_model_parameters



def logistic_regression(origin_X, origin_Y, lr, dim, tracking_or_not):

#     dim = X.shape
    
    epoch = 0
    
#     last_theta = None
    
    
    mini_batch_epoch = 0
    
#     batch_size = 100
    
    
#     model = lr_model(dim[1])
#     
#     
#     optimizer = torch.optim.SGD(model.parameters(), lr=alpha, weight_decay=2*beta)
    
    theta_list = []
    
    grad_list = []
    
    while epoch < max_epoch:
        
#         for para in model.parameters():
#                 print(para.data)

#         permutation = torch.randperm(X.size()[0])

        print('epoch::', epoch)
#         gap_to_be_averaged = []
#         permutation = list(range(origin_X.shape[0]))
# 
#         grad_vector = []
        
        end = False
        
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
            
            
            if tracking_or_not:
                
                
                '''batch_size*1'''
                
                res_prod = torch.mm(batch_x.mul(batch_y).view(-1, dim[1]), lr.theta)
                
                '''batch_size*(t*(n/batch_size))'''
                global res_prod_seq
                  
                res_prod_seq.append(res_prod.view(-1))  
                  
                
#                 if res_prod_seq.shape == 0:
#                     res_prod_seq = res_prod
#         #             res_prod_seq.append(lr.theta.clone())
#                 else:
#                     
#                     print(res_prod.shape)
#                     
#                     res_prod_seq = torch.cat((res_prod_seq, res_prod), 1)

            
            
            # in case you wanted a semi-full example
#             outputs = model.forward(batch_x)
#             loss = loss_function2(outputs,batch_y, dim)
            
            loss = loss_function1(batch_x, batch_y, lr.theta, batch_x.shape[0], beta)
    
#             optimizer.zero_grad()
            loss.backward()
            
            with torch.no_grad():
#                 theta_list.append(lr.theta.clone())

                lr.theta -= alpha * lr.theta.grad
                
#                 grad_list.append(lr.theta.grad.clone())
                
                gap = torch.norm(lr.theta.grad)
#                 print('id::', i)
#                 
#                 print('gradient::', lr.theta.grad.view(-1))
#                 
#                 print('updated_weight::', lr.theta.view(-1))
                
                lr.theta.grad.zero_()
#             optimizer.step()
            
#             updated_para = format_model_parameters(model.parameters(), dim[1] + 1)
#             
#             gradient = format_model_gradient(model.parameters(), dim[1] + 1)
            
#             for para in model.parameters():
#                 print(para.data)
#                 print('gradient::', para.grad)
#                 
#                 grad_vector.append(para.grad)
            
            
            
#             if last_theta is not None:
            print(mini_batch_epoch, gap)
             
#             if last_theta is not None:
                 
#                 gap = torch.norm(last_theta - lr.theta)
                 
            if gap < threshold:
                end = True
                break
           
           
#                 if gap < threshold*5:
#                     tracking_or_not = False
           
#             if tracking_or_not:
#                 gap_to_be_averaged.append(gap.item())
#                
#                
# #                 if len(gap_to_be_averaged) >= (X.shape[0] - 1)/batch_size:
#                 if end_id == X.shape[0]:
#                     
#                     average_gap = np.sum(gap_to_be_averaged)/len(gap_to_be_averaged)
#                     
#                     
#                     print('avg_gap::', average_gap, len(gap_to_be_averaged))
#                     
#                     if average_gap < prov_record_rate:
#                         tracking_or_not = False    
# 
#                 
#                     del gap_to_be_averaged
#                     gap_to_be_averaged.pop(0)

                
            
#             last_theta = lr.theta.clone()
            
            epoch = epoch + 1
            
            mini_batch_epoch += 1
            
            if epoch >= max_epoch:
                end = True
                break
            
#             mini_batch_epoch = mini_batch_epoch + 1
            
            
#         epoch = epoch + 1
        del X
        
        del Y
        
#         del random_ids
        
        if end:
            break
    
#             loss.backward()
#             optimizer.step()
        
        
#     return format_model_parameters(model.parameters, dim[1] + 1), epoch

    return lr.theta, epoch, mini_batch_epoch, theta_list, grad_list
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#         if tracking_or_not:
#             global res_prod_seq
#              
#             if res_prod_seq.shape == 0:
#                 res_prod_seq = lr.theta.clone()
#     #             res_prod_seq.append(lr.theta.clone())
#             else:
#                 res_prod_seq = torch.cat((res_prod_seq, lr.theta.clone()), 1)
#         
#         loss = loss_function2(X, Y, lr.theta, dim, beta)
#    
#         loss.backward()
#        
#         with torch.no_grad():
#             lr.theta -= alpha * lr.theta.grad
#             
#             gap = torch.norm(lr.theta.grad)
#             
#             if gap < threshold:
#                 break
#             if gap < prov_record_rate:
#                 tracking_or_not = False
#             
#             print(gap)
#             
#             lr.theta.grad.zero_()
#             
#         epoch = epoch + 1
# 
#             
# #         if last_theta is not None:
# #             print(torch.norm(last_theta - lr.theta))
# #         
# #         if last_theta is not None:
# #             
# #             gap = torch.norm(last_theta - lr.theta)
# #             
# #             if gap < threshold:
# #                 break
# #             if gap < prov_record_rate:
# #                 tracking_or_not = False
#         
#         
#         
#             
#         last_theta = lr.theta.clone()
            
            
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
      
#     return lr.theta, epoch

def logistic_regression_by_standard_library(X, Y, lr, dim, max_epoch, alpha, beta, batch_size, mini_batch_epoch):

#     dim = X.shape
    
    
    num = 0
    
    for epoch in range(max_epoch):
        
        end = False
        
        for i in range(0,X.size()[0], batch_size):
            
            batch_X, batch_Y = X[i:i+batch_size], Y[i:i+batch_size]
            
            
        
        
#             loss = loss_function2(X, Y, lr.theta, dim, beta)

            loss = loss_function1(batch_X, batch_Y, lr.theta, batch_X.shape, beta)
       
            loss.backward()
           
            with torch.no_grad():
                lr.theta -= alpha * lr.theta.grad
                
#                 print('gradient::', lr.theta.grad)
#                  
#                 print('theta::',lr.theta)
                
                lr.theta.grad.zero_()
                
            num += 1
            
            if num >= mini_batch_epoch:
                end = True
                break
                
        
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
      
    return lr.theta


def logistic_regression_by_standard_library2(origin_X, origin_Y, lr, dim, max_epoch, alpha, beta, batch_size, selected_data_ids, mini_batch_epoch, random_ids_multi_super_iterations):

#     dim = X.shape
    
#     for epoch in range(mini_batch_epoch):

    epoch = 0

#     while epoch < mini_batch_epoch:


    selected_rows_set = set(selected_data_ids.view(-1).tolist())
#     id_mappings = {}
    end = False
#     for i in range(max_epoch):
    k = 0
    while epoch < mini_batch_epoch:
        
        random_ids = random_ids_multi_super_iterations[k]
        
        X = origin_X
        
        Y = origin_Y
        
#         selected_data_ids_this_super_iterations = (random_ids.view(-1,1) == selected_data_ids.view(1,-1))
#         
#         selected_data_ids_this_super_iterations = torch.nonzero(selected_data_ids_this_super_iterations)[:,0]

        
        for i in range(0, dim[0], batch_size):
            

            end_id = i + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            
            
            curr_rand_ids = random_ids[i:end_id]
            
            curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
            
            if curr_matched_ids.shape[0] <= 0:
                continue
#             if i not in id_mappings:
#                 
#                 curr_selected_data_ids = selected_data_ids_this_super_iterations[(torch.nonzero((selected_data_ids_this_super_iterations >= i)*(selected_data_ids_this_super_iterations < end_id))).view(-1)]
# 
#                 id_mappings[i] = curr_selected_data_ids
# 
#             else:
#                 
#                 curr_selected_data_ids = id_mappings[i]
#     
#             if curr_selected_data_ids.shape[0] <= 0:
#                 continue
            
            
            batch_X, batch_Y = X[curr_matched_ids], Y[curr_matched_ids]
            
            
        
        
#             loss = loss_function2(X, Y, lr.theta, dim, beta)
#             print(curr_matched_ids.shape)
#             
#             print(i, end_id)
            loss = loss_function1(batch_X, batch_Y, lr.theta, curr_matched_ids.shape[0], beta)
       
            loss.backward()
           
            with torch.no_grad():
                lr.theta -= alpha * lr.theta.grad
                
                
#                 print('gradient::', lr.theta.grad)
#                    
#                    
#                 print('theta::', lr.theta)
                
                
                lr.theta.grad.zero_()
                
            
#             if epoch >= mini_batch_epoch:
#                 break
            
#             if epoch >= 1430:
#                 y = 1
#                 y+=1
            
                
            epoch = epoch + 1
            
            
            if epoch >= mini_batch_epoch:
                end = True
                break
        if end:
            break
                
            
            
        k = k + 1
#             print('epoch', epoch, mini_batch_epoch, batch_X.shape)
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
      
    print('epoch::', epoch)  
    
    return lr.theta



def compute_parameters(X, Y, lr, dim, tracking_or_not):
    
    
    lr.theta, epoch, mini_batch_epoch, theta_list, grad_list = logistic_regression(X, Y, lr, dim, tracking_or_not)
    
    print('res_real:::', lr.theta)
    
    print('mini_batch_epoch::', mini_batch_epoch)
    
    return lr.theta, epoch, mini_batch_epoch, theta_list, grad_list
    
    
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


def compute_model_parameter_by_iteration(dim, theta,  X_Y_mult, max_epoch, alpha, beta, mini_batch_epoch, selected_data_ids, batch_size):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    epoch = 0
    
    
    id_mappings = {}
    
    for i in range(max_epoch):
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


        for i in range(0,dim[0], batch_size):
            

            end_id = i + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
                
            
            
            if i not in id_mappings:
                curr_selected_data_ids = selected_data_ids[(torch.nonzero((selected_data_ids >= i)*(selected_data_ids < end_id))).view(-1)]
                id_mappings[i] = curr_selected_data_ids
            else:
                curr_selected_data_ids = id_mappings[i]
            
            
            if curr_selected_data_ids.shape[0] <= 0:
                continue
            
            
            batch_X_Y_mult = X_Y_mult[curr_selected_data_ids]
        
#         for j in range(X_Y_mult.shape[0]):
            gradient = torch.sum((-batch_X_Y_mult*(1 - sig_layer(torch.mm(batch_X_Y_mult, theta))))/(end_id - i), 0)
            
            gradient = gradient.view(theta.shape) + beta*theta
            
            theta = theta - alpha*gradient
            
            
#             print('gradient::', gradient)
#              
#             print('theta::', theta)
            
#             if epoch >= mini_batch_epoch:
#                 break
            
            epoch = epoch + 1
            
            
            
            
            
            
#             print('id::', j)
#             
#             
#             print('gradient::', gradient.view(-1))
#             
#             print('updated_weight::', theta.view(-1))
            
        
        
        
#         non_linear_term = X_Y_mult*(1 - sig_layer(torch.mm(X_Y_mult, theta)))
#         
# #         
# #         if i == max_epoch - 1:
# #             for j in range(non_linear_term.shape[0]):
# #                 print(j, non_linear_term[j], theta)
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
# 
#         gradient = -torch.sum(non_linear_term, dim=0).view(theta.shape)/dim[0] + beta*theta
# 
# #         print('iteration_gradient::', gradient)
# 
#         theta = theta - alpha*gradient#(1-alpha*beta)*theta + (alpha/dim[0])*torch.sum(non_linear_term, dim=0).view(theta.shape)
        
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

def compute_model_parameter_by_iteration_2(random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, dim, theta,  X_Y_mult, max_epoch, alpha, beta, mini_batch_epoch, selected_data_ids, batch_size):
    
    total_time = 0.0

    epoch = 0
    
#     selected_rows_set = set(selected_data_ids.view(-1).tolist())

#     id_mappings = {}
    
    theta_list = []
    
    gradient_list = []
    end = False
#     for j in range(max_epoch):

    k = 0
    while epoch < mini_batch_epoch:
        
        random_ids = random_ids_multi_super_iterations[k]
        
#         sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
#         all_indexes = np.sort(np.searchsorted(random_ids.numpy(), delta_ids.numpy()))
        
#         all_indexes = np.sort(sort_idx[np.searchsorted(random_ids.numpy(),selected_data_ids.numpy(),sorter = sort_idx)])

        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
        all_indexes = np.sort(sort_idx[selected_data_ids])

        id_start = 0
        
        id_end = 0

        for i in range(0,dim[0], batch_size):
            

            end_id = i + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
                
#             curr_rand_ids = random_ids[i:end_id]
#             
#             curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
            
            
            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id.item())
#             while 1:
#                 if id_end >= all_indexes.shape[0] or all_indexes[id_end] >= end_id:
#                     break
#                 
#                 id_end = id_end + 1
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
            
            curr_matched_ids_size = curr_matched_ids.shape[0]
            
            
            if curr_matched_ids_size <= 0:
                continue
            
#             print(curr_matched_ids)
            
#             batch_X = X[curr_matched_ids]
#             
#             batch_Y = Y[curr_matched_ids]
            
            
#             if i not in id_mappings:
#                 curr_selected_data_ids = selected_data_ids[(torch.nonzero((selected_data_ids >= i)*(selected_data_ids < end_id))).view(-1)]
#                 id_mappings[i] = curr_selected_data_ids
#             else:
#                 curr_selected_data_ids = id_mappings[i]
#             
#             
#             if curr_selected_data_ids.shape[0] <= 0:
#                 continue
            
            
            batch_X_Y_mult = X_Y_mult[curr_matched_ids]
        
#         for j in range(X_Y_mult.shape[0]):
            gradient = torch.sum((-batch_X_Y_mult*(1 - sig_layer(torch.mm(batch_X_Y_mult, theta)))), 0)
#             gradient_list.append(gradient)
            gradient = gradient.view(theta.shape)/(curr_matched_ids_size) + beta*theta
            
            theta = theta - alpha*gradient
#             theta_list.append(theta)
            epoch = epoch + 1
            
            id_start = id_end
            
            
            if epoch >= mini_batch_epoch:
                end = True
                break
        
        if end:
            break
        
        k = k + 1   

            
            
    print('epoch::', epoch)
            
    return theta, total_time, theta_list, gradient_list


def compute_model_parameter_by_iteration_2_sparse(X, Y, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, dim, theta,  X_Y_mult, max_epoch, alpha, beta, mini_batch_epoch, selected_data_ids, batch_size):
    
    total_time = 0.0

    epoch = 0
    
    selected_rows_set = set(selected_data_ids.view(-1).tolist())

    theta_list = []
    
    gradient_list = []
    
    theta = theta.numpy()
    
#     for j in range(max_epoch):

    end = False
    
    overhead = 0
    
    for k in range(random_ids_multi_super_iterations.shape[0]):
        
        random_ids = random_ids_multi_super_iterations[k]

#         sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
#         
#         all_indexes = np.sort(sort_idx[selected_data_ids])

        id_start = 0
        
        id_end = 0

        for i in range(0,dim[0], batch_size):
            
#             print('epoch::', epoch)
            end_id = i + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            
#             while 1:
#                 if id_end >= all_indexes.shape[0] or all_indexes[id_end] >= end_id:
#                     break
#                 
#                 id_end = id_end + 1
            
#             if all_indexes[-1] < end_id:
#                 id_end = all_indexes.shape[0]
#             else:
#                 id_end = np.argmax(all_indexes >= end_id.item())
            
#             print(id_end - id_end2)
            
#             curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
#             
#             curr_matched_ids_size = curr_matched_ids.shape[0]
                
            curr_rand_ids = random_ids[i:end_id]
             
            curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
            curr_matched_ids_size = curr_matched_ids.shape[0]            
            if curr_matched_ids_size <= 0:
                continue
            
            batch_X_Y_mult = X_Y_mult[curr_matched_ids]
        
            
            
            t1 = time.time()
            
        
#             gradient = np.sum((-batch_X_Y_mult.multiply(1 - sigmoid_np(batch_X_Y_mult.dot(theta)))), axis = 0)
            gradient = 0-batch_X_Y_mult.transpose().dot(1 - sigmoid_np(batch_X_Y_mult.dot(theta)))# np.sum((-batch_X_Y_mult.multiply(1 - sigmoid_np(batch_X_Y_mult.dot(theta)))), axis = 0)
        
        
#             gradient = torch.sum((-batch_X_Y_mult*(1 - sigmoid(torch.mm(batch_X_Y_mult, theta)))), 0)
#             gradient_list.append(gradient)
            
            
            gradient = gradient/(curr_matched_ids_size) + beta*theta
            
            theta -= alpha*gradient
            
            t2 = time.time()
            overhead += (t2 - t1)
            
            id_start = id_end
            epoch += 1
#             theta_list.append(theta)
            
            if epoch >= mini_batch_epoch:
                end = True
                break
        
        if end == True:
            break
            
            
    print('overhead::', overhead)       
            
            
    return torch.from_numpy(theta).type(torch.DoubleTensor), total_time, theta_list, gradient_list

def compute_parameters_sparse(origin_X, origin_Y, lr, dim, tracking_or_not):
    
    total_time = 0.0

    epoch = 0
    
#     selected_rows_set = set(selected_data_ids.view(-1).tolist())

    theta_list = []
    
    gradient_list = []
    
    theta = lr.theta.detach().numpy()
    
    
    epoch = 0
#     for j in range(max_epoch):

    end = False
    while epoch < max_epoch:
        
#         random_ids = torch.randperm(dim[0])
        random_ids = np.random.permutation(dim[0])
#         random_ids = torch.tensor(list(range(dim[0])))
        
#         print('rand_ids::', random_ids)
        
        X = origin_X[random_ids]
        
        Y = origin_Y[random_ids]
        
        random_ids_multi_super_iterations.append(random_ids)
        
        
        gap_to_be_averaged = []

        
#         random_ids = random_ids_multi_super_iterations[j]

        for i in range(0,dim[0], batch_size):
            

            end_id = i + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
                
            batch_X = X[i:end_id]
            
            batch_Y = Y[i:end_id]
                
#             curr_rand_ids = random_ids[i:end_id]
#             
#             curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, selected_rows_set)
#             
#             batch_X_Y_mult = X_Y_mult[curr_matched_ids]
        
            
            batch_X_Y_mult = batch_X.multiply(batch_Y.numpy())
        
            if tracking_or_not:
                
                
                '''batch_size*1'''
                
                res_prod = batch_X_Y_mult.dot(theta)
                
                '''batch_size*(t*(n/batch_size))'''
                global res_prod_seq
                  
                curr_res = np.copy(res_prod).flatten()#np.reshape(res_prod, [batch_X.shape[0]])
#                 print('res_prod_shape::', res_prod.shape)
#                 print('theta_shape::', theta.shape)
#                 print('curr_res_shape::', curr_res.shape)
#                 print('batch_X_Y_mult_shape::', batch_X_Y_mult.shape)    
                
                res_prod_seq.append(curr_res)       
        
        
        
            gradient = np.sum((-batch_X_Y_mult.multiply(1 - sigmoid_np(batch_X_Y_mult.dot(theta)))), axis = 0)
        
        
#             gradient = torch.sum((-batch_X_Y_mult*(1 - sigmoid(torch.mm(batch_X_Y_mult, theta)))), 0)
#             gradient_list.append(gradient)
            gradient = np.reshape(gradient, (theta.shape))/(end_id - i) + beta*theta
            
            
            gap = np.linalg.norm(gradient)
            
            theta = theta - alpha*gradient
            
            print('epoch::', epoch, gap)
            
#             if tracking_or_not:
#                 gap_to_be_averaged.append(gap.item())
#                
#                
# #                 if len(gap_to_be_averaged) >= (X.shape[0] - 1)/batch_size:
#                 if end_id == X.shape[0]:
#                     
#                     average_gap = np.sum(gap_to_be_averaged)/len(gap_to_be_averaged)
#                     
#                     
#                     print('avg_gap::', average_gap, len(gap_to_be_averaged))
#                     
#                     if average_gap < prov_record_rate:
#                         tracking_or_not = False    
# 
#                 
#                     del gap_to_be_averaged
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            epoch = epoch + 1
            
            if gap < threshold:
                end = True
                break
#             theta_list.append(theta)
            
            if epoch >= max_epoch:
                end = True
                break
    
    
        if end == True:
            break
       
            
    return torch.from_numpy(theta).type(torch.DoubleTensor), epoch, epoch, theta_list, gradient_list


def compute_model_parameter_by_iteration2(dim, theta,  X_Y_mult, max_epoch, alpha, beta, mini_batch_epoch, selected_data_ids, batch_size):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    epoch = 0
    for i in range(max_epoch):
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

        end = False

        for i in range(0,dim[0], batch_size):
            

#             end_id = i + batch_size
#             
#             if end_id > dim[0]:
#                 end_id = dim[0]
#                 
#                 
#                 
#             curr_selected_data_ids = (torch.nonzero((selected_data_ids >= i)*(selected_data_ids < end_id))).view(-1)
#             
#             
#             if curr_selected_data_ids.shape[0] <= 0:
#                 continue
            
            
            batch_X_Y_mult = X_Y_mult[i:i+batch_size]
        
#         for j in range(X_Y_mult.shape[0]):
            gradient = torch.sum((-batch_X_Y_mult*(1 - sig_layer(torch.mm(batch_X_Y_mult, theta))))/batch_X_Y_mult.shape[0], 0)
            
            gradient = gradient.view(theta.shape) + beta*theta
            
            theta = theta - alpha*gradient
            
            
#             print('gradient::', gradient)
#               
#             print('theta::', theta)
            
            if epoch >= mini_batch_epoch:
                
                end = True
                
                break
            
            epoch = epoch + 1
            
            
        if end:
            break    
            
            
            
#             print('id::', j)
#             
#             
#             print('gradient::', gradient.view(-1))
#             
#             print('updated_weight::', theta.view(-1))
            
        
        
        
#         non_linear_term = X_Y_mult*(1 - sig_layer(torch.mm(X_Y_mult, theta)))
#         
# #         
# #         if i == max_epoch - 1:
# #             for j in range(non_linear_term.shape[0]):
# #                 print(j, non_linear_term[j], theta)
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
# 
#         gradient = -torch.sum(non_linear_term, dim=0).view(theta.shape)/dim[0] + beta*theta
# 
# #         print('iteration_gradient::', gradient)
# 
#         theta = theta - alpha*gradient#(1-alpha*beta)*theta + (alpha/dim[0])*torch.sum(non_linear_term, dim=0).view(theta.shape)
        
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


# def compute_model_parameter_by_iteration2(dim, theta,  X_Y_mult):
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


def compute_model_parameter_by_approx_incremental_3(A, B, term1, term2, dim, theta, max_epoch, cut_off_epoch, batch_size, alpha, beta, mini_batch_epoch):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)

    avg_term1 = torch.sum(term1, 0)/term1.shape[0]
    
    avg_term2 = torch.sum(term2, 0)/term2.shape[0]


    theta = torch.mm(A, theta) + B


    for i in range(cut_off_epoch):
        gradient = -(torch.mm(avg_term1, theta) + (avg_term2).view(theta.shape))/batch_size + beta*theta
              
        theta = theta - alpha * gradient

#     for i in range(mini_batch_epoch):
#          
#         if i <= cut_off_epoch:
#             gradient = -(torch.mm(term1[i], theta) + (term2[i]).view(theta.shape))/dim[0] + beta*theta
#                  
#             theta = theta - alpha * gradient
#              
# #             print('gradient::', gradient)    
# #                  
# #             print('theta::', theta)
#              
#         else:
#             gradient = -(torch.mm(avg_term1, theta) + (avg_term2).view(theta.shape))/dim[0] + beta*theta
#              
#             theta = theta - alpha * gradient
#         
#     print('gap::', theta - theta2)

    
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
# 
#         if i < cut_off_epoch:
#             
#             for j in range(0, dim[0], batch_size):
#             
#                 gradient = -(torch.mm(term1[i*batch_size + j], theta) + (term2[i*batch_size + j]).view(theta.shape))/dim[0] + beta*theta
#                 
#                 theta = theta - alpha * gradient
#         
#         else:
#             
#             
#             for j in range(batch_size):
#                 gradient = -(torch.mm(term1[(cut_off_epoch - 1)*i + j], theta) + (term2[(cut_off_epoch - 1)*i + j]).view(theta.shape))/dim[0] + beta*theta
#                 theta = theta - alpha * gradient
#         print('approx_gradient::', gradient)

        
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

def compute_sub_term_2_by_epoch(X_Y_prod, b_seq):
    
    res = torch.t(torch.mm(torch.t(X_Y_prod), b_seq.view(-1,1)))
    
    return res


def compute_sub_term_2_by_epoch_sparse(X_Y_prod, b_seq):
    
    
    res = X_Y_prod.transpose().dot(b_seq.view(-1,1).numpy())
    
#     res = torch.t(torch.mm(torch.t(X_Y_prod), b_seq.view(-1,1)))
    
    return res


def prepare_term_1_batch2_theta(X, X_mult_theta, w_seq_this_epoch):
    
    res = torch.mm(torch.t(X), X_mult_theta.view(-1,1)*w_seq_this_epoch.view(-1,1))
    
    return res



def prepare_term_1_batch2_theta_sparse(X, X_mult_theta, w_seq_this_epoch):
    
    res = X.transpose().dot(np.multiply(np.reshape(X_mult_theta, (-1,1)), (w_seq_this_epoch.view(-1,1).numpy())))
    
    
#     res = torch.mm(torch.t(X), X_mult_theta.view(-1,1)*w_seq_this_epoch.view(-1,1))
    
    return res



def prepare_term_1_batch2_theta_sparse2(X_w_prod, X_mult_theta):
    
    
#     print(X_w_prod.shape)
#     
#     print(X_mult_theta.shape)
    
    res = X_w_prod.transpose().dot(np.reshape(X_mult_theta, (-1,1)))
    
#     res = X.transpose().dot(np.multiply(np.reshape(X_mult_theta, (-1,1)), (w_seq_this_epoch.view(-1,1).numpy())))
    
    
#     res = torch.mm(torch.t(X), X_mult_theta.view(-1,1)*w_seq_this_epoch.view(-1,1))
    
    return res

def prepare_term_1_batch2_sparse(X, w_seq_this_epoch):
    
    res = X.transpose().dot(X.multiply(w_seq_this_epoch.view(-1,1).numpy()))
    
    
#     res = torch.mm(torch.t(X), X_mult_theta.view(-1,1)*w_seq_this_epoch.view(-1,1))
    
    return res


def prepate_term_1_batch_by_epoch(X, w_seq):
    
    
    res1 = X*w_seq.view(w_seq.shape[0], -1)

    res = torch.mm(torch.t(X), res1)
    
    
    return res


def prepate_term_1_batch_by_epoch_with_eigen_matrix(delta_X, w_seq, remaining_batch_size, M, M_inverse, s):

    if delta_X.shape[0] < delta_X.shape[1]:
        delta_s = torch.mm(torch.mm(M_inverse, torch.t(delta_X*w_seq.view(-1,1))), torch.mm(delta_X, M))
        
    else:
        delta_s = torch.mm(torch.mm(M_inverse, torch.mm(torch.t(delta_X*w_seq.view(-1,1)), delta_X)), M)
    
    
    updated_s = (1 - alpha*beta) + alpha*(s - torch.diag(delta_s))/remaining_batch_size
#     updated_s[updated_s > 1] = 1
    
    return updated_s

def convert_coo_matrix_to_sparse_tensor(curr_term1):
    
    values = curr_term1.data
    indices = np.vstack((curr_term1.row, curr_term1.col))
    
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = curr_term1.shape
    
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))
    
    
    
    
def compute_model_parameter_by_provenance_sparse(theta_list, gradient_list, origin_X, origin_Y, origin_X_Y_prod, weights, offsets, delta_ids, random_ids_multi_super_iterations, term1, term2, dim, theta, mini_batch_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, avg_term1, avg_term2):
    
    total_time = 0.0
    
    min_batch_num_per_epoch = int((origin_X.shape[0] - 1)/batch_size) + 1
    cut_off_super_iteration = int((weights.shape[0]-1)/dim[0])

    '''T, |delta_X|, q^2'''

#     delta_X = origin_X[delta_ids]


    epoch = 0

    overhead = 0
     
    overhead2 = 0
     
    overhead3 = 0
    overhead4 = 0
     
    t_time = 0
    t_time2 = 0

    overhead3 = 0
        
    delta_ids_set = set(delta_ids.view(-1).tolist())    
    
    X = origin_X
        
        
    Y = origin_Y
    
#     vectorized_theta = Variable(torch.reshape(torch.t(theta), [-1,1]))
    
#     theta = Variable(theta)
    theta = theta.detach().numpy()
    
    avg_sub_term_1 = 0
    
    avg_sub_term_2 = 0
    
#     avg_term1 = None
#     
#     avg_term2 = None
#     
    
    end = False
    
    avg_batch_weights = 0
    
#     for k in range(max_epoch):
    for k in range(random_ids_multi_super_iterations.shape[0]):

        random_ids = random_ids_multi_super_iterations[k]
        
        
        super_iter_id = k
        
        if k > cut_off_super_iteration:
            super_iter_id = cut_off_super_iteration
        
        end_id_super_iteration = (super_iter_id + 1)*dim[0]
        
        
        if end_id_super_iteration > weights.shape[0]:
            end_id_super_iteration = weights.shape[0]
            
        
        weights_this_super_iteration = weights[super_iter_id*dim[0]:end_id_super_iteration]
        
        offsets_this_super_iteration = offsets[super_iter_id*dim[0]:end_id_super_iteration]
        
        for i in range(0, X.shape[0], batch_size):
        
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            

            curr_rand_ids = Variable(random_ids[i:end_id])
            
            
            curr_matched_ids = (get_subset_data_per_epoch(curr_rand_ids, delta_ids_set))
            
            if curr_matched_ids.shape[0] > 0:
                
                
                if (end_id - i - curr_matched_ids.shape[0]) <= 0:
                    continue

                
                batch_delta_X = (X[curr_matched_ids])
                
                batch_delta_Y = (Y[curr_matched_ids])
                
                batch_delta_X_Y_prod = origin_X_Y_prod[curr_matched_ids]

            
            if epoch < cut_off_epoch:
                
                
                
                
                sub_term2 = 0
                
                sub_term_1 = 0
            
                if curr_matched_ids.shape[0] > 0:
                    batch_weights = weights_this_super_iteration[curr_matched_ids]
                    
                    
                    batch_offsets = offsets_this_super_iteration[curr_matched_ids]
                
#                 batch_X_multi_theta = Variable(torch.mm(batch_delta_X, theta))
                    t1 = time.time()
                    batch_X_multi_theta = batch_delta_X.dot(theta)
                    
                    t2 = time.time()
                    
                    overhead3 += (t2 - t1)
            
            
                    t1 = time.time()
                    sub_term2 = compute_sub_term_2_by_epoch_sparse(batch_delta_X_Y_prod, batch_offsets)
                    
                    t2 = time.time()
                     
                    overhead4 += (t2 - t1)
                    
            
#                 sub_term2 = (prepare_sub_term_2(batch_delta_X, batch_offsets, batch_delta_X.shape, num_class))
                full_term1 = term1[epoch]
                
                full_term2 = term2[epoch]
                
                if epoch >= cut_off_epoch - min_batch_num_per_epoch:
                    
#                     sub_term_1_without_weights = prepate_term_1_batch_by_epoch(batch_delta_X, batch_weights)
                    t1 = time.time()
                    if curr_matched_ids.shape[0] > 0:
                        sub_term_1 =  prepare_term_1_batch2_theta_sparse(batch_delta_X, batch_X_multi_theta, batch_weights)
                    t2 = time.time()
                     
                    overhead += (t2 - t1)
                    
#                     sub_term_1_without_weights = prepare_term_1_batch2_1(batch_delta_X, batch_weights, batch_delta_X.shape, num_class)
#                     avg_sub_term_1 += (sub_term_1)
#                     avg_sub_term_2 += (sub_term2)
                     
                    t0_1 = time.time()
                    dot_res = full_term1.dot(theta)
#                     if avg_term1 is None:
#                         avg_term1 = full_term1
#                         avg_term2 = full_term2
#                     else:
#                         avg_term1 += full_term1
#                         avg_term2 += full_term2
                    t1_1 = time.time()
                    
                    t_time += (t1_1 - t0_1)
                    t3 = time.time()
                    
#                     output = 0 - (torch.mm((full_term1), theta) - sub_term_1 + (full_term2.view(-1,1) - sub_term2.view(-1,1)))
                    output = 0 - (dot_res - sub_term_1 + (np.reshape(full_term2, (-1,1)) - np.reshape(sub_term2, (-1,1))))
                    
                    t4 = time.time()
#                     
#                     overhead2 += (t4  -t3)
                    
                    del sub_term_1
                
                else:
                    
                    t1 = time.time()
                    if curr_matched_ids.shape[0] > 0:
                        sub_term_1 =  prepare_term_1_batch2_theta_sparse(batch_delta_X, batch_X_multi_theta, batch_weights)
                    
                    
                    t2 = time.time()
                     
                    overhead += (t2 - t1)
#                     sub_term_1 = (torch.t(compute_sub_term_1(batch_X_multi_theta, batch_delta_X, batch_weights, batch_X_multi_theta.shape, num_class)))
                    
#                     vectorized_sub_term_1 = Variable(torch.reshape(sub_term_1, [-1,1]))
                    
#                     output = 0 - (torch.mm((full_term1), theta) - sub_term_1 + (full_term2.view(-1,1) - sub_term2.view(-1,1)))
                    
                    t0_1 = time.time()
                    dot_res = full_term1.dot(theta)
                    
                    t1_1 = time.time()
                    
                    t_time += (t1_1 - t0_1)
                    
                    
                    
                    t3 = time.time()

                    output = 0 - (dot_res - sub_term_1 + (np.reshape(full_term2, (-1,1)) - np.reshape(sub_term2, (-1,1))))
                    t4 = time.time()
                     
                    overhead2 += (t4  -t3)
                    
                    
                    del sub_term_1

            
                
                
#                 delta_x_sum_by_class = compute_x_sum_by_class(batch_delta_X, batch_delta_Y, num_class, batch_delta_X.shape)
                
#                 x_sum_by_class = x_sum_by_class_list[epoch] - delta_x_sum_by_class
#                 print(batch_delta_X)
#                 print(beta)
#                 
#                 print(theta)
#                 
#                 print(output)
#                 
#                 print(batch_delta_X.shape)
                
#                 computed_origin_grad_this_epoch =  (0 - ((full_term1.dot(theta)) + (np.reshape(full_term2, (-1,1)))))/(end_id - i) + beta*theta
#     
#                 exp_gradient = np.sum((-origin_X_Y_prod[curr_rand_ids].multiply(1 - sigmoid_np(origin_X_Y_prod[curr_rand_ids].dot(theta)))), axis = 0)
#                 
#                 exp_gradient = np.reshape(exp_gradient, (-1,1))/(end_id - i) + beta*theta
#                 
#                 print(np.linalg.norm(exp_gradient - computed_origin_grad_this_epoch))
                
                gradient = output/(end_id - i - curr_matched_ids.shape[0]) + theta*beta     
    
#                 print('gradient_diff:', np.linalg.norm(np.reshape(output, (-1,1)) - np.reshape(gradient_list[epoch], (-1,1))))
    
                del output
                
                del sub_term2
                                
                theta = (theta - gradient*alpha)  
                
                
#                 print('theta_diff:', np.linalg.norm(np.reshape(theta, (-1,1)) - np.reshape(theta_list[epoch], (-1,1))))
                
                
                del gradient
                
#                 theta = torch.t(vectorized_theta.view(num_class, dim[1]))
                
#                 t2 = time.time()
#                  
#                 overhead += (t2 - t1)
                
                epoch = epoch + 1
                
                if epoch >= mini_batch_epoch:
                    end = True
                    break
                
                
                       
            else:
                
#                 t2_1 = time.time()
                
                sub_term_1 = 0
                
                sub_term2 = 0
                
#                 if epoch == cut_off_epoch:
#                     avg_sub_term_1 = avg_sub_term_1/min_batch_num_per_epoch
#                     
#                     avg_sub_term_2 = avg_sub_term_2/min_batch_num_per_epoch
                
#                     avg_term1 = avg_term1/min_batch_num_per_epoch
#                     
#                     avg_term2 = avg_term2/min_batch_num_per_epoch
                
#                     full_term1 = Variable(avg_term1 - avg_sub_term_1)
#                 
#                     full_term2 = Variable(avg_term2 - avg_sub_term_2)
                
                batch_weights = weights_this_super_iteration[curr_matched_ids]
                
                
                batch_offsets = offsets_this_super_iteration[curr_matched_ids]
                
#                 batch_X_multi_theta = Variable(torch.mm(batch_delta_X, theta))
                t1 = time.time()
                batch_X_multi_theta = batch_delta_X.dot(theta)
                t2 = time.time()
                     
                overhead3 += (t2 - t1)
                
                
                if curr_matched_ids.shape[0] > 0:
                    t1 = time.time()
                    sub_term_1 =  prepare_term_1_batch2_theta_sparse(batch_delta_X, batch_X_multi_theta, batch_weights)

                    t2 = time.time()
                
                    overhead += (t2 - t1)
                
                    t1 = time.time()
                    sub_term2 = np.reshape(compute_sub_term_2_by_epoch_sparse(batch_delta_X_Y_prod, batch_offsets), (-1,1))
                    t2 = time.time()
                    
                    overhead4 += (t2 - t1)
                
                
                
                t0_1 = time.time()
                
                dot_res = avg_term1.dot(theta)
                
                t1_1 = time.time()
                    
                t_time += (t1_1 - t0_1)
                
                t3 = time.time()
                output = 0 - (dot_res - sub_term_1 + (avg_term2 - sub_term2))
                
                t4 = time.time()
                    
                overhead2 += (t4  -t3)
#                 delta_x_sum_by_class = compute_x_sum_by_class(batch_delta_X, batch_delta_Y, num_class, batch_delta_X.shape)
                
#                 x_sum_by_class = x_sum_by_class_list[epoch] - delta_x_sum_by_class
#                 
#                 del delta_x_sum_by_class
    
                
                gradient = (output)/(end_id - i - curr_matched_ids.shape[0]) + theta*beta     
    
#                 print('gradient_diff:', torch.norm(output - gradient_list[epoch]))

                del output      
                
#                 del x_sum_by_class
                
                theta = (theta - gradient*alpha)  
#                 print('theta_diff:', torch.norm(theta - theta_list[epoch]))

                del gradient
                
#                 theta = torch.t(vectorized_theta.view(num_class, dim[1]))
                
                epoch = epoch + 1
            
            
#                 t2_2 = time.time()
#                 
#                 t_time2 += (t2_2 - t2_1)
            
            
            
            
                if epoch >= mini_batch_epoch:
                    end = True
                    
                    break
                
        
        if end == True:
            break
        
    
    print('overhead::', overhead)
     
    print('overhead2::', overhead2)
     
    print('overhead3::', overhead3)
     
    print('overhead4::', overhead4)
     
    print('t_time::', t_time)
     
    print('t_time2::', t_time2)
    
    final_res = torch.from_numpy(theta).type(torch.DoubleTensor) 
    
    return final_res


def compute_model_parameter_by_provenance_sparse0(X_w_prod, X_off_prod, theta_list, gradient_list, X, Y, origin_X_Y_prod, weights, offsets, selected_rows, random_ids_multi_super_iterations, dim, theta, mini_batch_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, term2, avg_term2):
    
    total_time = 0.0
    
#     min_batch_num_per_epoch = int((origin_X.shape[0] - 1)/batch_size) + 1
    cut_off_super_iteration = int((weights.shape[0]-1)/dim[0])

    '''T, |delta_X|, q^2'''

#     delta_X = origin_X[delta_ids]


    epoch = 0

    overhead = 0
     
    overhead2 = 0
     
    overhead3 = 0
    overhead4 = 0
     
    t_time = 0
    t_time2 = 0

#     overhead3 = 0
        
    delta_ids_set = set(selected_rows.view(-1).tolist())    
    
#     X = origin_X
        
        
#     Y = origin_Y
    
#     vectorized_theta = Variable(torch.reshape(torch.t(theta), [-1,1]))
    
#     theta = Variable(theta)
    theta = theta.numpy()
    
    avg_sub_term_1 = 0
    
    avg_sub_term_2 = 0
    
#     avg_term1 = None
#     
#     avg_term2 = None
#     
    
    end = False
    
    avg_batch_weights = 0
    
#     for k in range(max_epoch):
    for k in range(random_ids_multi_super_iterations.shape[0]):

        random_ids = random_ids_multi_super_iterations[k]
        
        
        super_iter_id = k
        
        if k > cut_off_super_iteration:
            super_iter_id = cut_off_super_iteration
        
        end_id_super_iteration = (super_iter_id + 1)*dim[0]
        
        
        if end_id_super_iteration > weights.shape[0]:
            end_id_super_iteration = weights.shape[0]
        
        
        
        weights_this_super_iteration = weights[super_iter_id*dim[0]:end_id_super_iteration]
#         
#         offsets_this_super_iteration = offsets[super_iter_id*dim[0]:end_id_super_iteration]
         
        
#         X_w_prod_this_super_iter = X_w_prod[super_iter_id]
        X_offset_prod_this_super_iter = X_off_prod[super_iter_id]
           
        for i in range(0, X.shape[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            

            curr_rand_ids = random_ids[i:end_id]
            
            
            curr_matched_ids, curr_non_matched_ids = get_subset_data_per_epoch2(curr_rand_ids, delta_ids_set)
            
            
            if curr_matched_ids.shape[0] <= 0:
                continue
            ttt1 = time.time()    

            batch_delta_X = X[curr_matched_ids]
            
#             batch_delta_X_Y_prod = origin_X_Y_prod[curr_matched_ids]

#             other_X_Y_prod = origin_X_Y_prod[curr_non_matched_ids].tocsc()
             
#             curr_X_w_prod = X_w_prod_this_super_iter[curr_matched_ids]
#             batch_offsets = offsets_this_super_iteration[curr_non_matched_ids]
            batch_weights = weights_this_super_iteration[curr_matched_ids]
            
            ttt2 = time.time()
        
            t_time2 += (ttt2 - ttt1) 
#             batch_X_multi_theta = batch_delta_X.dot(theta)
            
#             sub_term2 = compute_sub_term_2_by_epoch_sparse(batch_delta_X_Y_prod, batch_offsets)

            t1 = time.time()
#             curr_off_sets = batch_offsets
            

#             sub_term2 = batch_delta_X_Y_prod.transpose().dot(curr_off_sets)
#             sub_term2 = other_X_Y_prod.transpose().dot(curr_off_sets)
#             sub_term2 = sparse_matrix_mult_sparseX_mod1(X_offset_prod_this_super_iter[curr_non_matched_ids], np.array(range(curr_non_matched_ids.shape[0])))#].sum(axis = 0).toarray()
            sub_term2 = np.asarray(X_offset_prod_this_super_iter[curr_non_matched_ids].sum(axis = 0))
            sub_term2 = np.reshape(sub_term2, theta.shape)
            
            X_theta_mult = batch_delta_X.dot(theta)
            
            x_theta_w = np.multiply(X_theta_mult, batch_weights)
            
            
#             curr_weights = batch_weights
            
            
            
#             t1 = time.time()
#             curr_sub_term2 = sub_term2#np.reshape(sub_term2, (-1,1))
#             t2 = time.time()
#             
#             overhead += (t2 - t1)
#             sub_term_1 = prepare_term_1_batch2_theta_sparse2(curr_X_w_prod, batch_X_multi_theta)
            
#             sub_term_1 = curr_X_w_prod.transpose().dot(X_theta_mult)
            
            
            sub_term_1 = batch_delta_X.transpose().dot(x_theta_w)
            
                 
#             sub_term_1 = curr_X_w_prod.transpose().dot(np.reshape(batch_X_multi_theta, (-1,1)))
            if epoch < cut_off_epoch:
                gradient = 0 - (sub_term_1 + term2[epoch] - sub_term2)
            else:
                gradient = 0 - (sub_term_1 + avg_term2 - sub_term2)
            
            
            
            

#             t3 = time.time()
#             t1 = time.time()
#             grad1 = output/(curr_matched_ids.shape[0])
# #             t4 = time.time()
# #             overhead3 += (t4 - t3)
#             
#             
# #             t1 = time.time()
#             grad2 = beta*theta
# #             t2 = time.time()
# #               
# #             overhead2 += (t2 - t1)
#             
# #             t1 = time.time()
#             gradient = grad1 + grad2
# #             t2 = time.time()
# #             overhead4 += (t2 - t1)
#             
#             
#             theta -= alpha*gradient
#             
#             t2 = time.time()
#             
#             overhead += (t2 - t1)
            
            
            
            gradient = gradient/(curr_matched_ids.shape[0]) + beta*theta
            
            theta = theta - alpha*gradient
            
            t2 = time.time()
            overhead += (t2 - t1)
            
            
            epoch = epoch + 1
            
            

            
            if epoch >= mini_batch_epoch:
                end = True
                break
                
                
                       
#             else:
#                 
# 
#                 batch_offsets = offsets_this_super_iteration[curr_matched_ids]
#                 
#                 batch_X_multi_theta = batch_delta_X.dot(theta)
# 
#                 sub_term_1 = prepare_term_1_batch2_theta_sparse2(curr_X_w_prod, batch_X_multi_theta)
# 
# 
#                 sub_term2 = compute_sub_term_2_by_epoch_sparse(batch_delta_X_Y_prod, batch_offsets)
# 
#                 output = 0 - (sub_term_1 + (np.reshape(sub_term2, (-1,1))))
# 
#                 
#                 gradient = (output)/(end_id - i - curr_matched_ids.shape[0]) + theta*beta     
#                 
#                 theta = (theta - gradient*alpha)  
#                 
#                 epoch = epoch + 1
#             
#                 if epoch >= mini_batch_epoch:
#                     end = True
#                     
#                     break
        
               
        
        if end == True:
            break
        
    
    print('overhead::', overhead)
     
    print('overhead2::', overhead2)
     
    print('overhead3::', overhead3)
     
    print('overhead4::', overhead4)
     
    print('t_time::', t_time)
    
     
    
    final_res = torch.from_numpy(theta).type(torch.DoubleTensor) 
    
    
    
    
    print('t_time2::', t_time2)
    
    return final_res


def compute_model_parameter_by_provenance_sparse5(X_w_prod, X_off_prod, theta_list, gradient_list, X, Y, origin_X_Y_prod, weights, offsets, selected_rows, random_ids_multi_super_iterations, dim, theta, mini_batch_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, term2, avg_term2):
    
    total_time = 0.0
    
#     min_batch_num_per_epoch = int((origin_X.shape[0] - 1)/batch_size) + 1
    cut_off_super_iteration = int((weights.shape[0]-1)/dim[0])

    '''T, |delta_X|, q^2'''

#     delta_X = origin_X[delta_ids]


    epoch = 0

    overhead = 0
     
    overhead2 = 0
     
    overhead3 = 0
    overhead4 = 0
     
    t_time = 0
    t_time2 = 0

#     overhead3 = 0
        
    delta_ids_set = set(selected_rows.view(-1).tolist())    
    
#     X = origin_X
        
        
#     Y = origin_Y
    
#     vectorized_theta = Variable(torch.reshape(torch.t(theta), [-1,1]))
    
#     theta = Variable(theta)
    theta = theta.numpy()
    
    avg_sub_term_1 = 0
    
    avg_sub_term_2 = 0
    
#     avg_term1 = None
#     
#     avg_term2 = None
#     
    
    end = False
    
    avg_batch_weights = 0
    
#     for k in range(max_epoch):
    for k in range(random_ids_multi_super_iterations.shape[0]):

        random_ids = random_ids_multi_super_iterations[k]
        
        
        super_iter_id = k
        
        if k > cut_off_super_iteration:
            super_iter_id = cut_off_super_iteration
        
        end_id_super_iteration = (super_iter_id + 1)*dim[0]
        
        
        if end_id_super_iteration > weights.shape[0]:
            end_id_super_iteration = weights.shape[0]
        
        
        
        weights_this_super_iteration = weights[super_iter_id*dim[0]:end_id_super_iteration]
#         
#         offsets_this_super_iteration = offsets[super_iter_id*dim[0]:end_id_super_iteration]
         
        
#         X_w_prod_this_super_iter = X_w_prod[super_iter_id]
        X_offset_prod_this_super_iter = X_off_prod[super_iter_id]
           
        for i in range(0, X.shape[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            

            curr_rand_ids = random_ids[i:end_id]
            
            
            curr_matched_ids, curr_non_matched_ids = get_subset_data_per_epoch2(curr_rand_ids, delta_ids_set)
            
            
            if curr_matched_ids.shape[0] <= 0:
                continue
            ttt1 = time.time()    

            batch_delta_X = X[curr_matched_ids]
            
#             batch_delta_X_Y_prod = origin_X_Y_prod[curr_matched_ids]

#             other_X_Y_prod = origin_X_Y_prod[curr_non_matched_ids].tocsc()
             
#             curr_X_w_prod = X_w_prod_this_super_iter[curr_matched_ids]
#             batch_offsets = offsets_this_super_iteration[curr_non_matched_ids]
            batch_weights = weights_this_super_iteration[curr_matched_ids]
            
            ttt2 = time.time()
        
            t_time2 += (ttt2 - ttt1) 
#             batch_X_multi_theta = batch_delta_X.dot(theta)
            
#             sub_term2 = compute_sub_term_2_by_epoch_sparse(batch_delta_X_Y_prod, batch_offsets)

            t1 = time.time()
#             curr_off_sets = batch_offsets
            

#             sub_term2 = batch_delta_X_Y_prod.transpose().dot(curr_off_sets)
#             sub_term2 = other_X_Y_prod.transpose().dot(curr_off_sets)
#             sub_term2 = sparse_matrix_mult_sparseX_mod1(X_offset_prod_this_super_iter[curr_non_matched_ids], np.array(range(curr_non_matched_ids.shape[0])))#].sum(axis = 0).toarray()
            sub_term2 = 0
            if curr_non_matched_ids.shape[0] > 0:
                sub_term2 = np.asarray(X_offset_prod_this_super_iter[curr_non_matched_ids].sum(axis = 0))
                sub_term2 = np.reshape(sub_term2, theta.shape)
            
            X_theta_mult = batch_delta_X.dot(theta)
            
            x_theta_w = np.multiply(X_theta_mult, batch_weights)
            
            
#             curr_weights = batch_weights
            
            
            
#             t1 = time.time()
#             curr_sub_term2 = sub_term2#np.reshape(sub_term2, (-1,1))
#             t2 = time.time()
#             
#             overhead += (t2 - t1)
#             sub_term_1 = prepare_term_1_batch2_theta_sparse2(curr_X_w_prod, batch_X_multi_theta)
            
#             sub_term_1 = curr_X_w_prod.transpose().dot(X_theta_mult)
            
            
            sub_term_1 = batch_delta_X.transpose().dot(x_theta_w)
            
                 
#             sub_term_1 = curr_X_w_prod.transpose().dot(np.reshape(batch_X_multi_theta, (-1,1)))
            if epoch < cut_off_epoch:
                gradient = 0 - (sub_term_1 + term2[epoch] - sub_term2)
            else:
                gradient = 0 - (sub_term_1 + avg_term2 - sub_term2)
            
            
            
            

#             t3 = time.time()
#             t1 = time.time()
#             grad1 = output/(curr_matched_ids.shape[0])
# #             t4 = time.time()
# #             overhead3 += (t4 - t3)
#             
#             
# #             t1 = time.time()
#             grad2 = beta*theta
# #             t2 = time.time()
# #               
# #             overhead2 += (t2 - t1)
#             
# #             t1 = time.time()
#             gradient = grad1 + grad2
# #             t2 = time.time()
# #             overhead4 += (t2 - t1)
#             
#             
#             theta -= alpha*gradient
#             
#             t2 = time.time()
#             
#             overhead += (t2 - t1)
            
            
            
            gradient = gradient/(curr_matched_ids.shape[0]) + beta*theta
            
            theta = theta - alpha*gradient
            
            t2 = time.time()
            overhead += (t2 - t1)
            
            
            epoch = epoch + 1
            
            

            
            if epoch >= mini_batch_epoch:
                end = True
                break
                
                
                       
#             else:
#                 
# 
#                 batch_offsets = offsets_this_super_iteration[curr_matched_ids]
#                 
#                 batch_X_multi_theta = batch_delta_X.dot(theta)
# 
#                 sub_term_1 = prepare_term_1_batch2_theta_sparse2(curr_X_w_prod, batch_X_multi_theta)
# 
# 
#                 sub_term2 = compute_sub_term_2_by_epoch_sparse(batch_delta_X_Y_prod, batch_offsets)
# 
#                 output = 0 - (sub_term_1 + (np.reshape(sub_term2, (-1,1))))
# 
#                 
#                 gradient = (output)/(end_id - i - curr_matched_ids.shape[0]) + theta*beta     
#                 
#                 theta = (theta - gradient*alpha)  
#                 
#                 epoch = epoch + 1
#             
#                 if epoch >= mini_batch_epoch:
#                     end = True
#                     
#                     break
        
               
        
        if end == True:
            break
        
    
    print('overhead::', overhead)
     
    print('overhead2::', overhead2)
     
    print('overhead3::', overhead3)
     
    print('overhead4::', overhead4)
     
    print('t_time::', t_time)
    
     
    
    final_res = torch.from_numpy(theta).type(torch.DoubleTensor) 
    
    
    
    
    print('t_time2::', t_time2)
    
    return final_res










def compute_model_parameter_by_provenance_sparse6(X_w_prod, X_off_prod, theta_list, gradient_list, X, Y, origin_X_Y_prod, weights, offsets, selected_rows, random_ids_multi_super_iterations, dim, theta, mini_batch_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, term2, avg_term2):
    
    total_time = 0.0
    
#     min_batch_num_per_epoch = int((origin_X.shape[0] - 1)/batch_size) + 1
    cut_off_super_iteration = int((weights.shape[0]-1)/dim[0])

    '''T, |delta_X|, q^2'''

#     delta_X = origin_X[delta_ids]


    epoch = 0

    overhead = 0
     
    overhead2 = 0
     
    overhead3 = 0
    overhead4 = 0
     
    t_time = 0
    t_time2 = 0

#     overhead3 = 0
        
    delta_ids_set = set(selected_rows.view(-1).tolist())    
    
#     X = origin_X
        
        
#     Y = origin_Y
    
#     vectorized_theta = Variable(torch.reshape(torch.t(theta), [-1,1]))
    
#     theta = Variable(theta)
    theta = theta.numpy()
    
    avg_sub_term_1 = 0
    
    avg_sub_term_2 = 0
    
#     avg_term1 = None
#     
#     avg_term2 = None
#     
    
    end = False
    
    avg_batch_weights = 0
    
#     for k in range(max_epoch):
    for k in range(random_ids_multi_super_iterations.shape[0]):

        random_ids = random_ids_multi_super_iterations[k]
        
        
        super_iter_id = k
        
        if k > cut_off_super_iteration:
            super_iter_id = cut_off_super_iteration
        
        end_id_super_iteration = (super_iter_id + 1)*dim[0]
        
        
        if end_id_super_iteration > weights.shape[0]:
            end_id_super_iteration = weights.shape[0]
        
        
        
#         weights_this_super_iteration = weights[super_iter_id*dim[0]:end_id_super_iteration]
#         
#         offsets_this_super_iteration = offsets[super_iter_id*dim[0]:end_id_super_iteration]
         
        
        X_w_prod_this_super_iter = X_w_prod[super_iter_id]
#         X_offset_prod_this_super_iter = X_off_prod[super_iter_id]
           
        for i in range(0, X.shape[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            

            curr_rand_ids = random_ids[i:end_id]
            
            
            curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, delta_ids_set)
            
            
            if curr_matched_ids.shape[0] <= 0:
                continue
#             ttt1 = time.time()    

#             batch_delta_X = X[curr_matched_ids]
            
            
            
#             batch_weights = weights_this_super_iteration[curr_matched_ids]
#             batch_offsets = offsets_this_super_iteration[curr_matched_ids]
            batch_offsets = offsets[curr_matched_ids + super_iter_id*dim[0]]


#             other_X_Y_prod = origin_X_Y_prod[curr_non_matched_ids].tocsc()
             
            curr_X_w_prod = X_w_prod_this_super_iter[curr_matched_ids]
            
            
#             ttt2 = time.time()
#         
#             t_time2 += (ttt2 - ttt1) 
#             batch_X_multi_theta = batch_delta_X.dot(theta)
            
#             sub_term2 = compute_sub_term_2_by_epoch_sparse(batch_delta_X_Y_prod, batch_offsets)
            batch_delta_X_Y_prod = origin_X_Y_prod[curr_matched_ids]
            t1 = time.time()
            
            
            
            
#             X_Y_prod_theta_mult = batch_delta_X_Y_prod.dot(theta)
            
#             gradient = 0-batch_delta_X_Y_prod.transpose().dot(np.multiply(X_Y_prod_theta_mult, batch_weights) + batch_offsets)
            gradient = 0-batch_delta_X_Y_prod.transpose().dot(curr_X_w_prod.dot(theta) + batch_offsets)
#             curr_off_sets = batch_offsets
            

#             sub_term2 = batch_delta_X_Y_prod.transpose().dot(curr_off_sets)
#             sub_term2 = other_X_Y_prod.transpose().dot(curr_off_sets)
#             sub_term2 = sparse_matrix_mult_sparseX_mod1(X_offset_prod_this_super_iter[curr_non_matched_ids], np.array(range(curr_non_matched_ids.shape[0])))#].sum(axis = 0).toarray()
#             sub_term2 = 0
#             if curr_non_matched_ids.shape[0] > 0:
#                 sub_term2 = np.asarray(X_offset_prod_this_super_iter[curr_non_matched_ids].sum(axis = 0))
#                 sub_term2 = np.reshape(sub_term2, theta.shape)
#             
#             X_theta_mult = batch_delta_X.dot(theta)
#             
#             x_theta_w = np.multiply(X_theta_mult, batch_weights)
            
            
#             curr_weights = batch_weights
            
            
            
#             t1 = time.time()
#             curr_sub_term2 = sub_term2#np.reshape(sub_term2, (-1,1))
#             t2 = time.time()
#             
#             overhead += (t2 - t1)
#             sub_term_1 = prepare_term_1_batch2_theta_sparse2(curr_X_w_prod, batch_X_multi_theta)
            
#             sub_term_1 = curr_X_w_prod.transpose().dot(X_theta_mult)
            
            
#             sub_term_1 = batch_delta_X.transpose().dot(x_theta_w)
#             
#                  
# #             sub_term_1 = curr_X_w_prod.transpose().dot(np.reshape(batch_X_multi_theta, (-1,1)))
#             if epoch < cut_off_epoch:
#                 gradient = 0 - (sub_term_1 + term2[epoch] - sub_term2)
#             else:
#                 gradient = 0 - (sub_term_1 + avg_term2 - sub_term2)
            
            
            
            

#             t3 = time.time()
#             t1 = time.time()
#             grad1 = output/(curr_matched_ids.shape[0])
# #             t4 = time.time()
# #             overhead3 += (t4 - t3)
#             
#             
# #             t1 = time.time()
#             grad2 = beta*theta
# #             t2 = time.time()
# #               
# #             overhead2 += (t2 - t1)
#             
# #             t1 = time.time()
#             gradient = grad1 + grad2
# #             t2 = time.time()
# #             overhead4 += (t2 - t1)
#             
#             
#             theta -= alpha*gradient
#             
#             t2 = time.time()
#             
#             overhead += (t2 - t1)
            
            
            
            gradient = gradient/(curr_matched_ids.shape[0]) + beta*theta
            
            theta = theta - alpha*gradient
            
            t2 = time.time()
            overhead += (t2 - t1)
            
            
            epoch = epoch + 1
            
            

            
            if epoch >= mini_batch_epoch:
                end = True
                break
                
                
                       
#             else:
#                 
# 
#                 batch_offsets = offsets_this_super_iteration[curr_matched_ids]
#                 
#                 batch_X_multi_theta = batch_delta_X.dot(theta)
# 
#                 sub_term_1 = prepare_term_1_batch2_theta_sparse2(curr_X_w_prod, batch_X_multi_theta)
# 
# 
#                 sub_term2 = compute_sub_term_2_by_epoch_sparse(batch_delta_X_Y_prod, batch_offsets)
# 
#                 output = 0 - (sub_term_1 + (np.reshape(sub_term2, (-1,1))))
# 
#                 
#                 gradient = (output)/(end_id - i - curr_matched_ids.shape[0]) + theta*beta     
#                 
#                 theta = (theta - gradient*alpha)  
#                 
#                 epoch = epoch + 1
#             
#                 if epoch >= mini_batch_epoch:
#                     end = True
#                     
#                     break
        
               
        
        if end == True:
            break
        
    
#     print('overhead::', overhead)
#       
#     print('overhead2::', overhead2)
#       
#     print('overhead3::', overhead3)
#       
#     print('overhead4::', overhead4)
#       
#     print('t_time::', t_time)
    
     
    
    final_res = torch.from_numpy(theta).type(torch.DoubleTensor) 
    
    
    
    
    print('t_time2::', t_time2)
    
    return final_res









def compute_model_parameter_by_provenance_sparse7(X_w_prod, X_off_prod, theta_list, gradient_list, X, Y, origin_X_Y_prod, weights, offsets, selected_rows, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, dim, theta, mini_batch_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, term2, avg_term2):
    
    total_time = 0.0
    
#     min_batch_num_per_epoch = int((origin_X.shape[0] - 1)/batch_size) + 1
    cut_off_super_iteration = int((weights.shape[0]-1)/dim[0])

    '''T, |delta_X|, q^2'''

#     delta_X = origin_X[delta_ids]


    epoch = 0

    overhead = 0
     
    overhead2 = 0
     
    overhead3 = 0
    overhead4 = 0
     
    t_time = 0
    t_time2 = 0

#     overhead3 = 0
        
    delta_ids_set = set(selected_rows.view(-1).tolist())    
    
#     X = origin_X
        
        
#     Y = origin_Y
    
#     vectorized_theta = Variable(torch.reshape(torch.t(theta), [-1,1]))
    
#     theta = Variable(theta)
    theta = theta.numpy()
    
    avg_sub_term_1 = 0
    
    avg_sub_term_2 = 0
    
#     avg_term1 = None
#     
#     avg_term2 = None
#     
    
    end = False
    
    avg_batch_weights = 0
    
#     for k in range(max_epoch):
    for k in range(random_ids_multi_super_iterations.shape[0]):

        random_ids = random_ids_multi_super_iterations[k]
        
        
#         sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
#         all_indexes = np.sort(np.searchsorted(random_ids.numpy(), delta_ids.numpy()))
        
#         found_res = np.searchsorted(random_ids.numpy(),delta_ids.numpy(),sorter = sort_idx)
#         
#         all_indexes = np.sort(sort_idx[found_res])
        
#         all_indexes = np.sort(sort_idx[selected_rows])
        
        
        
        end_id_super_iteration = (k + 1)*dim[0]
        
        id_start = 0
    
        id_end = 0
        
        if end_id_super_iteration > weights.shape[0]:
            end_id_super_iteration = weights.shape[0]
        
        
#         weights_this_super_iteration = weights[super_iter_id*dim[0]:end_id_super_iteration]
#         
#         offsets_this_super_iteration = offsets[super_iter_id*dim[0]:end_id_super_iteration]
         
        
#         X_w_prod_this_super_iter = X_w_prod[super_iter_id]
#         X_offset_prod_this_super_iter = X_off_prod[super_iter_id]
           
        for i in range(0, X.shape[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            
            
            
#             while 1:
#                 if id_end >= all_indexes.shape[0] or all_indexes[id_end] >= end_id:
#                     break
#                 
#                 id_end = id_end + 1

#             if all_indexes[-1] < end_id:
#                 id_end = all_indexes.shape[0]
#             else:
#                 id_end = np.argmax(all_indexes >= end_id)
#             
#             curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
#             
#             curr_matched_ids_size = curr_matched_ids.shape[0]



            curr_rand_ids = random_ids[i:end_id]
             
             
            curr_matched_ids = get_subset_data_per_epoch(curr_rand_ids, delta_ids_set)
            
            curr_matched_ids_size = curr_matched_ids.shape[0]            
            if curr_matched_ids_size <= 0:
                continue
#             ttt1 = time.time()    

#             batch_delta_X = X[curr_matched_ids]
            
            curr_rand_ids = curr_matched_ids + k*dim[0]
            
            batch_weights = weights[curr_rand_ids]
#             batch_offsets = offsets_this_super_iteration[curr_matched_ids]
            batch_offsets = offsets[curr_rand_ids]


#             other_X_Y_prod = origin_X_Y_prod[curr_non_matched_ids].tocsc()
             
#             curr_X_w_prod = X_w_prod_this_super_iter[curr_matched_ids]
             
            
#             ttt2 = time.time()
#         
#             t_time2 += (ttt2 - ttt1) 
#             batch_X_multi_theta = batch_delta_X.dot(theta)
            
#             sub_term2 = compute_sub_term_2_by_epoch_sparse(batch_delta_X_Y_prod, batch_offsets)
            batch_delta_X_Y_prod = origin_X_Y_prod[curr_matched_ids]
            t1 = time.time()
            
            
            
            
            X_Y_prod_theta_mult = batch_delta_X_Y_prod.dot(theta)
            
            gradient = 0-batch_delta_X_Y_prod.transpose().dot(X_Y_prod_theta_mult*batch_weights + batch_offsets)
            
            
#             del batch_delta_X_Y_prod
#             gradient = 0-batch_delta_X_Y_prod.transpose().dot(curr_X_w_prod.dot(theta) + batch_offsets)
#             curr_off_sets = batch_offsets
            

#             sub_term2 = batch_delta_X_Y_prod.transpose().dot(curr_off_sets)
#             sub_term2 = other_X_Y_prod.transpose().dot(curr_off_sets)
#             sub_term2 = sparse_matrix_mult_sparseX_mod1(X_offset_prod_this_super_iter[curr_non_matched_ids], np.array(range(curr_non_matched_ids.shape[0])))#].sum(axis = 0).toarray()
#             sub_term2 = 0
#             if curr_non_matched_ids.shape[0] > 0:
#                 sub_term2 = np.asarray(X_offset_prod_this_super_iter[curr_non_matched_ids].sum(axis = 0))
#                 sub_term2 = np.reshape(sub_term2, theta.shape)
#             
#             X_theta_mult = batch_delta_X.dot(theta)
#             
#             x_theta_w = np.multiply(X_theta_mult, batch_weights)
            
            
#             curr_weights = batch_weights
            
            
            
#             t1 = time.time()
#             curr_sub_term2 = sub_term2#np.reshape(sub_term2, (-1,1))
#             t2 = time.time()
#             
#             overhead += (t2 - t1)
#             sub_term_1 = prepare_term_1_batch2_theta_sparse2(curr_X_w_prod, batch_X_multi_theta)
            
#             sub_term_1 = curr_X_w_prod.transpose().dot(X_theta_mult)
            
            
#             sub_term_1 = batch_delta_X.transpose().dot(x_theta_w)
#             
#                  
# #             sub_term_1 = curr_X_w_prod.transpose().dot(np.reshape(batch_X_multi_theta, (-1,1)))
#             if epoch < cut_off_epoch:
#                 gradient = 0 - (sub_term_1 + term2[epoch] - sub_term2)
#             else:
#                 gradient = 0 - (sub_term_1 + avg_term2 - sub_term2)
            
            
            
            

#             t3 = time.time()
#             t1 = time.time()
#             grad1 = output/(curr_matched_ids.shape[0])
# #             t4 = time.time()
# #             overhead3 += (t4 - t3)
#             
#             
# #             t1 = time.time()
#             grad2 = beta*theta
# #             t2 = time.time()
# #               
# #             overhead2 += (t2 - t1)
#             
# #             t1 = time.time()
#             gradient = grad1 + grad2
# #             t2 = time.time()
# #             overhead4 += (t2 - t1)
#             
#             
#             theta -= alpha*gradient
#             
#             t2 = time.time()
#             
#             overhead += (t2 - t1)
            
            
            
            gradient = gradient/(curr_matched_ids_size) + beta*theta
            
            theta -= alpha*gradient
            
            t2 = time.time()
            overhead += (t2 - t1)
            
            
            epoch += 1
            
            
            id_start = id_end
            
            if epoch >= mini_batch_epoch:
                end = True
                break
                
                
                       
#             else:
#                 
# 
#                 batch_offsets = offsets_this_super_iteration[curr_matched_ids]
#                 
#                 batch_X_multi_theta = batch_delta_X.dot(theta)
# 
#                 sub_term_1 = prepare_term_1_batch2_theta_sparse2(curr_X_w_prod, batch_X_multi_theta)
# 
# 
#                 sub_term2 = compute_sub_term_2_by_epoch_sparse(batch_delta_X_Y_prod, batch_offsets)
# 
#                 output = 0 - (sub_term_1 + (np.reshape(sub_term2, (-1,1))))
# 
#                 
#                 gradient = (output)/(end_id - i - curr_matched_ids.shape[0]) + theta*beta     
#                 
#                 theta = (theta - gradient*alpha)  
#                 
#                 epoch = epoch + 1
#             
#                 if epoch >= mini_batch_epoch:
#                     end = True
#                     
#                     break
        
               
        
        if end == True:
            break
        
    
    print('overhead::', overhead)
#       
#     print('overhead2::', overhead2)
#       
#     print('overhead3::', overhead3)
#       
#     print('overhead4::', overhead4)
#       
#     print('t_time::', t_time)
    
     
    
    final_res = torch.from_numpy(theta).type(torch.DoubleTensor) 
    
    
    
    
#     print('t_time2::', t_time2)
    
    return final_res






def compute_model_parameter_by_provenance_sparse2(X_w_prod, X_offset_prod, theta_list, gradient_list, X, Y, origin_X_Y_prod, weights, offsets, selected_rows, random_ids_multi_super_iterations, dim, theta, mini_batch_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, term2, avg_term2):
    
    total_time = 0.0
    
#     min_batch_num_per_epoch = int((origin_X.shape[0] - 1)/batch_size) + 1
    cut_off_super_iteration = int((weights.shape[0]-1)/dim[0])

    '''T, |delta_X|, q^2'''

#     delta_X = origin_X[delta_ids]


    epoch = 0

    overhead = 0
     
    overhead2 = 0
     
    overhead3 = 0
    overhead4 = 0
     
    t_time = 0
    t_time2 = 0

#     overhead3 = 0
        
    delta_ids_set = set(selected_rows.view(-1).tolist())    
    
#     X = origin_X
        
        
#     Y = origin_Y
    
#     vectorized_theta = Variable(torch.reshape(torch.t(theta), [-1,1]))
    
#     theta = Variable(theta)
    theta = theta.numpy()
    
    avg_sub_term_1 = 0
    
    avg_sub_term_2 = 0
    
#     avg_term1 = None
#     
#     avg_term2 = None
#     
    
    end = False
    
    avg_batch_weights = 0
    
#     for k in range(max_epoch):
    for k in range(random_ids_multi_super_iterations.shape[0]):

        random_ids = random_ids_multi_super_iterations[k]
        
        
        super_iter_id = k
        
        if k > cut_off_super_iteration:
            super_iter_id = cut_off_super_iteration
        
        end_id_super_iteration = (super_iter_id + 1)*dim[0]
        
        
        if end_id_super_iteration > weights.shape[0]:
            end_id_super_iteration = weights.shape[0]
        
        
        
        weights_this_super_iteration = weights[super_iter_id*dim[0]:end_id_super_iteration]
#         
        offsets_this_super_iteration = offsets[super_iter_id*dim[0]:end_id_super_iteration]
         
        
#         X_w_prod_this_super_iter = X_w_prod[super_iter_id]
#         X_offset_prod_this_super_iter = X_offset_prod[super_iter_id]
           
        for i in range(0, X.shape[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            

            curr_rand_ids = random_ids[i:end_id]
            
            
            curr_matched_ids, curr_non_matched_ids = get_subset_data_per_epoch2(curr_rand_ids, delta_ids_set)
            
            
            if curr_matched_ids.shape[0] <= 0:
                continue
            ttt1 = time.time()    

            batch_delta_X = X[curr_matched_ids]
            
#             batch_delta_X_Y_prod = origin_X_Y_prod[curr_matched_ids]

            sub_term2 = 0
            if curr_non_matched_ids.shape[0] > 0:
                other_X_Y_prod = origin_X_Y_prod[curr_non_matched_ids].tocsc()
             
#             curr_X_w_prod = X_w_prod_this_super_iter[curr_matched_ids]
                batch_offsets = offsets_this_super_iteration[curr_non_matched_ids]
                
                sub_term2 = other_X_Y_prod.transpose().dot(batch_offsets)
            
            batch_weights = weights_this_super_iteration[curr_matched_ids]
            
            ttt2 = time.time()
        
            t_time2 += (ttt2 - ttt1) 
#             batch_X_multi_theta = batch_delta_X.dot(theta)
            
#             sub_term2 = compute_sub_term_2_by_epoch_sparse(batch_delta_X_Y_prod, batch_offsets)

            t1 = time.time()
#             curr_off_sets = batch_offsets
            

#             sub_term2 = batch_delta_X_Y_prod.transpose().dot(curr_off_sets)
            
            
            X_theta_mult = batch_delta_X.dot(theta)
            
            x_theta_w = np.multiply(X_theta_mult, batch_weights)
            
            
#             curr_weights = batch_weights
            
            
            
#             t1 = time.time()
#             curr_sub_term2 = sub_term2#np.reshape(sub_term2, (-1,1))
#             t2 = time.time()
#             
#             overhead += (t2 - t1)
#             sub_term_1 = prepare_term_1_batch2_theta_sparse2(curr_X_w_prod, batch_X_multi_theta)
            
#             sub_term_1 = curr_X_w_prod.transpose().dot(X_theta_mult)
            
            
            sub_term_1 = batch_delta_X.transpose().dot(x_theta_w)
            
                 
#             sub_term_1 = curr_X_w_prod.transpose().dot(np.reshape(batch_X_multi_theta, (-1,1)))
            if epoch < cut_off_epoch:
                gradient = 0 - (sub_term_1 + term2[epoch] - sub_term2)
            else:
                gradient = 0 - (sub_term_1 + avg_term2 - sub_term2)
            
            
            
            

#             t3 = time.time()
#             t1 = time.time()
#             grad1 = output/(curr_matched_ids.shape[0])
# #             t4 = time.time()
# #             overhead3 += (t4 - t3)
#             
#             
# #             t1 = time.time()
#             grad2 = beta*theta
# #             t2 = time.time()
# #               
# #             overhead2 += (t2 - t1)
#             
# #             t1 = time.time()
#             gradient = grad1 + grad2
# #             t2 = time.time()
# #             overhead4 += (t2 - t1)
#             
#             
#             theta -= alpha*gradient
#             
#             t2 = time.time()
#             
#             overhead += (t2 - t1)
            
            
            
            gradient = gradient/(curr_matched_ids.shape[0]) + beta*theta
            
            theta = theta - alpha*gradient
            
            t2 = time.time()
            overhead += (t2 - t1)
            
            
            epoch = epoch + 1
            
            

            
            if epoch >= mini_batch_epoch:
                end = True
                break
                
                
                       
#             else:
#                 
# 
#                 batch_offsets = offsets_this_super_iteration[curr_matched_ids]
#                 
#                 batch_X_multi_theta = batch_delta_X.dot(theta)
# 
#                 sub_term_1 = prepare_term_1_batch2_theta_sparse2(curr_X_w_prod, batch_X_multi_theta)
# 
# 
#                 sub_term2 = compute_sub_term_2_by_epoch_sparse(batch_delta_X_Y_prod, batch_offsets)
# 
#                 output = 0 - (sub_term_1 + (np.reshape(sub_term2, (-1,1))))
# 
#                 
#                 gradient = (output)/(end_id - i - curr_matched_ids.shape[0]) + theta*beta     
#                 
#                 theta = (theta - gradient*alpha)  
#                 
#                 epoch = epoch + 1
#             
#                 if epoch >= mini_batch_epoch:
#                     end = True
#                     
#                     break
        
               
        
        if end == True:
            break
        
    
    print('overhead::', overhead)
     
    print('overhead2::', overhead2)
     
    print('overhead3::', overhead3)
     
    print('overhead4::', overhead4)
     
    print('t_time::', t_time)
    
     
    
    final_res = torch.from_numpy(theta).type(torch.DoubleTensor) 
    
    
    
    
    print('t_time2::', t_time2)
    
    return final_res
    
def compute_model_parameter_by_provenance_sparse4(X_w_prod, X_offset_prod, theta_list, X_cat_X_Y_mult, X, Y, origin_X_Y_prod, weights, offsets, selected_rows, random_ids_multi_super_iterations, dim, theta, mini_batch_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, term2, avg_term2):
    
    total_time = 0.0
    
#     min_batch_num_per_epoch = int((origin_X.shape[0] - 1)/batch_size) + 1
    cut_off_super_iteration = int((weights.shape[0]-1)/dim[0])

    '''T, |delta_X|, q^2'''

#     delta_X = origin_X[delta_ids]


    epoch = 0

    overhead = 0
     
    overhead2 = 0
     
    overhead3 = 0
    overhead4 = 0
     
    t_time = 0
    t_time2 = 0

#     overhead3 = 0
        
    delta_ids_set = set(selected_rows.view(-1).tolist())    
    
#     X = origin_X
        
        
#     Y = origin_Y
    
#     vectorized_theta = Variable(torch.reshape(torch.t(theta), [-1,1]))
    
#     theta = Variable(theta)
    theta = theta.numpy()
    
    avg_sub_term_1 = 0
    
    avg_sub_term_2 = 0
    
#     avg_term1 = None
#     
#     avg_term2 = None
#     
    
    end = False
    
    avg_batch_weights = 0
    
#     for k in range(max_epoch):
    for k in range(random_ids_multi_super_iterations.shape[0]):

        random_ids = random_ids_multi_super_iterations[k]
        
        
        super_iter_id = k
        
        if k > cut_off_super_iteration:
            super_iter_id = cut_off_super_iteration
        
        end_id_super_iteration = (super_iter_id + 1)*dim[0]
        
        
        if end_id_super_iteration > weights.shape[0]:
            end_id_super_iteration = weights.shape[0]
        
        
        
        weights_this_super_iteration = weights[super_iter_id*dim[0]:end_id_super_iteration]
#         
        offsets_this_super_iteration = offsets[super_iter_id*dim[0]:end_id_super_iteration]
         
        
#         X_w_prod_this_super_iter = X_w_prod[super_iter_id]
#         X_offset_prod_this_super_iter = X_offset_prod[super_iter_id]
           
        for i in range(0, X.shape[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            

            curr_rand_ids = random_ids[i:end_id]
            
            
            curr_matched_ids, curr_non_matched_ids = get_subset_data_per_epoch2(curr_rand_ids, delta_ids_set)
            
            
            if curr_matched_ids.shape[0] <= 0:
                continue
            ttt1 = time.time()    

            curr_X_Y_mult = X_cat_X_Y_mult[curr_matched_ids]
            
#             batch_delta_X_Y_prod = origin_X_Y_prod[curr_matched_ids]

#             other_X_Y_prod = origin_X_Y_prod[curr_non_matched_ids]
             
#             curr_X_w_prod = X_w_prod_this_super_iter[curr_matched_ids]
            batch_offsets = offsets_this_super_iteration[curr_matched_ids]
            batch_weights = weights_this_super_iteration[curr_matched_ids]
            
            ttt2 = time.time()
        
            t_time2 += (ttt2 - ttt1) 
#             batch_X_multi_theta = batch_delta_X.dot(theta)
            
#             sub_term2 = compute_sub_term_2_by_epoch_sparse(batch_delta_X_Y_prod, batch_offsets)

            t1 = time.time()
            curr_off_sets = batch_offsets
            

#             sub_term2 = batch_delta_X_Y_prod.transpose().dot(curr_off_sets)
            sub_term2 = curr_X_Y_mult[:,X.shape[1]:].transpose().dot(curr_off_sets)
            
            
            curr_X = curr_X_Y_mult[:,:X.shape[1]]
            
            X_theta_mult = curr_X.dot(theta)
            
            
            
            
            curr_weights = batch_weights
            
            
            
#             t1 = time.time()
#             curr_sub_term2 = sub_term2#np.reshape(sub_term2, (-1,1))
#             t2 = time.time()
#             
#             overhead += (t2 - t1)
#             sub_term_1 = prepare_term_1_batch2_theta_sparse2(curr_X_w_prod, batch_X_multi_theta)
            
#             sub_term_1 = curr_X_w_prod.transpose().dot(X_theta_mult)
            
            
            sub_term_1 = curr_X.transpose().dot(np.multiply(X_theta_mult, curr_weights))
            
                 
#             sub_term_1 = curr_X_w_prod.transpose().dot(np.reshape(batch_X_multi_theta, (-1,1)))
            if epoch < cut_off_epoch:
                gradient = 0 - (sub_term_1 + term2[epoch] - sub_term2)
            else:
                gradient = 0 - (sub_term_1 + avg_term2 - sub_term2)
            
            
            
            

#             t3 = time.time()
#             t1 = time.time()
#             grad1 = output/(curr_matched_ids.shape[0])
# #             t4 = time.time()
# #             overhead3 += (t4 - t3)
#             
#             
# #             t1 = time.time()
#             grad2 = beta*theta
# #             t2 = time.time()
# #               
# #             overhead2 += (t2 - t1)
#             
# #             t1 = time.time()
#             gradient = grad1 + grad2
# #             t2 = time.time()
# #             overhead4 += (t2 - t1)
#             
#             
#             theta -= alpha*gradient
#             
#             t2 = time.time()
#             
#             overhead += (t2 - t1)
            
            
            
            gradient = np.reshape(gradient, (theta.shape))/(curr_matched_ids.shape[0]) + beta*theta
            
            theta = theta - alpha*gradient
            
            t2 = time.time()
            overhead += (t2 - t1)
            
            
            epoch = epoch + 1
            
            

            
            if epoch >= mini_batch_epoch:
                end = True
                break
                
                
                       
#             else:
#                 
# 
#                 batch_offsets = offsets_this_super_iteration[curr_matched_ids]
#                 
#                 batch_X_multi_theta = batch_delta_X.dot(theta)
# 
#                 sub_term_1 = prepare_term_1_batch2_theta_sparse2(curr_X_w_prod, batch_X_multi_theta)
# 
# 
#                 sub_term2 = compute_sub_term_2_by_epoch_sparse(batch_delta_X_Y_prod, batch_offsets)
# 
#                 output = 0 - (sub_term_1 + (np.reshape(sub_term2, (-1,1))))
# 
#                 
#                 gradient = (output)/(end_id - i - curr_matched_ids.shape[0]) + theta*beta     
#                 
#                 theta = (theta - gradient*alpha)  
#                 
#                 epoch = epoch + 1
#             
#                 if epoch >= mini_batch_epoch:
#                     end = True
#                     
#                     break
        
               
        
        if end == True:
            break
        
    
    print('overhead::', overhead)
     
    print('overhead2::', overhead2)
     
    print('overhead3::', overhead3)
     
    print('overhead4::', overhead4)
     
    print('t_time::', t_time)
    
     
    
    final_res = torch.from_numpy(theta).type(torch.DoubleTensor) 
    
    
    
    
    print('t_time2::', t_time2)
    
    return final_res


def compute_model_parameter_by_provenance_sparse3(X_w_prod, X_offset_prod, theta_list, gradient_list, X, Y, origin_X_Y_prod, weights, offsets, selected_rows, random_ids_multi_super_iterations, dim, theta, mini_batch_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, term2, avg_term2):
    
    total_time = 0.0
    
#     min_batch_num_per_epoch = int((origin_X.shape[0] - 1)/batch_size) + 1
    cut_off_super_iteration = int((weights.shape[0]-1)/dim[0])

    '''T, |delta_X|, q^2'''

#     delta_X = origin_X[delta_ids]


    epoch = 0

    overhead = 0
     
    overhead2 = 0
     
    overhead3 = 0
    overhead4 = 0
     
    t_time = 0
    t_time2 = 0

#     overhead3 = 0
        
    delta_ids_set = set(selected_rows.view(-1).tolist())    
    
#     X = origin_X
        
        
#     Y = origin_Y
    
#     vectorized_theta = Variable(torch.reshape(torch.t(theta), [-1,1]))
    
#     theta = Variable(theta)
    theta = theta.numpy()
    
    avg_sub_term_1 = 0
    
    avg_sub_term_2 = 0
    
#     avg_term1 = None
#     
#     avg_term2 = None
#     
    
    end = False
    
    avg_batch_weights = 0
    
#     for k in range(max_epoch):
    for k in range(random_ids_multi_super_iterations.shape[0]):

        random_ids = random_ids_multi_super_iterations[k]
        
        
        super_iter_id = k
        
        if k > cut_off_super_iteration:
            super_iter_id = cut_off_super_iteration
        
        end_id_super_iteration = (super_iter_id + 1)*dim[0]
        
        
        if end_id_super_iteration > weights.shape[0]:
            end_id_super_iteration = weights.shape[0]
        
        
        
#         weights_this_super_iteration = weights[super_iter_id*dim[0]:end_id_super_iteration]
#         
        offsets_this_super_iteration = offsets[super_iter_id*dim[0]:end_id_super_iteration]
         
         
        X_w_prod_this_super_iter = X_w_prod[super_iter_id]
#         X_offset_prod_this_super_iter = X_offset_prod[super_iter_id]
           
        for i in range(0, X.shape[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            

            curr_rand_ids = random_ids[i:end_id]
            
            
            curr_matched_ids, curr_non_matched_ids = get_subset_data_per_epoch2(curr_rand_ids, delta_ids_set)
            
            
            if curr_matched_ids.shape[0] <= 0:
                continue
            ttt1 = time.time()    

            batch_delta_X = X[curr_matched_ids]
            
#             batch_delta_X_Y_prod = origin_X_Y_prod[curr_matched_ids]

            other_X_Y_prod = origin_X_Y_prod[curr_non_matched_ids].tocsc()
             
            curr_X_w_prod = X_w_prod_this_super_iter[curr_matched_ids].tocsc()
            batch_offsets = offsets_this_super_iteration[curr_non_matched_ids]
#             batch_weights = weights_this_super_iteration[curr_matched_ids]
            
            ttt2 = time.time()
        
            t_time2 += (ttt2 - ttt1) 
#             batch_X_multi_theta = batch_delta_X.dot(theta)
            
#             sub_term2 = compute_sub_term_2_by_epoch_sparse(batch_delta_X_Y_prod, batch_offsets)

            t1 = time.time()
            curr_off_sets = batch_offsets
            

#             sub_term2 = batch_delta_X_Y_prod.transpose().dot(curr_off_sets)
            sub_term2 = other_X_Y_prod.transpose().dot(curr_off_sets)
            
            
            X_theta_mult = batch_delta_X.dot(theta)
            
            
            
            
#             curr_weights = batch_weights
            
            
            
#             t1 = time.time()
#             curr_sub_term2 = sub_term2#np.reshape(sub_term2, (-1,1))
#             t2 = time.time()
#             
#             overhead += (t2 - t1)
#             sub_term_1 = prepare_term_1_batch2_theta_sparse2(curr_X_w_prod, batch_X_multi_theta)
            
            sub_term_1 = curr_X_w_prod.transpose().dot(X_theta_mult)
            
            
#             sub_term_1 = batch_delta_X.transpose().dot(np.multiply(X_theta_mult, curr_weights))
            
                 
#             sub_term_1 = curr_X_w_prod.transpose().dot(np.reshape(batch_X_multi_theta, (-1,1)))
            if epoch < cut_off_epoch:
                gradient = 0 - (sub_term_1 + term2[epoch] - sub_term2)
            else:
                gradient = 0 - (sub_term_1 + avg_term2 - sub_term2)
            
            
            
            

#             t3 = time.time()
#             t1 = time.time()
#             grad1 = output/(curr_matched_ids.shape[0])
# #             t4 = time.time()
# #             overhead3 += (t4 - t3)
#             
#             
# #             t1 = time.time()
#             grad2 = beta*theta
# #             t2 = time.time()
# #               
# #             overhead2 += (t2 - t1)
#             
# #             t1 = time.time()
#             gradient = grad1 + grad2
# #             t2 = time.time()
# #             overhead4 += (t2 - t1)
#             
#             
#             theta -= alpha*gradient
#             
#             t2 = time.time()
#             
#             overhead += (t2 - t1)
            
            
            
            gradient = np.reshape(gradient, (theta.shape))/(curr_matched_ids.shape[0]) + beta*theta
            
            theta = theta - alpha*gradient
            
            t2 = time.time()
            overhead += (t2 - t1)
            
            
            epoch = epoch + 1
            
            

            
            if epoch >= mini_batch_epoch:
                end = True
                break
                
                
                       
#             else:
#                 
# 
#                 batch_offsets = offsets_this_super_iteration[curr_matched_ids]
#                 
#                 batch_X_multi_theta = batch_delta_X.dot(theta)
# 
#                 sub_term_1 = prepare_term_1_batch2_theta_sparse2(curr_X_w_prod, batch_X_multi_theta)
# 
# 
#                 sub_term2 = compute_sub_term_2_by_epoch_sparse(batch_delta_X_Y_prod, batch_offsets)
# 
#                 output = 0 - (sub_term_1 + (np.reshape(sub_term2, (-1,1))))
# 
#                 
#                 gradient = (output)/(end_id - i - curr_matched_ids.shape[0]) + theta*beta     
#                 
#                 theta = (theta - gradient*alpha)  
#                 
#                 epoch = epoch + 1
#             
#                 if epoch >= mini_batch_epoch:
#                     end = True
#                     
#                     break
        
               
        
        if end == True:
            break
        
    
    print('overhead::', overhead)
     
    print('overhead2::', overhead2)
     
    print('overhead3::', overhead3)
     
    print('overhead4::', overhead4)
     
    print('t_time::', t_time)
    
     
    
    final_res = torch.from_numpy(theta).type(torch.DoubleTensor) 
    
    
    
    
    print('t_time2::', t_time2)
    
    return final_res


def compute_model_parameter_by_provenance(mini_batch_epoch, theta_list, gradient_list, origin_X, origin_Y, origin_X_Y_prod, weights, offsets, delta_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, term1, term2, dim, theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, avg_term_1, avg_term_2, origin_theta_list, origin_grad_list, u_list, v_s_list):
    
    total_time = 0.0
    
    min_batch_num_per_epoch = int((origin_X.shape[0] - 1)/batch_size) + 1
    cut_off_super_iteration = int((weights.shape[0]-1)/dim[0])

    '''T, |delta_X|, q^2'''

    delta_X = origin_X[delta_ids]


    epoch = 0

    overhead = 0

#     
        
    X = origin_X
        
        
    Y = origin_Y
    
#     vectorized_theta = Variable(torch.reshape(torch.t(theta), [-1,1]))
    
    theta = Variable(theta)
    
    avg_sub_term_1 = 0
    
    avg_sub_term_2 = 0
    
    
#     avg_term_1 = 0
#     
#     avg_term_2 = 0
    
#     for k in range(max_epoch):
    end = False
    k = 0
    
    res_list = []
#     
#     output_list = []
#     
#     sub_term_1_list = []
#     
#     sub_term_2_list = []
#     
#     term_1_list = []
#     
#     term_2_list = []
    
#     while epoch < mini_batch_epoch:
    for k in range(random_ids_multi_super_iterations.shape[0]):
    
        random_ids = random_ids_multi_super_iterations[k]
        
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
#         all_indexes = np.sort(np.searchsorted(random_ids.numpy(), delta_ids.numpy()))
        
#         found_res = np.searchsorted(random_ids.numpy(),delta_ids.numpy(),sorter = sort_idx)
#         
#         all_indexes = np.sort(sort_idx[found_res])
        
        all_indexes = np.sort(sort_idx[delta_ids])
        
        
#         super_iter_id = k
#         
#         if k > cut_off_super_iteration:
#             super_iter_id = cut_off_super_iteration
        
        end_id_super_iteration = (k + 1)*dim[0]
        
        id_start = 0
    
        id_end = 0
        
        if end_id_super_iteration > weights.shape[0]:
            end_id_super_iteration = weights.shape[0]
            
        
        
#         weights_this_super_iteration = weights[super_iter_id*dim[0]:end_id_super_iteration]
#         
#         offsets_this_super_iteration = offsets[super_iter_id*dim[0]:end_id_super_iteration]
        
        for i in range(0, X.shape[0], batch_size):
        
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            
            
#             curr_rand_ids = Variable(random_ids[i:end_id])
#             
#             
#             curr_matched_ids = (get_subset_data_per_epoch(curr_rand_ids, delta_ids_set))
            
            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id.item())
#             while 1:
#                 if id_end >= all_indexes.shape[0] or all_indexes[id_end] >= end_id:
#                     break
#                 
#                 id_end = id_end + 1
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
            
            curr_matched_ids_size = curr_matched_ids.shape[0]
            
            
            
            if curr_matched_ids_size > 0:
                
                if (end_id - i - curr_matched_ids.shape[0]) <= 0:
                    
                    epoch += 1
                    
                    continue
                
                
                
                batch_delta_X = (X[curr_matched_ids])
                
#                 batch_delta_Y = (Y[curr_matched_ids])
                
                batch_delta_X_Y_prod = origin_X_Y_prod[curr_matched_ids]

            
#             if epoch < cut_off_epoch:
            sub_term2 = 0
        
            if curr_matched_ids_size > 0:
                coeff_rand_ids = curr_matched_ids + k*dim[0]
                
                batch_weights = weights[coeff_rand_ids]
            
                batch_offsets = offsets[coeff_rand_ids]
            
#                 t1 = time.time()
                
                batch_X_multi_theta = Variable(torch.mm(batch_delta_X, theta))
        
                sub_term2 = compute_sub_term_2_by_epoch(batch_delta_X_Y_prod, batch_offsets).view(-1,1)
        
#                 sub_term2 = (prepare_sub_term_2(batch_delta_X, batch_offsets, batch_delta_X.shape, num_class))
#             full_term1 = term1[epoch]
            
            full_term2 = term2[epoch]
        
#             if epoch >= cut_off_epoch - min_batch_num_per_epoch:
#                 
#                 sub_term_1_without_weights = 0
#                 
#                 if curr_matched_ids.shape[0] > 0:
#                     sub_term_1_without_weights = prepate_term_1_batch_by_epoch(batch_delta_X, batch_weights)
#                 
# #                     sub_term_1_without_weights = prepare_term_1_batch2_1(batch_delta_X, batch_weights, batch_delta_X.shape, num_class)
# 
#                 avg_sub_term_1 += (sub_term_1_without_weights)
#                 avg_sub_term_2 += (sub_term2)
# #                     avg_term_1 += full_term1
# #                     avg_term_2 += full_term2
#                 
#                 output = 0 - (torch.mm((full_term1 - sub_term_1_without_weights), theta) + (full_term2.view(-1,1) - sub_term2))
#                 
#                 del sub_term_1_without_weights
#             
#             else:
            sub_term_1 = 0
            if curr_matched_ids_size > 0:
                sub_term_1 =  prepare_term_1_batch2_theta(batch_delta_X, batch_X_multi_theta, batch_weights)
                
#                     sub_term_1 = (torch.t(compute_sub_term_1(batch_X_multi_theta, batch_delta_X, batch_weights, batch_X_multi_theta.shape, num_class)))
                
#                     vectorized_sub_term_1 = Variable(torch.reshape(sub_term_1, [-1,1]))
#             print(epoch)
            output = 0 - (torch.mm(u_list[epoch], torch.mm(v_s_list[epoch], theta)) - sub_term_1 + (full_term2.view(-1,1) - sub_term2))
#                     sub_term_1_list.append(sub_term_1)

            del sub_term_1

            
#                 term_1_list.append(batch_delta_X)
#                 
#                 term_2_list.append(batch_X_multi_theta)
#                 
#                 sub_term_2_list.append(batch_weights)
                
#                 delta_x_sum_by_class = compute_x_sum_by_class(batch_delta_X, batch_delta_Y, num_class, batch_delta_X.shape)
                
#                 x_sum_by_class = x_sum_by_class_list[epoch] - delta_x_sum_by_class
                
#                 origin_theta_this_epoch = origin_theta_list[epoch - 1]
#                 
#                 origin_grad_this_epoch = origin_grad_list[epoch]
#                 
#                 if epoch > 0:
#                     computed_origin_grad_this_epoch =  (0 - (torch.mm((full_term1), origin_theta_this_epoch) + (full_term2.view(-1,1))))/(end_id - i) + beta*origin_theta_this_epoch
#                 else:
#                     computed_origin_grad_this_epoch =  (0 - (torch.mm((full_term1), theta) + (full_term2.view(-1,1))))/(end_id - i) + beta*theta
                
#                 print(torch.norm(computed_origin_grad_this_epoch - origin_grad_this_epoch))
                
                
            if curr_matched_ids.shape[0] > 0:
                gradient = output/(end_id - i - curr_matched_ids_size) + beta*theta
            else:
                gradient = output/(end_id - i) + beta*theta
    
#                 print('gradient_diff:', torch.norm(output - gradient_list[epoch]))
    
            del output
            
            del sub_term2
                            
            theta = (theta - alpha*gradient)  
            
            
#                 print('theta_diff:', torch.norm(theta - theta_list[epoch]))
            
            
            del gradient
            
#                 theta = torch.t(vectorized_theta.view(num_class, dim[1]))
            
#                 t2 = time.time()
#                  
#                 overhead += (t2 - t1)
            
            epoch = epoch + 1
            
            id_start = id_end
                
#                 res_list.append(theta)

                        
#             else:
#                 
#                 if epoch == cut_off_epoch:
#                     avg_sub_term_1 = avg_sub_term_1/min_batch_num_per_epoch
#                     
#                     avg_sub_term_2 = avg_sub_term_2/min_batch_num_per_epoch
#                 
# #                     avg_term_1 = avg_term_1/min_batch_num_per_epoch
# #                     
# #                     avg_term_2 = avg_term_2/min_batch_num_per_epoch
#                 
#                     full_term1 = (avg_term_1 - avg_sub_term_1)
#                 
#                     full_term2 = (avg_term_2.view(-1,1) - avg_sub_term_2)
#                 
#                 
#                 output = 0 - (torch.mm(full_term1, theta) + full_term2.view(-1,1))
#                 
#                 
# #                 delta_x_sum_by_class = compute_x_sum_by_class(batch_delta_X, batch_delta_Y, num_class, batch_delta_X.shape)
#                 
# #                 x_sum_by_class = x_sum_by_class_list[epoch] - delta_x_sum_by_class
# #                 
# #                 del delta_x_sum_by_class
#     
#                 if curr_matched_ids.shape[0] > 0:
#                     gradient = (output)/(end_id - i - batch_delta_X.shape[0]) + beta*theta
#                 else:
#                     gradient = (output)/(end_id - i) + beta*theta
#     
# #                 print('gradient_diff:', torch.norm(output - gradient_list[epoch]))
# 
#                 del output      
#                 
# #                 del x_sum_by_class
#                 
#                 theta = (theta - alpha*gradient)  
#                 
# #                 res_list.append(theta)
# #                 print('theta_diff:', torch.norm(theta - theta_list[epoch]))
#                 
# #                 print('angle::', torch.dot(theta.view(-1), theta_list[epoch].view(-1))/(torch.norm(theta.view(-1))*torch.norm(theta_list[epoch].view(-1))))
# 
#                 del gradient
                
#                 theta = torch.t(vectorized_theta.view(num_class, dim[1]))
                
#                 epoch = epoch + 1
                
                
            
            if epoch >= mini_batch_epoch:
                end = True
                
                break
        if end:
            break
            
            
#         k = k + 1
    
    print('overhead::', overhead)
    
    print('epoch ::', epoch)
    
#     return theta, res_list, sub_term_1_list, sub_term_2_list, term_1_list, term_2_list
    return theta
    

def compute_model_parameter_by_provenance_2(mini_batch_epoch, theta_list, M, M_inverse, s, origin_X, origin_Y, origin_X_Y_prod, weights, offsets, delta_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, term1, term2, dim, theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, avg_term1, avg_term2, u_list, v_s_list):
    
    total_time = 0.0
    
    min_batch_num_per_epoch = int((origin_X.shape[0] - 1)/batch_size) + 1
    cut_off_super_iteration = int((weights.shape[0]-1)/dim[0]) + 1

    '''T, |delta_X|, q^2'''

    epoch = 0

    overhead = 0
        
    delta_ids_set = set(delta_ids.view(-1).tolist())    
    
    X = origin_X
        
        
    Y = origin_Y
    
#     vectorized_theta = Variable(torch.reshape(torch.t(theta), [-1,1]))
    
    theta = Variable(theta)
    
#     avg_sub_term_1 = 0
#     
#     avg_sub_term_2 = 0
    
    avg_B = 0
    
    avg_s = 0
    
    
    for k in range(cut_off_super_iteration):
    
        random_ids = random_ids_multi_super_iterations[k]
        
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
#         all_indexes = np.sort(np.searchsorted(random_ids.numpy(), delta_ids.numpy()))
        
#         found_res = np.searchsorted(random_ids.numpy(),delta_ids.numpy(),sorter = sort_idx)
#         
#         all_indexes = np.sort(sort_idx[found_res])
        
        all_indexes = np.sort(sort_idx[delta_ids])
        
        
        end_id_super_iteration = (k + 1)*dim[0]
        
        id_start = 0
    
        id_end = 0
        
        if end_id_super_iteration > weights.shape[0]:
            end_id_super_iteration = weights.shape[0]
        
#         super_iter_id = k
#         
#         if k > cut_off_super_iteration:
#             super_iter_id = cut_off_super_iteration
#         
#         end_id_super_iteration = (super_iter_id + 1)*dim[0]
#         
#         
#         if end_id_super_iteration > weights.shape[0]:
#             end_id_super_iteration = weights.shape[0]
#             
#         
#         
#         weights_this_super_iteration = weights[super_iter_id*dim[0]:end_id_super_iteration]
#         
#         offsets_this_super_iteration = offsets[super_iter_id*dim[0]:end_id_super_iteration]
        
        for i in range(0, X.shape[0], batch_size):
        
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            
            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id.item())
#             while 1:
#                 if id_end >= all_indexes.shape[0] or all_indexes[id_end] >= end_id:
#                     break
#                 
#                 id_end = id_end + 1
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
            
            curr_matched_ids_size = curr_matched_ids.shape[0]
            
            
#             curr_rand_ids = Variable(random_ids[i:end_id])
#             
#             
#             curr_matched_ids = (get_subset_data_per_epoch(curr_rand_ids, delta_ids_set))
            
            if curr_matched_ids_size > 0:
                if (end_id - i - curr_matched_ids.shape[0]) <= 0:
                    epoch += 1
                    
                    continue

                
                batch_delta_X = (X[curr_matched_ids])
                
#                 batch_delta_Y = (Y[curr_matched_ids])
                
                batch_delta_X_Y_prod = origin_X_Y_prod[curr_matched_ids]

            
            if epoch < cut_off_epoch:
            
                sub_term2 = 0
                sub_term_1 = 0
                
                if curr_matched_ids_size > 0:

                    coeff_rand_ids = curr_matched_ids + k*dim[0]
                
                    batch_weights = weights[coeff_rand_ids]
                
                    batch_offsets = offsets[coeff_rand_ids]
                    
    #                 t1 = time.time()
                        
                    batch_X_multi_theta = Variable(torch.mm(batch_delta_X, theta))
            
                    sub_term2 = compute_sub_term_2_by_epoch(batch_delta_X_Y_prod, batch_offsets).view(-1,1)
            
#                 sub_term2 = (prepare_sub_term_2(batch_delta_X, batch_offsets, batch_delta_X.shape, num_class))
#                 full_term1 = term1[epoch]
                
                full_term2 = term2[epoch]
            
                if epoch >= cut_off_epoch - min_batch_num_per_epoch:
                    
#                     sub_term_1_without_weights = prepate_term_1_batch_by_epoch(batch_delta_X, batch_weights)
                    
#                     sub_term_1_without_weights = prepare_term_1_batch2_1(batch_delta_X, batch_weights, batch_delta_X.shape, num_class)

#                     avg_sub_term_1 += (sub_term_1_without_weights)
#                     avg_sub_term_2 += (sub_term2)
                    
                    if curr_matched_ids_size > 0:
                        
                        updated_s = prepate_term_1_batch_by_epoch_with_eigen_matrix(batch_delta_X, batch_weights, end_id - i - batch_delta_X.shape[0], M, M_inverse, s*(end_id-i)/dim[0])
                        
#                         print(updated_s)
                        
                        avg_s = avg_s + updated_s
#                         avg_s = avg_s + prepate_term_1_batch_by_epoch_with_eigen_matrix(batch_delta_X, batch_weights, end_id - i, M, M_inverse, s)
                        avg_B = avg_B + alpha/(end_id - i - batch_delta_X.shape[0])*(full_term2.view(-1,1) - sub_term2)
                    else:
                        avg_B = avg_B + alpha/(end_id - i)*(full_term2.view(-1,1) - sub_term2)
                        
                        
                        updated_s = (1 - alpha*beta) + alpha*(s)/dim[0]
#                         updated_s[updated_s > 1] = 1
                        avg_s = avg_s + updated_s
                        
                        
#                     output = torch.mm((full_term1 - sub_term_1_without_weights), vectorized_theta) + (full_term2.view(-1,1) - sub_term2.view(-1,1))
#                     
#                     del sub_term_1_without_weights
                
#                 else:
                if curr_matched_ids_size > 0:

                    sub_term_1 =  prepare_term_1_batch2_theta(batch_delta_X, batch_X_multi_theta, batch_weights)
                
#                     sub_term_1 = (torch.t(compute_sub_term_1(batch_X_multi_theta, batch_delta_X, batch_weights, batch_X_multi_theta.shape, num_class)))
                
#                     vectorized_sub_term_1 = Variable(torch.reshape(sub_term_1, [-1,1]))
                
                output = 0 - (torch.mm(u_list[epoch], torch.mm(v_s_list[epoch], theta)) - sub_term_1 + (full_term2.view(-1,1) - sub_term2))
                
#                 print('sub_term_1_diff::', torch.norm(sub_term_1_list[epoch] - sub_term_1))
#                 
#                 print('sub_term_2_diff::', torch.norm(sub_term_2_list[epoch] - batch_weights))
#                 
#                 print('term_1_diff::', torch.norm(term_1_list[epoch] - batch_delta_X))
#                 
#                 print('term_2_diff::', torch.norm(term_2_list[epoch] - batch_X_multi_theta))
                
                del sub_term_1

            
                
                
#                 delta_x_sum_by_class = compute_x_sum_by_class(batch_delta_X, batch_delta_Y, num_class, batch_delta_X.shape)
                
#                 x_sum_by_class = x_sum_by_class_list[epoch] - delta_x_sum_by_class
                
                if curr_matched_ids_size > 0:
                    gradient = output/(end_id - i - curr_matched_ids_size) + beta*theta
                else:
                    gradient = output/(end_id - i) + beta*theta
    
                del output
                
                del sub_term2
                                
                theta = (theta - alpha*gradient)  
                
#                 print(torch.norm(theta - theta_list[epoch]))
                
                del gradient
                
#                 theta = torch.t(vectorized_theta.view(num_class, dim[1]))
                
#                 t2 = time.time()
#                  
#                 overhead += (t2 - t1)
#                 print('theta_diff::', torch.norm(theta-theta_list[epoch]))
                epoch = epoch + 1
                
                id_start = id_end
                        
#             else:
                
    if mini_batch_epoch > cut_off_epoch:
        avg_s = avg_s/min_batch_num_per_epoch
                    
        avg_B = avg_B/min_batch_num_per_epoch            
        
        avg_s[avg_s > 1] = 1-1e-15
    
        s_power = torch.pow(avg_s, float(mini_batch_epoch - cut_off_epoch))
        
        res1 = M.mul(s_power.view(1,-1))
    
        sub_sum = (1-s_power)/(1-avg_s)
        
        res2 = M.mul(sub_sum.view(1, -1))
           
        theta = torch.mm(res1, torch.mm(M_inverse,theta)) + torch.mm(res2, torch.mm(M_inverse, avg_B))
                
                
                
                
                
#                 if epoch == cut_off_epoch:
#                     avg_sub_term_1 = avg_sub_term_1/min_batch_num_per_epoch
#                     
#                     avg_sub_term_2 = avg_sub_term_2/min_batch_num_per_epoch
#                 
#                     full_term1 = Variable(avg_term1 - avg_sub_term_1)
#                 
#                     full_term2 = Variable(avg_term2 - avg_sub_term_2)
#                 
#                 
#                 output = torch.mm((full_term1), vectorized_theta) + (full_term2.view(-1,1))
#                 
#                 
# #                 delta_x_sum_by_class = compute_x_sum_by_class(batch_delta_X, batch_delta_Y, num_class, batch_delta_X.shape)
#                 
# #                 x_sum_by_class = x_sum_by_class_list[epoch] - delta_x_sum_by_class
# #                 
# #                 del delta_x_sum_by_class
#     
#                 
#                 gradient = (output)/(end_id - i - batch_delta_X.shape[0]) + beta*vectorized_theta     
#     
#                 
#                 del output      
#                 
# #                 del x_sum_by_class
#                 
#                 vectorized_theta = (vectorized_theta - alpha*gradient)  
#                 
#                 del gradient
#                 
#                 theta = torch.t(vectorized_theta.view(num_class, dim[1]))
#                 
#                 
#                 epoch = epoch + 1
            
        
    
    print('overhead::', overhead)
    
    return theta



def compute_model_parameter_by_approx_incremental_0(A, B, last_A, last_B, term1, term2, dim, theta, max_epoch, cut_off_epoch, alpha, beta, batch_size):
    
    total_time = 0.0
    

    
    
    theta = torch.mm(A, theta) + B

    
    print(A.shape, B.shape, theta.shape)
    
    min_batch_num_per_epoch = int((dim[0] - 1)/batch_size) + 1
     
    
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
#      
#     last_A = torch.eye(dim[1], dtype = torch.double)
#     
#     
#     last_B = torch.zeros([dim[1], 1], dtype = torch.double) 
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
#             gradient = -(torch.mm(term1[num], theta) + (term2[num]).view(theta.shape)) + beta*theta
#             
#             
#             
#             curr_A = (1-alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*term1[num]
#             
#             
#             curr_B = term2[num]*alpha
#             
#             
#         else:
#             gradient = -(torch.mm(avg_term1, theta) + (avg_term2).view(theta.shape)) + beta*theta
#             
#             curr_A = (1-alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*avg_term1
#             
#             curr_B = avg_term2*alpha
#         
# #             if num < cut_off_epoch:
# 
#         last_A = torch.mm(last_A, curr_A)
#                 
#         last_B = torch.mm(curr_A, last_B) + curr_B.view(dim[1], 1)
# 
#         theta = theta - alpha * gradient
# 
# 
# #             print('gradient::', gradient)
#          
# #             print('theta::', theta)
#         
#         num += 1

         
    cut_off_super_threshold = int(cut_off_epoch/min_batch_num_per_epoch)
    
     
    for i in range(max_epoch - cut_off_super_threshold):
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


    
    
        
    print('total_time::', total_time)
    
    return theta

def compute_model_parameter_by_approx_incremental_1(A, B, term1, term2, dim, theta, max_epoch, cut_off_epoch, alpha, beta, batch_size):
    
    total_time = 0.0
    
#     pid = os.getpid()
    
#     prev_mem=0
#     
#     print('pid::', pid)
    
    
    num = 0
     
    for i in range(A.shape[0]):
        theta = torch.mm(A[i], theta) + B[i]
         
     
    if A.shape[0] >= max_epoch:
        
        return theta
     
    min_batch_num_per_epoch = int((dim[0] - 1)/batch_size) + 1
     
         
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
     
    last_A = torch.eye(dim[1], dtype = torch.double)
    
    
    last_B = torch.zeros([dim[1], 1], dtype = torch.double) 
    
    for j in range(0, dim[0],batch_size):
        
        end_id = j + batch_size
        
        if end_id > dim[0]:
            end_id = dim[0]



        if num < cut_off_epoch:
            gradient = -(torch.mm(term1[num], theta) + (term2[num]).view(theta.shape)) + beta*theta
            
            
            
            curr_A = (1-alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*term1[num]
            
            
            curr_B = term2[num]*alpha
            
            
        else:
            gradient = -(torch.mm(avg_term1, theta) + (avg_term2).view(theta.shape)) + beta*theta
            
            curr_A = (1-alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*avg_term1
            
            curr_B = avg_term2*alpha
        
#             if num < cut_off_epoch:

        last_A = torch.mm(last_A, curr_A)
                
        last_B = torch.mm(curr_A, last_B) + curr_B.view(dim[1], 1)

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


    
    
        
    print('total_time::', total_time)
    
    return theta


def compute_model_parameter_by_approx_incremental_4(term1, term2, dim, theta, max_epoch, cut_off_epoch, alpha, beta):
    
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
    
    
    s, M = torch.eig(A, True)
    
    s = s[:,0]
    
    s_power = torch.pow(s, float(max_epoch - cut_off_epoch))
    
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
    
    theta = torch.mm(res1, theta) + torch.mm(res2, B)


    
    
        
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
    theta = Variable(torch.zeros([shape[1],1], dtype = torch.float64))
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
    
    
def compute_linear_approx_parameters_sparse(X, Y, X_Y_mult, max_epoch, curr_res_prod_seq):
    
    Pi = create_piecewise_linea_class()

    t1 = time.time()
    
    print(X.shape, len(curr_res_prod_seq))
    
    print(curr_res_prod_seq[0].shape)
    
    print(curr_res_prod_seq[1].shape)
    
    res_prod_tensor = np.concatenate(curr_res_prod_seq, axis = 0)
    

    t2 = time.time()
    
    print('multi_time::', (t2 - t1))
    
    
    '''batch_size*(t*n/batch_size)'''
    
    
    
    
    w_res, b_res = Pi.piecewise_linear_interpolate_coeff_batch2_sparse(res_prod_tensor)
    
#     delta = w_res*res_prod_tensor + b_res - non_linear_function(x)(res_prod_tensor)
#     
#     print('interpolation delta::', delta)
    
    cut_off_epoch = len(curr_res_prod_seq)
     
     
    print('cut_off_epoch::', cut_off_epoch)
    torch.save(cut_off_epoch, git_ignore_folder + 'cut_off_epoch')
    
    
    print('w_res_shape::', w_res.shape)
    
    print('b_res_shape::', b_res.shape)
    
    
    '''batch_num*t + residule, 1'''
    
    return w_res.view(-1,1), b_res.view(-1,1)
    
    

def compute_linear_approx_parameters(X, Y, X_Y_mult, max_epoch, curr_res_prod_seq):
    
    Pi = create_piecewise_linea_class()
    
#     w_seq = []
#     
#     b_seq = []
    
    t1 = time.time()
    
#     print('res_prod_seq::', res_prod_seq)
    
    
    print(X.shape, len(curr_res_prod_seq))
    
    
    res_prod_tensor = torch.cat(curr_res_prod_seq)
    
#     res = torch.mm(X, res_prod_seq)*(Y.repeat(1, epoch))
    
    
    '''n*t'''
    
#     res = torch.mm(X_Y_mult, res_prod_seq)
    
#     print(torch.norm(res2 - res))
    
    t2 = time.time()
    
    print('multi_time::', (t2 - t1))
    
#     print(res_prod_seq.shape)
    
#     print(res)
    
    
    '''batch_size*(t*n/batch_size)'''
    
    
    
    
    w_res, b_res = Pi.piecewise_linear_interpolate_coeff_batch2(res_prod_tensor)
    
    delta = w_res*res_prod_tensor + b_res - non_linear_function(res_prod_tensor)
    
    print('interpolation delta::', delta)
    
#     cut_off_epoch = max_epoch
    cut_off_epoch = len(curr_res_prod_seq)
     
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
     
    print('cut_off_epoch::', cut_off_epoch)
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
    
    
    '''batch_num*t + residule, 1'''
    
    return w_res.view(-1,1), b_res.view(-1,1)
    
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
    
     
#     if torch.sum(Y == 1) > torch.sum(Y == -1):
#         expected_selected_label = 1
#         
#     else:
#         expected_selected_label = -1
#         coeff = 1/coeff 
    
    
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
        
        curr_coeff = torch.sum(curr_coeff[curr_coeff == curr_coeff])
         
#         curr_coeff = torch.sum(curr_coeff[curr_coeff != np.inf and np.isnan(curr_coeff.numpy())])
        
#         print(curr_coeff)
        
        selected_point = -curr_coeff*(X[indices[i]].clone()).clone()
        
        
        if multi_res[indices[i],0]*Y[indices[i], 0] > 0:
            selected_label = curr_class
        else:
            selected_label = -curr_class
        
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



def add_noise_data_sparse(X, Y, num, res):
    
    
#     X_distance = torch.sqrt(torch.bmm(X.view(dim[0], 1, dim[1]), X.view(dim[0],dim[1], 1))).view(-1,1)
    
    positive_X_mean = np.mean(X[Y.view(-1)==1], axis = 0)
    
    negative_X_mean = np.mean(X[Y.view(-1)==-1], axis = 0)
    
    coeff = positive_X_mean/negative_X_mean
    
    coeff = coeff[coeff != np.inf]
        
    coeff = np.sum(coeff[coeff == coeff])
    
    
    print('coeff::', coeff)
    
#     coeff = torch.sum(coeff[coeff != np.inf])
    
    
    
    expected_selected_label =0
    
     
#     if torch.sum(Y == 1) > torch.sum(Y == -1):
#         expected_selected_label = 1
#         
#     else:
#         expected_selected_label = -1
#         coeff = 1/coeff 
    
    
#     multi_res = torch.mm(X, res)
    multi_res = X.dot(res.detach().numpy())
    
    indices = np.argsort(np.absolute(multi_res), axis = 0)[::-1][:X.shape[0]]
    
    sorted = multi_res[indices]
    
#     sorted, indices = torch.sort(torch.abs(multi_res), dim = 0, descending = True)
    
    
    selected_point = None
    
    selected_label = None
    
    
    selected_id = 0
    
#     noise_data_X = torch.zeros((num, X.shape[1]), dtype = torch.double)
    print(num)
    noise_data_X = scipy.sparse.csr_matrix(np.zeros((num, X.shape[1])))

    noise_data_Y = torch.zeros((num, 1), dtype = torch.double)
    
    for i in range(num):
        
        curr_class = Y[indices[i], 0]
        
        if curr_class == 1:
            curr_coeff = positive_X_mean/negative_X_mean#mean_list[curr_class]/mean_list[(curr_class + 1)%(num_class)]        
        
        
        if curr_class == -1:
            curr_coeff = negative_X_mean/positive_X_mean
         
        curr_coeff = curr_coeff[curr_coeff != np.inf]
        
        curr_coeff = np.sum(curr_coeff[curr_coeff == curr_coeff])
         
#         curr_coeff = torch.sum(curr_coeff[curr_coeff != np.inf and np.isnan(curr_coeff.numpy())])
        
#         print(curr_coeff)
        
        selected_point = -curr_coeff*(X[indices[i]].copy())
        
        
        if multi_res[indices[i],0]*Y[indices[i], 0].numpy() > 0:
            selected_label = curr_class
        else:
            selected_label = -curr_class
        
        noise_data_X[i] = selected_point
        
        noise_data_Y[i] = selected_label
        
    
    print(X.shape)
    
    print(noise_data_X.shape)
        
        
    X = scipy.sparse.vstack([X, noise_data_X])
    
    Y = torch.cat([Y.view(-1,1), noise_data_Y], 0)
#     X = np.concatenate([X, noise_data_X], 0)
#         
#     Y = np.concatenate([Y, noise_data_Y], 0)    
    
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
    
    
def random_deletion(X, Y, delta_num):
    delta_data_ids = random_generate_subset_ids(X.shape, delta_num)     
#     torch.save(delta_data_ids, git_ignore_folder+'delta_data_ids')
    return X, Y, delta_data_ids    
    
    
    
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




def eigen_decomposition(avg_term1, dim, batch_size):
    
    
#     A = (1-alpha*beta)*torch.eye(dim[1], dtype = torch.double) - alpha*(term1[cut_off_epoch - 1])/dim[0]
    
    
#     A = A.type(torch.FloatTensor)
#     min_batch_num_per_epoch = int((dim[0] - 1)/batch_size) + 1
    
#     avg_term1 = torch.mean(torch.stack(term1[-min_batch_num_per_epoch:-1], 0), 0)
    
    s, M = torch.eig(avg_term1, True)
        
    s = s[:,0]
    
    print('eigen_values::', s)
        
    torch.save(M, git_ignore_folder + 'eigen_vectors')
    
    M_inverse = torch.tensor(np.linalg.inv(M.numpy()), dtype = torch.double)
    
#     M_inverse = torch.inverse(M)
    
    
    print('inverse_gap::', torch.norm(torch.mm(M, M_inverse) - torch.eye(dim[1], dtype = torch.double)))
    
    torch.save(M_inverse, git_ignore_folder + 'eigen_vectors_inverse')
    
    torch.save(s, git_ignore_folder + 'eigen_values')

#     torch.save(A, git_ignore_folder + 'expected_A')



def save_term1(term1, name):
    
    
#     try:
#         torch.save(term1, git_ignore_folder + 'term1')
#     except:
    directory = git_ignore_folder + name + '_folder'
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i in range(len(term1)):
        
        
        np.save(directory + '/' + str(i), term1[i])
            
            
    torch.save(torch.tensor([len(term1)]), directory + '/' + name + '_len')
    
def load_term1(directory, name):
    
#     try:
#         return torch.load(git_ignore_folder + 'term1')
#     
#     except:
    
    
    term1_len = torch.load(directory + '/' + name +'_len')[0]
    
    term1 = []
    
    for i in range(term1_len):
        print(i)
#         try:
#             term1.append(torch.load(directory + '/' + str(i)))
#         except:
        term1.append(np.load(directory + '/' + str(i) + '.npy'))
            
    return term1


def verify_results(X, Y, random_ids_multi_super_iterations, theta_list, grad_list, weights, offsets, term1_list, term2_list):
    
    epoch = 0
    for k in range(len(random_ids_multi_super_iterations)):
    
        random_ids = random_ids_multi_super_iterations[k]
        
        
        super_iter_id = k
        
#         if k > cut_off_super_iteration:
#             super_iter_id = cut_off_super_iteration
        
        end_id_super_iteration = (super_iter_id + 1)*dim[0]
        
        
        if end_id_super_iteration > weights.shape[0]:
            end_id_super_iteration = weights.shape[0]
            
        
        
        weights_this_super_iteration = weights[super_iter_id*dim[0]:end_id_super_iteration]
        
        offsets_this_super_iteration = offsets[super_iter_id*dim[0]:end_id_super_iteration]
        
        for i in range(0, X.shape[0], batch_size):
        
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
                
            origin_theta_this_epoch = theta_list[epoch]
            
#             origin_grad_this_epoch = grad_list[epoch]
            
            batch_X = X[random_ids[i:end_id]]
            
            batch_Y = Y[random_ids[i:end_id]]
            
            res = torch.mm(batch_X.mul(batch_Y).view(-1, dim[1]), origin_theta_this_epoch)
            
            weight_batch = weights_this_super_iteration[random_ids[i:end_id]]
            
            offset_batch = offsets_this_super_iteration[random_ids[i:end_id]]
            
            delta = res.view(-1)*weight_batch.view(-1) + offset_batch.view(-1) - (non_linear_function(res)).view(-1)
                
            print(torch.norm(delta))
            
            
            
            origin_grad_this_epoch = grad_list[epoch]
                
#             if epoch > 0:
            computed_origin_grad_this_epoch =  (0 - (torch.mm((term1_list[epoch]), origin_theta_this_epoch) + (term2_list[epoch].view(-1,1))))/(end_id - i) + beta*origin_theta_this_epoch
#             else:
#                 computed_origin_grad_this_epoch =  (0 - (torch.mm((term1_list[epoch]), origin_theta_this_epoch) + (full_term2.view(-1,1))))/(end_id - i) + beta*theta


            print('gradient_diff::', torch.norm(origin_grad_this_epoch.view(-1) - computed_origin_grad_this_epoch.view(-1)))

            epoch = epoch + 1

            y = 0
            
            y = y+ 1


def compute_X_weight_product(weights, offsets, X_Y_mult):
    
    end = False
    
#     X_weight_prod = []
    
    X_offset_prod = []
    
    print(random_ids_multi_super_iterations.shape[0])
    
    print('weights.shape::', weights.shape[0])
    
    print('x shape::', dim[0])
    
    for k in range(random_ids_multi_super_iterations.shape[0]):

        random_ids = random_ids_multi_super_iterations[k]
        
        print(k)
        
        super_iter_id = k
        
#         if k > cut_off_super_iteration:
#             super_iter_id = cut_off_super_iteration
        
        end_id_super_iteration = (super_iter_id + 1)*dim[0]
        
        
        if end_id_super_iteration >= weights.shape[0]:
            end_id_super_iteration = weights.shape[0]
            end = True
        
        t1 = time.time()    
        
        weights_this_super_iteration = weights[super_iter_id*dim[0]:end_id_super_iteration]
        
        offsets_this_super_iteration = offsets[super_iter_id*dim[0]:end_id_super_iteration]
        
        
#         print(X.shape)
#         
#         print(weights_this_super_iteration.view(-1).numpy().shape)
        
#         curr_X_weight_prod = scipy.sparse.csr_matrix(np.multiply(X, weights_this_super_iteration))

#         print(X_Y_mult.shape)
#         
#         print(offsets_this_super_iteration.shape)
        curr_X_offset_prod = scipy.sparse.csr_matrix(np.multiply(X_Y_mult, offsets_this_super_iteration))
        
#         X_weight_prod.append(curr_X_weight_prod)
        
        X_offset_prod.append(curr_X_offset_prod)
        
        
        if end == True:
            break
        
    del X_Y_mult
        
    return X_offset_prod


def compute_single_svd(i, term1, batch_size):
    
    if batch_size < term1.shape[1]:
        upper_bound = int(batch_size/svd_ratio)
    else:
        upper_bound = int(term1.shape[1]/svd_ratio)
    
    if upper_bound <= 0:
        upper_bound = 1
    
    curr_term1 = term1.numpy()
        
#     u,s,vt = np.linalg.svd(curr_term1)
    
    
    
    u, s, vt = randomized_svd(curr_term1, n_components=upper_bound, random_state=None)

    
#         upper_bound = compute_approx_dimension(s)
#         non_zero_ids = (s >= 1)
    
    sub_s = s[0:upper_bound]
    
#     if sub_s.shape[0] <= 0:
# #             non_zero_ids = np.array([0,1])
#         upper_bound = 1
#         
#         sub_s = s[0:upper_bound]
        
    
    sub_u = u[:,0:upper_bound]
     
    
     
    sub_v = vt[0:upper_bound]
    
    res = np.dot(sub_u*sub_s, sub_v)
    
    print(i, upper_bound, np.linalg.norm(res - curr_term1), np.linalg.norm(res - curr_term1)/np.linalg.norm(curr_term1))
    
    return torch.from_numpy(sub_u*sub_s), torch.from_numpy(sub_v)

def save_random_id_orders(random_ids_multi_super_iterations):
    sorted_ids_multi_super_iterations = []
    
    
    for i in range(random_ids_multi_super_iterations.shape[0]):
        sorted_ids_multi_super_iterations.append(random_ids_multi_super_iterations[i].numpy().argsort())
        
        
    torch.save(sorted_ids_multi_super_iterations, git_ignore_folder + 'sorted_ids_multi_super_iterations')


def save_random_id_orders_np(random_ids_multi_super_iterations):
    sorted_ids_multi_super_iterations = []
    
    
    for i in range(random_ids_multi_super_iterations.shape[0]):
        sorted_ids_multi_super_iterations.append(random_ids_multi_super_iterations[i].argsort())
        
        
    torch.save(sorted_ids_multi_super_iterations, git_ignore_folder + 'sorted_ids_multi_super_iterations')



def capture_provenance(theta_list, grad_list, run_rc1, X, Y, dim, epoch, batch_size, mini_batch_epoch, random_ids_multi_super_iterations, mini_epochs_per_super_iteration):
    
#     global res_prod_seq
    
    super_iteration = (int((len(res_prod_seq) - 1)/mini_epochs_per_super_iteration) + 1)
    
#     print('cutoff_epoch::', cut_off_epoch)
    if not run_rc1:
        save_random_id_orders(random_ids_multi_super_iterations)
    else:
        save_random_id_orders_np(random_ids_multi_super_iterations)
        
    t3 = time.time()
    
    if not run_rc1:
        X_Y_mult = X.mul(Y)
    else:
        X_Y_mult = X.multiply(Y.numpy()).tocsr()
    
    
    t3_1 = time.time()
    
    
    
    cut_off_epoch = len(res_prod_seq)
    cut_off_super_iteration = (int((cut_off_epoch - 1)/mini_epochs_per_super_iteration) + 1)
    
#     global res_prod_seq
    curr_res_prod_seq = res_prod_seq[0:cut_off_epoch]
    
    curr_rand_ids_multi_super_iterations = random_ids_multi_super_iterations[0:(int((cut_off_epoch - 1)/mini_epochs_per_super_iteration) + 1)*dim[0]]

    if not run_rc1:
        w_seq, b_seq = compute_linear_approx_parameters(X, Y, X_Y_mult, cut_off_epoch, curr_res_prod_seq)
    
        '''T*dim[0]'''
    
        curr_rand_ids_multi_super_iterations = curr_rand_ids_multi_super_iterations.view(-1, dim[0])
    
        _, sorted_ids_multi_super_iterations = torch.sort(curr_rand_ids_multi_super_iterations)
    else:
        w_seq, b_seq = compute_linear_approx_parameters_sparse(X, Y, X_Y_mult, cut_off_epoch, curr_res_prod_seq)
        
        curr_rand_ids_multi_super_iterations = np.reshape(curr_rand_ids_multi_super_iterations, (-1, dim[0]))
    
        sorted_ids_multi_super_iterations = np.argsort(curr_rand_ids_multi_super_iterations)

    w_seq_copy = torch.zeros([dim[0]*cut_off_super_iteration, 1], dtype = torch.double)
    
    b_seq_copy = torch.zeros([dim[0]*cut_off_super_iteration, 1], dtype = torch.double)
    
    w_seq_copy[0:w_seq.shape[0]] = w_seq
    
    b_seq_copy[0:w_seq.shape[0]] = b_seq
    
    w_seq_copy = w_seq_copy.view(cut_off_super_iteration, dim[0])
    
    b_seq_copy = b_seq_copy.view(cut_off_super_iteration, dim[0])
    
    
    for i in range(cut_off_super_iteration):
        w_seq_copy[i, :] = w_seq_copy[i, sorted_ids_multi_super_iterations[i]]
        b_seq_copy[i, :] = b_seq_copy[i, sorted_ids_multi_super_iterations[i]]
    
    
    w_seq_copy = w_seq_copy.view(-1)
    
    b_seq_copy = b_seq_copy.view(-1)
    

    t3_2 = time.time()

#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     if run_rc1:
#         X = check_and_convert_to_sparse_tensor(X)
#         
#         X_Y_mult = check_and_convert_to_sparse_tensor(X_Y_mult)
    
    
    
    # term1, term1_inter_result = prepare_term_1_serial(X, w_seq, dim)
    if not run_rc1:
        avg_term1 = prepare_term_1_batch2(run_rc1, X, w_seq_copy, dim, batch_size, cut_off_super_iteration, mini_batch_epoch, cut_off_epoch, random_ids_multi_super_iterations, mini_epochs_per_super_iteration)
        term2 = prepare_term_2_batch2(run_rc1, X_Y_mult, b_seq_copy, dim, cut_off_super_iteration, mini_batch_epoch, batch_size, cut_off_epoch, random_ids_multi_super_iterations)

    # term2, term2_inter_result = prepare_term_2_serial(X, Y, b_seq, dim)
    
#     prepare_term_1_2_batch2(beta, theta_list, grad_list, run_rc1, X, X_Y_mult, w_seq_copy, b_seq_copy, dim, batch_size, max_epoch, mini_batch_epoch, cut_off_epoch, random_ids_multi_super_iterations)
# 
#     verify_results(X, Y, random_ids_multi_super_iterations, theta_list, grad_list, w_seq_copy, b_seq_copy, term1, term2)

#     torch.save(term2, git_ignore_folder + 'term2')
    cut_off_super_iteration = int(super_iteration*prov_record_rate)#(int((cut_off_epoch - 1)/mini_epochs_per_super_iteration) + 1)
    
    
    cut_off_epoch = cut_off_super_iteration*mini_epochs_per_super_iteration

    print('super_iteration::', super_iteration)
    
    print('cut_off_super_iteration::', cut_off_super_iteration)
    
    print('cut_off_epoch::', cut_off_epoch)
    
    torch.save(w_seq_copy, git_ignore_folder + 'w_seq')
    
    torch.save(b_seq_copy, git_ignore_folder + 'b_seq')
    
#     print('X_Y_mult shape::', X_Y_mult.shape)
    
#     torch.save(X_Y_mult, git_ignore_folder + 'X_Y_mult')
    
    del w_seq_copy, b_seq_copy
    
    if not run_rc1:
        eigen_decomposition(avg_term1, dim, batch_size)
    
#         torch.save(term1, git_ignore_folder + 'term1')
        
        torch.save(term2, git_ignore_folder + 'term2')
        
        torch.save(X_Y_mult, git_ignore_folder + 'X_Y_mult')
        
        del X_Y_mult
        
        del term2
    else:
#         save_term1(term1, 'term1')
#         save_term1(term2, 'term2')
        scipy.sparse.save_npz(git_ignore_folder + 'X_Y_mult', X_Y_mult)
        
        
#         avg_term2 = None
#      
#         for i in range(mini_epochs_per_super_iteration):
#             if avg_term2 is None:
#     #             avg_term1 = term1[-(i + 1)]
#                 avg_term2 = term2[-(i + 1)]
#             else:
#     #             avg_term1 = avg_term1 + term1[-(i + 1)]
#                 avg_term2 = avg_term2 + term2[-(i + 1)]
#         
#         np.save(git_ignore_folder + 'avg_term2', avg_term2)
        
        del X_Y_mult
    
        
    
    
    torch.save(cut_off_epoch, git_ignore_folder + 'cut_off_epoch')
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

def precomptation_influence_function_sparse(X, Y, res, dim):
    
    t5 = time.time()
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
    X_Y_mult = X.multiply(Y.numpy())
    
#     Hessin_matrix = compute_hessian_matrix(X, X_Y_mult, res, dim, X_product)


#     Hessin_matrix = compute_hessian_matrix_2(X, X_Y_mult, res, dim, X_product)
    
    Hessin_matrix = compute_hessian_matrix_4_sparse(X, X_Y_mult, res.detach().numpy(), dim)

    print('hessian_matrix shape::', Hessin_matrix.shape)
    
#     Hessin_matrix2 = compute_hessian_matrix_3(X, X_Y_mult, res, dim)
    
#     print(Hessin_matrix)
#     
#     print(Hessin_matrix - Hessin_matrix2)
    

    Hessian_inverse = torch.tensor(scipy.sparse.linalg.inv(Hessin_matrix).todense(), dtype = torch.double)
    
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
        
    coeff = torch.sum(coeff[coeff == coeff])
    
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


def change_data_values_sparse(X, Y, num, res):
    
    positive_X_mean = np.mean(X[Y.view(-1)==1], axis = 0)
    
    negative_X_mean = np.mean(X[Y.view(-1)==-1], axis = 0)
    
#     multi_res = torch.mm(X, res)
    
    multi_res = X.dot(res.detach().numpy())
     
    indices = np.argsort(np.absolute(multi_res),axis = 0)[::-1][:X.shape[0]]
    
    sorted = multi_res[indices] 
    
#     sorted, indices = torch.sort(torch.abs(multi_res), dim = 0, descending = True)
     
    p1 = None
     
    p2 = None
    
    delta_data_ids = set()
    
    
    coeff = positive_X_mean/negative_X_mean
    
    coeff = coeff[coeff != np.inf]
        
    coeff = np.sum(coeff[coeff == coeff])
    
    print(coeff)
    
#     coeff = torch.sum(coeff[coeff != np.inf])
    
     
    for i in range(num):
        if Y[indices[i],0].numpy()[0] == 1:
            X[indices[i]] = coeff*X[indices[i]]
         
        if Y[indices[i],0].numpy()[0] == -1:
            X[indices[i]] = X[indices[i]]/coeff
        
        delta_data_ids.add(indices[i].item())
    
#         if p1 is not None and p2 is not None:
#             break
    
#     middle_point = (positive_X_mean + negative_X_mean)/2
#     
#     X[Y.view(-1) == 1] = X[Y.view(-1)==1]*torch.mean(positive_X_mean/negative_X_mean)
    
#     return X, Y, torch.tensor(list(delta_data_ids))
    return X, Y, np.array(list(delta_data_ids))

    
def deleting_data_values(X, Y, num, res):
    
    positive_X_mean = torch.mean(X[Y.view(-1)==1], 0)
    
    negative_X_mean = torch.mean(X[Y.view(-1)==-1], 0)
    
    multi_res = torch.mm(X, res)
     
    sorted, indices = torch.sort(torch.abs(multi_res), dim = 0, descending = True)
     
    p1 = None
     
    p2 = None
    
    delta_data_ids = set()
    
    
    coeff = positive_X_mean/negative_X_mean
    
    coeff = coeff[coeff != np.inf]
        
    coeff = torch.sum(coeff[coeff == coeff])
    
    print(coeff)
    
#     coeff = torch.sum(coeff[coeff != np.inf])
    
     
    for i in range(num):
        if multi_res[indices[i]]*Y[indices[i],0] >= 0:
            X[indices[i]] = -X[indices[i]]
        
        
#         else:
            
#         if Y[indices[i],0].numpy()[0] == -1:
#             X[indices[i]] = X[indices[i]]/coeff
        
        delta_data_ids.add(indices[i])
    
#         if p1 is not None and p2 is not None:
#             break
    
#     middle_point = (positive_X_mean + negative_X_mean)/2
#     
#     X[Y.view(-1) == 1] = X[Y.view(-1)==1]*torch.mean(positive_X_mean/negative_X_mean)
    
    return X, Y, torch.tensor(list(delta_data_ids))    
    
def deleting_data_values_sparse(X, Y, num, res):
    
    positive_X_mean = np.mean(X[Y.view(-1)==1], axis = 0)
    
    negative_X_mean = np.mean(X[Y.view(-1)==-1], axis = 0)
    
    multi_res = X.dot(res.detach().numpy())
    
#     multi_res = torch.mm(X, res)
     
    indices = np.argsort(np.absolute(multi_res), axis = 0)[::-1][:X.shape[0]]
    
    sorted = multi_res[indices]
    
#     sorted, indices = torch.sort(torch.abs(multi_res), dim = 0, descending = True)
     
    p1 = None
     
    p2 = None
    
    delta_data_ids = set()
    
    
    coeff = positive_X_mean/negative_X_mean
    
    coeff = coeff[coeff != np.inf]
        
    coeff = np.sum(coeff[coeff == coeff])
    
    print(coeff)
    
#     coeff = torch.sum(coeff[coeff != np.inf])
    
     
    for i in range(num):
        if multi_res[indices[i]]*Y[indices[i],0].numpy() >= 0:
            X[indices[i]] = -X[indices[i]]
        
        
#         else:
            
#         if Y[indices[i],0].numpy()[0] == -1:
#             X[indices[i]] = X[indices[i]]/coeff
        
        delta_data_ids.add(indices[i].item())
    
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
    
    batch_size =int(sys_args[6])
    
    max_epoch = int(sys_args[7])
    
#     global alpha, beta, threshold
    noise_rate = float(sys_args[8])
#     

    random_deletion_or_not = bool(int(sys_args[9]))

    add_noise_or_not = bool(int(sys_args[10]))
    
    
    prov_record_rate = float(sys_args[11])
    
    run_rc1 = bool(int(sys_args[12]))
    
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
         
        
#         X = extended_by_constant_terms(X)
#         
#         test_X = extended_by_constant_terms(test_X)
        
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
        
        
        
        
        
        
        
        
        
        
        X = extended_by_constant_terms(X, False)

        test_X = extended_by_constant_terms(test_X, False)
        
        
        
        dim = X.shape
        
        print(dim)
        
        
        random_ids = torch.randperm(dim[0])
        
        X = X[random_ids]
        
        
        Y = Y[random_ids]
    
    
        t1  = time.time()
        
    #     for i in range(repetition):
    
        lr = initialize(X)
            
        res2, epoch, mini_batch_epoch, _, _ = compute_parameters(X, Y, lr, dim, False)
    
#         lr = initialize(X)
#         
#         compute_model_parameter_by_iteration(dim, lr.theta, X.mul(Y), max_epoch, alpha, beta)
        
        t2 = time.time()
        
        print(res2)
        
        print('epoch::', epoch)
        
        torch.save(res2, git_ignore_folder + 'model_without_noise')
         
         
         
         
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
         
         
         
#         torch.save(X, git_ignore_folder + 'X')
         
        torch.save(Y, git_ignore_folder + 'Y')
         
#         torch.save(test_X, git_ignore_folder + 'test_X')
      
        torch.save(test_Y, git_ignore_folder + 'test_Y')
        
        torch.save(torch.tensor(batch_size), git_ignore_folder + 'batch_size')
        
#         torch.save(torch.tensor(epoch), git_ignore_folder + 'epoch')
        
    else:
        
        res1 = torch.load(git_ignore_folder + 'model_without_noise')
        
#         X = torch.load(git_ignore_folder + 'X')
        
        Y = torch.load(git_ignore_folder + 'Y')
        
#         test_X = torch.load(git_ignore_folder + 'test_X')
        
        test_Y = torch.load(git_ignore_folder + 'test_Y')
        
        
        
        
        if run_rc1:
#             try:
            X = scipy.sparse.load_npz(git_ignore_folder + 'X.npz')
#             X = convert_coo_matrix2_dense_tensor(sparse_X)
            test_X = scipy.sparse.load_npz(git_ignore_folder + 'test_X.npz')
#             except:
    #             X = torch.load(git_ignore_folder + 'X').to_dense()
    #             test_X = torch.load(git_ignore_folder + 'test_X').to_dense()
    
#                 X = torch.load(git_ignore_folder + 'X')
#                 test_X = torch.load(git_ignore_folder + 'test_X')

        else:
            
            X = torch.load(git_ignore_folder + 'X')
            test_X = torch.load(git_ignore_folder + 'test_X')
        
        
        
        
        
        
        
        
        print('model_without_noise::', res1)
        
        dim = X.shape
        
        
        if random_deletion_or_not:
#             X, Y, noise_data_ids = random_deletion(X, Y, int(X.shape[0]*noise_rate))
            if not run_rc1:
                X, Y, noise_data_ids =  deleting_data_values(X, Y, int(X.shape[0]*noise_rate), res1)
                
            else:
                X, Y, noise_data_ids =  deleting_data_values_sparse(X, Y, int(X.shape[0]*noise_rate), res1)
        else:
            if add_noise_or_not:
    #             X, Y = add_noise_data(X, Y, int(X.shape[0]*noise_rate), res1)
                if not run_rc1:
                    X, Y = add_noise_data(X, Y, int(X.shape[0]*noise_rate), res1)
                    
                else:
                    X, Y = add_noise_data_sparse(X, Y, int(X.shape[0]*noise_rate), res1)
                
                
                noise_data_ids = torch.tensor(list(set(range(X.shape[0])) - set(range(dim[0]))))
            else:
                
                if not run_rc1:
                    X, Y, noise_data_ids = change_data_values(X, Y, int(X.shape[0]*noise_rate), res1)
                else:
                    X, Y, noise_data_ids = change_data_values_sparse(X, Y, int(X.shape[0]*noise_rate), res1)
        

        
#     X, Y = add_noise_data2(X, Y, added_x, added_y, 1000)
#         X, Y, noise_data_ids = change_instance_labels(X, Y, int(X.shape[0]*0.01), dim, res1)
#         X, Y = add_noise_data(X, Y, int(X.shape[0]*0.3), res1)
    
        
        dim = X.shape

#
         
#         random_ids = torch.randperm(dim[0])
        
        
        random_ids = np.random.permutation(dim[0])
          
        X = X[random_ids]
          
          
        Y = Y[random_ids]
        
#         matched_ids = (random_ids.view(-1,1) == noise_data_ids.view(1,-1))
        
        
        
#         shuffled_noise_data_ids = torch.zeros(noise_data_ids.shape)
        shuffled_noise_data_ids = torch.argsort(torch.tensor(random_ids))[noise_data_ids]#random_ids[noise_data_ids]
#         for i in range(noise_data_ids.shape[0]):
#               
#             shuffled_id = torch.nonzero(random_ids == noise_data_ids[i])
#               
# #             print(shuffled_id)
#               
#             shuffled_noise_data_ids[i] = shuffled_id 
#         
#         
# #         shuffled_noise_data_ids = torch.nonzero(torch.sum(matched_ids, 1)).view(-1).type(torch.IntTensor)
#         
# #         noise_data_ids[random_ids]
#          
#          
# #         shuffled_noise_data_ids = torch.zeros(noise_data_ids.shape)
# #          
# #         for i in range(noise_data_ids.shape[0]):
# #              
# #             shuffled_id = torch.nonzero(random_ids == noise_data_ids[i])
# #              
# # #             print(shuffled_id)
# #              
# #             shuffled_noise_data_ids[i] = shuffled_id 
# #          
# #         
# #         shuffled_noise_data_ids,_ = torch.sort(shuffled_noise_data_ids)
#          
#         print(shuffled_noise_data_ids[:100])
         
        
         
        torch.save(shuffled_noise_data_ids, git_ignore_folder + 'noise_data_ids')
         
        print(dim)
        
    #     X, Y, noise_data_ids = change_data_labels2(X, Y, 0.8, res) 
    #                    
    #     torch.save(noise_data_ids, git_ignore_folder + 'noise_data_ids')
    
        t1 = time.time()  
        if not run_rc1:
            lr = initialize(X)
            res2, epoch, mini_batch_epoch, theta_list, grad_list = compute_parameters(X, Y, lr, dim, True)
        else:
            
            lr = initialize_by_size(X.shape)
            res2,epoch,mini_batch_epoch,theta_list, grad_list = compute_parameters_sparse(X, Y, lr, dim, True)
        
        t2 = time.time()
        
#         print('epoch::', epoch)
        
        print(res2 - res1)
        
        if not run_rc1:
            torch.save(theta_list, git_ignore_folder + 'origin_theta_list')
        
            torch.save(grad_list, git_ignore_folder + 'origin_grad_list')
        
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
        
        
        
        torch.save(torch.tensor(epoch), git_ignore_folder + 'epoch')
        
        torch.save(torch.tensor(mini_batch_epoch), git_ignore_folder + 'mini_batch_epoch')
        
        torch.save(res2, git_ignore_folder + 'model_origin')
    
        torch.save(alpha, git_ignore_folder + 'alpha')
        
        torch.save(beta, git_ignore_folder + 'beta')
        
        if not run_rc1:
            random_ids_multi_super_iterations_tensors = torch.stack(random_ids_multi_super_iterations)
        
            torch.save(random_ids_multi_super_iterations_tensors, git_ignore_folder + 'random_ids_multi_super_iterations')
        else:
            random_ids_multi_super_iterations_tensors = np.stack(random_ids_multi_super_iterations, axis = 0)
            
            np.save(git_ignore_folder + 'random_ids_multi_super_iterations', random_ids_multi_super_iterations_tensors)
        
        torch.save(torch.tensor(batch_size), git_ignore_folder + 'batch_size')
        mini_epochs_per_super_iteration = int((dim[0] - 1)/batch_size) + 1
        
        
        
        if run_rc1:
#             X = check_and_convert_to_sparse_tensor(X)
             
             
#             try:
            scipy.sparse.save_npz(git_ignore_folder + 'noise_X', X)    
            
            
#             X_tensor = torch.from_numpy(X.todense()).type(torch.double)
#             
#             precomptation_influence_function(X_tensor, Y, res2, dim)   
#             
#             
#             del X_tensor
#             except:
#                 torch.save(X, git_ignore_folder + 'noise_X')  
                
        else:
            precomptation_influence_function(X, Y, res2, dim)
            
            torch.save(X, git_ignore_folder + 'noise_X')
#         torch.save(X, git_ignore_folder + 'noise_X')
        capture_provenance(theta_list, grad_list, run_rc1, X, Y, dim, epoch, batch_size, mini_batch_epoch, random_ids_multi_super_iterations_tensors, mini_epochs_per_super_iteration)

        torch.save(Y, git_ignore_folder + 'noise_Y')
        
        print('training_time::', t2 - t1)

    if not run_rc1:
        print('training_accuracy::',compute_accuracy2(X, Y, res2))
        
        print('test_accuracy::', compute_accuracy2(test_X, test_Y, res2))
    else:
        print('training_accuracy::',compute_accuracy2_sparse(X, Y, res2))
        
        print('test_accuracy::', compute_accuracy2_sparse(test_X, test_Y, res2))
         
         
        
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
