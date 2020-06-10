'''
Created on Feb 5, 2019

'''
import os
import sys
from torch.autograd import Variable

from torch import nn, optim
import torch

from main.watcher import Watcher
import matplotlib.pyplot as plt
import numpy as np


# sample_level = True
# 
# if sample_level:
#     from main.matrix_prov_sample_level import M_prov
# else:
#     from main.matrix_prov_entry_level import M_prov
# from main.add_prov import add_prov_token_per_row
torch.set_printoptions(precision=10)

# x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
#                     [9.779], [6.182], [7.59], [2.167], [7.042],
#                     [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

x_train = np.array([[0, 1], [4.4, 0], [5.5, 3]], dtype=np.float32)

# y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
#                     [3.366], [2.596], [2.53], [1.221], [2.827],
#                     [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

y_train = np.array([[1.7], [2.06], [2.59]], dtype=np.float32)


x_train = torch.from_numpy(x_train)

y_train = torch.from_numpy(y_train)

X = Variable(x_train)

Y = Variable(y_train)

shape = list(X.size())

# x_train[0][0].a[0] = 1



class logistic_regressor_parameter:
    def __init__(self, theta):
        self.theta = theta
        
        
def binary_cross_entropy(x_i, y_i, theta):
    return 1/(1+torch.exp(-y_i*torch.dot(x_i, theta)))

def non_linear_terms(x_i, y_i, theta):
    return 1-1/(1+torch.exp(-y_i*torch.dot(x_i, theta)))

def gradient(X, Y, dim, theta):
    
    res = torch.zeros(theta.shape, dtype = torch.DoubleTensor)
    
    for i in range(dim[0]):
        res = res + Y[i]*X[i,:]*non_linear_terms(X[i,:], Y[i], theta)
    
    return res
    

def logistic_regression(X, Y, lr):

    max_epoch = 6
      
    alpha = 0.01
      
    beta = 0
      
    dim = X.shape
      
    
    for epoch in range(max_epoch):
        print('epoch', epoch)
#         print('start', lr.theta)
#         print('step 0', (torch.mm(X, lr.theta)))
#         print('step 1', (torch.mm(X, lr.theta) - Y))
#         print('step 2', alpha*torch.mm(torch.transpose(X, 0, 1),(torch.mm(X, lr.theta) - Y)))
#         print('theta!!!!', lr.theta)
#         lr.theta = lr.theta - 2*alpha*(torch.mm(torch.transpose(X, 0, 1),(torch.mm(X, lr.theta) - Y)) + beta*lr.theta)

        lr.theta = lr.theta - alpha*(gradient(X, Y, dim, lr.theta) + beta*lr.theta)
        
        print('theta!!!!', lr.theta)
#         err = Y - torch.mm(X, lr.theta)
#         error = torch.mm(torch.transpose(err, 0, 1), err)# + beta*torch.matmul(torch.transpose(theta, 0, 1), theta)
        
#         print('error', error)
      
    return lr.theta

theta = Variable(torch.zeros([shape[1],1]))
theta[0][0] = 0.5
print(theta)
lr = logistic_regressor_parameter(theta)
print('path::', os.path.dirname(__file__))
'''current_module = sys.modules[__name__] for variable without class name'''
w = Watcher(output_var_name = 'lr.theta', obj = lr, attr = 'theta', input_data=X, input_data_name='X', log_file='log.txt', include =[os.path.dirname(__file__)], enabled= True, file_name=os.path.basename(__file__), annotate_input_by_row = True, annotate_para_by_row = False)
sys.settrace(w.trace_command)
logistic_regression(X, Y, lr)

result = torch.mm(torch.inverse(torch.mm(torch.transpose(x_train, 0 , 1), x_train)), torch.mm(torch.transpose(x_train,0,1), y_train))

print('exact_result::')
print(result)