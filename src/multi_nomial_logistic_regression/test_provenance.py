'''
Created on Jan 12, 2020


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
from sensitivity_analysis_SGD.multi_nomial_logistic_regression import Multi_logistic_regression


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_IO.Load_data import *
    from Interpolation.piecewise_linear_interpolation_multi_dimension import *
    from sensitivity_analysis_SGD.multi_nomial_logistic_regression.incremental_updates_logistic_regression_multi_dim import *
    from sensitivity_analysis_SGD.multi_nomial_logistic_regression.evaluating_test_samples import *
    from Multi_logistic_regression import *
except ImportError:
    from Load_data import *
    from piecewise_linear_interpolation_multi_dimension import *
    from incremental_updates_logistic_regression_multi_dim import *
    from evaluating_test_samples import *
    from Multi_logistic_regression import *

import gc 
import sys

X_theta_prod_softmax_seq = []
X_theta_prod_seq = []

def loss_function(X, Y, theta, dim, beta):
    

    X_theta_prod = torch.mm(X, theta)
    
    
    X_theta_prod_softmax = softmax_layer(X_theta_prod)
    
    
    
    X_theta_prod_softmax_seq.append(X_theta_prod_softmax)
    X_theta_prod_seq.append(X_theta_prod)
    
    res = -torch.sum(torch.log(torch.gather(X_theta_prod_softmax, 1, Y.view(-1,1))))/dim[0]

    
    return res + beta/2*torch.sum(torch.reshape(theta, (-1,1))*torch.reshape(theta, (-1,1)))



def compute_intermediate_results(lr, X, Y, random_ids_multi_super_iterations, learning_rate_all_epochs, beta, batch_size):
    
    Y = Y.type(torch.LongTensor)
    
    epoch = 0
    
    last_theta = None
    
    
    last_recorded_theta = None
    
#     for epoch in range(max_epoch):

    mini_batch_epoch = 0

    for k in range(len(random_ids_multi_super_iterations)):
        
        random_ids = random_ids_multi_super_iterations[k]
        
        origin_X = X[random_ids]
        
        origin_Y = Y[random_ids]
#         gap_to_be_averaged = []

        for i in range(0, dim[0], batch_size):
            
            
            end_id = i + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            
            
            batch_x = origin_X[i: end_id]
            
            batch_y = origin_Y[i: end_id]
        
        
            loss = loss_function(batch_x, batch_y, lr.theta, batch_x.shape, beta)
       
            loss.backward()
           
            with torch.no_grad():
                lr.theta -= learning_rate_all_epochs[epoch] * lr.theta.grad
                gap = torch.norm(lr.theta.grad)
                
                lr.theta.grad.zero_()
        
                
            print(epoch, gap)
            
            mini_batch_epoch += 1
            
            epoch = epoch + 1
                
    return lr.theta, epoch
    
    
    
    


if __name__ == '__main__':
    X = torch.load(git_ignore_folder + 'noise_X')
        
    Y = torch.load(git_ignore_folder + 'noise_Y')

    max_epoch = torch.load(git_ignore_folder+'epoch')

    num_class = torch.unique(Y).shape[0]
    
    batch_size = torch.load(git_ignore_folder + 'batch_size')

    random_ids_multi_super_iterations = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations')

    dim = X.shape

    learning_rate_all_epochs = torch.load(git_ignore_folder + 'learning_rate_all_epochs')
    
    beta = torch.load(git_ignore_folder + 'beta')

    mini_epochs_per_super_iteration = int((dim[0] - 1)/batch_size) + 1
    
    random_ids_multi_super_iterations_tensors = torch.stack(random_ids_multi_super_iterations)

    init_para_list = list(torch.load(git_ignore_folder + 'init_para'))
    lr = initialize(X, num_class)
    
    lr.theta = init_para_list[0].T
    
    lr.theta.requires_grad = True
    
#     init_theta = Variable(lr.theta)

    compute_intermediate_results(lr, X, Y, random_ids_multi_super_iterations, learning_rate_all_epochs, beta, batch_size)

#     Multi_logistic_regression.random_ids_multi_super_iterations = random_ids_multi_super_iterations

    x_sum_by_class_by_batch = compute_x_sum_by_class_by_batch(X, Y, batch_size, num_class, random_ids_multi_super_iterations)


    torch.save(x_sum_by_class_by_batch, git_ignore_folder+'x_sum_by_class')

    capture_provenance(X, Y, X.shape, max_epoch, num_class, batch_size, mini_epochs_per_super_iteration, random_ids_multi_super_iterations_tensors, X_theta_prod_softmax_seq, X_theta_prod_seq)
    
    
    
    
    
    
    
    