'''
Created on Feb 4, 2019

'''
from torch import nn, optim
import torch
from sensitivity_analysis.Load_data import load_data
import random
from sensitivity_analysis.Linear_regression import *
import time


def get_subset_training_data(X, dim, delta_data_ids):
    selected_rows = torch.tensor(list(set(range(dim[0])) - delta_data_ids))
    print(selected_rows)
    update_X = torch.index_select(X, 0, selected_rows)
    return update_X



def random_generate_subset_ids(dim, delta_size):
    
    delta_data_ids = set()
    
    for i in range(delta_size):
    
        id = random.randint(0, dim[0]-1)
        
        if id in delta_data_ids:
            i = i-1
            
            continue
    #     print(id, i)
    #     print(update_X_product)
        delta_data_ids.add(id)
    
    return delta_data_ids
#         temp = X[id,:].resize_(dim[1],1)
#     #     print(torch.transpose(temp, 0, 1))
#         
#         update_X_product = update_X_product - torch.mm(X[id,:].resize_(dim[1],1), X[id, :].resize_(1,dim[1]))


def model_para_initialization(dim):
    theta = Variable(torch.zeros([dim[1],1])).type(torch.DoubleTensor)
    # theta[0][0] = 0
    lr = linear_regressor(theta)
    
    return lr

def update_model_parameters_from_the_scratch(update_X, update_Y):
    dim = update_X.shape
    
    
    lr = model_para_initialization(dim)
    
#     theta = Variable(torch.zeros([dim[1],1]))
#     # theta[0][0] = 0
#     lr = linear_regressor(theta)
    
    
    res = linear_regression_iteration(update_X, update_Y, lr)
    
    return res


def compute_geometry_seq_ratio(U, S, V, update_X):
    
    update_X_product = torch.mm(torch.t(update_X), update_X)
    
    left_product = torch.mm(torch.t(U), update_X_product)
    
    l_shape = left_product.shape

    res = torch.bmm(left_product.view(l_shape[0], 1, l_shape[1]), torch.t(V).view(l_shape[0], l_shape[1], 1)).resize_(4)


#     print('size', res.shape)
# 
#     print('re!!!!!!!', res)
#     
#     print('re!!!!!!!', res)

    ratio = (1-beta*alpha - 2*alpha*res)
    
    return ratio


def update_model_parameters_incrementally(U, S, V, update_X, update_Y, max_epoch, dim):
    
    ratio = compute_geometry_seq_ratio(U, S, V, update_X)

    lr = model_para_initialization(dim)

#     print('ratio', ratio)

    ratio_t = torch.pow(ratio, max_epoch)
    
#     print(ratio_t)
    
    diag_matrix = torch.diag(ratio_t)
    
    diag_matrix = diag_matrix.type(torch.DoubleTensor)
    
#     print(diag_matrix)
    
    left_product = torch.mm(U, torch.diag(ratio_t))
    
    restore_svd_res = torch.mm(left_product, torch.t(V)) 
    
    term1 = torch.mm(restore_svd_res, lr.theta)
    
    term2 =  2*alpha*torch.mm(torch.mm(torch.mm(U, torch.diag((1-ratio_t)/(1-ratio))), torch.t(V)), torch.mm(torch.t(update_X), update_Y))
    
    return term1 + term2
    

delta_size = 1000

repetition = 10

# alpha = 0.00001
#       
# beta = 0.5
[X, Y] = load_data(False)


dim = X.shape


delta_data_ids = random_generate_subset_ids(dim, delta_size)

update_X = get_subset_training_data(X, dim, delta_data_ids)

update_Y = get_subset_training_data(Y, Y.shape, delta_data_ids)

print(X.shape)

print(update_X.shape)

t1 = time.time()

for i in range(repetition):
    res1 = update_model_parameters_from_the_scratch(update_X, update_Y)

t2 = time.time()

print('res1', res1)


t5 = time.time()

for i in range(repetition):
    [U, S, V] = torch.svd(torch.mm(torch.transpose(X, 0, 1), X))

t6 = time.time()

t3 = time.time()


for i in range(repetition):
    res2 = update_model_parameters_incrementally(U, S, V, update_X, update_Y, max_epoch, dim)

t4 = time.time()

time1 = (t2 - t1)/repetition

time2 = (t4 - t3)/repetition

time3 = (t6 - t5)/repetition

print('res2', res2)

print('time1', time1)

print('time2', time2)

print('svd_time', time3)

# ratio = compute_geometry_seq_ratio(U, S, V, update_X)
# 
# print(ratio)
# 
# update_X_product = torch.mm(torch.t(update_X), update_X)
# 
# [U_d, S_d, V_d] = torch.svd(update_X_product)
# 
# print(S_d)




# update_X_product = torch.mm(torch.transpose(X, 0 ,1), X)

# update_X = X.clone()









# print(delta_data_ids)
# 
# print(selected_rows)
# 
# print(len(selected_rows))


# print(update_X.shape)

# print(torch.mm(torch.transpose(update_X, 0, 1), update_X) - update_X_product)















