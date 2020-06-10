'''
Created on Feb 4, 2019

'''
from torch import nn, optim
import torch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


try:
    from sensitivity_analysis.Load_data import *
    from sensitivity_analysis.multi_nomial_logistic_regression.Multi_logistic_regression import *
except ImportError:
    from Load_data import *
    from Multi_logistic_regression import *
import random
import time


def get_subset_training_data(X, dim, delta_data_ids):
    selected_rows = torch.tensor(list(set(range(dim[0])) - set(delta_data_ids.tolist())))
#     print(selected_rows)
    update_X = torch.index_select(X, 0, selected_rows)
    return update_X, selected_rows

def get_subset_parameter_list(selected_rows, delta_data_ids, para_list, dim, axis):
    para_list_tensor = torch.tensor(para_list, dtype = torch.double)
    update_para_list_tensor = torch.index_select(para_list_tensor, axis, selected_rows)
    return update_para_list_tensor
    

def random_generate_subset_ids(dim, delta_size):
    
    delta_data_ids = set()
    
    while len(delta_data_ids) < delta_size:
        id = random.randint(0, dim[0]-1)
        delta_data_ids.add(id)
    
#     for i in range(delta_size):
#     
#         id = random.randint(0, dim[0]-1)
#         
#         if id in delta_data_ids:
#             i = i-1
#             
#             continue
    #     print(id, i)
    #     print(update_X_product)
        
    
    return torch.tensor(list(delta_data_ids))
#         temp = X[id,:].resize_(dim[1],1)
#     #     print(torch.transpose(temp, 0, 1))
#         
#         update_X_product = update_X_product - torch.mm(X[id,:].resize_(dim[1],1), X[id, :].resize_(1,dim[1]))


def model_para_initialization(dim):
    theta = Variable(torch.zeros([dim[1],1])).type(torch.FloatTensor)
    theta[0][0] = -1
    lr = logistic_regression(X, Y, lr)
    
    return lr

def update_model_parameters_from_the_scratch(update_X, update_Y):
#     dim = update_X.shape
    
    
    lr = initialize(update_X)
    
#     theta = Variable(torch.zeros([dim[1],1]))
#     # theta[0][0] = 0
#     lr = linear_regressor(theta)
    
    
    res = logistic_regression(update_X, update_Y, lr)
    
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
    
    diag_matrix = diag_matrix.type(torch.FloateTensor)
    
#     print(diag_matrix)
    
    left_product = torch.mm(U, torch.diag(ratio_t))
    
    restore_svd_res = torch.mm(left_product, torch.t(V)) 
    
    term1 = torch.mm(restore_svd_res, lr.theta)
    
    term2 =  2*alpha*torch.mm(torch.mm(torch.mm(U, torch.diag((1-ratio_t)/(1-ratio))), torch.t(V)), torch.mm(torch.t(update_X), update_Y))
    
    return term1 + term2

def prepare_term_1(term1_inter_result, delta_ids):
    
#     w_dim = w_seq.shape
#     
#     print(w_dim)
#     
#     print(dim)
    
    inter_res_dim = term1_inter_result.shape
    
    print(inter_res_dim)
    
    res = torch.zeros(inter_res_dim[1], inter_res_dim[2], inter_res_dim[3], dtype = torch.float)
    
    for id in delta_ids:
        res += term1_inter_result[id]
    
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     res = torch.bmm(X_product.view(dim[0], dim[1]*dim[1], 1), torch.t(w_seq).view(dim[0], 1, w_dim[0]))
#     
# #     print(res.shape)
# 
#     res = torch.transpose(torch.transpose(torch.transpose(res.view(dim[0], dim[1], dim[1], w_dim[0]), 2, 3), 1, 2), 0, 1)
#     
#     res = torch.sum(res, dim = 1)
#     
#     print('final_res_shape::', res.shape)
    
    
    
    
    return res

def prepare_term_1_batch(X_product, w_seq, dim, batch_size):
    
    w_dim = w_seq.shape
    
    print(w_dim)
    
    print(dim)
    
    w_seq_transpose = torch.t(w_seq)
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
    batch_num = int(dim[0]/batch_size)
    
    res = torch.zeros(w_dim[0], dim[1]*dim[1], dtype = torch.float)
    
    for i in range(batch_num):
        X_product_subset = X_product[i*batch_size:(i+1)*batch_size, :]
        
        w_seq_subset = w_seq_transpose[i*batch_size:(i+1)*batch_size, :]
        
        curr_res = torch.bmm(X_product_subset.view(batch_size, dim[1]*dim[1], 1), w_seq_subset.view(batch_size, 1, w_dim[0]))
        
#         curr_res = torch.transpose(torch.transpose(curr_res, 1, 2), 0, 1)
        
        curr_res = torch.sum(curr_res, dim = 0)
        
        curr_res = torch.t(curr_res)#.view(w_dim[0], dim[1], dim[1])
        
        res += curr_res
    
    if batch_num*batch_size < dim[0]:
        curr_batch_size = dim[0] - batch_num*batch_size
        
        X_product_subset = X_product[batch_num*batch_size:dim[0], :]
        
        w_seq_subset = w_seq_transpose[batch_num*batch_size:dim[0], :]
        
        curr_res = torch.bmm(X_product_subset.view(curr_batch_size, dim[1]*dim[1], 1), w_seq_subset.view(curr_batch_size, 1, w_dim[0]))
        
#         curr_res = torch.transpose(torch.transpose(curr_res, 1, 2), 0, 1)
        
        curr_res = torch.sum(curr_res, dim = 0)
        
        curr_res = torch.t(curr_res)#.view(w_dim[0], dim[1], dim[1])
        
        res += curr_res
    
#     res = torch.bmm(X_product.view(dim[0], dim[1]*dim[1], 1), torch.t(w_seq).view(dim[0], 1, w_dim[0]))
#     
# #     print(res.shape)
# 
#     res = torch.transpose(torch.transpose(torch.transpose(res.view(dim[0], dim[1], dim[1], w_dim[0]), 2, 3), 1, 2), 0, 1)
#     
#     res = torch.sum(res, dim = 1)
#     
#     print('final_res_shape::', res.shape)
    
    
    
    
    return res.view(w_dim[0], dim[1], dim[1])




#     res = torch.bmm(X_product.view(dim[0], dim[1]*dim[1], 1), torch.t(w_seq).view(dim[0], 1, w_dim[0]))
#     
# #     print(res.shape)
# 
#     res = torch.transpose(torch.transpose(torch.transpose(res.view(dim[0], dim[1], dim[1], w_dim[0]), 2, 3), 1, 2), 0, 1)
#     
#     res = torch.sum(res, dim = 1)
#     
#     print('final_res_shape::', res.shape)


def prepare_term_1_batch2(X_product, weights, dim, max_epoch, num_class):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    
    
    
    w_dim = weights.shape
    
    print('w_dim::', w_dim)
    
    print(dim)
    
#     w_seq_transpose = torch.t(w_seq)
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     batch_num = int(dim[0]/batch_size)
    
#     res = torch.zeros(w_dim[0], dim[1], dim[1], dtype = torch.double)
    
    X_product1 = torch.t(X_product.view(dim[0], dim[1]*dim[1]))
    
#     del X_product
    
    
    t1 = time.time()
    
    res1 = torch.mm(X_product1, torch.reshape(weights, [dim[0], w_dim[1]*num_class*num_class]))
    
    t2 = time.time()
    
    del X_product1
    
    res2 = torch.transpose(torch.transpose((res1.view(dim[1]*dim[1], w_dim[1], num_class*num_class)), 1, 0), 1, 2)

#     del res1
    
    res3 = res2.view(w_dim[1], num_class, num_class, dim[1], dim[1])

    del res2
#     res = torch.transpose(res, 1, 2)
    
    res4 = torch.transpose(res3, 2, 3)
    
    del res3
    
    print(res4.shape)
    
    res = torch.reshape(res4, [w_dim[1], num_class*dim[1], dim[1]*num_class])
    
    del res4
    
    print('time::', t2 - t1)
    
    
    size = 100
    
    
    X_product1 = torch.t(X_product[0:size].view(size, dim[1]*dim[1]))
    
#     del X_product
    
    
    t1 = time.time()
    
    res1_1 = torch.mm(X_product1, torch.reshape(weights[0:size], [size, w_dim[1]*num_class*num_class]))
    
    t2 = time.time()
    
    del X_product1
    
    res2 = torch.transpose(torch.transpose((res1_1.view(dim[1]*dim[1], w_dim[1], num_class*num_class)), 1, 0), 1, 2)

#     del res1
    
    res3 = res2.view(w_dim[1], num_class, num_class, dim[1], dim[1])

    del res2
#     res = torch.transpose(res, 1, 2)
    
    res4 = torch.transpose(res3, 2, 3)
    
    del res3
    
    print(res4.shape)
    
    res_1 = torch.reshape(res4, [w_dim[1], num_class*dim[1], dim[1]*num_class])
    
    del res4
    
    print('time::', t2 - t1)
    
    
    
    
    
    
    
    X_product1 = torch.t(X_product[size:].view(dim[0] - size, dim[1]*dim[1]))
    
#     del X_product
    
    
    t1 = time.time()
    
    res1_2 = torch.mm(X_product1, torch.reshape(weights[size:], [dim[0] - size, w_dim[1]*num_class*num_class]))
    
    t2 = time.time()
    
    del X_product1
    
    res2 = torch.transpose(torch.transpose((res1_2.view(dim[1]*dim[1], w_dim[1], num_class*num_class)), 1, 0), 1, 2)

#     del res1_2
    
    res3 = res2.view(w_dim[1], num_class, num_class, dim[1], dim[1])

    del res2
#     res = torch.transpose(res, 1, 2)
    
    res4 = torch.transpose(res3, 2, 3)
    
    del res3
    
    print(res4.shape)
    
    res_2 = torch.reshape(res4, [w_dim[1], num_class*dim[1], dim[1]*num_class])
    
    del res4
    
    print('time::', t2 - t1)
    
    
    
#     res = res.view([max_epoch, num_class, dim[1], dim[1]*num_class])
#     
#     res = res.view()
    
    
     
    
    
    
#     for t in range(w_dim[0]):
# #         print(X_product.shape)
# #         print(w_seq[t].shape)
#         curr_res = torch.mm(X_product, w_seq[t].view(dim[0], 1))
# #         print(curr_res.shape)
#         res[t] = curr_res.view(dim[1], dim[1])
    
    
    
#     for i in range(batch_num):
#         X_product_subset = X_product[i*batch_size:(i+1)*batch_size, :]
#         
#         w_seq_subset = w_seq_transpose[i*batch_size:(i+1)*batch_size, :]
#         
#         curr_res = torch.bmm(X_product_subset.view(batch_size, dim[1]*dim[1], 1), w_seq_subset.view(batch_size, 1, w_dim[0]))
#         
# #         curr_res = torch.transpose(torch.transpose(curr_res, 1, 2), 0, 1)
#         
#         curr_res = torch.sum(curr_res, dim = 0)
#         
#         curr_res = torch.t(curr_res)#.view(w_dim[0], dim[1], dim[1])
#         
#         res += curr_res
#     
#     if batch_num*batch_size < dim[0]:
#         curr_batch_size = dim[0] - batch_num*batch_size
#         
#         X_product_subset = X_product[batch_num*batch_size:dim[0], :]
#         
#         w_seq_subset = w_seq_transpose[batch_num*batch_size:dim[0], :]
#         
#         curr_res = torch.bmm(X_product_subset.view(curr_batch_size, dim[1]*dim[1], 1), w_seq_subset.view(curr_batch_size, 1, w_dim[0]))
#         
# #         curr_res = torch.transpose(torch.transpose(curr_res, 1, 2), 0, 1)
#         
#         curr_res = torch.sum(curr_res, dim = 0)
#         
#         curr_res = torch.t(curr_res)#.view(w_dim[0], dim[1], dim[1])
#         
#         res += curr_res
    
#     res = torch.bmm(X_product.view(dim[0], dim[1]*dim[1], 1), torch.t(w_seq).view(dim[0], 1, w_dim[0]))
#     
# #     print(res.shape)
# 
#     res = torch.transpose(torch.transpose(torch.transpose(res.view(dim[0], dim[1], dim[1], w_dim[0]), 2, 3), 1, 2), 0, 1)
#     
#     res = torch.sum(res, dim = 1)
#     
#     print('final_res_shape::', res.shape)
    
    
    
    
    return res


def prepare_term_1_batch2_0(X, weights, dim, max_epoch, num_class):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    
    
    
    w_dim = weights.shape
    
    print('w_dim::', w_dim)
    
    print(dim)
    
#     w_seq_transpose = torch.t(w_seq)
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     batch_num = int(dim[0]/batch_size)
    
#     res = torch.zeros(w_dim[0], dim[1], dim[1], dtype = torch.double)
    
#     X_product1 = torch.t(X_product.view(dim[0], dim[1]*dim[1]))
    
#     del X_product
    
    
    t1 = time.time()
    
    res1 = torch.bmm(X.view(dim[0], dim[1]), weights.view(dim[0], w_dim[1]*num_class*num_class))
    
    res1 = res1.view(dim[0], dim[1]*w_dim[1]*num_class*num_class)
    
    res1 = torch.mm(torch.t(X), res1)
    
#     res1 = torch.mm(X_product1, weights.view(dim[0], w_dim[1]*num_class*num_class))
    
    t2 = time.time()
    
#     del X_product1
    
    res2 = torch.transpose(torch.transpose((res1.view(dim[1]*dim[1], w_dim[1], num_class*num_class)), 1, 0), 1, 2)

    del res1
    
    res3 = res2.view(w_dim[1], num_class, num_class, dim[1], dim[1])

    del res2
#     res = torch.transpose(res, 1, 2)
    
    res4 = torch.transpose(res3, 2, 3)
    
    del res3
    
    print(res4.shape)
    
    res = torch.reshape(res4, [w_dim[1], num_class*dim[1], dim[1]*num_class])
    
    del res4
    
    print('time::', t2 - t1)
    
#     res = res.view([max_epoch, num_class, dim[1], dim[1]*num_class])
#     
#     res = res.view()
    
    
     
    
    
    
#     for t in range(w_dim[0]):
# #         print(X_product.shape)
# #         print(w_seq[t].shape)
#         curr_res = torch.mm(X_product, w_seq[t].view(dim[0], 1))
# #         print(curr_res.shape)
#         res[t] = curr_res.view(dim[1], dim[1])
    
    
    
#     for i in range(batch_num):
#         X_product_subset = X_product[i*batch_size:(i+1)*batch_size, :]
#         
#         w_seq_subset = w_seq_transpose[i*batch_size:(i+1)*batch_size, :]
#         
#         curr_res = torch.bmm(X_product_subset.view(batch_size, dim[1]*dim[1], 1), w_seq_subset.view(batch_size, 1, w_dim[0]))
#         
# #         curr_res = torch.transpose(torch.transpose(curr_res, 1, 2), 0, 1)
#         
#         curr_res = torch.sum(curr_res, dim = 0)
#         
#         curr_res = torch.t(curr_res)#.view(w_dim[0], dim[1], dim[1])
#         
#         res += curr_res
#     
#     if batch_num*batch_size < dim[0]:
#         curr_batch_size = dim[0] - batch_num*batch_size
#         
#         X_product_subset = X_product[batch_num*batch_size:dim[0], :]
#         
#         w_seq_subset = w_seq_transpose[batch_num*batch_size:dim[0], :]
#         
#         curr_res = torch.bmm(X_product_subset.view(curr_batch_size, dim[1]*dim[1], 1), w_seq_subset.view(curr_batch_size, 1, w_dim[0]))
#         
# #         curr_res = torch.transpose(torch.transpose(curr_res, 1, 2), 0, 1)
#         
#         curr_res = torch.sum(curr_res, dim = 0)
#         
#         curr_res = torch.t(curr_res)#.view(w_dim[0], dim[1], dim[1])
#         
#         res += curr_res
    
#     res = torch.bmm(X_product.view(dim[0], dim[1]*dim[1], 1), torch.t(w_seq).view(dim[0], 1, w_dim[0]))
#     
# #     print(res.shape)
# 
#     res = torch.transpose(torch.transpose(torch.transpose(res.view(dim[0], dim[1], dim[1], w_dim[0]), 2, 3), 1, 2), 0, 1)
#     
#     res = torch.sum(res, dim = 1)
#     
#     print('final_res_shape::', res.shape)
    
    
    
    
    return res


def prepare_term_1_batch2_epoch(X, weights, dim, cut_off_epoch, num_class):
    
    term1 = torch.zeros([cut_off_epoch, num_class*dim[1], num_class*dim[1]], dtype = torch.double)
    
    for i in range(cut_off_epoch):
#         sub_term_1 = prepare_term_1(delta_X, sub_weights[:, cut_off_epoch - 1], delta_X.shape, num_class)

        res1 = torch.bmm(X.view(dim[0], dim[1], 1), weights[:,i].view(dim[0], 1, num_class*num_class))
    
        '''dim[1],dim[1]*t*num_class*num_class'''
        res2 = torch.mm(torch.t(X), res1.view(dim[0], dim[1]*num_class*num_class)).view(dim[1]*dim[1], num_class*num_class)
        
        del res1
        
        res3 = torch.reshape(torch.t(res2), [num_class, num_class, dim[1], dim[1]])
        
        del res2
        
        res4 = torch.reshape(torch.transpose(res3, 1, 2), [num_class*dim[1], dim[1]*num_class])
            
        del res3
        
        term1[i] = res4
        
    return term1


def prepare_term_1(X, weights, dim, max_epoch, num_class):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    
    
    
    w_dim = weights.shape
    
    print(w_dim)
    
    print(dim)
    
#     w_seq_transpose = torch.t(w_seq)
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     batch_num = int(dim[0]/batch_size)
    
#     res = torch.zeros(w_dim[0], dim[1], dim[1], dtype = torch.double)
    
    '''dim[0]*dim[1]*(max_epoch*num_class*num_class)'''
    
    t1 = time.time()
    
    res1 = torch.bmm(X.view(dim[0], dim[1], 1), weights.view(dim[0], 1, w_dim[1]*num_class*num_class))
    
    '''dim[1],dim[1]*t*num_class*num_class'''
    res2 = torch.mm(torch.t(X), res1.view(dim[0], dim[1]*w_dim[1]*num_class*num_class)).view(dim[1]*dim[1], w_dim[1], num_class*num_class)
    
    del res1
    
    res3 = torch.transpose(torch.transpose(res2, 0, 1), 2, 1).view(w_dim[1], num_class, num_class, dim[1], dim[1])
    
    del res2
    
    res4 = torch.reshape(torch.transpose(res3, 2, 3), [w_dim[1], num_class*dim[1], dim[1]*num_class])
    
#     res4 = torch.transpose(res3, 2, 3).view(w_dim[1], dim[1]*num_class, dim[1]*num_class)
    
    del res3
    
    t2 = time.time()
    
#     del X_product1
#     
#     res2 = torch.transpose(torch.transpose((res1.view(dim[1]*dim[1], w_dim[1], num_class*num_class)), 1, 0), 1, 2)
# 
#     del res1
#     
#     res3 = res2.view(w_dim[1], num_class, num_class, dim[1], dim[1])
# 
#     del res2
# #     res = torch.transpose(res, 1, 2)
#     
#     res4 = torch.transpose(res3, 2, 3)
#     
#     del res3
#     
#     print(res4.shape)
#     
#     res = torch.reshape(res4, [w_dim[1], num_class*dim[1], dim[1]*num_class])
#     
#     del res4
    
    print('time::', t2 - t1)
    
#     res = res.view([max_epoch, num_class, dim[1], dim[1]*num_class])
#     
#     res = res.view()
    
    
     
    
    
    
#     for t in range(w_dim[0]):
# #         print(X_product.shape)
# #         print(w_seq[t].shape)
#         curr_res = torch.mm(X_product, w_seq[t].view(dim[0], 1))
# #         print(curr_res.shape)
#         res[t] = curr_res.view(dim[1], dim[1])
    
    
    
#     for i in range(batch_num):
#         X_product_subset = X_product[i*batch_size:(i+1)*batch_size, :]
#         
#         w_seq_subset = w_seq_transpose[i*batch_size:(i+1)*batch_size, :]
#         
#         curr_res = torch.bmm(X_product_subset.view(batch_size, dim[1]*dim[1], 1), w_seq_subset.view(batch_size, 1, w_dim[0]))
#         
# #         curr_res = torch.transpose(torch.transpose(curr_res, 1, 2), 0, 1)
#         
#         curr_res = torch.sum(curr_res, dim = 0)
#         
#         curr_res = torch.t(curr_res)#.view(w_dim[0], dim[1], dim[1])
#         
#         res += curr_res
#     
#     if batch_num*batch_size < dim[0]:
#         curr_batch_size = dim[0] - batch_num*batch_size
#         
#         X_product_subset = X_product[batch_num*batch_size:dim[0], :]
#         
#         w_seq_subset = w_seq_transpose[batch_num*batch_size:dim[0], :]
#         
#         curr_res = torch.bmm(X_product_subset.view(curr_batch_size, dim[1]*dim[1], 1), w_seq_subset.view(curr_batch_size, 1, w_dim[0]))
#         
# #         curr_res = torch.transpose(torch.transpose(curr_res, 1, 2), 0, 1)
#         
#         curr_res = torch.sum(curr_res, dim = 0)
#         
#         curr_res = torch.t(curr_res)#.view(w_dim[0], dim[1], dim[1])
#         
#         res += curr_res
    
#     res = torch.bmm(X_product.view(dim[0], dim[1]*dim[1], 1), torch.t(w_seq).view(dim[0], 1, w_dim[0]))
#     
# #     print(res.shape)
# 
#     res = torch.transpose(torch.transpose(torch.transpose(res.view(dim[0], dim[1], dim[1], w_dim[0]), 2, 3), 1, 2), 0, 1)
#     
#     res = torch.sum(res, dim = 1)
#     
#     print('final_res_shape::', res.shape)
    
    
    
    
    return res4


def prepare_term_1_batch4(X_theta_prod_softmax_seq_tensor, X_theta_prod_seq_tensor, X, dim, max_epoch, num_class):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    
    
#     X_theta_prod_softmax_seq_tensor = torch.transpose(X_theta_prod_softmax_seq_tensor, 0 ,1)
    
#     w_dim = weights.shape
    
#     print(w_dim)
    
    print(dim)
    
    res = Variable(torch.zeros([max_epoch, dim[1]*num_class, dim[1]*num_class], dtype = torch.float))
    
    weights = torch.zeros([dim[0], max_epoch, num_class, num_class], dtype = torch.float)
    
    offsets = torch.zeros([dim[0], max_epoch, num_class], dtype = torch.float)
    
#     last_weight = None
#     
#     last_offsets = None
    
    for i in range(max_epoch):
        '''X_theta_prod_softmax_seq_tensor[i]: n*q'''
        
        curr_weight = Variable(torch.bmm(X_theta_prod_softmax_seq_tensor[i].view(dim[0],num_class ,1), X_theta_prod_softmax_seq_tensor[i].view(dim[0],1,num_class)))
        
        
        '''n*q*q'''
        
        curr_weight = torch.diag_embed(torch.diagonal(curr_weight, dim1=1, dim2=2)) - curr_weight
        
        
        
        curr_weight = torch.diag_embed(-torch.sum(curr_weight, dim = 1)) + curr_weight
        
        weights[:,i,:,:] = curr_weight
        
        offsets[:,i,:] = X_theta_prod_softmax_seq_tensor[i] - (torch.bmm(X_theta_prod_seq_tensor[i].view(dim[0], 1, num_class), curr_weight.view(dim[0], num_class, num_class))).view(dim[0], num_class)
        
#         if i >= 1:
#             print('weight_changed::', i, torch.norm(curr_weight-last_weight))
#             print('offsets_changed::', i, torch.norm(offsets[:,i,:] - last_offsets))
#         last_weight = curr_weight
#         
#         last_offsets = offsets[:,i,:]
        
#         print(weights[:,i,:,:] - curr_weight)
        
        curr_res = Variable(torch.bmm(X.view(dim[0], dim[1], 1), curr_weight.view(dim[0], 1, num_class*num_class)))
        
        
        curr_res = torch.transpose(curr_res, 1, 2)
        
        curr_res = torch.mm(torch.t(X), curr_res.contiguous().view(dim[0], dim[1]*num_class*num_class))
        
        curr_res = torch.transpose(curr_res.view(dim[1], num_class, num_class, dim[1]), 0,1)
        
        curr_res = curr_res.contiguous().view(dim[1]*num_class, dim[1]*num_class)
        
        res[i] = curr_res
        
        
    return res, weights, offsets



def prepare_term_1_batch3(X_theta_prod_softmax_seq_tensor, X_theta_prod_seq_tensor, X, dim, max_epoch, num_class):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    
    
#     X_theta_prod_softmax_seq_tensor = torch.transpose(X_theta_prod_softmax_seq_tensor, 0 ,1)
    
#     w_dim = weights.shape
    
#     print(w_dim)
    
    print(dim)
    
    res = Variable(torch.zeros([max_epoch, dim[1]*num_class, dim[1]*num_class], dtype = torch.float))
    
    weights = torch.zeros([dim[0], max_epoch, num_class, num_class], dtype = torch.float)
    
    offsets = torch.zeros([dim[0], max_epoch, num_class], dtype = torch.float)
    
#     last_weight = None
#     
#     last_offsets = None
    
    for i in range(max_epoch):
        '''X_theta_prod_softmax_seq_tensor[i]: n*q'''
        
        curr_weight = Variable(torch.bmm(X_theta_prod_softmax_seq_tensor[i].view(dim[0],num_class ,1), X_theta_prod_softmax_seq_tensor[i].view(dim[0],1,num_class)))
        
        
        '''n*q*q'''
        
        curr_weight = torch.diag_embed(torch.diagonal(curr_weight, dim1=1, dim2=2)) - curr_weight
        
        
        
        curr_weight = torch.diag_embed(-torch.sum(curr_weight, dim = 1)) + curr_weight
        
        weights[:,i,:,:] = curr_weight
        
        offsets[:,i,:] = X_theta_prod_softmax_seq_tensor[i] - (torch.bmm(X_theta_prod_seq_tensor[i].view(dim[0], 1, num_class), curr_weight.view(dim[0], num_class, num_class))).view(dim[0], num_class)
        
#         if i >= 1:
#             print('weight_changed::', i, torch.norm(curr_weight-last_weight))
#             print('offsets_changed::', i, torch.norm(offsets[:,i,:] - last_offsets))
#         last_weight = curr_weight
#         
#         last_offsets = offsets[:,i,:]
        
#         print(weights[:,i,:,:] - curr_weight)
        
        curr_res = Variable(torch.bmm(X.view(dim[0], dim[1], 1), curr_weight.view(dim[0], 1, num_class*num_class)))
         
         
        curr_res = torch.transpose(curr_res, 1, 2)
         
        curr_res = torch.mm(torch.t(X), curr_res.contiguous().view(dim[0], dim[1]*num_class*num_class))
         
        curr_res = torch.transpose(curr_res.view(dim[1], num_class, num_class, dim[1]), 0,1)
         
        curr_res = curr_res.contiguous().view(dim[1]*num_class, dim[1]*num_class)
         
        res[i] = curr_res
        
        
    return res, weights, offsets
        
        
    
#     print('done')
    
    
    
#     w_seq_transpose = torch.t(w_seq)
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     batch_num = int(dim[0]/batch_size)
    
#     res = torch.zeros(w_dim[0], dim[1], dim[1], dtype = torch.double)
    


def prepare_term_1_batch3_0(X_theta_prod_softmax_seq_tensor, X_theta_prod_seq_tensor, X, dim, max_epoch, num_class, cut_off_epoch):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    
    
#     X_theta_prod_softmax_seq_tensor = torch.transpose(X_theta_prod_softmax_seq_tensor, 0 ,1)
    
#     w_dim = weights.shape
    
#     print(w_dim)
    
#     print(dim)
    
#     res = Variable(torch.zeros([max_epoch, dim[1]*num_class, dim[1]*num_class], dtype = torch.double))
    
    weights = []#Variable(torch.zeros([dim[0], max_epoch, num_class, num_class], dtype = torch.double))
    
    offsets = []#Variable(torch.zeros([dim[0], max_epoch, num_class], dtype = torch.double))
    
#     last_weight = None
#     
#     last_offsets = None
    
    
    for i in range(cut_off_epoch):
        '''X_theta_prod_softmax_seq_tensor[i]: n*q'''
        
        curr_weight = Variable(torch.bmm(X_theta_prod_softmax_seq_tensor[i].view(dim[0],num_class ,1), X_theta_prod_softmax_seq_tensor[i].view(dim[0],1,num_class)))
        
        
        '''n*q*q'''
        curr_weight_sum = torch.sum(curr_weight, dim = 1)
        
        '''curr_weight_sum:: n*q'''
        
        
        curr_weight3 = torch.diag_embed(curr_weight_sum) - curr_weight
        
        '''curr_weight3: n*q*q'''
#         curr_weight1 = torch.diag_embed(torch.diagonal(curr_weight, dim1=1, dim2=2)) - curr_weight
#         
#         
#         
#         curr_weight2 = torch.diag_embed(-torch.sum(curr_weight1, dim = 1)) + curr_weight1
#         
#         print(curr_weight3 - curr_weight2)
        
        del curr_weight
        
        del curr_weight_sum
        
        
        
        weights.append(curr_weight3)
        
#         weights[:,i,:,:] = curr_weight3
        
#         offsets[:,i,:] = X_theta_prod_softmax_seq_tensor[i] - (torch.bmm(X_theta_prod_seq_tensor[i].view(dim[0], 1, num_class), curr_weight3.view(dim[0], num_class, num_class))).view(dim[0], num_class)
        
        offsets.append(X_theta_prod_softmax_seq_tensor[i] - (torch.bmm(X_theta_prod_seq_tensor[i].view(dim[0], 1, num_class), curr_weight3.view(dim[0], num_class, num_class))).view(dim[0], num_class))
        
        
        
        
#         if i > 0:
#             curr_gap = torch.norm(weights[i - 1] - curr_weight3)
#             
#             if curr_gap < cut_off_threshold:
#                 cut_off_epoch = i
#                 break
        
#     print(cut_off_epoch)    
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

    del X_theta_prod_softmax_seq_tensor[:]

    del X_theta_prod_softmax_seq_tensor
    
    del X_theta_prod_seq_tensor[:]
    
    del X_theta_prod_seq_tensor

    '''weights:: n*t*q*q'''

    weights = torch.transpose(torch.stack(weights), 0, 1)
    
    offsets = torch.transpose(torch.stack(offsets), 0, 1)
    
    
        
    return weights, offsets        
#     return weights[:, 0:cut_off_epoch, :, :], offsets[:, 0:cut_off_epoch, :], cut_off_epoch
        
        
    
#     print('done')
    
    
    
#     w_seq_transpose = torch.t(w_seq)
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     batch_num = int(dim[0]/batch_size)
    
#     res = torch.zeros(w_dim[0], dim[1], dim[1], dtype = torch.double)



def prepare_term_1_batch3_2(sub_weights, X, dim, max_epoch, num_class):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    
    
#     X_theta_prod_softmax_seq_tensor = torch.transpose(X_theta_prod_softmax_seq_tensor, 0 ,1)
    
#     w_dim = weights.shape
    
#     print(w_dim)
    
    print(dim)
    
    res = Variable(torch.zeros([max_epoch, dim[1]*num_class, dim[1]*num_class], dtype = torch.float))
    
    
    for i in range(max_epoch):
        '''X_theta_prod_softmax_seq_tensor[i]: n*q'''
        
#         curr_weight = Variable(torch.bmm(X_theta_prod_softmax_seq_tensor[i].view(dim[0],num_class ,1), X_theta_prod_softmax_seq_tensor[i].view(dim[0],1,num_class)))
#         
#         
#         '''n*q*q'''
#         
#         curr_weight = torch.diag_embed(torch.diagonal(curr_weight, dim1=1, dim2=2)) - curr_weight
#         
#         
#         
#         curr_weight = torch.diag_embed(-torch.sum(curr_weight, dim = 1)) + curr_weight

        curr_weight = sub_weights[:,i,:,:]
        
#         print(weights[:,i,:,:] - curr_weight)
        
        curr_res = Variable(torch.bmm(X.view(dim[0], dim[1], 1), curr_weight.view(dim[0], 1, num_class*num_class)))
        
        
        curr_res = torch.transpose(curr_res, 1, 2)
        
        curr_res = torch.mm(torch.t(X), curr_res.contiguous().view(dim[0], dim[1]*num_class*num_class))
        
        curr_res = torch.transpose(curr_res.view(dim[1], num_class, num_class, dim[1]), 0,1)
        
        curr_res = curr_res.contiguous().view(dim[1]*num_class, dim[1]*num_class)
        
        res[i] = curr_res
        
        
    return res
        
        
    
    print('done')
    
    
    
#     w_seq_transpose = torch.t(w_seq)
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     batch_num = int(dim[0]/batch_size)
    
#     res = torch.zeros(w_dim[0], dim[1], dim[1], dtype = torch.double)



def prepare_term_1_serial(X, w_seq, dim):
    
    w_dim = w_seq.shape
    
    print(w_dim)
    
    print(dim)
    
#     t1 = time.time()
    
    X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     t2 = time.time()
#     
#     print('X_product_time::', (t2 - t1))
    
    res = torch.zeros(w_dim[0], dim[1], dim[1],  dtype = torch.float)
    
#     inter_result = torch.zeros(dim[0], w_dim[0], dim[1], dim[1], dtype = torch.double)
    
    
    for i in range(dim[0]):
        curr_res = torch.mm(X_product[i].view(dim[1]*dim[1], 1), w_seq[:,i].view(1, w_dim[0]))
        
        curr_res = torch.t(curr_res)
        
        curr_res = curr_res.view(w_dim[0], dim[1], dim[1])
        
        res += curr_res
        
#         inter_result[i] = curr_res
        
#         if i == 0:
#             inter_result = curr_res
#         else:        
#             torch.cat((inter_result, curr_res), 0)
        
#     res = torch.transpose(torch.transpose(torch.transpose(res.view(dim[0], dim[1], dim[1], w_dim[0]), 2, 3), 1, 2), 0, 1)
    
#     res = torch.transpose(torch.transpose(res.view(dim[1], dim[1], w_dim[0]), 1, 2), 0, 1)
    
#     res = torch.sum(res, dim = 1)
    
    print('final_res_shape::', res.shape)
    
    
    
    
#     return res, inter_result
    return res
    
#     print(w_dim)
#     
#     print(X.shape)
#     
#     sum_x_product = torch.zeros(w_dim[1], dim[1],dim[1])
#     
#     
#     X_product_batch = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
#     
#     for i in range(dim[0]):
#         
#         curr_x_product = X_product_batch[i]
#         
#         sum_x_product += torch.mm(X[i].view(dim[1], 1))
#         
#     
#     X_product_batch = torch.transpose(X_product_batch, 0, 1).repeat(w_dim[1],1, 1)
# 
#     X_product_batch = X_product_batch.view(w_dim[1], dim[0], dim[1], dim[1])
# 
#     res = X_product_batch*(w_seq.view(w_dim[1], dim[0], 1, 1))
#     
#     return res

def prepare_term_2(term2_inter_res, delta_ids):
    
#     b_dim = b_seq.shape
#     
#     print('b_dim::', b_dim)
    
    term2_inter_res_dim = term2_inter_res.shape
    
    res = torch.zeros(term2_inter_res_dim[1], term2_inter_res_dim[2], dtype = torch.float)
    
    for id in delta_ids:
        res += term2_inter_res[id]
    
    
#     X_Y_prod = X.mul(Y)
#     dim[0]*dim[1]
#     X_Y_prod = X_Y_prod.repeat(b_dim[1], 1)


#     res = torch.bmm(X_Y_prod.view(dim[0], dim[1], 1), torch.t(b_seq).view(dim[0], 1, b_dim[0]))
    
#     X_Y_prod = X_Y_prod.view(b_dim[1], dim[0], dim[1])
#     
#     res = X_Y_prod*(b_seq.view(b_dim[1], dim[0], 1))
    
#     print(res.shape)
#     
#     res = torch.transpose(torch.transpose(res, 1, 2), 0, 1)
#     
#     res = torch.sum(res, dim = 1)
#     
#     print('final_res_shape::',res.shape)
    
    return res


def prepare_term_2_batch(X_Y_prod, b_seq, dim, batch_size):
    
    b_dim = b_seq.shape
    
    print('b_dim::', b_dim)
    
    b_seq_transpose = torch.t(b_seq)
#     X_Y_prod = X.mul(Y)
#     dim[0]*dim[1]
#     X_Y_prod = X_Y_prod.repeat(b_dim[1], 1)

    batch_num = int(dim[0]/batch_size)
    
    res = torch.zeros(b_dim[0], dim[1], dtype = torch.float)
    
    for i in range(batch_num):
        X_Y_prod_subset = X_Y_prod[i*batch_size:(i+1)*batch_size, :]
        
        b_seq_subset = b_seq_transpose[i*batch_size:(i+1)*batch_size, :]
        
        curr_res = torch.bmm(X_Y_prod_subset.view(batch_size, dim[1], 1), b_seq_subset.view(batch_size, 1, b_dim[0]))
        
#         curr_res = torch.transpose(torch.transpose(curr_res, 1, 2), 0, 1)
        
        curr_res = torch.sum(curr_res, dim = 0)
        
        curr_res = torch.t(curr_res)#.view(b_dim[0], dim[1])
        
        res += curr_res
    
    if batch_num*batch_size < dim[0]:
        curr_batch_size = dim[0] - batch_num*batch_size
        
        X_Y_prod_subset = X_Y_prod[batch_num*batch_size:dim[0], :]
        
        b_seq_subset = b_seq_transpose[batch_num*batch_size:dim[0], :]
        
        curr_res = torch.bmm(X_Y_prod_subset.view(curr_batch_size, dim[1], 1), b_seq_subset.view(curr_batch_size, 1, b_dim[0]))
        
#         curr_res = torch.transpose(torch.transpose(curr_res, 1, 2), 0, 1)
        
        curr_res = torch.sum(curr_res, dim = 0)
        
        curr_res = torch.t(curr_res)#.view(b_dim[0], dim[1])
        
        res += curr_res

#     res = torch.bmm(X_Y_prod.view(dim[0], dim[1], 1), torch.t(b_seq).view(dim[0], 1, b_dim[0]))
#     
# #     X_Y_prod = X_Y_prod.view(b_dim[1], dim[0], dim[1])
# #     
# #     res = X_Y_prod*(b_seq.view(b_dim[1], dim[0], 1))
#     
#     print(res.shape)
#     
#     res = torch.transpose(torch.transpose(res, 1, 2), 0, 1)
#     
#     res = torch.sum(res, dim = 1)
#     
#     print('final_res_shape::',res.shape)
    
    return res


def prepare_term_2_batch2(X, offsets, dim, max_epoch, num_class):
    
    
    '''offsets:: dim[0], max_epoch, num_class, 1'''
    '''x:: dim[0], dim[1]'''
    b_dim = offsets.shape
    
    print('b_dim::', b_dim)
    
#     b_seq_transpose = torch.t(b_seq)
# #     X_Y_prod = X.mul(Y)
# #     dim[0]*dim[1]
# #     X_Y_prod = X_Y_prod.repeat(b_dim[1], 1)
# 
#     batch_num = int(dim[0]/batch_size)
#     
#     res = torch.zeros(b_dim[0], dim[1], dtype = torch.double)
    
    print(X.shape)
    
    print('offsets_shape::', offsets.shape)
    
#     print('offsets::', offsets)
    
    res = torch.t(torch.mm(torch.t(X), torch.reshape(offsets, (dim[0], offsets.shape[1]*num_class))))
    
#     print(res/dim[0])
    
    res = torch.reshape(res, [offsets.shape[1], num_class*dim[1]])
    
#     res = res.view(max_epoch, num_class*dim[1])
    
#     print('term2::', res)
    
    
    
#     res = torch.t(torch.mm(torch.t(X), torch.t(b_seq)))
    
    
#     for i in range(batch_num):
#         X_Y_prod_subset = X_Y_prod[i*batch_size:(i+1)*batch_size, :]
#         
#         b_seq_subset = b_seq_transpose[i*batch_size:(i+1)*batch_size, :]
#         
#         curr_res = torch.bmm(X_Y_prod_subset.view(batch_size, dim[1], 1), b_seq_subset.view(batch_size, 1, b_dim[0]))
#         
# #         curr_res = torch.transpose(torch.transpose(curr_res, 1, 2), 0, 1)
#         
#         curr_res = torch.sum(curr_res, dim = 0)
#         
#         curr_res = torch.t(curr_res)#.view(b_dim[0], dim[1])
#         
#         res += curr_res
#     
#     if batch_num*batch_size < dim[0]:
#         curr_batch_size = dim[0] - batch_num*batch_size
#         
#         X_Y_prod_subset = X_Y_prod[batch_num*batch_size:dim[0], :]
#         
#         b_seq_subset = b_seq_transpose[batch_num*batch_size:dim[0], :]
#         
#         curr_res = torch.bmm(X_Y_prod_subset.view(curr_batch_size, dim[1], 1), b_seq_subset.view(curr_batch_size, 1, b_dim[0]))
#         
# #         curr_res = torch.transpose(torch.transpose(curr_res, 1, 2), 0, 1)
#         
#         curr_res = torch.sum(curr_res, dim = 0)
#         
#         curr_res = torch.t(curr_res)#.view(b_dim[0], dim[1])
#         
#         res += curr_res
# 
# #     res = torch.bmm(X_Y_prod.view(dim[0], dim[1], 1), torch.t(b_seq).view(dim[0], 1, b_dim[0]))
# #     
# # #     X_Y_prod = X_Y_prod.view(b_dim[1], dim[0], dim[1])
# # #     
# # #     res = X_Y_prod*(b_seq.view(b_dim[1], dim[0], 1))
# #     
# #     print(res.shape)
# #     
# #     res = torch.transpose(torch.transpose(res, 1, 2), 0, 1)
# #     
# #     res = torch.sum(res, dim = 1)
# #     
# #     print('final_res_shape::',res.shape)
    
    return res

def prepare_term_2_serial(X, Y, b_seq, dim):
    
    b_dim = b_seq.shape
    
    print('b_dim::', b_dim)
    
    X_Y_prod = X.mul(Y)
#     dim[0]*dim[1]
#     X_Y_prod = X_Y_prod.repeat(b_dim[1], 1)

    res = torch.zeros(b_dim[0], dim[1], dtype = torch.float)

#     inter_res = torch.zeros(dim[0], b_dim[0], dim[1], dtype = torch.double)

    for i in range(dim[0]):
        
        curr_res = torch.mm(X_Y_prod[i].view(dim[1], 1), b_seq[:,i].view(1, b_dim[0]))
        
        curr_res = torch.t(curr_res)
        
#         inter_res[i] = curr_res
        
#         if i == 0:
#             inter_res = curr_res
#         
#         else:
#             torch.cat((inter_res, curr_res), 0)
        
#         res += torch.mm(X_Y_prod[i].view(dim[1], 1), b_seq[:,i].view(1, b_dim[0]))
        res += curr_res
    
#     X_Y_prod = X_Y_prod.view(b_dim[1], dim[0], dim[1])
#     
#     res = X_Y_prod*(b_seq.view(b_dim[1], dim[0], 1))
    
    print(res.shape)
    
#     res = torch.transpose(res, 0, 1)
    
#     res = torch.sum(res, dim = 1)
    
    print('final_res_shape::',res.shape)
    
#     return res, inter_res
    return res

def verity_term_1(weights, theta, X, epoch, num_class, dim):
    
    
#     print(theta)
    
    coeff_sum = torch.zeros([num_class, X.shape[1]], dtype = torch.float)
    
    for i in range(X.shape[0]):
        for j in range(num_class):
            
            temp = torch.mm(theta[epoch][:,j].view(1, -1), X[i].view(-1, 1))
            
            
            temp = (temp*weights[i][epoch][j]).view(-1, 1)
            
            coeff_sum += torch.mm(temp, X[i].view(1, -1))
    
    
    print('expected_output::', coeff_sum.view(-1,1)/dim[0])
    
    
        
def generate_theta_seq(res_prod_seq, dim, num_class, max_epoch):
    
    theta_seq_tensor = torch.zeros([dim[1], num_class*max_epoch], dtype = torch.float)
    
    for i in range(len(res_prod_seq)):
        theta_seq_tensor[:, i*num_class:(i+1)*num_class] = res_prod_seq[i]
        
    return theta_seq_tensor

def compute_hessian_matrix_2(theta, X, dim, num_class, X_product):
    
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
    
    X_product1 = torch.t(X_product.view(dim[0], dim[1]*dim[1]))
    
    res1 = Variable(torch.mm(X_product1, curr_weight3.view(dim[0], num_class*num_class)))
    
    del X_product1
    
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
        
        
    return res/dim[0] + beta*torch.eye(num_class*dim[1], dtype = torch.float)


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


def compute_first_derivative(X, Y, dim, theta, num_class):
    x_sum_by_class = compute_x_sum_by_class(X, Y, num_class, X.shape)
    output = softmax_layer(torch.mm(X, theta))
        
        
    output = torch.mm(torch.t(X), output)
    
    
    output = torch.reshape(torch.t(output), [-1,1])
    
#     print(output.shape)
    
#     print(theta.shape)
    
#         print(i, theta)
    
#         inter_result2.append(output)
#         
#         res = torch.mm(torch.t(torch.gather(output, 1, Y.view(dim[0], 1))), X)
    
    reshape_theta = torch.reshape(torch.t(theta), (-1, 1))
    
    
    return (output - x_sum_by_class) + beta*theta.view(-1, 1)*X.shape[0]
    


if __name__ == '__main__':
    
    
    delta_size = 140000
    
    repetition = 1
    
    # alpha = 0.00001
    #       
    # beta = 0.5
    # [X, Y] = clean_sensor_data()
    
    
    [X, Y] = load_data_multi_classes(True)
    
    X = extended_by_constant_terms(X)
    
    dim = X.shape
    
    num_class = torch.unique(Y).shape[0]
    
    '''X, Y, lr, dim, num_class'''
    
    print(X.shape)
    
    print(Y.shape)
    
    res, max_epoch = compute_parameters(X, Y, initialize(X, num_class), dim, num_class)
    
    print('max_epoch::', max_epoch)
    
    '''weights:: dim[0], max_epoch, num_class, num_class......first num_class '''
    '''offsets:: dim[0], max_epoch, num_class, 1'''
    # offsets = compute_linear_approx_parameters1(X, Y, dim, num_class)
    # 
    # # print('weights::', weights[1])
    # 
    # print('offsets::', offsets[1])
    
    t_01 = time.time()
     
     
    # X_Y_mult = X.mul(Y)
     
    # X_product = torch.zeros(dim[0], dim[1], dim[1])
    # 
    # for i in range(dim[0]):
    #     X_product[i] = torch.mm(X[i].view(dim[1]), X[i].view(1, dim[1]))
     
     
     
    X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
     
     
    # term1, term1_inter_result = prepare_term_1_serial(X, w_seq, dim)
    # term1 = prepare_term_1_batch2(X_product, weights, dim, max_epoch, num_class)
     
    X_theta_prod_softmax_seq_tensor = torch.stack(X_theta_prod_softmax_seq, dim = 0)
     
    X_theta_prod_seq_tensor = torch.stack(X_theta_prod_seq, dim = 0)
    
    weights, offsets, cut_off_epoch = prepare_term_1_batch3_0(X_theta_prod_softmax_seq_tensor, X_theta_prod_seq_tensor, X, dim, max_epoch, num_class) 
     
    term_1_time_1 = time.time() 
     
    term1 = prepare_term_1_batch2(X_product, weights, dim, max_epoch, num_class)
    
    term_1_time_2 = time.time()
     
    # print(torch.max(torch.abs(term1 - term1_2)))
     
     
    # term2, term2_inter_result = prepare_term_2_serial(X, Y, b_seq, dim)
    term2 = prepare_term_2_batch2(X, offsets, dim, max_epoch, num_class)
     
     
    x_sum_by_class = compute_x_sum_by_class(X, Y, num_class, dim)
     
     
    t_02 = time.time()
    
    dim = X.shape
    
    delta_data_ids = random_generate_subset_ids(dim, delta_size)
    
    print(delta_data_ids)
    
    
    # selected_rows = torch.tensor(list(set(range(dim[0])) - delta_data_ids))
    
    update_X, selected_rows = get_subset_training_data(X, dim, delta_data_ids)
    
    update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
    print(X.shape)
    
    print(update_X.shape)
    
    # origin_res1,_ = compute_model_parameter_by_iteration(dim, initialize(update_X, num_class).theta, X, Y, x_sum_by_class, num_class)
    # 
    
    # res_prod_seq = []
    # 
    # for i in range(repetition):
    #      
    #     lr = initialize(X, num_class)
    # #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
    #     origin_res = logistic_regression_by_standard_library(X, Y, lr, dim, res_prod_seq)
    # 
    # verity_term_1(weights, res_prod_seq, X, 1, num_class, dim)
    # 
    # 
    # origin_res2 = compute_model_parameter_by_approx_incremental_2(term1, term2, x_sum_by_class, dim, initialize(update_X, num_class).theta, num_class)
    # 
    # 
    # print('origin_gap1::', origin_res1 - origin_res2)
    #   
    # print('origin_gap2::', origin_res1 - res)
    
    t1 = time.time()
    
    res1 = torch.zeros(res.shape)
    
    # res_prod_seq = []
    
    for i in range(repetition):
        
        lr = initialize(update_X, num_class)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
        res1= logistic_regression_by_standard_library(update_X, update_Y, lr, dim, max_epoch)
    
    
    
    # theta_seq_tensor = generate_theta_seq(res_prod_seq, dim, num_class, max_epoch)    
    # 
    # 
    # exp_weights, exp_offsets = compute_linear_approx_parameters2(update_X, update_Y, update_X.shape, num_class, theta_seq_tensor)
    
    t2 = time.time()
    
    t3 = time.time()
    
    total_time = 0
    
    res2 = torch.zeros(res.shape)
    # 
    for i in range(repetition):
        init_theta = Variable(initialize(update_X, num_class).theta)
        
        update_x_sum_by_class = compute_x_sum_by_class(update_X, update_Y, num_class, update_X.shape)
    #     update_X_Y_mult = update_X.mul(update_Y)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)dim, theta,  X, Y, X_sum_by_class, num_class
        res2, total_time = compute_model_parameter_by_iteration(dim, init_theta, update_X, update_Y, update_x_sum_by_class, num_class, max_epoch)
    
    
    t4 = time.time()
    # t5 = time.time()
    # 
    # for i in range(repetition):
    #     [U, S, V] = torch.svd(torch.mm(torch.transpose(X, 0, 1), X))
    # 
    # t6 = time.time()
    
    
    
    
    step_time_1 = 0
    
    step_time_2 = 0
    
    
    t5 = time.time()
    
    res3 = torch.zeros(res.shape)
    
    
    for i in range(repetition):
    #     print(X_Y_mult.shape)
    #     
    #     print('w_seq_shape::', w_seq.shape)
    #      
    #     print(b_seq.shape)
         
         
        if len(delta_data_ids) < (dim[0])/2:
             
    #         print(weights.shape)
    #         
    #         print(delta_data_ids)
             
            sub_weights = torch.index_select(weights, 0, delta_data_ids)#get_subset_parameter_list(selected_rows, delta_data_ids, w_seq, dim, 1)
              
            sub_offsets = torch.index_select(offsets, 0, delta_data_ids)#get_subset_parameter_list(b_seq, delta_data_ids, b_seq, dim, 1)
              
              
    #         sub_x_theta_prod_softmax_tensor = torch.index_select(X_theta_prod_softmax_seq_tensor, 1, delta_data_ids)
              
    #         selected_weights = torch.index_select(weights, 0, selected_rows)
    #         
    #         selected_offsets = torch.index_select(offsets, 0, selected_rows)
    #         
    #         print('weights_gap::', torch.max(torch.abs(selected_weights - exp_weights)))
    #         
    #         print('offsets_gap::', torch.max(torch.abs(selected_offsets - exp_offsets)))
             
             
        #     delta_X = torch.index_select(X, 0, delta_data_ids)
        #      
        #     delta_Y = torch.index_select(Y, 0, delta_data_ids)
        #     sub_X_Y_mult = get_subset_parameter_list(selected_rows, delta_data_ids, X_Y_mult, dim, 0)
              
             
               
    #         delta_X_Y_mult = torch.index_select(X_Y_mult, 0, delta_data_ids)
             
            delta_X = torch.index_select(X, 0, delta_data_ids)
             
            delta_Y = torch.index_select(Y, 0, delta_data_ids)
             
            delta_X_product = torch.bmm(delta_X.view(delta_X.shape[0], dim[1], 1), delta_X.view(delta_X.shape[0], 1, dim[1]))
             
    #         delta_X_product = torch.index_select(X_product, 0, delta_data_ids)
             
    #         expected_sum = torch.zeros([num_class, dim[1]], dtype = torch.double)
    # #     
    # #     
    # #         
    #         select_offsets = torch.index_select(offsets, 0, selected_rows)
    #         
    #         delta_sum = torch.zeros([num_class, dim[1]], dtype = torch.double)
    #         
    #         for i in range(sub_offsets.shape[0]):
    #             delta_sum += torch.mm(sub_offsets[i][0].view(num_class, 1), delta_X[i].view(1, dim[1]))
    #              
    #         print('expected_sum::',delta_sum.view(-1,1)/dim[0])
             
              
        #     sub_term_1 = prepare_term_1_serial(delta_X, sub_weights, delta_X.shape)#(delta_X_product, sub_weights, delta_X.shape)
        #      
        #     sub_term_2 = prepare_term_2_serial(delta_X, delta_Y, sub_offsets, delta_X.shape)#(delta_X_Y_mult, sub_offsets, delta_X.shape)
             
        #     update_X_dim = update_X.dim
             
    #         curr_delta_dim = [X.shape[0] - update_X.shape[0], X.shape[1]]
             
            s_1 = time.time()
             
             
            #X_product, weights, dim, max_epoch, num_class
    #         sub_term_1 = prepare_term_1_batch3_2(sub_weights, delta_X, delta_X.shape, max_epoch, num_class)
              
            sub_term_1 = prepare_term_1_batch2(delta_X_product, sub_weights, delta_X.shape, max_epoch, num_class)
    #         sub_term_1 = prepare_term_1_batch2(sub_x_theta_prod_softmax_tensor, delta_X, delta_X.shape, max_epoch, num_class)
    #         exp_sub_term_2 = prepare_term_2_batch2(update_X, exp_offsets, update_X.shape, max_epoch)
             
            sub_term_2 = prepare_term_2_batch2(delta_X, sub_offsets, delta_X.shape, max_epoch)     
              
    #         print('sub_term2_gap::', torch.max(torch.abs(((term2 - sub_term_2) - exp_sub_term_2)))) 
             
              
            sub_x_sum_by_class = compute_x_sum_by_class(delta_X, delta_Y, num_class, delta_X.shape) 
             
    #         print((sub_term_2[0].view(-1,1))/dim[0])
     
    #         selected_weights = torch.index_select(weights, 0, selected_rows)
    # 
    #         verity_term_1(selected_weights, res_prod_seq, update_X, 1, num_class, dim)
              
            s_2 = time.time()
             
            step_time_1 += s_2  -s_1
        #     sub_term_2 = prepare_term_2_batch(delta_X_Y_mult, sub_offsets, delta_X.shape)
         
        #     sub_term_1 = prepare_term_1(term1_inter_result, delta_data_ids)
        #     
        #     sub_term_2 = prepare_term_2(term2_inter_result, delta_data_ids)
             
            init_theta = Variable(initialize(update_X, num_class).theta)
             
        #     res2 = update_model_parameters_incrementally(U, S, V, update_X, update_Y, max_epoch, dim)
         
        #     print(sub_weights.shape)
        #     
        #     print(sub_offsets.shape)
        #     
        #     print(update_X.shape)
        #     
        #     print(update_Y.shape)
        #     
        #     print(lr.theta.shape)
         
        #     update_X_products = compute_sample_products(update_X, update_X.shape)
        #      
        #     update_X_Y_products = compute_sample_label_products(update_X, update_Y)
         
        #     res2 = compute_model_parameter_by_approx(sub_weights, sub_offsets, update_X, update_Y, update_X.shape, lr.theta, update_X_products, update_X_Y_products)
             
            print('this_dim::', dim)
             
            print('init_theta::', init_theta)
             
            s_3 = time.time()
             
            res3 = compute_model_parameter_by_approx_incremental_2(term1 - sub_term_1, term2 - sub_term_2, x_sum_by_class - sub_x_sum_by_class, dim, init_theta, num_class, max_epoch)
             
            s_4 = time.time()
             
            step_time_2 += s_4  -s_3
         
        else:
            sub_weights = torch.index_select(weights, 1, selected_rows)#get_subset_parameter_list(selected_rows, delta_data_ids, w_seq, dim, 1)
               
            sub_offsets = torch.index_select(offsets, 1, selected_rows)#get_subset_parameter_list(b_seq, delta_data_ids, b_seq, dim, 1)
               
        #     delta_X = torch.index_select(X, 0, delta_data_ids)
        #      
        #     delta_Y = torch.index_select(Y, 0, delta_data_ids)
        #     sub_X_Y_mult = get_subset_parameter_list(selected_rows, delta_data_ids, X_Y_mult, dim, 0)
               
    #         delta_X_product = torch.index_select(X_product, 0, selected_rows)
             
            delta_X = torch.index_select(X, 0, selected_rows)
             
                
    #         delta_X_Y_mult = torch.index_select(X_Y_mult, 0, selected_rows)
               
        #     sub_term_1 = prepare_term_1_serial(delta_X, sub_weights, delta_X.shape)#(delta_X_product, sub_weights, delta_X.shape)
        #      
        #     sub_term_2 = prepare_term_2_serial(delta_X, delta_Y, sub_offsets, delta_X.shape)#(delta_X_Y_mult, sub_offsets, delta_X.shape)
              
        #     update_X_dim = update_X.dim
              
            curr_delta_dim = [update_X.shape[0], X.shape[1]]
              
            s_1 = time.time()
              
            sub_term_1 = prepare_term_1_batch2(delta_X, sub_weights, curr_delta_dim)
               
            sub_term_2 = prepare_term_2_batch2(delta_X, sub_offsets, curr_delta_dim)     
               
               
            s_2 = time.time()
              
            step_time_1 += s_2  -s_1
        #     sub_term_2 = prepare_term_2_batch(delta_X_Y_mult, sub_offsets, delta_X.shape)
          
        #     sub_term_1 = prepare_term_1(term1_inter_result, delta_data_ids)
        #     
        #     sub_term_2 = prepare_term_2(term2_inter_result, delta_data_ids)
              
            init_theta = Variable(initialize(update_X, num_class).theta)
              
        #     res2 = update_model_parameters_incrementally(U, S, V, update_X, update_Y, max_epoch, dim)
          
        #     print(sub_weights.shape)
        #     
        #     print(sub_offsets.shape)
        #     
        #     print(update_X.shape)
        #     
        #     print(update_Y.shape)
        #     
        #     print(lr.theta.shape)
          
        #     update_X_products = compute_sample_products(update_X, update_X.shape)
        #      
        #     update_X_Y_products = compute_sample_label_products(update_X, update_Y)
          
        #     res2 = compute_model_parameter_by_approx(sub_weights, sub_offsets, update_X, update_Y, update_X.shape, lr.theta, update_X_products, update_X_Y_products)
              
            print('this_dim::', dim)
              
            s_3 = time.time()
              
            res3 = compute_model_parameter_by_approx_incremental_2(sub_term_1, sub_term_2, dim, init_theta)
              
            s_4 = time.time()
              
            step_time_2 += s_4  -s_3
             
             
     
    t6 = time.time()
     
    time1 = (t2 - t1)/repetition
     
    time2 = (t4 - t3)/repetition
     
    time3 = (t6 - t5)/repetition
    
    term_1_time = (term_1_time_2 - term_1_time_1)
     
    compute_sub_term_time = t_02 - t_01
     
     
    print('res::', res)
     
    print('res1::', res1)
     
    print('res2::', res2)
     
    print('res3::', res3)
     
    print('delta::', res - res1)
     
    # print(torch.max(torch.abs(torch.mm(X, res-res1))))
    # 
    # print(torch.argmax(torch.abs(torch.mm(X, res-res1))))
    # 
    # 
    # curr_softmax_layer = torch.nn.Softmax(dim = 1)
    # 
    # 
    # print(torch.max(torch.abs(curr_softmax_layer(torch.mm(X, res)) - curr_softmax_layer(torch.mm(X, res1)))))
     
    print(res3 - res2)
     
    print(res3 - res1)
     
    print(torch.norm(res3, p=2))
     
    print(torch.norm(res3 - res2, p=2))
     
    # print('res2', res2)
     
    print('time1', time1)
     
    print('time2', time2)
     
    print('time3', time3)
     
    print('compute_sub_term_time::', compute_sub_term_time)
     
    print('sigmoid_time::', total_time)
     
    print('step_time_1::', step_time_1)
     
    print('step_time_2::', step_time_2)
    
    print('term_1_time::', term_1_time)





# compute_model_parameter_by_iteration2(dim, Variable(initialize(X).theta), X_Y_mult)




# print('delta_ids::', delta_data_ids)

# hessian_matrix = compute_hessian_matrix(X, Y, res, dim, X_product)
# 
# res4 = torch.zeros(res.shape)
# 
# 
# t7 = time.time()
# 
# first_derivative = compute_first_derivative_single_data(X_Y_mult, delta_data_ids, res, dim)
# 
# res4 = res + torch.mm(torch.inverse(hessian_matrix), first_derivative)/dim[0]
# 
# 
# t8 = time.time()
# 
# print(hessian_matrix)
# 
# print(first_derivative)
# 
# 
# 
# print(res4)
# 
# print(res4  - res1)
# 
# print('time4::', (t8  -t7))
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















