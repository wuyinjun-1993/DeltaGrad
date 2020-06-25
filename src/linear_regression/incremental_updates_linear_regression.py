'''
Created on Feb 4, 2019


'''
from torch import nn, optim
import torch

# try:
#     from ...data_IO import *
# except ImportError:
#     from .data_IO import *

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_IO.Load_data import *
    from sensitivity_analysis_SGD.linear_regression.Linear_regression import *
except ImportError:
    from Load_data import *
    from Linear_regression import *

import random

# try:
#     from .Logistic_regression import *
# except ImportError:
#     from Logistic_regression import *
    
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
    theta = Variable(torch.zeros([dim[1],1])).type(torch.DoubleTensor)
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
    
    diag_matrix = diag_matrix.type(torch.DoubleTensor)
    
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
    
    res = torch.zeros(inter_res_dim[1], inter_res_dim[2], inter_res_dim[3], dtype = torch.double)
    
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
    
    res = torch.zeros(w_dim[0], dim[1]*dim[1], dtype = torch.double)
    
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

def prepare_term_1_batch2(X_product, w_seq, dim, batch_size, max_epoch, mini_batch_epoch, cut_off_epoch):
    
    
    '''batch_num*t + residule, 1'''
    
    
    w_dim = w_seq.shape
    
    
    batch_num = int(dim[0]/batch_size) + 1
#     print(w_dim)
#     
#     print(dim)
    
#     w_seq_transpose = torch.t(w_seq)
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     batch_num = int(dim[0]/batch_size)
    
#     res = torch.zeros(w_dim[0], dim[1], dim[1], dtype = torch.double)

    '''batch_size*'''
    
    
    
#     X_product = X_product[0:batch_size*batch_num].view(-1, dim[1]*dim[1])
#     
#     X_product = torch.reshape(X_product, [batch_num, batch_size, dim[1]*dim[1]])
#     
#     
#     X_product = torch.transpose(X_product, 0, 1)
#     
#     X_product = X_product.view(batch_size, -1)
#     
#     
#     res = torch.mm(torch.t(w_seq), X_product).view(w_dim[1], dim[1], dim[1])
    
    
    
    
    res = torch.zeros((cut_off_epoch, dim[1], dim[1]), dtype = torch.double)
    
    
    num = 0
    
    for t in range(max_epoch):
        
        end = False
        
        for i in range(0, dim[0], batch_size):
        
#             batch_id = i%(batch_num)

            if t*dim[0] + i+ batch_size < w_dim[0]:


                curr_X_prod = X_product[i:i + batch_size]
                

                curr_w_seq = w_seq[t*dim[0] + i: t*dim[0] + i + curr_X_prod.shape[0]]
                
#                 print(i*batch_size, (i + 1)*batch_size)
#                 print(curr_w_seq.shape)
                
                curr_X_prod = curr_X_prod.view(curr_X_prod.shape[0], dim[1]*dim[1])
                
                res[num] = torch.mm(torch.t(curr_X_prod), curr_w_seq).view(dim[1], dim[1])
                
            else:
                
                curr_X_prod = X_product[i:i + batch_size]
                
                curr_w_seq = w_seq[t*dim[0] + i: w_dim[0]]
                
                
                curr_X_prod = curr_X_prod.view(curr_X_prod.shape[0], dim[1]*dim[1])
                
                res[num] = torch.mm(torch.t(curr_X_prod), curr_w_seq).view(dim[1], dim[1])
                
                end = True
                
                break
            
            
            num = num + 1 
    
            if num >= cut_off_epoch:
                end = True
                break
    
        if end:
            break
    
#     X_product = torch.t(X_product.view(dim[0], dim[1]*dim[1]))
#     
#     res = torch.t(torch.mm(X_product, w_seq)).view(w_dim[1], dim[1], dim[1])
    
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


def prepare_term_1_batch2_delta(alpha, beta, term1, term2, delta_X_product, delta_X_Y_prod, w_seq, b_seq, dim, batch_num, batch_size, delta_data_ids, mini_batch_epoch, max_epoch, cut_off_epoch, init_theta):
    
    
    '''batch_size*cut_off_epoch = batch_size*(t*(n/batch_size))'''
    
    
    w_dim = w_seq.shape
    
    
#      = dim[0]/batch_size
#     print(w_dim)
#     
#     print(dim)
    
#     w_seq_transpose = torch.t(w_seq)
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     batch_num = int(dim[0]/batch_size)
    
#     res = torch.zeros(w_dim[0], dim[1], dim[1], dtype = torch.double)

    '''batch_size*'''
    
    
    
#     X_product = X_product[0:batch_size*batch_num].view(-1, dim[1]*dim[1])
#     
#     X_product = torch.reshape(X_product, [batch_num, batch_size, dim[1]*dim[1]])
#     
#     
#     X_product = torch.transpose(X_product, 0, 1)
#     
#     X_product = X_product.view(batch_size, -1)
#     
#     
#     res = torch.mm(torch.t(w_seq), X_product).view(w_dim[1], dim[1], dim[1])
    
    
    
    
#     res = torch.zeros((w_dim[1], dim[1], dim[1]))
    
    
#     res = torch.zeros((int(mini_batch_epoch/max_epoch), dim[1], dim[1]), dtype = torch.double)
#     
#     res2 = torch.zeros([int(mini_batch_epoch/max_epoch), dim[1]], dtype = torch.double)
    
    
#     num = 0
    
    
    A = torch.eye(dim[1], dtype = torch.double)
    
    B = torch.zeros([dim[1], 1], dtype = torch.double)
    
    
    curr_mini_epoch = 0
    
    
    inter_res1_list = []
    
    inter_res2_list = []
    
    for t in range(max_epoch):
        
        end = False
        
#         print('epoch::', t)
        
        for i in range(0, dim[0], batch_size):
        
#             batch_id = i%(batch_num)

#             print('batch::', i)

            

            end_id = i + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]


            satisfiable_delta_ids = ((i<= delta_data_ids)*(delta_data_ids < end_id)).view(-1)

            '''delta_data_id_ids'''
            curr_delta_ids = torch.nonzero(satisfiable_delta_ids)

            curr_term1 = 0
            
            curr_term2 = 0
            
            
            if curr_delta_ids.shape[0] == end_id - i:
                continue
            
            
            

            if curr_delta_ids.shape[0] > 0:

                curr_origin_delta_ids = delta_data_ids[curr_delta_ids]%batch_size + t*dim[0] + i
    
                if t*dim[0] + i+ batch_size < w_dim[0]:
    
    
                    curr_X_prod = delta_X_product[curr_delta_ids.view(-1)]
                    
                    curr_X_Y_prod = delta_X_Y_prod[curr_delta_ids.view(-1)]
    
                    curr_w_seq = w_seq[curr_origin_delta_ids.view(-1)]
                    
                    curr_b_seq = b_seq[curr_origin_delta_ids.view(-1)]
                    
    #                 print(i*batch_size, (i + 1)*batch_size)
    #                 print(curr_w_seq.shape)
                    curr_X_Y_prod = curr_X_Y_prod.view(curr_X_Y_prod.shape[0], dim[1])
                    
                    curr_X_prod = curr_X_prod.view(curr_X_prod.shape[0], dim[1]*dim[1])


                    
                    
                    curr_term1 = torch.mm(torch.t(curr_X_prod), curr_w_seq).view(dim[1], dim[1])
                    
#                     curr_A = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*(term1[num] - res[num])/dim[0]
                    
                    curr_term2 = torch.mm(torch.t(curr_X_Y_prod), curr_b_seq).view(dim[1])
                    
                    
                    
                else:
                    
                    curr_X_prod = delta_X_product[curr_delta_ids.view(-1)]
                    
                    curr_X_Y_prod = delta_X_Y_prod[curr_delta_ids.view(-1)]
                    
                    curr_w_seq = w_seq[curr_origin_delta_ids[curr_origin_delta_ids < w_dim[0]].view(-1)]
                    
                    curr_b_seq = b_seq[curr_origin_delta_ids[curr_origin_delta_ids < w_dim[0]].view(-1)]
                    
                    
                    curr_X_Y_prod = curr_X_Y_prod.view(curr_X_Y_prod.shape[0], dim[1])
                    
                    curr_X_prod = curr_X_prod.view(curr_X_prod.shape[0], dim[1]*dim[1])
                    
                    curr_term1 = torch.mm(torch.t(curr_X_prod), curr_w_seq).view(dim[1], dim[1])
                    
#                     curr_A = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*(term1[num] - res[num])/dim[0]
                    
                    curr_term2 = torch.mm(torch.t(curr_X_Y_prod), curr_b_seq).view(dim[1])
                    
                    end = True
                    
                    break
            
            
            inter_res1_list.append(term1[curr_mini_epoch] - curr_term1)
                
            inter_res2_list.append(term2[curr_mini_epoch] - curr_term2)
            
            
            if len(inter_res1_list) > int((dim[0] - 1)/batch_size) + 1:
                inter_res1_list.pop(0)
                inter_res2_list.pop(0)
            
            
#             if curr_mini_epoch > cut_off_epoch - int(mini_batch_epoch/max_epoch):
#                 res[curr_mini_epoch - 1 - (cut_off_epoch - int(mini_batch_epoch/max_epoch))] = term1[curr_mini_epoch] - curr_term1
#                 
#                 res2[curr_mini_epoch - 1 - (cut_off_epoch - int(mini_batch_epoch/max_epoch))] = term2[curr_mini_epoch] - curr_term2
                
#             if i >= 17300:
#                 print('here')
                
            curr_A = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*(term1[curr_mini_epoch] - curr_term1)/(end_id - i - (curr_delta_ids.shape[0]))
                    
            A = torch.mm(A, curr_A)
            
            B = torch.mm(curr_A, B) + alpha*(term2[curr_mini_epoch] - curr_term2).view([dim[1],1])/(end_id - i - (curr_delta_ids.shape[0]))    
            
#             num = num + 1 

            print('curr_term1::', curr_term1)
             
            print('curr_term2::', curr_term2)
 
            print('theta::', torch.mm(A, init_theta) + B)

            
            curr_mini_epoch += 1
            
#             if curr_mini_epoch >= cut_off_epoch:
#                 end = True
#                 
#                 break
        
        if end:
            break
        
        
        
    
    
    
#     for i in range(w_dim[1]):
#         
#         batch_id = i%(batch_num)
#         
#         
#         satisfiable_delta_ids = (batch_id*batch_size<= delta_data_ids)*(delta_data_ids < (batch_id + 1)*batch_size)
#         
#         curr_delta_ids = torch.nonzero(satisfiable_delta_ids)
#         
#         
#         curr_origin_delta_ids = delta_data_ids[satisfiable_delta_ids]%batch_size
#         
#         
#         
#         if curr_delta_ids.shape[0] > 0:
#             curr_X_prod = delta_X_product[curr_delta_ids].view(curr_delta_ids, dim[1]*dim[1])
#         
#             res[i] = torch.mm(torch.t(curr_X_prod), w_seq[curr_origin_delta_ids,i]).view(dim[1], dim[1])
    
#     X_product = torch.t(X_product.view(dim[0], dim[1]*dim[1]))
#     
#     res = torch.t(torch.mm(X_product, w_seq)).view(w_dim[1], dim[1], dim[1])
    
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
    
    res = torch.stack(inter_res1_list)
    
    res2 = torch.stack(inter_res2_list)
    
    
    return res, res2, A, B, max_epoch*(int((dim[0]-1)/batch_size)+1) - curr_mini_epoch


def prepare_term_1_batch2_delta1(alpha, beta, term1, term2, delta_X_product, delta_X_Y_prod, w_seq, b_seq, dim, batch_num, batch_size, delta_data_ids, mini_batch_epoch, max_epoch, cut_off_epoch, theta):
    
    
    '''batch_size*cut_off_epoch = batch_size*(t*(n/batch_size))'''
    
    
    w_dim = w_seq.shape
    
    
#      = dim[0]/batch_size
#     print(w_dim)
#     
#     print(dim)
    
#     w_seq_transpose = torch.t(w_seq)
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     batch_num = int(dim[0]/batch_size)
    
#     res = torch.zeros(w_dim[0], dim[1], dim[1], dtype = torch.double)

    '''batch_size*'''
    
    
    
#     X_product = X_product[0:batch_size*batch_num].view(-1, dim[1]*dim[1])
#     
#     X_product = torch.reshape(X_product, [batch_num, batch_size, dim[1]*dim[1]])
#     
#     
#     X_product = torch.transpose(X_product, 0, 1)
#     
#     X_product = X_product.view(batch_size, -1)
#     
#     
#     res = torch.mm(torch.t(w_seq), X_product).view(w_dim[1], dim[1], dim[1])
    
    
    
    
#     res = torch.zeros((w_dim[1], dim[1], dim[1]))
    
    
#     res = torch.zeros((int(mini_batch_epoch/max_epoch), dim[1], dim[1]), dtype = torch.double)
#     
#     res2 = torch.zeros([int(mini_batch_epoch/max_epoch), dim[1]], dtype = torch.double)
    
    
#     num = 0
    
    
    
    curr_mini_epoch = 0
    
    
    inter_res1_list = []
    
    inter_res2_list = []
    

    res = term1.clone()#torch.zeros([cut_off_epoch, dim[1], dim[1]], dtype = torch.double)
    
    
    res2 = term2.clone()#torch.zeros(cut_off_epoch, dim[1], dtype = torch.double)
    
    
    min_batch_num_per_epoch = int((dim[0] - 1)/batch_size) + 1


    A = torch.zeros([int((cut_off_epoch)/min_batch_num_per_epoch), dim[1], dim[1]], dtype = torch.double)
    
    B = torch.zeros([int((cut_off_epoch)/min_batch_num_per_epoch), dim[1], 1], dtype = torch.double)

    
    for i in range(0, dim[0], batch_size):
        
        end_id = i + batch_size
            
        if end_id > dim[0]:
            end_id = dim[0]

        
        
        satisfiable_delta_ids = ((i<= delta_data_ids)*(delta_data_ids < end_id)).view(-1)

        '''delta_data_id_ids'''
        curr_delta_ids = torch.nonzero(satisfiable_delta_ids)


        curr_X_prod = delta_X_product[curr_delta_ids.view(-1)]
                    
        curr_X_Y_prod = delta_X_Y_prod[curr_delta_ids.view(-1)]


        curr_batch_delta_id_offset = (delta_data_ids[curr_delta_ids]%batch_size).view(1,-1) 


        curr_origin_delta_ids = curr_batch_delta_id_offset + (torch.tensor(list(range(max_epoch)))*dim[0] + i).view(-1, 1)
        
        curr_origin_delta_ids = curr_origin_delta_ids.view(-1)
        
        ids = ((torch.tensor(list(range(max_epoch)))*min_batch_num_per_epoch + int(i/batch_size))).view(-1)


        if curr_origin_delta_ids.shape[0] <= 0:
            
            res[ids[ids < cut_off_epoch]] = (term1[ids[ids < cut_off_epoch]] - 0)/(end_id - i)
        
            res2[ids[ids < cut_off_epoch]] = (term2[ids[ids < cut_off_epoch]] - 0)/(end_id - i)
            
            curr_A = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*res[ids[ids<cut_off_epoch]]
                
            curr_B = alpha*res2[ids[ids < cut_off_epoch]]
            
            if i == 0:
                A[:] = curr_A[0:A.shape[0]]
                B[:] = curr_B[0:B.shape[0]].view(-1, dim[1], 1)
                
            else:
                
                
    #             if ids[ids < cut_off_epoch].shape[0] == A.shape[0]:
    #                 curr_A = (1 - alpha*beta)*torch.eye(dim[1]) + alpha*res[ids[ids<cut_off_epoch]]
    #                 
    #                 curr_B = alpha*res2
                A = torch.bmm(A, curr_A[0:A.shape[0]])
                
                B = torch.bmm(curr_A[0:A.shape[0]], B) + curr_B[0:B.shape[0]].view(-1, dim[1], 1)

            
            continue
        
        curr_w_seq = torch.t(w_seq[curr_origin_delta_ids[curr_origin_delta_ids<w_seq.shape[0]]].view(-1, curr_batch_delta_id_offset.view(-1).shape[0]))
                    
        curr_b_seq = torch.t(b_seq[curr_origin_delta_ids[curr_origin_delta_ids<b_seq.shape[0]]].view(-1, curr_batch_delta_id_offset.view(-1).shape[0]))
        
        
        curr_X_Y_prod = curr_X_Y_prod.view(curr_X_Y_prod.shape[0], dim[1])
                    
        curr_X_prod = curr_X_prod.view(curr_X_prod.shape[0], dim[1]*dim[1])


        
        
        curr_term1 = torch.t(torch.mm(torch.t(curr_X_prod), curr_w_seq)).view(-1, dim[1], dim[1])
        
#                     curr_A = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*(term1[num] - res[num])/dim[0]
        
        curr_term2 = torch.t(torch.mm(torch.t(curr_X_Y_prod), curr_b_seq)).view(-1, dim[1])
        
        
#         print(i)
        
#         print(curr_term1)
#         
#         print(curr_term2)


        res[ids[ids < cut_off_epoch]] = (term1[ids[ids < cut_off_epoch]] - curr_term1)/(end_id - i - curr_delta_ids.shape[0])
        
        res2[ids[ids < cut_off_epoch]] = (term2[ids[ids < cut_off_epoch]] - curr_term2)/(end_id - i- curr_delta_ids.shape[0])
        
        
        curr_A = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*res[ids[ids<cut_off_epoch]]
                
        curr_B = alpha*res2[ids[ids < cut_off_epoch]]
        
        if i == 0:
            A[:] = curr_A[0:A.shape[0]]
            B[:] = curr_B[0:B.shape[0]].view(-1, dim[1], 1)
            
        else:
            
            
#             if ids[ids < cut_off_epoch].shape[0] == A.shape[0]:
#                 curr_A = (1 - alpha*beta)*torch.eye(dim[1]) + alpha*res[ids[ids<cut_off_epoch]]
#                 
#                 curr_B = alpha*res2
            A = torch.bmm(A, curr_A[0:A.shape[0]])
            
            B = torch.bmm(curr_A[0:A.shape[0]], B) + curr_B[0:B.shape[0]].view(-1, dim[1], 1)
            
                
#         print('curr_term1::', curr_term1)
#              
#         print('curr_term2::', curr_term2)
#     
#         print('batch::', i)
#         print('theta::', torch.mm(A[0], theta) + B[0])
    
    
#     for t in range(max_epoch):
#         
#         end = False
#         
#         for i in range(0, dim[0], batch_size):
#         
# #             batch_id = i%(batch_num)
# 
# 
#             
# 
#             end_id = i + batch_size
#             
#             if end_id > dim[0]:
#                 end_id = dim[0]
# 
# 
#             satisfiable_delta_ids = ((i<= delta_data_ids)*(delta_data_ids < end_id)).view(-1)
# 
#             '''delta_data_id_ids'''
#             curr_delta_ids = torch.nonzero(satisfiable_delta_ids)
# 
#             curr_term1 = 0
#             
#             curr_term2 = 0
#             
#             
#             if curr_delta_ids.shape[0] == end_id - i:
#                 continue
#             
#             
#             
# 
#             if curr_delta_ids.shape[0] > 0:
# 
#                 curr_origin_delta_ids = delta_data_ids[curr_delta_ids]%batch_size + t*dim[0] + i
#     
#                 if t*dim[0] + i+ batch_size < w_dim[0]:
#     
#     
#                     curr_X_prod = delta_X_product[curr_delta_ids.view(-1)]
#                     
#                     curr_X_Y_prod = delta_X_Y_prod[curr_delta_ids.view(-1)]
#     
#                     curr_w_seq = w_seq[curr_origin_delta_ids.view(-1)]
#                     
#                     curr_b_seq = b_seq[curr_origin_delta_ids.view(-1)]
#                     
#     #                 print(i*batch_size, (i + 1)*batch_size)
#     #                 print(curr_w_seq.shape)
#                     curr_X_Y_prod = curr_X_Y_prod.view(curr_X_Y_prod.shape[0], dim[1])
#                     
#                     curr_X_prod = curr_X_prod.view(curr_X_prod.shape[0], dim[1]*dim[1])
# 
# 
#                     
#                     
#                     curr_term1 = torch.mm(torch.t(curr_X_prod), curr_w_seq).view(dim[1], dim[1])
#                     
# #                     curr_A = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*(term1[num] - res[num])/dim[0]
#                     
#                     curr_term2 = torch.mm(torch.t(curr_X_Y_prod), curr_b_seq).view(dim[1])
#                     
#                     
#                     
#                 else:
#                     
#                     curr_X_prod = delta_X_product[curr_delta_ids.view(-1)]
#                     
#                     curr_X_Y_prod = delta_X_Y_prod[curr_delta_ids.view(-1)]
#                     
#                     curr_w_seq = w_seq[curr_origin_delta_ids[curr_origin_delta_ids < w_dim[0]].view(-1)]
#                     
#                     curr_b_seq = b_seq[curr_origin_delta_ids[curr_origin_delta_ids < w_dim[0]].view(-1)]
#                     
#                     
#                     curr_X_Y_prod = curr_X_Y_prod.view(curr_X_Y_prod.shape[0], dim[1])
#                     
#                     curr_X_prod = curr_X_prod.view(curr_X_prod.shape[0], dim[1]*dim[1])
#                     
#                     curr_term1 = torch.mm(torch.t(curr_X_prod), curr_w_seq).view(dim[1], dim[1])
#                     
# #                     curr_A = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*(term1[num] - res[num])/dim[0]
#                     
#                     curr_term2 = torch.mm(torch.t(curr_X_Y_prod), curr_b_seq).view(dim[1])
#                     
#                     end = True
#                     
#                     break
#             
#             
#             inter_res1_list.append(term1[curr_mini_epoch] - curr_term1)
#                 
#             inter_res2_list.append(term2[curr_mini_epoch] - curr_term2)
#             
#             
#             if len(inter_res1_list) > int((dim[0] - 1)/batch_size) + 1:
#                 inter_res1_list.pop(0)
#                 inter_res2_list.pop(0)
#             
#             
# #             if curr_mini_epoch > cut_off_epoch - int(mini_batch_epoch/max_epoch):
# #                 res[curr_mini_epoch - 1 - (cut_off_epoch - int(mini_batch_epoch/max_epoch))] = term1[curr_mini_epoch] - curr_term1
# #                 
# #                 res2[curr_mini_epoch - 1 - (cut_off_epoch - int(mini_batch_epoch/max_epoch))] = term2[curr_mini_epoch] - curr_term2
#                 
#             curr_A = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*(term1[curr_mini_epoch] - curr_term1)/(end_id - i - (curr_delta_ids.shape[0]))
#                     
#             A = torch.mm(A, curr_A)
#             
#             B = torch.mm(curr_A, B) + alpha*(term2[curr_mini_epoch] - curr_term2).view([dim[1],1])/(end_id - i - (curr_delta_ids.shape[0]))    
#             
# #             num = num + 1 
#             
#             curr_mini_epoch += 1
#             
# #             if curr_mini_epoch >= cut_off_epoch:
# #                 end = True
# #                 
# #                 break
#         
#         if end:
#             break
        
        
        
    
    
    
#     for i in range(w_dim[1]):
#         
#         batch_id = i%(batch_num)
#         
#         
#         satisfiable_delta_ids = (batch_id*batch_size<= delta_data_ids)*(delta_data_ids < (batch_id + 1)*batch_size)
#         
#         curr_delta_ids = torch.nonzero(satisfiable_delta_ids)
#         
#         
#         curr_origin_delta_ids = delta_data_ids[satisfiable_delta_ids]%batch_size
#         
#         
#         
#         if curr_delta_ids.shape[0] > 0:
#             curr_X_prod = delta_X_product[curr_delta_ids].view(curr_delta_ids, dim[1]*dim[1])
#         
#             res[i] = torch.mm(torch.t(curr_X_prod), w_seq[curr_origin_delta_ids,i]).view(dim[1], dim[1])
    
#     X_product = torch.t(X_product.view(dim[0], dim[1]*dim[1]))
#     
#     res = torch.t(torch.mm(X_product, w_seq)).view(w_dim[1], dim[1], dim[1])
    
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
    
#     res = torch.stack(inter_res1_list)
#     
#     res2 = torch.stack(inter_res2_list)
    
    
    return res, res2, A, B


def prepare_term_1_batch2_delta2(alpha, beta, update_X, update_Y, w_seq, b_seq, dim, batch_num, batch_size, mini_batch_epoch, max_epoch):
    
    
    '''batch_size*cut_off_epoch = batch_size*(t*(n/batch_size))'''
    
    
    w_dim = w_seq.shape
    
    
#      = dim[0]/batch_size
#     print(w_dim)
#     
#     print(dim)
    
#     w_seq_transpose = torch.t(w_seq)
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     batch_num = int(dim[0]/batch_size)
    
#     res = torch.zeros(w_dim[0], dim[1], dim[1], dtype = torch.double)

    '''batch_size*'''
    
    
    
#     X_product = X_product[0:batch_size*batch_num].view(-1, dim[1]*dim[1])
#     
#     X_product = torch.reshape(X_product, [batch_num, batch_size, dim[1]*dim[1]])
#     
#     
#     X_product = torch.transpose(X_product, 0, 1)
#     
#     X_product = X_product.view(batch_size, -1)
#     
#     
#     res = torch.mm(torch.t(w_seq), X_product).view(w_dim[1], dim[1], dim[1])
    
    
    
    
#     res = torch.zeros((w_dim[1], dim[1], dim[1]))
    
    
    res = torch.zeros((int(mini_batch_epoch/max_epoch), dim[1], dim[1]), dtype = torch.double)
    
    res2 = torch.zeros([int(mini_batch_epoch/max_epoch), dim[1]], dtype = torch.double)
    
    
#     num = 0
    
    
    A = torch.eye(dim[1], dtype = torch.double)
    
    B = torch.zeros([dim[1], 1], dtype = torch.double)
    
    
    curr_mini_epoch = 0
    
    
    cut_off_epoch = int(w_dim[0]/update_X.shape[0]) + (w_dim[0] - int(w_dim[0]/update_X.shape[0])*update_X.shape[0])/batch_size
    
    X_prod = torch.bmm(update_X.view(update_X.shape[0], update_X.shape[1], 1), update_X.view(update_X.shape[0], 1, update_X.shape[1]))
     
    X_Y_prod = update_X.mul(update_Y)
    
    
    inter_res1_list = []
    
    inter_res2_list = []
    
    
    for t in range(max_epoch):
        
        end = False
        
        
        
        
        
        
        for i in range(0, dim[0], batch_size):
        
#             batch_id = i%(batch_num)

#             curr_update_X = update_X[i:i+batch_size]
#             
#             curr_update_Y = update_Y[i:i+batch_size]
# 
#             curr_X_prod = torch.bmm(curr_update_X.view(curr_update_X.shape[0], curr_update_X.shape[1], 1), curr_update_X.view(curr_update_X.shape[0], 1, curr_update_X.shape[1]))
#                     
#             curr_X_Y_prod = curr_update_X.mul(curr_update_Y)
            
            
            curr_X_prod = X_prod[i:i+batch_size]
            
            curr_X_Y_prod = X_Y_prod[i:i+batch_size]
            
            
            if t*dim[0] + i+ batch_size < w_dim[0]:
                
                curr_w_seq = w_seq[t*dim[0] + i: t*dim[0] + i + curr_X_prod.shape[0]]
            
                curr_b_seq = b_seq[t*dim[0] + i: t*dim[0] + i + curr_X_prod.shape[0]]

            else:
                curr_w_seq = w_seq[t*dim[0] + i: w_dim[0]]
            
                curr_b_seq = b_seq[t*dim[0] + i: w_dim[0]]
                
                end = True


#                 print(i*batch_size, (i + 1)*batch_size)
#             print(curr_w_seq.shape)
            curr_X_Y_prod = curr_X_Y_prod.view(curr_X_Y_prod.shape[0], dim[1])
            
            curr_X_prod = curr_X_prod.view(curr_X_prod.shape[0], dim[1]*dim[1])


            
            
            curr_term1 = torch.mm(torch.t(curr_X_prod), curr_w_seq).view(dim[1], dim[1])
            
#                     curr_A = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*(term1[num] - res[num])/dim[0]
            
            curr_term2 = torch.mm(torch.t(curr_X_Y_prod), curr_b_seq).view(dim[1])
            
            

#             end_id = i + batch_size
#             
#             if end_id > dim[0]:
#                 end_id = dim[0]
# 
# 
#             satisfiable_delta_ids = ((i*batch_size<= delta_data_ids)*(delta_data_ids < end_id)).view(-1)
# 
# 
#             curr_delta_ids = torch.nonzero(satisfiable_delta_ids)
# 
#             curr_term1 = 0
#             
#             curr_term2 = 0
#             
#             
#             if satisfiable_delta_ids.shape[0] == curr_delta_ids.shape[0]:
#                 continue
#             
#             
#             
# 
#             if curr_delta_ids.shape[0] > 0:
# 
#                 curr_origin_delta_ids = delta_data_ids[satisfiable_delta_ids]%batch_size + t*dim[0] + i
#     
#                 if t*dim[0] + i+ batch_size < w_dim[0]:
#     
#     
#                     curr_X_prod = delta_X_product[curr_delta_ids]
#                     
#                     curr_X_Y_prod = delta_X_Y_prod[curr_delta_ids]
#     
#                     curr_w_seq = w_seq[curr_origin_delta_ids]
#                     
#                     curr_b_seq = b_seq[curr_origin_delta_ids]
#                     
#     #                 print(i*batch_size, (i + 1)*batch_size)
#     #                 print(curr_w_seq.shape)
#                     curr_X_Y_prod = curr_X_Y_prod.view(curr_X_Y_prod.shape[0], dim[1])
#                     
#                     curr_X_prod = curr_X_prod.view(curr_X_prod.shape[0], dim[1]*dim[1])
# 
# 
#                     
#                     
#                     curr_term1 = torch.mm(torch.t(curr_X_prod), curr_w_seq).view(dim[1], dim[1])
#                     
# #                     curr_A = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*(term1[num] - res[num])/dim[0]
#                     
#                     curr_term2 = torch.mm(torch.t(curr_X_Y_prod), curr_b_seq).view(dim[1])
#                     
#                     
#                     
#                 else:
#                     
#                     curr_X_prod = delta_X_product[curr_delta_ids]
#                     
#                     curr_X_Y_prod = delta_X_Y_prod[curr_delta_ids]
#                     
#                     curr_w_seq = w_seq[curr_origin_delta_ids[curr_origin_delta_ids < w_dim[0]]]
#                     
#                     curr_b_seq = b_seq[curr_origin_delta_ids[curr_origin_delta_ids < w_dim[0]]]
#                     
#                     
#                     curr_X_Y_prod = curr_X_Y_prod.view(curr_X_Y_prod.shape[0], dim[1])
#                     
#                     curr_X_prod = curr_X_prod.view(curr_X_prod.shape[0], dim[1]*dim[1])
#                     
#                     curr_term1 = torch.mm(torch.t(curr_X_prod), curr_w_seq).view(dim[1], dim[1])
#                     
# #                     curr_A = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*(term1[num] - res[num])/dim[0]
#                     
#                     curr_term2 = torch.mm(torch.t(curr_X_Y_prod), curr_b_seq).view(dim[1])
#                     
#                     end = True
#                     
#                     break
                
            if curr_mini_epoch > cut_off_epoch - int(mini_batch_epoch/max_epoch):
                
                inter_res1_list.append(curr_term1)
                
                inter_res2_list.append(curr_term2)
                
                
                if len(inter_res1_list) > int(dim[0]/batch_size) + 1:
                    inter_res1_list.pop(0)
                    inter_res2_list.pop(0)
                
#                 res[curr_mini_epoch - 1 - (cut_off_epoch - int(mini_batch_epoch/max_epoch))] = curr_term1
#                 
#                 res2[curr_mini_epoch - 1 - (cut_off_epoch - int(mini_batch_epoch/max_epoch))] = curr_term2
                
            curr_A = (1 - alpha*beta)*torch.eye(dim[1], dtype = torch.double) + alpha*(curr_term1)/curr_X_prod.shape[0]
                    
            A = torch.mm(A, curr_A)
            
            B = torch.mm(curr_A, B) + alpha*(curr_term2).view([dim[1],1])/curr_X_prod.shape[0]
            
#             num = num + 1 
            
            curr_mini_epoch += 1
            
#             if curr_mini_epoch >= cut_off_epoch:
            if end:
#                 end = True
                
                break
        
        if end:
            break
        
        
        
    
    
    
#     for i in range(w_dim[1]):
#         
#         batch_id = i%(batch_num)
#         
#         
#         satisfiable_delta_ids = (batch_id*batch_size<= delta_data_ids)*(delta_data_ids < (batch_id + 1)*batch_size)
#         
#         curr_delta_ids = torch.nonzero(satisfiable_delta_ids)
#         
#         
#         curr_origin_delta_ids = delta_data_ids[satisfiable_delta_ids]%batch_size
#         
#         
#         
#         if curr_delta_ids.shape[0] > 0:
#             curr_X_prod = delta_X_product[curr_delta_ids].view(curr_delta_ids, dim[1]*dim[1])
#         
#             res[i] = torch.mm(torch.t(curr_X_prod), w_seq[curr_origin_delta_ids,i]).view(dim[1], dim[1])
    
#     X_product = torch.t(X_product.view(dim[0], dim[1]*dim[1]))
#     
#     res = torch.t(torch.mm(X_product, w_seq)).view(w_dim[1], dim[1], dim[1])
    
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
    
    res = torch.stack(inter_res1_list)
    
    res2 = torch.stack(inter_res2_list)
    
    
    return res, res2, A, B, max_epoch*(int((dim[0]-1)/batch_size)+1) - curr_mini_epoch


def prepare_term_1_serial(X, w_seq, dim):
    
    w_dim = w_seq.shape
    
    print(w_dim)
    
    print(dim)
    
#     t1 = time.time()
    
    X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     t2 = time.time()
#     
#     print('X_product_time::', (t2 - t1))
    
    res = torch.zeros(w_dim[0], dim[1], dim[1],  dtype = torch.double)
    
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
    
    res = torch.zeros(term2_inter_res_dim[1], term2_inter_res_dim[2], dtype = torch.double)
    
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
    
    res = torch.zeros(b_dim[0], dim[1], dtype = torch.double)
    
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


def prepare_term_2_batch2(X_Y_prod, b_seq, dim, max_epoch, mini_batch_epoch, batch_size, cut_off_epoch):
    
    
    '''batch_size*cut_off_epoch = batch_size*(t*(n/batch_size))'''
    
    b_dim = b_seq.shape
    
#     print('b_dim::', b_dim)
    
#     b_seq_transpose = torch.t(b_seq)
# #     X_Y_prod = X.mul(Y)
# #     dim[0]*dim[1]
# #     X_Y_prod = X_Y_prod.repeat(b_dim[1], 1)
# 
#     batch_num = int(dim[0]/batch_size)
#     
#     res = torch.zeros(b_dim[0], dim[1], dtype = torch.double)
    
#     print(X_Y_prod.shape)
#     
#     print(b_seq.shape)


#     batch_num = int(dim[0]/batch_size)


    res = torch.zeros([cut_off_epoch, dim[1]], dtype = torch.double)
    
    num = 0
    
    for t in range(max_epoch):
        
        end = False
        
        for i in range(0, dim[0], batch_size):
        
#             batch_id = i%(batch_num)

            if t*dim[0] + i + batch_size < b_dim[0]:
                
                
                curr_X_Y_prod = X_Y_prod[i:i + batch_size]

                curr_b_seq = b_seq[t*dim[0] + i: t*dim[0] + i+curr_X_Y_prod.shape[0]]
                
                
                curr_X_Y_prod = curr_X_Y_prod.view(curr_X_Y_prod.shape[0], dim[1])
                
                res[num] = torch.mm(torch.t(curr_X_Y_prod), curr_b_seq).view(dim[1])
                
            else:
                
                curr_X_Y_prod = X_Y_prod[i:i + batch_size]
                
                curr_b_seq = b_seq[t*dim[0] + i: b_dim[0]]
                
                
                print(t*dim[0] + i, b_dim[0])
                
                curr_X_Y_prod = curr_X_Y_prod.view(curr_X_Y_prod.shape[0], dim[1])
                
                res[num] = torch.mm(torch.t(curr_X_Y_prod), curr_b_seq).view(dim[1])
                
                end = True
                
                break
    
            num = num + 1
            
            
            if num >= cut_off_epoch:
                end = True
                
                break
            
            
        if end:
            break

#     for i in range(b_dim[1]):
#         batch_id = int(i%batch_num)
#         
#         res[i] = torch.t(torch.mm(torch.t(X_Y_prod[batch_id*batch_size:(batch_id + 1)*batch_size]), b_seq[:,i])) 

    
#     res = torch.t(torch.mm(torch.t(X_Y_prod), b_seq))
    
    
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



def prepare_term_2_batch2_delta(delta_X_Y_prod, b_seq, dim, batch_num, batch_size, mini_batch_epoch, max_epoch, delta_data_ids, cut_off_epoch):
    
    
    '''batch_size*cut_off_epoch = batch_size*(t*(n/batch_size))'''
    
    b_dim = b_seq.shape
    
#     print('b_dim::', b_dim)
    
#     b_seq_transpose = torch.t(b_seq)
# #     X_Y_prod = X.mul(Y)
# #     dim[0]*dim[1]
# #     X_Y_prod = X_Y_prod.repeat(b_dim[1], 1)
# 
#     batch_num = int(dim[0]/batch_size)
#     
#     res = torch.zeros(b_dim[0], dim[1], dtype = torch.double)
    
#     print(X_Y_prod.shape)
#     
#     print(b_seq.shape)


#     batch_num = int(dim[0]/batch_size)


#     res = torch.zeros([b_dim[1], dim[1]])
    
    res = torch.zeros([mini_batch_epoch, dim[1]], dtype = torch.double)

    
    
    num = 0
    
    for t in range(max_epoch):
        
        end = False
        
        for i in range(0, dim[0], batch_size):
        
#             batch_id = i%(batch_num)


            end_id = i + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]


            satisfiable_delta_ids = ((i*batch_size<= delta_data_ids)*(delta_data_ids < end_id)).view(-1)


            curr_delta_ids = torch.nonzero(satisfiable_delta_ids)
            
            
            if curr_delta_ids.shape[0] > 0:
            
                curr_origin_delta_ids = delta_data_ids[satisfiable_delta_ids]%batch_size + t*dim[0] + i
    
                if t*dim[0] + i + batch_size < b_dim[0]:
                    
                    
                    curr_X_Y_prod = delta_X_Y_prod[curr_delta_ids]
    
                    curr_b_seq = b_seq[curr_origin_delta_ids]
                    
                    
                    curr_X_Y_prod = curr_X_Y_prod.view(curr_X_Y_prod.shape[0], dim[1])
                    
                    res[num] = torch.mm(torch.t(curr_X_Y_prod), curr_b_seq).view(dim[1])
                    
                else:
                    
                    curr_X_Y_prod = delta_X_Y_prod[curr_delta_ids]
                    
                    curr_b_seq = b_seq[curr_origin_delta_ids[curr_origin_delta_ids < b_dim[0]]]
                    
                    
                    print(t*dim[0] + i, b_dim[0])
                    
                    curr_X_Y_prod = curr_X_Y_prod.view(curr_X_Y_prod.shape[0], dim[1])
                    
                    res[num] = torch.mm(torch.t(curr_X_Y_prod), curr_b_seq).view(dim[1])
                    
                    end = True
                    
                    break
    
            num = num + 1
    
            if num >= cut_off_epoch:
                end  = True
                break
            
        if end:
            break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

#     for i in range(b_dim[1]):
#         
#         batch_id = i%(batch_num)
#         
#         
#         satisfiable_delta_ids = (batch_id*batch_size<= delta_data_ids)*(delta_data_ids < (batch_id + 1)*batch_size)
#         
#         curr_delta_ids = torch.nonzero(satisfiable_delta_ids)
#         
#         
#         curr_origin_delta_ids = delta_data_ids[satisfiable_delta_ids]%batch_size
#         
#         
#         if curr_delta_ids.shape[0] > 0:
#             curr_X_Y_prod = delta_X_Y_prod[curr_delta_ids].view(curr_delta_ids, dim[1])
#         
#             res[i] = torch.mm(torch.t(curr_X_Y_prod), b_seq[curr_origin_delta_ids,i]).view(dim[1])
        
        
        
#         res[i] = torch.t(torch.mm(torch.t(X_Y_prod[batch_id*batch_size:(batch_id + 1)*batch_size]), b_seq[:,i])) 

    
#     res = torch.t(torch.mm(torch.t(X_Y_prod), b_seq))
    
    
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

    res = torch.zeros(b_dim[0], dim[1], dtype = torch.double)

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
    
    
if __name__ == '__main__':
    
    
    delta_size = 10000
    
    repetition = 1
    
    # alpha = 0.00001
    #       
    # beta = 0.5
    [X, Y] = load_data(True)
    
    dim = X.shape
    
    
    res = compute_parameters(X, Y, initialize(X), dim)
    
    
    w_seq, b_seq = compute_linear_approx_parameters(X, Y, dim, res)
    
    # print('w_seq::', w_seq)
    # 
    # print('b_seq::', b_seq)
    
    t_01 = time.time()
    
    
    X_Y_mult = X.mul(Y)
    
    X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
    
    # term1, term1_inter_result = prepare_term_1_serial(X, w_seq, dim)
    term1 = prepare_term_1_batch2(X_product, w_seq, dim)
    
    # term2, term2_inter_result = prepare_term_2_serial(X, Y, b_seq, dim)
    term2 = prepare_term_2_batch2(X_Y_mult, b_seq, dim)
    
    t_02 = time.time()
    
    # dim = X.shape
    
    
    delta_data_ids = random_generate_subset_ids(dim, delta_size)
    
    
    
    
    # selected_rows = torch.tensor(list(set(range(dim[0])) - delta_data_ids))
    
    update_X, selected_rows = get_subset_training_data(X, dim, delta_data_ids)
    
    update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
    print(X.shape)
    
    print(update_X.shape)
    
    t1 = time.time()
    
    res1 = torch.zeros(res.shape)
    
    for i in range(repetition):
        
        lr = initialize(update_X)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
        res1 = logistic_regression_by_standard_library(update_X, update_Y, lr, dim)
    
    t2 = time.time()
    
    t3 = time.time()
    
    total_time = 0
    
    res2 = torch.zeros(res.shape)
    # 
    for i in range(repetition):
        init_theta = Variable(initialize(update_X).theta)
        
        update_X_Y_mult = update_X.mul(update_Y)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
        res2, total_time = compute_model_parameter_by_iteration(dim, init_theta, update_X_Y_mult)
    
    
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
        print('w_seq_shape::', w_seq.shape)
         
        print(b_seq.shape)
        
        
        if len(delta_data_ids) < (dim[0])/2:
            
            sub_weights = torch.index_select(w_seq, 1, delta_data_ids)#get_subset_parameter_list(selected_rows, delta_data_ids, w_seq, dim, 1)
             
            sub_offsets = torch.index_select(b_seq, 1, delta_data_ids)#get_subset_parameter_list(b_seq, delta_data_ids, b_seq, dim, 1)
             
        #     delta_X = torch.index_select(X, 0, delta_data_ids)
        #      
        #     delta_Y = torch.index_select(Y, 0, delta_data_ids)
        #     sub_X_Y_mult = get_subset_parameter_list(selected_rows, delta_data_ids, X_Y_mult, dim, 0)
             
            delta_X_product = torch.index_select(X_product, 0, delta_data_ids)
              
            delta_X_Y_mult = torch.index_select(X_Y_mult, 0, delta_data_ids)
             
        #     sub_term_1 = prepare_term_1_serial(delta_X, sub_weights, delta_X.shape)#(delta_X_product, sub_weights, delta_X.shape)
        #      
        #     sub_term_2 = prepare_term_2_serial(delta_X, delta_Y, sub_offsets, delta_X.shape)#(delta_X_Y_mult, sub_offsets, delta_X.shape)
            
        #     update_X_dim = update_X.dim
            
            curr_delta_dim = [X.shape[0] - update_X.shape[0], X.shape[1]]
            
            s_1 = time.time()
            
            sub_term_1 = prepare_term_1_batch2(delta_X_product, sub_weights, curr_delta_dim)
             
            sub_term_2 = prepare_term_2_batch2(delta_X_Y_mult, sub_offsets, curr_delta_dim)     
             
             
            s_2 = time.time()
            
            step_time_1 += s_2  -s_1
        #     sub_term_2 = prepare_term_2_batch(delta_X_Y_mult, sub_offsets, delta_X.shape)
        
        #     sub_term_1 = prepare_term_1(term1_inter_result, delta_data_ids)
        #     
        #     sub_term_2 = prepare_term_2(term2_inter_result, delta_data_ids)
            
            init_theta = Variable(initialize(update_X).theta)
            
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
            
            res3 = compute_model_parameter_by_approx_incremental_2(term1 - sub_term_1, term2 - sub_term_2, dim, init_theta)
            
            s_4 = time.time()
            
            step_time_2 += s_4  -s_3
        
        else:
            sub_weights = torch.index_select(w_seq, 1, selected_rows)#get_subset_parameter_list(selected_rows, delta_data_ids, w_seq, dim, 1)
             
            sub_offsets = torch.index_select(b_seq, 1, selected_rows)#get_subset_parameter_list(b_seq, delta_data_ids, b_seq, dim, 1)
             
        #     delta_X = torch.index_select(X, 0, delta_data_ids)
        #      
        #     delta_Y = torch.index_select(Y, 0, delta_data_ids)
        #     sub_X_Y_mult = get_subset_parameter_list(selected_rows, delta_data_ids, X_Y_mult, dim, 0)
             
            delta_X_product = torch.index_select(X_product, 0, selected_rows)
              
            delta_X_Y_mult = torch.index_select(X_Y_mult, 0, selected_rows)
             
        #     sub_term_1 = prepare_term_1_serial(delta_X, sub_weights, delta_X.shape)#(delta_X_product, sub_weights, delta_X.shape)
        #      
        #     sub_term_2 = prepare_term_2_serial(delta_X, delta_Y, sub_offsets, delta_X.shape)#(delta_X_Y_mult, sub_offsets, delta_X.shape)
            
        #     update_X_dim = update_X.dim
            
            curr_delta_dim = [update_X.shape[0], X.shape[1]]
            
            s_1 = time.time()
            
            sub_term_1 = prepare_term_1_batch2(delta_X_product, sub_weights, curr_delta_dim)
             
            sub_term_2 = prepare_term_2_batch2(delta_X_Y_mult, sub_offsets, curr_delta_dim)     
             
             
            s_2 = time.time()
            
            step_time_1 += s_2  -s_1
        #     sub_term_2 = prepare_term_2_batch(delta_X_Y_mult, sub_offsets, delta_X.shape)
        
        #     sub_term_1 = prepare_term_1(term1_inter_result, delta_data_ids)
        #     
        #     sub_term_2 = prepare_term_2(term2_inter_result, delta_data_ids)
            
            init_theta = Variable(initialize(update_X).theta)
            
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
    
    compute_sub_term_time = t_02 - t_01
    
    
    print('res::', res)
    
    print('res1::', res1)
    
    print('res2::', res2)
    
    print('res3::', res3)
    
    print('delta::', res - res1)
    
    diff = res2 - res1
    
    print(diff)
    
    print(res3 - res2)
    
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















