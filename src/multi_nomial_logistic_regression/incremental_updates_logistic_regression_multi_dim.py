'''
Created on Feb 4, 2019


'''
from torch import nn, optim
import torch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')


try:
    from data_IO.Load_data import *
    from sensitivity_analysis_SGD.multi_nomial_logistic_regression.Multi_logistic_regression import *
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



def prepare_term_1_mini_batch(X, weights, dim, num_class, batch_size):
     
    '''weights: dim[0], max_epoch, num_class, num_class''' 
    
    w_dim = weights.shape
    
    batch_num = int(X.shape[0]/batch_size)
    
    term_1 = Variable(torch.zeros([w_dim[1], num_class*dim[1], dim[1]*num_class], dtype = torch.double))
    
    print(batch_num)
     
    for i in range(batch_num):
         
         
        X_product_subset = Variable(torch.bmm(X[i*batch_size:(i+1)*batch_size].view(batch_size, dim[1], 1), X[i*batch_size:(i+1)*batch_size].view(batch_size, 1, dim[1])))
         
         
        X_product1 = torch.t(X_product_subset.view(batch_size, dim[1]*dim[1]))
     
        del X_product_subset
         
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
         
        res = torch.reshape(res4, [w_dim[1], num_class*dim[1], dim[1]*num_class])
         
        term_1 += res 
        
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
    
    del X_product
    
    
    t1 = time.time()
    
    res1 = torch.mm(X_product1, weights.view(dim[0], w_dim[1]*num_class*num_class))
    
    t2 = time.time()
    
    del X_product1
    
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



def prepare_term_1_batch2_0_delta(alpha, beta, X, weights, offsets, dim, max_epoch, num_class, cut_off_epoch, batch_size, delta_data_ids, term1, term2, x_sum_by_class_list, delta_x_sum_by_class_list):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    
    
    
    w_dim = weights.shape
    
    print(w_dim)
    
    print(dim)
    
    batch_num_per_epoch = int((X.shape[0] - 1)/batch_size) + 1
    
#     torch.tensor(list(range(max_epoch)))*batch_num_per_epoch

#     total_mini_epochs = batch_num_per_epoch * max_epoch
    
    
    delta_term1 = term1.clone()#torch.zeros([cut_off_epoch, num_class*dim[1], dim[1]*num_class], dtype = torch.double)
    
    delta_term2 = term2.clone()#torch.zeros([cut_off_epoch, num_class*dim[1]], dtype = torch.double)
    
    A = Variable(torch.zeros([int((cut_off_epoch)/batch_num_per_epoch), num_class*dim[1], dim[1]*num_class], dtype = torch.double))
    
    B = Variable(torch.zeros([int((cut_off_epoch)/batch_num_per_epoch), dim[1]*num_class, 1], dtype = torch.double))
    
    
    for i in range(0, X.shape[0], batch_size):
        
        end_id = i + batch_size
        
        if end_id > X.shape[0]:
            end_id = X.shape[0]
        
        
#         batch_x = X[i:end_id]
        
        
        satisfiable_delta_ids = ((i<= delta_data_ids)*(delta_data_ids < end_id)).view(-1)

        '''delta_data_id_ids'''
        curr_delta_ids = torch.nonzero(satisfiable_delta_ids)


        batch_x = X[delta_data_ids[curr_delta_ids.view(-1)].view(-1)]
#         batch_x = delta_X[curr_delta_ids].view(-1)
                    
#         curr_X_Y_prod = delta_X_Y_prod[curr_delta_ids.view(-1)]


        curr_batch_delta_id_offset = (delta_data_ids[curr_delta_ids]%batch_size).view(1,-1) 


        curr_origin_delta_ids = curr_batch_delta_id_offset + (torch.tensor(list(range(max_epoch)))*dim[0] + i).view(-1, 1)
        
        curr_origin_delta_ids = curr_origin_delta_ids.view(-1)
        
        ids = ((torch.tensor(list(range(max_epoch)))*batch_num_per_epoch + int(i/batch_size))).view(-1)
        
        if curr_delta_ids.shape[0] == 0:
            delta_term1[ids[ids < cut_off_epoch]] = (term1[ids[ids < cut_off_epoch]] - 0)/(end_id - i)
        
            delta_term2[ids[ids < cut_off_epoch]] = (term2[ids[ids < cut_off_epoch]] - torch.t(x_sum_by_class_list[int(i/batch_size)]))/(end_id - i)
            
            curr_A = Variable((1 - alpha*beta)*torch.eye(dim[1]*num_class, dtype = torch.double) - alpha*delta_term1[ids[ids<cut_off_epoch]])
                
            curr_B = Variable(-alpha*delta_term2[ids[ids < cut_off_epoch]])
            
            if i == 0:
                A[:] = curr_A[0:A.shape[0]]
                B[:] = curr_B[0:B.shape[0]].view(-1, dim[1]*num_class, 1)
                
            else:
                
                
    #             if ids[ids < cut_off_epoch].shape[0] == A.shape[0]:
    #                 curr_A = (1 - alpha*beta)*torch.eye(dim[1]) + alpha*res[ids[ids<cut_off_epoch]]
    #                 
    #                 curr_B = alpha*res2
                A = torch.bmm(A, curr_A[0:A.shape[0]])
                
                B = torch.bmm(curr_A[0:A.shape[0]], B) + curr_B[0:B.shape[0]].view(-1, dim[1]*num_class, 1)

#             print('batch::', i)
#             print('theta::', torch.t((torch.mm(A[1], theta.view(-1,1)) + B[1]).view(num_class, dim[1])))

            
            continue
        
        
#         curr_w_seq = torch.t(w_seq[curr_origin_delta_ids[curr_origin_delta_ids<w_seq.shape[0]]].view(-1, curr_batch_delta_id_offset.view(-1).shape[0]))
#                     
#         curr_b_seq = torch.t(b_seq[curr_origin_delta_ids[curr_origin_delta_ids<b_seq.shape[0]]].view(-1, curr_batch_delta_id_offset.view(-1).shape[0]))

        
        
        
        
        
        curr_seq_ids = torch.tensor(list(range(i, end_id))).view(-1, 1) + (torch.tensor(list(range(max_epoch)))*X.shape[0]).view(1,-1)
        
        curr_seq_ids = curr_seq_ids.view(-1)
        
        curr_seq_ids = curr_seq_ids[curr_seq_ids < weights.shape[0]]
         
        curr_batch_size = batch_x.shape[0]
        
        
        
        '''(batch_size*t)*q^2'''
        curr_weights = weights[curr_origin_delta_ids[curr_origin_delta_ids<weights.shape[0]]].view(-1, curr_batch_delta_id_offset.view(-1).shape[0], num_class*num_class)
        
        curr_weights = torch.transpose(curr_weights, 0, 1)
        
        curr_offsets = offsets[curr_origin_delta_ids[curr_origin_delta_ids<weights.shape[0]]].view(-1, curr_batch_delta_id_offset.view(-1).shape[0], num_class)
        
        curr_offsets = torch.transpose(curr_offsets, 0, 1)
        
#         print(curr_batch_size, curr_offsets.shape[1], num_class, dim[1])
#         
#         print(curr_weights.shape)
#         
#         print(batch_x.shape)
        
        curr_term2 = torch.reshape(torch.t(torch.mm(torch.t(batch_x), torch.reshape(curr_offsets, (curr_batch_size, curr_offsets.shape[1]*num_class)))), [curr_offsets.shape[1], num_class*dim[1]])
        
#         expect_term2 = torch.reshape(torch.t(torch.mm(torch.t(X[i:end_id]), torch.reshape(offsets[i:end_id], (end_id - i, num_class)))), [1, num_class*dim[1]])
        
        
        
        
        res1 = torch.bmm(batch_x.view(curr_batch_size, dim[1], 1), torch.reshape(curr_weights, (curr_batch_size, 1, curr_weights.shape[1]*num_class*num_class)))
    
        res2 = torch.mm(torch.t(batch_x), res1.view(curr_batch_size, dim[1]*curr_weights.shape[1]*num_class*num_class)).view(dim[1]*dim[1], curr_weights.shape[1], num_class*num_class)
        
        del res1
        
        res3 = torch.transpose(torch.transpose(res2, 0, 1), 2, 1).view(curr_weights.shape[1], num_class, num_class, dim[1], dim[1])
    
        del res2
    
        res4 = torch.reshape(torch.transpose(res3, 2, 3), [curr_weights.shape[1], num_class*dim[1], dim[1]*num_class])
        
        del res3
        
#         curr_res1 = torch.bmm(X[i:end_id].view(end_id - i, dim[1], 1), weights[i:end_id].view(end_id - i, 1, 1*num_class*num_class))
#     
#         curr_res2 = torch.mm(torch.t(X[i:end_id]), curr_res1.view(end_id - i, dim[1]*1*num_class*num_class)).view(dim[1]*dim[1], 1, num_class*num_class)
#         
#         curr_res3 = torch.transpose(torch.transpose(curr_res2, 0, 1), 2, 1).view(1, num_class, num_class, dim[1], dim[1])
#     
#         curr_res4 = torch.reshape(torch.transpose(curr_res3, 2, 3), [1, num_class*dim[1], dim[1]*num_class])
        
        
        
        
        delta_term1[ids[ids < cut_off_epoch]] = (term1[ids[ids < cut_off_epoch]] - res4)/(end_id - i - curr_delta_ids.shape[0])
        
#         print(term2[ids[ids < cut_off_epoch]].shape)
#          
#         print(curr_term2.shape)
#          
#         print(x_sum_by_class_list[int(i/batch_size)].shape)
#         
#         print(delta_x_sum_by_class_list[int(i/batch_size)].shape)
#          
#         print(ids[ids < cut_off_epoch])
#          
#         print(int(i/batch_size))
#          
#         print(delta_x_sum_by_class_list.keys())
        
        delta_term2[ids[ids < cut_off_epoch]] = (term2[ids[ids < cut_off_epoch]] - curr_term2 - torch.t(x_sum_by_class_list[int(i/batch_size)] - delta_x_sum_by_class_list[int(i/batch_size)]))/(end_id - i - curr_delta_ids.shape[0])
        
        
        curr_A = Variable((1 - alpha*beta)*torch.eye(dim[1]*num_class, dtype = torch.double) - alpha*delta_term1[ids[ids<cut_off_epoch]])
                
        curr_B = Variable(-alpha*delta_term2[ids[ids < cut_off_epoch]])
        
        if i == 0:
            A[:] = curr_A[0:A.shape[0]]
            B[:] = curr_B[0:B.shape[0]].view(-1, dim[1]*num_class, 1)
        else:
            
            A = torch.bmm(A, curr_A[0:A.shape[0]])
            
            B = torch.bmm(curr_A[0:A.shape[0]], B) + curr_B[0:B.shape[0]].view(-1, dim[1]*num_class, 1)
            
            
#         print('batch::', i)
#         print('theta::', torch.t((torch.mm(A[0], theta.view(-1,1)) + B[0]).view(num_class, dim[1])))

    
    return delta_term1, delta_term2, A, B

# def prepare_term_1(random_ids_multi_super_iterations, alpha, beta, X, weights, offsets, dim, max_epoch, num_class, cut_off_epoch, batch_size, term1, term2, x_sum_by_class_list, delta_x_sum_by_class_list):
#     
#     '''weights: dim[0]*max_epoch, num_class, num_class'''
#     
#     
#     
#     w_dim = weights.shape
#     
#     print(w_dim)
#     
#     print(dim)
#     
#     batch_num_per_epoch = int((X.shape[0] - 1)/batch_size) + 1
#     
#     cut_off_super_iteration = int((weights.shape[0]-1)/dim[0])+1
#     
#     print('cut_off_super_iterations::', cut_off_super_iteration)
#     
# 
#     min_batch_num_per_epoch = int((dim[0] - 1)/batch_size) + 1
# 
#     '''T,n'''
#     cut_off_random_ids_multi_super_iterations = random_ids_multi_super_iterations[0:cut_off_super_iteration]
#     
#     
# 
#     
#     '''T*n,|delta_X|'''
# 
#     matched_ids = (cut_off_random_ids_multi_super_iterations.view(-1,1) == delta_data_ids.view(1,-1))
#     
#     '''T, n, |delta_X|'''
#     
#     matched_ids = matched_ids.view(cut_off_super_iteration, dim[0], delta_data_ids.shape[0])
#         
#         
#     '''n, T, |delta_X|'''
#     matched_ids = torch.transpose(matched_ids, 1, 0)
#     
#     '''ids of [n, T, delta_X]'''
#     total_time = 0
#     
#     t1 = time.time()
#     
#     nonzero_ids = torch.nonzero(matched_ids)
#     
#     
#     all_noise_data_ids = delta_data_ids.view(1,-1) + (torch.tensor(list(range(cut_off_super_iteration)))*dim[0]).view(-1, 1)
#      
#      
#     all_noise_data_ids = all_noise_data_ids.view(-1)
#     
#     '''T, |delta_X|, q^2'''
#     curr_weights = weights.view(-1, dim[0], num_class*num_class)
#     
#     '''T, |delta_X|, q'''    
#     curr_offsets = offsets.view(-1, dim[0], num_class)
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
# #     torch.tensor(list(range(max_epoch)))*batch_num_per_epoch
# 
# #     total_mini_epochs = batch_num_per_epoch * max_epoch
#     
#     
#     delta_term1 = term1.clone()#torch.zeros([cut_off_epoch, num_class*dim[1], dim[1]*num_class], dtype = torch.double)
#     
#     delta_term2 = term2.clone()#torch.zeros([cut_off_epoch, num_class*dim[1]], dtype = torch.double)
#     
#     A = Variable(torch.zeros([int((cut_off_epoch)/batch_num_per_epoch), num_class*dim[1], dim[1]*num_class], dtype = torch.double))
#     
#     B = Variable(torch.zeros([int((cut_off_epoch)/batch_num_per_epoch), dim[1]*num_class, 1], dtype = torch.double))
#     
#     offset_mini_epochs = cut_off_epoch%min_batch_num_per_epoch
#     
#     if offset_mini_epochs == 0:
#         offset_mini_epochs = min_batch_num_per_epoch
#     
#     
#     base_data_ids = torch.tensor(list(range(cut_off_super_iteration)))*min_batch_num_per_epoch
#     
#     delta_X = X[delta_data_ids]
# 
#     for T in range(max_epoch):
#         
#         curr_rand_ids = random_ids_multi_super_iterations[T]
#         
#         for i in range(0, X.shape[0], batch_size):
#             end_id = i + batch_size
#         
#             if end_id > X.shape[0]:
#                 end_id = X.shape[0]
#                 
#             
#             
#             
#             if T*dim[0] + i+ batch_size < w_dim[0]:
# 
# 
#                 curr_X_prod = X_product[curr_rand_ids[i:i + batch_size]]
#                 
# 
#                 curr_w_seq = w_seq[t*dim[0] + i: t*dim[0] + i + curr_X_prod.shape[0]]
#                 
#             batch_X = X[curr_rand_ids[i:end_id]]
#             
#             res1 = torch.bmm(batch_X.view(batch_X.shape[0], dim[1], 1), torch.reshape(curr_weights, (delta_data_ids.shape[0], 1, curr_weights.shape[1]*num_class*num_class)))
#     
#             res2 = torch.mm(torch.t(delta_X), res1.view(delta_data_ids.shape[0], dim[1]*curr_weights.shape[1]*num_class*num_class)).view(dim[1]*dim[1], curr_weights.shape[1], num_class*num_class)
#             
#             del res1
#             
#             res3 = torch.transpose(torch.transpose(res2, 0, 1), 2, 1).view(curr_weights.shape[1], num_class, num_class, dim[1], dim[1])
#         
#             del res2
#         
#             res4 = torch.reshape(torch.transpose(res3, 2, 3), [curr_weights.shape[1], num_class*dim[1], dim[1]*num_class])
#             
#             del res3
#             
#     
#     for i in range(0, X.shape[0], batch_size):
#         
#         end_id = i + batch_size
#         
#         if end_id > X.shape[0]:
#             end_id = X.shape[0]
#         
#         
#         
#         curr_nonzero_ids = torch.nonzero(((nonzero_ids[:, 0] >= i)*(nonzero_ids[:, 0] < end_id))).view(-1)
#         
#         curr_nonzero_ids_this_batch = nonzero_ids[curr_nonzero_ids][:, 1:3]
#         
#         curr_nonzero_ids_count_this_batch = (curr_nonzero_ids_this_batch[:,0].view(-1,1) == torch.tensor(list(range(cut_off_super_iteration))).view(1,-1))
#         
#         '''count per super_iterations'''
#         
#         total_curr_matched_ids = torch.sum(curr_nonzero_ids_count_this_batch, 0)
#         
#         total_curr_matched_ids = total_curr_matched_ids.type(torch.DoubleTensor)
#         
#         ids = (base_data_ids + int(i/batch_size)).view(-1)
#         
#         curr_weights_this_batch = curr_weights.clone()
#         
#         '''T, |delta_X|, q*q'''
#         
#         curr_weights_this_batch[curr_nonzero_ids_this_batch[:,0], curr_nonzero_ids_this_batch[:, 1]] = torch.zeros([num_class*num_class], dtype = torch.double)
#         
#         curr_offsets_this_batch = curr_offsets.clone()
# 
#         '''T, |delta_X|, q'''
#         
#         curr_offsets_this_batch[curr_nonzero_ids_this_batch[:,0], curr_nonzero_ids_this_batch[:, 1]] = torch.zeros(num_class, dtype = torch.double)
#         
# #         batch_x = X[i:end_id]
#         
#         
# #         satisfiable_delta_ids = ((i<= delta_data_ids)*(delta_data_ids < end_id)).view(-1)
# # 
# #         '''delta_data_id_ids'''
# #         curr_delta_ids = torch.nonzero(satisfiable_delta_ids)
# 
# 
# #         batch_x = X[delta_data_ids[curr_delta_ids.view(-1)].view(-1)]
# #         batch_x = delta_X[curr_delta_ids].view(-1)
#                     
# #         curr_X_Y_prod = delta_X_Y_prod[curr_delta_ids.view(-1)]
# 
# 
# #         curr_batch_delta_id_offset = (delta_data_ids[curr_delta_ids]%batch_size).view(1,-1) 
# # 
# # 
# #         curr_origin_delta_ids = curr_batch_delta_id_offset + (torch.tensor(list(range(max_epoch)))*dim[0] + i).view(-1, 1)
# #         
# #         curr_origin_delta_ids = curr_origin_delta_ids.view(-1)
# #         
# #         ids = ((torch.tensor(list(range(max_epoch)))*batch_num_per_epoch + int(i/batch_size))).view(-1)
# #         
# #         curr_seq_ids = torch.tensor(list(range(i, end_id))).view(-1, 1) + (torch.tensor(list(range(max_epoch)))*X.shape[0]).view(1,-1)
# #         
# #         curr_seq_ids = curr_seq_ids.view(-1)
# #         
# #         curr_seq_ids = curr_seq_ids[curr_seq_ids < weights.shape[0]]
#          
# #         curr_batch_size = batch_x.shape[0]
#         
#         
#         
#         '''(batch_size*t)*q^2'''
# #         curr_weights = weights[curr_origin_delta_ids[curr_origin_delta_ids<weights.shape[0]]].view(-1, curr_batch_delta_id_offset.view(-1).shape[0], num_class*num_class)
# #         
# #         curr_weights = torch.transpose(curr_weights, 0, 1)
# #         
# #         curr_offsets = offsets[curr_origin_delta_ids[curr_origin_delta_ids<weights.shape[0]]].view(-1, curr_batch_delta_id_offset.view(-1).shape[0], num_class)
# #         
# #         curr_offsets = torch.transpose(curr_offsets, 0, 1)
#         
# #         print(curr_batch_size, curr_offsets.shape[1], num_class, dim[1])
# #         
# #         print(curr_weights.shape)
# #         
# #         print(batch_x.shape)
#         
#         curr_term2 = torch.reshape(torch.t(torch.mm(torch.t(delta_X), torch.reshape(curr_offsets, (delta_data_ids.shape[0], curr_offsets.shape[1]*num_class)))), [curr_offsets.shape[1], num_class*dim[1]])
#         
# #         expect_term2 = torch.reshape(torch.t(torch.mm(torch.t(X[i:end_id]), torch.reshape(offsets[i:end_id], (end_id - i, num_class)))), [1, num_class*dim[1]])
#         
#         
#         
#         
#         res1 = torch.bmm(delta_X.view(delta_data_ids.shape[0], dim[1], 1), torch.reshape(curr_weights, (delta_data_ids.shape[0], 1, curr_weights.shape[1]*num_class*num_class)))
#     
#         res2 = torch.mm(torch.t(delta_X), res1.view(delta_data_ids.shape[0], dim[1]*curr_weights.shape[1]*num_class*num_class)).view(dim[1]*dim[1], curr_weights.shape[1], num_class*num_class)
#         
#         del res1
#         
#         res3 = torch.transpose(torch.transpose(res2, 0, 1), 2, 1).view(curr_weights.shape[1], num_class, num_class, dim[1], dim[1])
#     
#         del res2
#     
#         res4 = torch.reshape(torch.transpose(res3, 2, 3), [curr_weights.shape[1], num_class*dim[1], dim[1]*num_class])
#         
#         del res3
#         
# #         curr_res1 = torch.bmm(X[i:end_id].view(end_id - i, dim[1], 1), weights[i:end_id].view(end_id - i, 1, 1*num_class*num_class))
# #     
# #         curr_res2 = torch.mm(torch.t(X[i:end_id]), curr_res1.view(end_id - i, dim[1]*1*num_class*num_class)).view(dim[1]*dim[1], 1, num_class*num_class)
# #         
# #         curr_res3 = torch.transpose(torch.transpose(curr_res2, 0, 1), 2, 1).view(1, num_class, num_class, dim[1], dim[1])
# #     
# #         curr_res4 = torch.reshape(torch.transpose(curr_res3, 2, 3), [1, num_class*dim[1], dim[1]*num_class])
#         
#         
#         
#         
#         delta_term1[ids[ids < cut_off_epoch]] = (term1[ids[ids < cut_off_epoch]] - res4)/(end_id - i - total_curr_matched_ids.view(-1, 1, 1))
#         
# #         print(term2[ids[ids < cut_off_epoch]].shape)
# #          
# #         print(curr_term2.shape)
# #          
# #         print(x_sum_by_class_list[int(i/batch_size)].shape)
# #         
# #         print(delta_x_sum_by_class_list[int(i/batch_size)].shape)
# #          
# #         print(ids[ids < cut_off_epoch])
# #          
# #         print(int(i/batch_size))
# #          
# #         print(delta_x_sum_by_class_list.keys())
#         
#         delta_term2[ids[ids < cut_off_epoch]] = (term2[ids[ids < cut_off_epoch]] - curr_term2 - torch.t(x_sum_by_class_list[int(i/batch_size)] - delta_x_sum_by_class_list[int(i/batch_size)]))/(end_id - i - total_curr_matched_ids.view(-1, 1))
#         
#         
#         curr_A = Variable((1 - alpha*beta)*torch.eye(dim[1]*num_class, dtype = torch.double) - alpha*delta_term1[ids[ids<cut_off_epoch]])
#                 
#         curr_B = Variable(-alpha*delta_term2[ids[ids < cut_off_epoch]])
#         
#         if i == 0:
#             A[:] = curr_A[0:A.shape[0]]
#             B[:] = curr_B[0:B.shape[0]].view(-1, dim[1]*num_class, 1)
#         else:
#             
#             A = torch.bmm(A, curr_A[0:A.shape[0]])
#             
#             B = torch.bmm(curr_A[0:A.shape[0]], B) + curr_B[0:B.shape[0]].view(-1, dim[1]*num_class, 1)
#             
#             
# #         print('batch::', i)
# #         print('theta::', torch.t((torch.mm(A[0], theta.view(-1,1)) + B[0]).view(num_class, dim[1])))
# 
#     
#     return delta_term1, delta_term2, A, B


def prepare_term_1_batch2_1_delta(random_ids_multi_super_iterations, alpha, beta, X, weights, offsets, dim, max_epoch, num_class, cut_off_epoch, batch_size, delta_data_ids, term1, term2, x_sum_by_class_list, delta_x_sum_by_class_list):
    
    '''weights: dim[0]*max_epoch, num_class, num_class'''
    
    
    
    w_dim = weights.shape
    
    print(w_dim)
    
    print(dim)
    
    batch_num_per_epoch = int((X.shape[0] - 1)/batch_size) + 1
    
    cut_off_super_iteration = int((weights.shape[0]-1)/dim[0])+1
    
    print('cut_off_super_iterations::', cut_off_super_iteration)
    

    min_batch_num_per_epoch = int((dim[0] - 1)/batch_size) + 1

    '''T,n'''
    cut_off_random_ids_multi_super_iterations = random_ids_multi_super_iterations[0:cut_off_super_iteration]
    
    
    
    
    '''T*n,|delta_X|'''

    
    matched_ids = (cut_off_random_ids_multi_super_iterations.view(-1,1) == delta_data_ids.view(1,-1))
    
    '''T, n, |delta_X|'''
    
    matched_ids = matched_ids.view(cut_off_super_iteration, dim[0], delta_data_ids.shape[0])
        
        
    '''n, T, |delta_X|'''
    matched_ids = torch.transpose(matched_ids, 1, 0)
    
    '''ids of [n, T, delta_X]'''
    total_time = 0
    
    t1 = time.time()
    
    nonzero_ids = torch.nonzero(matched_ids)
    
    
    all_noise_data_ids = delta_data_ids.view(1,-1) + (torch.tensor(list(range(cut_off_super_iteration)))*dim[0]).view(-1, 1)
     
     
    all_noise_data_ids = all_noise_data_ids.view(-1)
    
    '''T, |delta_X|, q^2'''
    curr_weights = weights[all_noise_data_ids].view(-1, delta_data_ids.view(-1).shape[0], num_class*num_class)
    
    '''T, |delta_X|, q'''    
    curr_offsets = offsets[all_noise_data_ids].view(-1, delta_data_ids.view(-1).shape[0], num_class)
    
    
    
    
    
    
    
    
    
    
    
    
#     torch.tensor(list(range(max_epoch)))*batch_num_per_epoch

#     total_mini_epochs = batch_num_per_epoch * max_epoch
    
    
    delta_term1 = term1.clone()#torch.zeros([cut_off_epoch, num_class*dim[1], dim[1]*num_class], dtype = torch.double)
    
    delta_term2 = term2.clone()#torch.zeros([cut_off_epoch, num_class*dim[1]], dtype = torch.double)
    
    A = Variable(torch.zeros([int((cut_off_epoch)/batch_num_per_epoch), num_class*dim[1], dim[1]*num_class], dtype = torch.double))
    
    B = Variable(torch.zeros([int((cut_off_epoch)/batch_num_per_epoch), dim[1]*num_class, 1], dtype = torch.double))
    
    offset_mini_epochs = cut_off_epoch%min_batch_num_per_epoch
    
    if offset_mini_epochs == 0:
        offset_mini_epochs = min_batch_num_per_epoch
    
    
    base_data_ids = torch.tensor(list(range(cut_off_super_iteration)))*min_batch_num_per_epoch
    
    delta_X = X[delta_data_ids]
    
    for i in range(0, X.shape[0], batch_size):
        
        end_id = i + batch_size
        
        if end_id > X.shape[0]:
            end_id = X.shape[0]
        
        
        
        curr_nonzero_ids = torch.nonzero(((nonzero_ids[:, 0] >= i)*(nonzero_ids[:, 0] < end_id))).view(-1)
        
        curr_nonzero_ids_this_batch = nonzero_ids[curr_nonzero_ids][:, 1:3]
        
        curr_nonzero_ids_count_this_batch = (curr_nonzero_ids_this_batch[:,0].view(-1,1) == torch.tensor(list(range(cut_off_super_iteration))).view(1,-1))
        
        '''count per super_iterations'''
        
        total_curr_matched_ids = torch.sum(curr_nonzero_ids_count_this_batch, 0)
        
        total_curr_matched_ids = total_curr_matched_ids.type(torch.DoubleTensor)
        
        ids = (base_data_ids + int(i/batch_size)).view(-1)
        
        curr_weights_this_batch = curr_weights.clone()
        
        '''T, |delta_X|, q*q'''
        
        curr_weights_this_batch[curr_nonzero_ids_this_batch[:,0], curr_nonzero_ids_this_batch[:, 1]] = torch.zeros([num_class*num_class], dtype = torch.double)
        
        curr_offsets_this_batch = curr_offsets.clone()

        '''T, |delta_X|, q'''
        
        curr_offsets_this_batch[curr_nonzero_ids_this_batch[:,0], curr_nonzero_ids_this_batch[:, 1]] = torch.zeros(num_class, dtype = torch.double)
        
#         batch_x = X[i:end_id]
        
        
#         satisfiable_delta_ids = ((i<= delta_data_ids)*(delta_data_ids < end_id)).view(-1)
# 
#         '''delta_data_id_ids'''
#         curr_delta_ids = torch.nonzero(satisfiable_delta_ids)


#         batch_x = X[delta_data_ids[curr_delta_ids.view(-1)].view(-1)]
#         batch_x = delta_X[curr_delta_ids].view(-1)
                    
#         curr_X_Y_prod = delta_X_Y_prod[curr_delta_ids.view(-1)]


#         curr_batch_delta_id_offset = (delta_data_ids[curr_delta_ids]%batch_size).view(1,-1) 
# 
# 
#         curr_origin_delta_ids = curr_batch_delta_id_offset + (torch.tensor(list(range(max_epoch)))*dim[0] + i).view(-1, 1)
#         
#         curr_origin_delta_ids = curr_origin_delta_ids.view(-1)
#         
#         ids = ((torch.tensor(list(range(max_epoch)))*batch_num_per_epoch + int(i/batch_size))).view(-1)
#         
#         curr_seq_ids = torch.tensor(list(range(i, end_id))).view(-1, 1) + (torch.tensor(list(range(max_epoch)))*X.shape[0]).view(1,-1)
#         
#         curr_seq_ids = curr_seq_ids.view(-1)
#         
#         curr_seq_ids = curr_seq_ids[curr_seq_ids < weights.shape[0]]
         
#         curr_batch_size = batch_x.shape[0]
        
        
        
        '''(batch_size*t)*q^2'''
#         curr_weights = weights[curr_origin_delta_ids[curr_origin_delta_ids<weights.shape[0]]].view(-1, curr_batch_delta_id_offset.view(-1).shape[0], num_class*num_class)
#         
#         curr_weights = torch.transpose(curr_weights, 0, 1)
#         
#         curr_offsets = offsets[curr_origin_delta_ids[curr_origin_delta_ids<weights.shape[0]]].view(-1, curr_batch_delta_id_offset.view(-1).shape[0], num_class)
#         
#         curr_offsets = torch.transpose(curr_offsets, 0, 1)
        
#         print(curr_batch_size, curr_offsets.shape[1], num_class, dim[1])
#         
#         print(curr_weights.shape)
#         
#         print(batch_x.shape)
        
        curr_term2 = torch.reshape(torch.t(torch.mm(torch.t(delta_X), torch.reshape(curr_offsets, (delta_data_ids.shape[0], curr_offsets.shape[0]*num_class)))), [curr_offsets.shape[0], num_class*dim[1]])
        
#         expect_term2 = torch.reshape(torch.t(torch.mm(torch.t(X[i:end_id]), torch.reshape(offsets[i:end_id], (end_id - i, num_class)))), [1, num_class*dim[1]])
        
        
        
        
        res1 = torch.bmm(delta_X.view(delta_data_ids.shape[0], dim[1], 1), torch.reshape(curr_weights, (delta_data_ids.shape[0], 1, curr_weights.shape[0]*num_class*num_class)))
    
        res2 = torch.mm(torch.t(delta_X), res1.view(delta_data_ids.shape[0], dim[1]*curr_weights.shape[0]*num_class*num_class)).view(dim[1]*dim[1], curr_weights.shape[0], num_class*num_class)
        
        del res1
        
        res3 = torch.transpose(torch.transpose(res2, 0, 1), 2, 1).view(curr_weights.shape[0], num_class, num_class, dim[1], dim[1])
    
        del res2
    
        res4 = torch.reshape(torch.transpose(res3, 2, 3), [curr_weights.shape[0], num_class*dim[1], dim[1]*num_class])
        
        del res3
        
#         curr_res1 = torch.bmm(X[i:end_id].view(end_id - i, dim[1], 1), weights[i:end_id].view(end_id - i, 1, 1*num_class*num_class))
#     
#         curr_res2 = torch.mm(torch.t(X[i:end_id]), curr_res1.view(end_id - i, dim[1]*1*num_class*num_class)).view(dim[1]*dim[1], 1, num_class*num_class)
#         
#         curr_res3 = torch.transpose(torch.transpose(curr_res2, 0, 1), 2, 1).view(1, num_class, num_class, dim[1], dim[1])
#     
#         curr_res4 = torch.reshape(torch.transpose(curr_res3, 2, 3), [1, num_class*dim[1], dim[1]*num_class])
        
        
        print(res4.shape)
        
        print(term1[ids[ids < cut_off_epoch]].shape)
        
        print(ids[ids < cut_off_epoch].shape)
        
        print(ids[ids < cut_off_epoch])
        
        delta_term1[ids[ids < cut_off_epoch]] = (term1[ids[ids < cut_off_epoch]] - res4)/(end_id - i - total_curr_matched_ids.view(-1, 1, 1))
        
#         print(term2[ids[ids < cut_off_epoch]].shape)
#          
#         print(curr_term2.shape)
#          
#         print(x_sum_by_class_list[int(i/batch_size)].shape)
#         
#         print(delta_x_sum_by_class_list[int(i/batch_size)].shape)
#          
#         print(ids[ids < cut_off_epoch])
#          
#         print(int(i/batch_size))
#          
#         print(delta_x_sum_by_class_list.keys())
        
        delta_term2[ids[ids < cut_off_epoch]] = (term2[ids[ids < cut_off_epoch]] - curr_term2 - torch.t(x_sum_by_class_list[int(i/batch_size)] - delta_x_sum_by_class_list[int(i/batch_size)]))/(end_id - i - total_curr_matched_ids.view(-1, 1))
        
        
        curr_A = Variable((1 - alpha*beta)*torch.eye(dim[1]*num_class, dtype = torch.double) - alpha*delta_term1[ids[ids<cut_off_epoch]])
                
        curr_B = Variable(-alpha*delta_term2[ids[ids < cut_off_epoch]])
        
        if i == 0:
            A[:] = curr_A[0:A.shape[0]]
            B[:] = curr_B[0:B.shape[0]].view(-1, dim[1]*num_class, 1)
        else:
            
            A = torch.bmm(A, curr_A[0:A.shape[0]])
            
            B = torch.bmm(curr_A[0:A.shape[0]], B) + curr_B[0:B.shape[0]].view(-1, dim[1]*num_class, 1)
            
            
#         print('batch::', i)
#         print('theta::', torch.t((torch.mm(A[0], theta.view(-1,1)) + B[0]).view(num_class, dim[1])))

    
    return delta_term1, delta_term2, A, B


def prepare_term_1_batch2_0_delta_0(alpha, beta, theta, X, delta_X, weights, offsets, dim, max_epoch, num_class, cut_off_epoch, batch_size, delta_data_ids, term1, term2, x_sum_by_class_list, delta_x_sum_by_class_list):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    
    
    
    w_dim = weights.shape
    
    print(w_dim)
    
    print(dim)
    
    batch_num_per_epoch = int((X.shape[0] - 1)/batch_size) + 1
    
#     torch.tensor(list(range(max_epoch)))*batch_num_per_epoch

#     total_mini_epochs = batch_num_per_epoch * max_epoch
    
    
    delta_term1 = term1.clone()#torch.zeros([cut_off_epoch, num_class*dim[1], dim[1]*num_class], dtype = torch.double)
    
    delta_term2 = term2.clone()#torch.zeros([cut_off_epoch, num_class*dim[1]], dtype = torch.double)
    
    A = torch.zeros([int((cut_off_epoch)/batch_num_per_epoch), num_class*dim[1], dim[1]*num_class], dtype = torch.double)
    
    B = torch.zeros([int((cut_off_epoch)/batch_num_per_epoch), dim[1]*num_class, 1], dtype = torch.double)
    
    mini_epochs = 0
    
    theta = theta.view(-1,1) 
    
    for epoch in range(max_epoch):
        
        end_flag = False
    
        for i in range(0, X.shape[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            
            
    #         batch_x = X[i:end_id]
            
            
            satisfiable_delta_ids = ((i<= delta_data_ids)*(delta_data_ids < end_id)).view(-1)
    
            '''delta_data_id_ids'''
            curr_delta_ids = torch.nonzero(satisfiable_delta_ids)
    
    
            batch_x = X[delta_data_ids[curr_delta_ids.view(-1)].view(-1)]
    #         batch_x = delta_X[curr_delta_ids].view(-1)
                        
    #         curr_X_Y_prod = delta_X_Y_prod[curr_delta_ids.view(-1)]
    
    
            curr_batch_delta_id_offset = (delta_data_ids[curr_delta_ids]%batch_size).view(1,-1) 
    
    
            curr_origin_delta_ids = curr_batch_delta_id_offset + i
            
            curr_origin_delta_ids = curr_origin_delta_ids.view(-1)
            
            ids = (epoch*batch_num_per_epoch + int(i/batch_size))
            
            if curr_delta_ids.shape[0] == 0:
                delta_term1[ids] = (term1[ids] - 0)/(end_id - i)
            
                delta_term2[ids] = (term2[ids] - torch.t(x_sum_by_class_list[int(i/batch_size)]))/(end_id - i)
                
                curr_A = (1 - alpha*beta)*torch.eye(dim[1]*num_class, dtype = torch.double) - alpha*delta_term1[ids]
                    
                curr_B = -alpha*delta_term2[ids]
                
                theta = torch.mm(curr_A, theta).view(theta.shape) + curr_B.view(theta.shape)
                
#                 if i == 0:
#                     A[:] = curr_A[0:A.shape[0]]
#                     B[:] = curr_B[0:B.shape[0]].view(-1, dim[1]*num_class, 1)
#                     
#                 else:
#                     
#                     
#         #             if ids[ids < cut_off_epoch].shape[0] == A.shape[0]:
#         #                 curr_A = (1 - alpha*beta)*torch.eye(dim[1]) + alpha*res[ids[ids<cut_off_epoch]]
#         #                 
#         #                 curr_B = alpha*res2
#                     A = torch.bmm(A, curr_A[0:A.shape[0]])
#                     
#                     B = torch.bmm(curr_A[0:A.shape[0]], B) + curr_B[0:B.shape[0]].view(-1, dim[1]*num_class, 1)
                print('batch::', i)
                print('theta::', torch.t(theta).view(num_class, dim[1]))
    #             print('theta::', torch.t((torch.mm(A[1], theta.view(-1,1)) + B[1]).view(num_class, dim[1])))
    
                
                continue
            
            
    #         curr_w_seq = torch.t(w_seq[curr_origin_delta_ids[curr_origin_delta_ids<w_seq.shape[0]]].view(-1, curr_batch_delta_id_offset.view(-1).shape[0]))
    #                     
    #         curr_b_seq = torch.t(b_seq[curr_origin_delta_ids[curr_origin_delta_ids<b_seq.shape[0]]].view(-1, curr_batch_delta_id_offset.view(-1).shape[0]))
    
            
            
            
            
            
            curr_seq_ids = torch.tensor(list(range(i, end_id))).view(-1, 1) + (torch.tensor(list(range(max_epoch)))*X.shape[0]).view(1,-1)
            
            curr_seq_ids = curr_seq_ids.view(-1)
            
            curr_seq_ids = curr_seq_ids[curr_seq_ids < weights.shape[0]]
             
            curr_batch_size = batch_x.shape[0]
            
            
            
            '''(batch_size*t)*q^2'''
            curr_weights = weights[curr_origin_delta_ids[curr_origin_delta_ids<weights.shape[0]]].view(curr_batch_delta_id_offset.view(-1).shape[0], -1, num_class*num_class)
            
            curr_offsets = offsets[curr_origin_delta_ids[curr_origin_delta_ids<weights.shape[0]]].view(curr_batch_delta_id_offset.view(-1).shape[0], -1, num_class)
            
    #         print(curr_batch_size, curr_offsets.shape[1], num_class, dim[1])
    #         
    #         print(curr_weights.shape)
    #         
    #         print(batch_x.shape)
    
            print('curr_offsets::', curr_offsets)
            
            print('delta_data_ids::', delta_data_ids[curr_delta_ids.view(-1)].view(-1))
            
            print('curr_origin_delta_ids::', curr_origin_delta_ids[curr_origin_delta_ids<weights.shape[0]])
    
            
            curr_term2 = torch.reshape(torch.t(torch.mm(torch.t(batch_x), torch.reshape(curr_offsets, (curr_batch_size, curr_offsets.shape[1]*num_class)))), [curr_offsets.shape[1], num_class*dim[1]])
            
            expect_term2 = torch.reshape(torch.t(torch.mm(torch.t(X[i:end_id]), torch.reshape(offsets[i:end_id], (end_id - i, num_class)))), [1, num_class*dim[1]])
            
            
            
            
            res1 = torch.bmm(batch_x.view(curr_batch_size, dim[1], 1), curr_weights.view(curr_batch_size, 1, curr_weights.shape[1]*num_class*num_class))
        
            res2 = torch.mm(torch.t(batch_x), res1.view(curr_batch_size, dim[1]*curr_weights.shape[1]*num_class*num_class)).view(dim[1]*dim[1], curr_weights.shape[1], num_class*num_class)
            
            del res1
            
            res3 = torch.transpose(torch.transpose(res2, 0, 1), 2, 1).view(curr_weights.shape[1], num_class, num_class, dim[1], dim[1])
        
            del res2
        
            res4 = torch.reshape(torch.transpose(res3, 2, 3), [curr_weights.shape[1], num_class*dim[1], dim[1]*num_class])
            
            del res3
            
            curr_res1 = torch.bmm(X[i:end_id].view(end_id - i, dim[1], 1), weights[i:end_id].view(end_id - i, 1, 1*num_class*num_class))
        
            curr_res2 = torch.mm(torch.t(X[i:end_id]), curr_res1.view(end_id - i, dim[1]*1*num_class*num_class)).view(dim[1]*dim[1], 1, num_class*num_class)
            
            curr_res3 = torch.transpose(torch.transpose(curr_res2, 0, 1), 2, 1).view(1, num_class, num_class, dim[1], dim[1])
        
            curr_res4 = torch.reshape(torch.transpose(curr_res3, 2, 3), [1, num_class*dim[1], dim[1]*num_class])
            
            
            
            
            delta_term1[ids] = (term1[ids] - res4)/(end_id - i - curr_delta_ids.shape[0])
            
    #         print(term2[ids[ids < cut_off_epoch]].shape)
    #          
    #         print(curr_term2.shape)
    #          
    #         print(x_sum_by_class_list[int(i/batch_size)].shape)
    #         
    #         print(delta_x_sum_by_class_list[int(i/batch_size)].shape)
    #          
    #         print(ids[ids < cut_off_epoch])
    #          
    #         print(int(i/batch_size))
    #          
    #         print(delta_x_sum_by_class_list.keys())
            
            delta_term2[ids] = (term2[ids] - curr_term2 - torch.t(x_sum_by_class_list[int(i/batch_size)] - delta_x_sum_by_class_list[int(i/batch_size)]))/(end_id - i - curr_delta_ids.shape[0])
            
            
            curr_A = (1 - alpha*beta)*torch.eye(dim[1]*num_class, dtype = torch.double) - alpha*delta_term1[ids]
                    
            curr_B = -alpha*delta_term2[ids]
            
            theta = torch.mm(curr_A, theta).view(theta.shape) + curr_B.view(theta.shape)
            
#             if i == 0:
#                 A[:] = curr_A[0:A.shape[0]]
#                 B[:] = curr_B[0:B.shape[0]].view(-1, dim[1]*num_class, 1)
#             else:
#                 
#                 A = torch.bmm(A, curr_A[0:A.shape[0]])
#                 
#                 B = torch.bmm(curr_A[0:A.shape[0]], B) + curr_B[0:B.shape[0]].view(-1, dim[1]*num_class, 1)
            
            
            mini_epoch = mini_epochs + 1
            
            
            if mini_epoch >= cut_off_epoch:
                end_flag = True
                
                break
            
            print('A::', curr_A)
            
            print('B::', curr_B)
            
            print(curr_term2)
            
            print(torch.t(delta_x_sum_by_class_list[int(i/batch_size)]))
                
            print('batch::', i)
            print('theta::', torch.t(theta).view(num_class, dim[1]))
            
        if end_flag:
            break
    
    return delta_term1, delta_term2, A, B



def prepare_term_1_batch2_0_1(X, weights, offsets, dim, max_epoch, num_class, cut_off_epoch, batch_size, curr_rand_ids_multi_super_iterations):

    term1 = torch.zeros([cut_off_epoch, num_class*dim[1], dim[1]*num_class], dtype = torch.double)
    
    term2 = torch.zeros([cut_off_epoch, num_class*dim[1]], dtype = torch.double)
    
    epoch = 0
    
    end = False
    
    for k in range(curr_rand_ids_multi_super_iterations.shape[0]):
    
        curr_rand_ids = curr_rand_ids_multi_super_iterations[k]
        
        
        weights_this_super_iter = weights[k*X.shape[0]:(k+1)*X.shape[0]]
        
        offsets_this_super_iter = offsets[k*X.shape[0]:(k+1)*X.shape[0]]
    
        for i in range(0, X.shape[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
    
            batch_X = X[curr_rand_ids[i:end_id]]
            
            batch_weights = weights_this_super_iter[curr_rand_ids[i:end_id]]
            
            batch_offsets = offsets_this_super_iter[curr_rand_ids[i:end_id]]
            
            batch_term1 = prepare_sub_term_1(batch_X, batch_weights, batch_X.shape, num_class)
            
            term1[epoch] = batch_term1
            
            batch_term2 = prepare_sub_term_2(batch_X, batch_offsets, batch_X.shape, num_class)
            
            term2[epoch] = batch_term2
            
            
            epoch = epoch + 1
            
            if epoch >= cut_off_epoch:
                end = True
                break
        if end == True:
            break
    return term1, term2
            
            
            

def prepare_term_1_batch2_0(X, weights, offsets, dim, max_epoch, num_class, cut_off_epoch, batch_size, curr_rand_ids_multi_super_iterations):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    
    
    
    w_dim = weights.shape
    
    print(w_dim)
    
    print(dim)
    
    batch_num_per_epoch = int((X.shape[0] - 1)/batch_size) + 1
    
#     torch.tensor(list(range(max_epoch)))*batch_num_per_epoch

#     total_mini_epochs = batch_num_per_epoch * max_epoch
    
    
    term1 = torch.zeros([cut_off_epoch, num_class*dim[1], dim[1]*num_class], dtype = torch.double)
    
    term2 = torch.zeros([cut_off_epoch, num_class*dim[1]], dtype = torch.double)
    
    for i in range(0, X.shape[0], batch_size):
        
        end_id = i + batch_size
        
        if end_id > X.shape[0]:
            end_id = X.shape[0]
        
        
        batch_x = X[i:end_id]
        
        curr_seq_ids = torch.tensor(list(range(i, end_id))).view(-1, 1) + (torch.tensor(list(range(max_epoch)))*X.shape[0]).view(1,-1)
        
        curr_seq_ids = curr_seq_ids.view(-1)
        
        curr_seq_ids = curr_seq_ids[curr_seq_ids < weights.shape[0]]
         
        curr_batch_size = end_id - i
        
        
        ids = ((torch.tensor(list(range(max_epoch)))*batch_num_per_epoch + int(i/batch_size))).view(-1)
        
        
        
        '''(batch_size*t)*q^2'''
        curr_weights = weights[curr_seq_ids].view(curr_batch_size, -1, num_class*num_class)
        
        curr_offsets = offsets[curr_seq_ids].view(curr_batch_size, -1, num_class)
        
        curr_term2 = torch.reshape(torch.t(torch.mm(torch.t(batch_x), torch.reshape(curr_offsets, (curr_batch_size, curr_offsets.shape[1]*num_class)))), [curr_offsets.shape[1], num_class*dim[1]])
        
        
        res1 = torch.bmm(batch_x.view(curr_batch_size, dim[1], 1), curr_weights.view(curr_batch_size, 1, curr_weights.shape[1]*num_class*num_class))
    
        res2 = torch.mm(torch.t(batch_x), res1.view(curr_batch_size, dim[1]*curr_weights.shape[1]*num_class*num_class)).view(dim[1]*dim[1], curr_weights.shape[1], num_class*num_class)
        
        del res1
        
        res3 = torch.transpose(torch.transpose(res2, 0, 1), 2, 1).view(curr_weights.shape[1], num_class, num_class, dim[1], dim[1])
    
        del res2
    
        res4 = torch.reshape(torch.transpose(res3, 2, 3), [curr_weights.shape[1], num_class*dim[1], dim[1]*num_class])
        
        del res3
        
        term1[ids[ids < cut_off_epoch]] = res4
        
        term2[ids[ids < cut_off_epoch]] = curr_term2
        
    
    return term1, term2


def prepare_term_1_batch4(X_theta_prod_softmax_seq_tensor, X_theta_prod_seq_tensor, X, dim, max_epoch, num_class):
    
    '''weights: dim[0], max_epoch, num_class, num_class'''
    
    
#     X_theta_prod_softmax_seq_tensor = torch.transpose(X_theta_prod_softmax_seq_tensor, 0 ,1)
    
#     w_dim = weights.shape
    
#     print(w_dim)
    
    print(dim)
    
    res = Variable(torch.zeros([max_epoch, dim[1]*num_class, dim[1]*num_class], dtype = torch.double))
    
    weights = torch.zeros([dim[0], max_epoch, num_class, num_class], dtype = torch.double)
    
    offsets = torch.zeros([dim[0], max_epoch, num_class], dtype = torch.double)
    
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
    
    res = Variable(torch.zeros([max_epoch, dim[1]*num_class, dim[1]*num_class], dtype = torch.double))
    
    weights = torch.zeros([dim[0], max_epoch, num_class, num_class], dtype = torch.double)
    
    offsets = torch.zeros([dim[0], max_epoch, num_class], dtype = torch.double)
    
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
    


def prepare_term_1_batch3_0(X_theta_prod_softmax_seq_tensor, X_theta_prod_seq_tensor, dim, max_epoch, num_class, cut_off_epoch, batch_size):
    
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

    num = 0


    for t in range(max_epoch):
        
        
        end = False
        
        for i in range(0, dim[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id > dim[0]:
                
                end_id = dim[0]
    
#     for i in range(cut_off_epoch):
            '''X_theta_prod_softmax_seq_tensor[i]: n*q'''
                
                
            curr_batch_size = end_id -  i
            
            curr_weight = Variable(torch.bmm(X_theta_prod_softmax_seq_tensor[num].view(curr_batch_size,num_class ,1), X_theta_prod_softmax_seq_tensor[num].view(curr_batch_size,1,num_class)))
            
            
            '''batch_size*q*q'''
            curr_weight_sum = torch.sum(curr_weight, dim = 1)
            
            curr_weight3 = torch.diag_embed(curr_weight_sum) - curr_weight
            
            
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
            
            offsets.append(X_theta_prod_softmax_seq_tensor[num] - (torch.bmm(X_theta_prod_seq_tensor[num].view(curr_batch_size, 1, num_class), curr_weight3.view(curr_batch_size, num_class, num_class))).view(curr_batch_size, num_class))
        
        
            num += 1
            
            
            if num >= cut_off_epoch:
                
                end = True
                
                break
        
        
        
        if end:
            break
        
        
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
    
    '''(t*batch_size)* q^2 -> T*n, q^2'''
    
    print(weights[0])
    
    weights = torch.cat(weights, 0)
    
    offsets = torch.cat(offsets, 0)
    

#     weights = torch.transpose(torch.stack(weights), 0, 1)
#     
#     offsets = torch.transpose(torch.stack(offsets), 0, 1)
    
    
        
    return weights, offsets        
#     return weights[:, 0:cut_off_epoch, :, :], offsets[:, 0:cut_off_epoch, :], cut_off_epoch
        
        
    
#     print('done')
    
    
    
#     w_seq_transpose = torch.t(w_seq)
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     batch_num = int(dim[0]/batch_size)
    
#     res = torch.zeros(w_dim[0], dim[1], dim[1], dtype = torch.double)

def prepare_term_1_batch3_1(random_ids_multi_super_iterations, theta_list, X, Y, dim, max_epoch, num_class, cut_off_epoch, batch_size):
    
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

    num = 0


    for k in range(random_ids_multi_super_iterations.shape[0]):
        
        random_ids = random_ids_multi_super_iterations[k]
        
        end = False
        
        for i in range(0, dim[0], batch_size):
            
            end_id = i + batch_size
            
            if end_id > dim[0]:
                
                end_id = dim[0]
    
#     for i in range(cut_off_epoch):
            '''X_theta_prod_softmax_seq_tensor[i]: n*q'''
            
            batch_X = X[random_ids[i:end_id]]
            
            batch_Y = Y[random_ids[i:end_id]]
            
            X_theta_prod = torch.mm(batch_X, theta_list[num])
    
    
            X_theta_prod_softmax = softmax_layer(X_theta_prod)
                
            curr_batch_size = end_id -  i
            
            curr_weight = Variable(torch.bmm(X_theta_prod_softmax.view(curr_batch_size,num_class ,1), X_theta_prod_softmax.view(curr_batch_size,1,num_class)))
            
            
            '''batch_size*q*q'''
            curr_weight_sum = torch.sum(curr_weight, dim = 1)
            
            curr_weight3 = torch.diag_embed(curr_weight_sum) - curr_weight
            
            
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
            
            offsets.append(X_theta_prod_softmax - (torch.bmm(X_theta_prod.view(curr_batch_size, 1, num_class), curr_weight3.view(curr_batch_size, num_class, num_class))).view(curr_batch_size, num_class))
        
        
            num += 1
            
            
            if num >= cut_off_epoch:
                
                end = True
                
                break
        
        
        
        if end:
            break
        
        
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
    
    '''(t*batch_size)* q^2 -> T*n, q^2'''
    weights = torch.cat(weights, 0)
    
    offsets = torch.cat(offsets, 0)
    

#     weights = torch.transpose(torch.stack(weights), 0, 1)
#     
#     offsets = torch.transpose(torch.stack(offsets), 0, 1)
    
    
        
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
    
    res = Variable(torch.zeros([max_epoch, dim[1]*num_class, dim[1]*num_class], dtype = torch.double))
    
    
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


def prepare_term_2_batch2(X, offsets, dim, max_epoch, num_class, cut_off_epoch, batch_size):
    
    
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
    
    
#     for i in range(0, X.shape[0], batch_size):
        
        
        
        
    
    
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

def verity_term_1(weights, theta, X, epoch, num_class, dim):
    
    
#     print(theta)
    
    coeff_sum = torch.zeros([num_class, X.shape[1]], dtype = torch.double)
    
    for i in range(X.shape[0]):
        for j in range(num_class):
            
            temp = torch.mm(theta[epoch][:,j].view(1, -1), X[i].view(-1, 1))
            
            
            temp = (temp*weights[i][epoch][j]).view(-1, 1)
            
            coeff_sum += torch.mm(temp, X[i].view(1, -1))
    
    
    print('expected_output::', coeff_sum.view(-1,1)/dim[0])
    
    
        
def generate_theta_seq(res_prod_seq, dim, num_class, max_epoch):
    
    theta_seq_tensor = torch.zeros([dim[1], num_class*max_epoch], dtype = torch.double)
    
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
        
        
    return res/dim[0] + beta*torch.eye(num_class*dim[1], dtype = torch.double)


def compute_first_derivative(X, Y, dim, theta, num_class):
    x_sum_by_class = compute_x_sum_by_class(X, Y, num_class, X.shape)
    output = softmax_layer(torch.mm(X, theta))
        
        
    output = torch.mm(torch.t(X), output)
    
    
    output = torch.reshape(torch.t(output), [-1,1])
    
#         print(i, theta)
    
#         inter_result2.append(output)
#         
#         res = torch.mm(torch.t(torch.gather(output, 1, Y.view(dim[0], 1))), X)
    
    reshape_theta = torch.reshape(torch.t(theta), (-1, 1))
    
    
    return (output - x_sum_by_class)
    


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















