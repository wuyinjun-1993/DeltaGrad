'''
Created on Feb 4, 2019

'''
from torch import nn, optim
import torch
from sensitivity_analysis.Load_data import load_data
import random
from sensitivity_analysis.Logistic_regression import *
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
    

def random_permutation(selected_rows, X, intensity, dim):
    update_X = X.clone()
    for i in selected_rows:
        print('before_change::', update_X[i])
        for j in range(dim[1]):
            
#             rand = random.uniform(0, intensity)
            rand = intensity
            
            positive = bool(random.getrandbits(1))
            
            update_value = 0
            
            if positive:
                update_value = update_X[i][j]*(1 + rand)
                
                if update_value > 1:
                    update_X[i][j] = 1
                else:
                    update_X[i][j] = update_value
                
            else:
                update_value = update_X[i][j]*(1 - rand)
                update_X[i][j] = update_value
        
        print('after_change::', update_X[i])    
    
    return update_X


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

def prepare_term_1_batch2(X_product, w_seq, dim, batch_size):
    
    w_dim = w_seq.shape
    
    print(w_dim)
    
    print(dim)
    
#     w_seq_transpose = torch.t(w_seq)
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
#     batch_num = int(dim[0]/batch_size)
    
#     res = torch.zeros(w_dim[0], dim[1], dim[1], dtype = torch.double)
    
    X_product = torch.t(X_product.view(dim[0], dim[1]*dim[1]))
    
    res = torch.t(torch.mm(X_product, torch.t(w_seq))).view(w_dim[0], dim[1], dim[1])
    
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


def prepare_term_2_batch2(X_Y_prod, b_seq, dim, batch_size):
    
    b_dim = b_seq.shape
    
    print('b_dim::', b_dim)
    
    b_seq_transpose = torch.t(b_seq)
#     X_Y_prod = X.mul(Y)
#     dim[0]*dim[1]
#     X_Y_prod = X_Y_prod.repeat(b_dim[1], 1)

    batch_num = int(dim[0]/batch_size)
    
    res = torch.zeros(b_dim[0], dim[1], dtype = torch.double)
    
    print(X_Y_prod.shape)
    
    print(b_seq.shape)
    
    res = torch.t(torch.mm(torch.t(X_Y_prod), torch.t(b_seq)))
    
    
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
    
    intensity = 0.1
    
    # alpha = 0.00001
    #       
    # beta = 0.5
    [X, Y] = load_data(True)
    
    dim = X.shape
    
    
    res = compute_parameters(X, Y, initialize(X), dim)
    
    
    w_seq, b_seq = compute_linear_approx_parameters(X, Y, dim, res)
    t_01 = time.time()
    # term1, term1_inter_result = prepare_term_1_serial(X, w_seq, dim)
#     term1 = prepare_term_1_serial(X, w_seq, dim)
#     
#     # term2, term2_inter_result = prepare_term_2_serial(X, Y, b_seq, dim)
#     term2 = prepare_term_2_serial(X, Y, b_seq, dim)
    
    t_02 = time.time()
    
    # dim = X.shape
    
    
    delta_data_ids = random_generate_subset_ids(dim, delta_size)
    
    update_X = random_permutation(delta_data_ids, X, intensity, dim)
    
    
    # selected_rows = torch.tensor(list(set(range(dim[0])) - delta_data_ids))
    
#     update_X, selected_rows = get_subset_training_data(X, dim, delta_data_ids)
    
#     update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    update_Y = Y
    
    print(X.shape)
    
    print(update_X.shape)
    
    t1 = time.time()
    
    res1 = torch.zeros(res.shape)
    
    for i in range(repetition):
        
        lr = initialize(update_X)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
        res1 = logistic_regression_by_standard_library(update_X, update_Y, lr, dim)
    
    t2 = time.time()
    
    
    update_X_Y_mult = update_X.mul(update_Y)

    
    t3 = time.time()
    
    total_time = 0
    
    res2 = torch.zeros(res.shape)
    # 
    for i in range(repetition):
        init_theta = Variable(initialize(update_X).theta)
        
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
        res2, total_time = compute_model_parameter_by_iteration(dim, init_theta, update_X_Y_mult)
    
    
    t4 = time.time()
    # t5 = time.time()
    # 
    # for i in range(repetition):
    #     [U, S, V] = torch.svd(torch.mm(torch.transpose(X, 0, 1), X))
    # 
    # t6 = time.time()
    
    X_Y_mult = X.mul(Y)
    
#     update_X_Y_mult = update_X.mult(Y)
    
    X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
    
    
    update_X_product = torch.bmm(update_X.view(dim[0], dim[1], 1), update_X.view(dim[0], 1, dim[1]))
    
    step_time_1 = 0
    
    step_time_2 = 0
    
    
    t5 = time.time()
    
    res3 = torch.zeros(res.shape)
    
    
    for i in range(repetition):
    #     print(X_Y_mult.shape)
    #     
        print('w_seq_shape::', w_seq.shape)
         
        print(b_seq.shape)
        
        
#         if len(delta_data_ids) < (dim[0])/2:
            
#             sub_w_seq = torch.index_select(w_seq, 1, delta_data_ids)#get_subset_parameter_list(selected_rows, delta_data_ids, w_seq, dim, 1)
#              
#             sub_b_seq = torch.index_select(b_seq, 1, delta_data_ids)#get_subset_parameter_list(b_seq, delta_data_ids, b_seq, dim, 1)
             
        #     delta_X = torch.index_select(X, 0, delta_data_ids)
        #      
        #     delta_Y = torch.index_select(Y, 0, delta_data_ids)
        #     sub_X_Y_mult = get_subset_parameter_list(selected_rows, delta_data_ids, X_Y_mult, dim, 0)
             
#             delta_X_product = torch.index_select(X_product, 0, delta_data_ids)
#               
#             delta_X_Y_mult = torch.index_select(X_Y_mult, 0, delta_data_ids)
             
        #     sub_term_1 = prepare_term_1_serial(delta_X, sub_w_seq, delta_X.shape)#(delta_X_product, sub_w_seq, delta_X.shape)
        #      
        #     sub_term_2 = prepare_term_2_serial(delta_X, delta_Y, sub_b_seq, delta_X.shape)#(delta_X_Y_mult, sub_b_seq, delta_X.shape)
            
        #     update_X_dim = update_X.dim
            
        curr_delta_dim = [X.shape[0], X.shape[1]]
        
        s_1 = time.time()
        
        sub_term_1 = prepare_term_1_batch2(update_X_product, w_seq, curr_delta_dim, 20)
         
        sub_term_2 = prepare_term_2_batch2(update_X_Y_mult, b_seq, curr_delta_dim, 20)     
         
         
        s_2 = time.time()
        
        step_time_1 += s_2  -s_1
    #     sub_term_2 = prepare_term_2_batch(delta_X_Y_mult, sub_b_seq, delta_X.shape)
    
    #     sub_term_1 = prepare_term_1(term1_inter_result, delta_data_ids)
    #     
    #     sub_term_2 = prepare_term_2(term2_inter_result, delta_data_ids)
        
        init_theta = Variable(initialize(update_X).theta)
        
    #     res2 = update_model_parameters_incrementally(U, S, V, update_X, update_Y, max_epoch, dim)
    
    #     print(sub_w_seq.shape)
    #     
    #     print(sub_b_seq.shape)
    #     
    #     print(update_X.shape)
    #     
    #     print(update_Y.shape)
    #     
    #     print(lr.theta.shape)
    
    #     update_X_products = compute_sample_products(update_X, update_X.shape)
    #      
    #     update_X_Y_products = compute_sample_label_products(update_X, update_Y)
    
    #     res2 = compute_model_parameter_by_approx(sub_w_seq, sub_b_seq, update_X, update_Y, update_X.shape, lr.theta, update_X_products, update_X_Y_products)
        
        print('this_dim::', dim)
        
        s_3 = time.time()
        
        res3 = compute_model_parameter_by_approx_incremental_2(sub_term_1, sub_term_2, dim, init_theta)
        
        s_4 = time.time()
        
        step_time_2 += s_4  -s_3
        
#         else:
#             sub_w_seq = torch.index_select(w_seq, 1, selected_rows)#get_subset_parameter_list(selected_rows, delta_data_ids, w_seq, dim, 1)
#              
#             sub_b_seq = torch.index_select(b_seq, 1, selected_rows)#get_subset_parameter_list(b_seq, delta_data_ids, b_seq, dim, 1)
#              
#         #     delta_X = torch.index_select(X, 0, delta_data_ids)
#         #      
#         #     delta_Y = torch.index_select(Y, 0, delta_data_ids)
#         #     sub_X_Y_mult = get_subset_parameter_list(selected_rows, delta_data_ids, X_Y_mult, dim, 0)
#              
#             delta_X_product = torch.index_select(X_product, 0, selected_rows)
#               
#             delta_X_Y_mult = torch.index_select(X_Y_mult, 0, selected_rows)
#              
#         #     sub_term_1 = prepare_term_1_serial(delta_X, sub_w_seq, delta_X.shape)#(delta_X_product, sub_w_seq, delta_X.shape)
#         #      
#         #     sub_term_2 = prepare_term_2_serial(delta_X, delta_Y, sub_b_seq, delta_X.shape)#(delta_X_Y_mult, sub_b_seq, delta_X.shape)
#             
#         #     update_X_dim = update_X.dim
#             
#             curr_delta_dim = [update_X.shape[0], X.shape[1]]
#             
#             s_1 = time.time()
#             
#             sub_term_1 = prepare_term_1_batch2(delta_X_product, sub_w_seq, curr_delta_dim, 20)
#              
#             sub_term_2 = prepare_term_2_batch2(delta_X_Y_mult, sub_b_seq, curr_delta_dim, 20)     
#              
#              
#             s_2 = time.time()
#             
#             step_time_1 += s_2  -s_1
#         #     sub_term_2 = prepare_term_2_batch(delta_X_Y_mult, sub_b_seq, delta_X.shape)
#         
#         #     sub_term_1 = prepare_term_1(term1_inter_result, delta_data_ids)
#         #     
#         #     sub_term_2 = prepare_term_2(term2_inter_result, delta_data_ids)
#             
#             init_theta = Variable(initialize(update_X).theta)
#             
#         #     res2 = update_model_parameters_incrementally(U, S, V, update_X, update_Y, max_epoch, dim)
#         
#         #     print(sub_w_seq.shape)
#         #     
#         #     print(sub_b_seq.shape)
#         #     
#         #     print(update_X.shape)
#         #     
#         #     print(update_Y.shape)
#         #     
#         #     print(lr.theta.shape)
#         
#         #     update_X_products = compute_sample_products(update_X, update_X.shape)
#         #      
#         #     update_X_Y_products = compute_sample_label_products(update_X, update_Y)
#         
#         #     res2 = compute_model_parameter_by_approx(sub_w_seq, sub_b_seq, update_X, update_Y, update_X.shape, lr.theta, update_X_products, update_X_Y_products)
#             
#             print('this_dim::', dim)
#             
#             s_3 = time.time()
#             
#             res3 = compute_model_parameter_by_approx_incremental_2(sub_term_1, sub_term_2, dim, init_theta)
#             
#             s_4 = time.time()
#             
#             step_time_2 += s_4  -s_3
            
            
    
    t6 = time.time()
    
    time1 = (t2 - t1)/repetition
    
    time2 = (t4 - t3)/repetition
    
    time3 = (t6 - t5)/repetition
    
    compute_sub_term_time = t_02 - t_01
    
    
    print('res::', res)
    
    print('res1::', res1)
    
    print('res2::', res2)
    
    print('res3::', res3)
    
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
    
#     hessian_matrix = compute_hessian_matrix(X, Y, res, dim, X_product)
#     
#     res4 = torch.zeros(res.shape)
#     
#     
#     t7 = time.time()
#     
#     first_derivative = compute_first_derivative_single_data(X_Y_mult, delta_data_ids, res, dim)
#     
#     res4 = res + torch.mm(torch.inverse(hessian_matrix), first_derivative)/dim[0]
#     
#     
#     t8 = time.time()
#     
#     print(hessian_matrix)
#     
#     print(first_derivative)
#     
#     
#     
#     print(res4)
#     
#     print(res4  - res1)
#     
#     print('time4::', (t8  -t7))
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
