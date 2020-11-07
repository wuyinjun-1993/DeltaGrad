'''
Created on Jun 24, 2020

'''


import sys, os
import torch
import time


import psutil

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/data_IO')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Interpolation')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Models')


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.abspath(__file__))




from Models.Data_preparer import *

from utils import *

from model_train import *




try:
    from data_IO.Load_data import *
    from utils import *
    from Models.DNN import DNNModel
    from Models.Data_preparer import *
    from Models.DNN_single import *
    from Models.DNN2 import DNNModel2
    from Models.DNN3 import DNNModel3
    from Models.ResNet import *
    from Models.Pretrained_models import *

except ImportError:
    from Load_data import *
    from utils import *
    from Models.DNN import DNNModel
    from Models.DNN2 import DNNModel2
    from Models.DNN3 import DNNModel3
    from Models.Data_preparer import *
    from Models.DNN_single import *
    from Models.ResNet import *
    from Models.Pretrained_models import *

def get_batch_train_data(dataset_train, ids):
    
    batch_x_train_cp, batch_y_train_cp = dataset_train.data[ids],dataset_train.labels[ids] 
    
    return batch_x_train_cp, batch_y_train_cp, ids


def compute_grad_final3(para, hessian_para_prod, gradient_dual, grad_list_tensor, para_list_tensor, size1, size2, alpha, beta, is_GPU, device):
    
    gradients = None
    
    if gradient_dual is not None:
        
        hessian_para_prod += grad_list_tensor 
        
        hessian_para_prod += beta*para_list_tensor 
        
        gradients = hessian_para_prod*size1
        
        
#         gradients = (hessian_para_prod[i]*size1 - (gradient_dual[i].to('cpu') + beta*para[i])*size2)/(size1 - size2)
        
        gradients -= (gradient_dual + beta*para_list_tensor)*size2
        
        gradients /= (size1 - size2)
            
    else:
        
        hessian_para_prod += (grad_list_tensor + beta*para_list_tensor)
        
        gradients = hessian_para_prod
        
        
    delta_para = para - para_list_tensor
    
    delta_grad = hessian_para_prod - (grad_list_tensor + beta*para_list_tensor)
    
    tmp_res = 0
    
    
    if torch.norm(delta_para) > torch.norm(delta_grad):
        return True, gradients

    else:
        return False, gradients
    


def cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, i, m, k, v_vec, period, is_GPU, device):
 
 
#     t3  = time.time()
    
    period_num = int(i/period)
    
    
    ids = torch.tensor(range(m)).view(-1)
    
    if period_num > 0:
        ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
#     else:
#         ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
    ids = ids - 1
    
    ids = ids[ids >= 0]
    
    if ids.shape[0] > k:
        ids = ids[-k:]
    
#     if i-k >= 1:
#         lb = i-k
#         
#         zero_mat_dim = ids.shape[0] + k
#         
#     else:
#         lb = 1
#         
#         zero_mat_dim = ids.shape[0] + i-1
    zero_mat_dim = k#ids.shape[0]
    
    
    
#     curr_S_k = torch.cat([S_k_list[:, ids],S_k_list[:,lb:i]], dim=1) 
#           
#     curr_Y_k = torch.cat([Y_k_list[:, ids],Y_k_list[:,lb:i]], dim=1)

#     print(ids)

#     curr_S_k = torch.cat([S_k_list[:, ids]], dim=1) 
#           
#     curr_Y_k = torch.cat([Y_k_list[:, ids]], dim=1)

#     curr_S_k = S_k_list[:, ids]
#           
#     curr_Y_k = Y_k_list[:, ids]
    
    curr_S_k = torch.t(torch.cat(list(S_k_list), dim = 0))
          
    curr_Y_k = torch.t(torch.cat(list(Y_k_list), dim = 0))
    
#     curr_S_k = S_k_list[:,k:m] 
#          
#     curr_Y_k = Y_k_list[:,k:m] 
    
    S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
    
    
    S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)
    
    
    if is_GPU:
        
        R_k = np.triu(S_k_time_Y_k.to('cpu').numpy())
        
        L_k = S_k_time_Y_k - torch.from_numpy(R_k).to(device)
        
    else:
        R_k = np.triu(S_k_time_Y_k.numpy())
    
        L_k = S_k_time_Y_k - torch.from_numpy(R_k)
    
    D_k_diag = torch.diag(S_k_time_Y_k)
    
    
    sigma_k = torch.mm(Y_k_list[-1],torch.t(S_k_list[-1]))/(torch.mm(S_k_list[-1], torch.t(S_k_list[-1])))
    
    
#     interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
#     
#     J_k = torch.from_numpy(np.linalg.cholesky(interm.detach().numpy())).type(torch.DoubleTensor)
    
    
#     v_vec = S_k_list[:,i-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
    
#         v_vec = torch.rand([para_num, 1], dtype = torch.double)
#     t1 = time.time()
    if is_GPU:
        p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double, device = device)
    else:
        p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double)
    
    tmp = torch.mm(torch.t(curr_Y_k), v_vec)
    
    p_mat[0:zero_mat_dim] = tmp
    
    p_mat[zero_mat_dim:zero_mat_dim*2] = torch.mm(torch.t(curr_S_k), v_vec)*sigma_k
    
#     t2 = time.time()
    
#     p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
    
    
#     D_k_sqr_root = torch.pow(D_k_diag, 0.5)
#     
#     D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
#     
#     upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
#     
#     lower_mat_1 = torch.cat([torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double), torch.t(J_k)], dim = 1)
#     
#     
#     mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
#     
#     
#     upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double)], dim = 1)
#     
#     lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
#     
#     mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
    
    
    
    
    
    upper_mat = torch.cat([-torch.diag(D_k_diag), torch.t(L_k)], dim = 1)
    
    lower_mat = torch.cat([L_k, sigma_k*S_k_time_S_k], dim = 1)
    
    mat = torch.cat([upper_mat, lower_mat], dim = 0)
    

    mat = np.linalg.inv(mat.cpu().numpy())
        
    inv_mat = torch.from_numpy(mat)
    
    if is_GPU:
        
        inv_mat = inv_mat.to(device)
        
        
    
    p_mat = torch.mm(inv_mat, p_mat)
    
    
    approx_prod = sigma_k*v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    
#     t4  = time.time()
    
    
#     print('time1::', t4 - t3)
#     
#     print('key time::', t2 - t1)
    
    
    return approx_prod,zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat


def compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, v_vec, is_GPU, device):
    
    
    if is_GPU:
        p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double, device =device)
    else:
        p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double)
     
    
    torch.mm(curr_Y_k, v_vec, out = p_mat[0:zero_mat_dim])
    
#     torch.sum(curr_S_k_transpose*v_vec*sigma_k, dim = 1, out = p_mat[zero_mat_dim:zero_mat_dim*2])
    
    
    torch.mm(curr_S_k, v_vec*sigma_k, out = p_mat[zero_mat_dim:zero_mat_dim*2])

#     torch.mm(mat, p_mat, out = p_mat)

    p_mat = torch.mm(mat, p_mat)
    
#     print(curr_Y_k_transpose.shape, curr_S_k_transpose.shape, v_vec.shape)
    
    approx_prod = sigma_k*v_vec
    
#     approx_prod -= torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
#     print((torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1)).shape)
#     
#     print((torch.sum((torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1))*p_mat.view(1, -1), dim = 1)).shape)
#     
#     print(approx_prod.shape)
    
    
    
#     approx_prod -= torch.sum((torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1))*p_mat.view(1, -1), dim = 1).view(-1,1)
    
    approx_prod -= (torch.mm(torch.t(curr_Y_k), p_mat[0:zero_mat_dim]) + torch.mm(sigma_k*torch.t(curr_S_k), p_mat[zero_mat_dim:zero_mat_dim*2]))
    
    
#     approx_prod -= torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    
    return approx_prod



def prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, m, k, is_GPU, device):
 
 
    zero_mat_dim = k#ids.shape[0]

    curr_S_k = torch.cat(list(S_k_list), dim = 0)
          
    curr_Y_k = torch.cat(list(Y_k_list), dim = 0)
#     curr_S_k = S_k_list[:,k:m] 
#          
#     curr_Y_k = Y_k_list[:,k:m] 
    
    S_k_time_Y_k = torch.mm(curr_S_k, torch.t(curr_Y_k))
    
    
    S_k_time_S_k = torch.mm(curr_S_k, torch.t(curr_S_k))
    
    if is_GPU:
        R_k = np.triu(S_k_time_Y_k.to('cpu').numpy())
        
    
        L_k = S_k_time_Y_k - (torch.from_numpy(R_k)).to(device)
    else:
        R_k = np.triu(S_k_time_Y_k.numpy())
        
    
        L_k = S_k_time_Y_k - torch.from_numpy(R_k)
    
    D_k_diag = torch.diag(S_k_time_Y_k)
    
    
    sigma_k = torch.mm(Y_k_list[-1],torch.t(S_k_list[-1]))/(torch.mm(S_k_list[-1], torch.t(S_k_list[-1])))
    
    
    upper_mat = torch.cat([-torch.diag(D_k_diag), torch.t(L_k)], dim = 1)
    
    lower_mat = torch.cat([L_k, sigma_k*S_k_time_S_k], dim = 1)
    
    mat = torch.cat([upper_mat, lower_mat], dim = 0)
    
    return zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat

def update_para_final2(para, gradient_list, alpha, beta, exp_gradient, exp_para):
    
    exp_grad_list = []
    
    vec_para = get_all_vectorized_parameters1(para)
    
    vec_para -= alpha*gradient_list
    
    if exp_gradient is not None:
        print("grad_diff::")
         
        compute_model_para_diff(gradient_list, exp_grad_list)
        
          
        print("here!!")
        
        
    return vec_para

def get_remaining_subset_data_per_epoch(curr_rand_ids, removed_rand_ids):
    
    
    curr_rand_id_set = set(curr_rand_ids.tolist())
    
    curr_remaining_ids_set = list(curr_rand_id_set.difference(set(removed_rand_ids.tolist())))
    
    res = torch.tensor(curr_remaining_ids_set)
    
    return res


def model_update_standard_lib(num_epochs, dataset_train, model, random_ids_multi_epochs, sorted_ids_multi_epochs, delta_data_ids, batch_size, learning_rate_all_epochs, criterion, optimizer, is_GPU, device):
    count = 0

    elapse_time = 0
    
    overhead = 0
 #     
    overhead3 = 0
    
    t1 = time.time()
    
    
    exp_gradient_list_all_epochs = []
      
    exp_para_list_all_epochs = []
    
    old_lr = -1
    

    random_ids_list_all_epochs = []
    
    t5 = time.time()
    
    for k in range(num_epochs):
        

    
        random_ids = random_ids_multi_epochs[k]
        
        sort_idx = sorted_ids_multi_epochs[k]#random_ids.numpy().argsort()
        
        if delta_data_ids.shape[0] > 1:
            all_indexes = np.sort(sort_idx[delta_data_ids])
        else:
            all_indexes = torch.tensor([sort_idx[delta_data_ids]])
                
        id_start = 0
    
        id_end = 0
        
        random_ids_list = []
        
        for j in range(0, len(dataset_train), batch_size):
        
            end_id = j + batch_size
            
            if end_id > len(dataset_train):
                end_id = len(dataset_train)
            
            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
            
            removed_ids = random_ids[all_indexes[id_start:id_end]]
            
            if removed_ids.shape[0] > 0:
                curr_matched_ids = get_remaining_subset_data_per_epoch(random_ids[j:end_id], removed_ids)
            else:
                curr_matched_ids = random_ids[j:end_id]
            random_ids_list.append(curr_matched_ids)
            id_start = id_end
                
        random_ids_list_all_epochs.append(random_ids_list)

    t6 = time.time()
    
    overhead2 = (t6 - t5)

    
    
            
            
    for k in range(num_epochs):   
        
        print("epoch::", k)
        
        random_ids_list = random_ids_list_all_epochs[k]
        
        for j in range(len(random_ids_list)):
        
        
            curr_random_ids = random_ids_list[j]
            
            if k == 0 and j == 0:
                print(curr_random_ids[0:50])
            
            curr_matched_ids_size = len(curr_random_ids)
            
            if curr_matched_ids_size <= 0:
                
                count += 1
                
                continue
            
            if not is_GPU:

                batch_X = dataset_train.data[curr_random_ids]
                
                batch_Y = dataset_train.labels[curr_random_ids]
                
            else:
                batch_X = dataset_train.data[curr_random_ids].to(device)
                
                batch_Y = dataset_train.labels[curr_random_ids].to(device)
                
            learning_rate = learning_rate_all_epochs[count]
            
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate
                
            optimizer.zero_grad()

            output = model(batch_X)
            
            loss = criterion(output, batch_Y)
            
            loss.backward()
            
#             if record_params:
#                 append_gradient_list(exp_gradient_list_all_epochs, None, exp_para_list_all_epochs, model, batch_X, is_GPU, device)

            optimizer.step()
            
            count += 1
        
    t2 = time.time()
        
    elapse_time += (t2 - t1)  

    print("training time is", elapse_time)
    
    print("overhead::", overhead)
     
    print("overhead2::", overhead2)
#     
    print("overhead3::", overhead3)

#     return model, count, exp_para_list_all_epochs, exp_gradient_list_all_epochs, random_ids_list_all_epochs
    return model

def model_update_deltagrad(max_epoch, period, length, init_epochs, dataset_train, model, gradient_list_all_epochs_tensor, para_list_all_epochs_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, delta_ids, m, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, criterion, optimizer, regularization_coeff, is_GPU, device):
    '''function to use deltagrad for incremental updates'''
    
    
    para = list(model.parameters())
    
    
    use_standard_way = False
    
    recorded = 0
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    if not is_GPU:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    else:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double, device=device)
    
    
    i = 0
    
    S_k_list = deque()
    
    Y_k_list = deque()
    
#     overhead2 = 0
#     
#     overhead3 = 0
#     
#     overhead4 = 0
#     
#     overhead5 = 0
    
    old_lr = 0
    
    cached_id = 0
    
    batch_id = 1
    
    
    random_ids_list_all_epochs = []

    removed_batch_empty_list = []
    
#     res_para_list = []
#     
#     res_grad_list = []
    
#     t5 = time.time()
    
    '''detect which samples are removed from each mini-batch'''
    
    for k in range(max_epoch):
        

    
        random_ids = random_ids_multi_super_iterations[k]
        
        sort_idx = sorted_ids_multi_super_iterations[k]#random_ids.numpy().argsort()
        
        if delta_ids.shape[0] > 1:
            all_indexes = np.sort(sort_idx[delta_ids])
        else:
            all_indexes = torch.tensor([sort_idx[delta_ids]])
                
        id_start = 0
    
        id_end = 0
        
        random_ids_list = []
        
        for j in range(0, dim[0], batch_size):
        
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]

            curr_matched_id_num = curr_matched_ids.shape[0]

            if curr_matched_id_num > 0:
                random_ids_list.append(curr_matched_ids)
                removed_batch_empty_list.append(False)
            else:
                random_ids_list.append(random_ids[0:1])
                removed_batch_empty_list.append(True)
            
            i += 1
            
            id_start = id_end
                
        random_ids_list_all_epochs.append(random_ids_list)        
    
    
#     t6 = time.time()
#     
#     overhead3 += (t6  -t5)
    
    '''main for loop of deltagrad'''
    
    i = 0
    
    for k in range(max_epoch):
    
        random_ids = random_ids_multi_super_iterations[k]
        
        
        random_ids_list = random_ids_list_all_epochs[k]
        
        id_start = 0
    
        id_end = 0
        
        j = 0
        
        curr_init_epochs = init_epochs
        
        for p in range(len(random_ids_list)):
            
#             curr_matched_ids = items[2]        
            curr_matched_ids = random_ids_list[p]
            
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]        
#             print(i,p)
            curr_matched_ids_size = 0
            if not removed_batch_empty_list[i]:
                
                if not is_GPU:
                
                    batch_delta_X = dataset_train.data[curr_matched_ids]
                    
                    batch_delta_Y = dataset_train.labels[curr_matched_ids]
                
                else:

                    batch_delta_X = dataset_train.data[curr_matched_ids].to(device)
                    
                    batch_delta_Y = dataset_train.labels[curr_matched_ids].to(device)
                
#                 curr_matched_ids_size = items[2].shape[0]
                curr_matched_ids_size = len(curr_matched_ids)
            
            learning_rate = learning_rate_all_epochs[i]
            
            
            if end_id - j - curr_matched_ids_size <= 0:
                
                i += 1
                
                continue            
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate    
                
            
            if (i-curr_init_epochs)%period == 0:
                
                recorded = 0
                
                use_standard_way = True
                
                
            if i< curr_init_epochs or use_standard_way == True:
#                 t7 = time.time()
                '''explicitly evaluate the gradient'''
                curr_rand_ids = random_ids[j:end_id]
                
                if not removed_batch_empty_list[i]:
                
                    curr_matched_ids2 = get_remaining_subset_data_per_epoch(curr_rand_ids, curr_matched_ids)
                else:
                    curr_matched_ids2 = curr_rand_ids
                
                batch_remaining_X, batch_remaining_Y, batch_remaining_ids = get_batch_train_data(dataset_train, curr_matched_ids2)
                
                if is_GPU:
                    batch_remaining_X = batch_remaining_X.to(device)
                    
                    batch_remaining_Y = batch_remaining_Y.to(device)
                
#                 t8 = time.time()
#             
#                 overhead4 += (t8 - t7)
#                 
#                 
#                 t5 = time.time()
                
                init_model(model, para)
                
                compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
                
                

                
                
                expect_gradients = get_all_vectorized_parameters1(model.get_all_gradient())
                
#                 t6 = time.time()
# 
#                 overhead3 += (t6 - t5)
                
                gradient_remaining = 0
#                 if curr_matched_ids_size > 0:
                if not removed_batch_empty_list[i]:
                    
#                     t3 = time.time()
                    
                    clear_gradients(model.parameters())
                        
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                
                
                    gradient_remaining = get_all_vectorized_parameters1(model.get_all_gradient())     
                    
                    
#                     t4 = time.time()
#                 
#                 
#                     overhead2 += (t4  -t3)
                
                with torch.no_grad():
                               
                
                    curr_para = get_all_vectorized_parameters1(para)
                
                    if k>0 or (p > 0 and k == 0):
                        
                        prev_para = para_list_GPU_tensor[cached_id]
                        
                        curr_s_list = (curr_para - prev_para) + 1e-16
                        
                        S_k_list.append(curr_s_list)
                        if len(S_k_list) > m:
                            removed_s_k = S_k_list.popleft()
                            
                            del removed_s_k
                        
                    gradient_full = (expect_gradients*curr_matched_ids2.shape[0] + gradient_remaining*curr_matched_ids_size)/(curr_matched_ids2.shape[0] + curr_matched_ids_size)

                    if k>0 or (p > 0 and k == 0):
                        
                        Y_k_list.append(gradient_full - grad_list_GPU_tensor[cached_id] + regularization_coeff*curr_s_list+ 1e-16)
                        
                        if len(Y_k_list) > m:
                            removed_y_k = Y_k_list.popleft()
                            
                            del removed_y_k
                    

                    para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_para - learning_rate*expect_gradients, full_shape_list, shape_list)
                    
                    recorded += 1
                    
                    
                    del gradient_full
                    
                    del gradient_remaining
                    
                    del expect_gradients
                    
                    del batch_remaining_X
                    
                    del batch_remaining_Y
                    
                    if not removed_batch_empty_list[i]:
                        
                        del batch_delta_X
                        
                        del batch_delta_Y
                    
                    if k>0 or (p > 0 and k == 0):
                        del prev_para
                    
                        del curr_para
                    
                    if recorded >= length:
                        use_standard_way = False
                
                
            else:
                
                '''use l-bfgs algorithm to evaluate the gradients'''
                
                gradient_dual = None
    
                if not removed_batch_empty_list[i]:
                
                    init_model(model, para)
                    
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                    
                    gradient_dual = model.get_all_gradient()
                    
                with torch.no_grad():
                
                    vec_para_diff = torch.t((get_all_vectorized_parameters1(para) - para_list_GPU_tensor[cached_id]))
                    
                    
                    if (i-curr_init_epochs)/period >= 1:
                        if (i-curr_init_epochs) % period == 1:
                            zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, is_GPU, device)
                            
                            mat = np.linalg.inv(mat_prime.cpu().numpy())
                            mat = torch.from_numpy(mat)
                            if is_GPU:
                                
                                
                                mat = mat.to(device)
                            
                    
                        hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff, is_GPU, device)
                        
                    else:
                        
                        hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period, is_GPU, device)
                    exp_gradient, exp_param = None, None
                    
                    if gradient_dual is not None:
                        is_positive, final_gradient_list = compute_grad_final3(get_all_vectorized_parameters1(para), torch.t(hessian_para_prod), get_all_vectorized_parameters1(gradient_dual), grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, is_GPU, device)
                        
                    else:
                        is_positive, final_gradient_list = compute_grad_final3(get_all_vectorized_parameters1(para), torch.t(hessian_para_prod), None, grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, is_GPU, device)
                    
                    
                    vec_para = update_para_final2(para, final_gradient_list, learning_rate, regularization_coeff, exp_gradient, exp_param)
                    
                    
                    para = get_devectorized_parameters(vec_para, full_shape_list, shape_list)
                    
            i = i + 1
            
            j += batch_size
            
            
            cached_id += 1
            
            if cached_id%cached_size == 0:
                
                GPU_tensor_end_id = (batch_id + 1)*cached_size
                
                if GPU_tensor_end_id > para_list_all_epochs_tensor.shape[0]:
                    GPU_tensor_end_id = para_list_all_epochs_tensor.shape[0] 
#                 print("end_tensor_id::", GPU_tensor_end_id)
                
                para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(para_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(gradient_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                batch_id += 1
                
                cached_id = 0
                
            
            id_start = id_end
                        
            
#     print('overhead::', overhead)
#     
#     print('overhead2::', overhead2)
#     
#     print('overhead3::', overhead3)
#     
#     print('overhead4::', overhead4)
#     
#     print('overhead5::', overhead5)
    
        
    init_model(model, para)
        
    return model



def model_update_del(args, method, lr_lists):
    
    
    model_name = args.model
    
    git_ignore_folder = args.repo
    
    dataset_name = args.dataset
    
    num_epochs = args.epochs
    
    batch_size = args.bz
    
    is_GPU = args.GPU
    
#     args.ratio


    regularization_coeff = args.wd
    
    if not is_GPU:
        device = torch.device("cpu")
    else:    
        GPU_ID = int(args.GID)
        device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    
    model_class = getattr(sys.modules[__name__], model_name)
    
    
    data_preparer = Data_preparer()
    
    
    dataset_train = torch.load(git_ignore_folder + "dataset_train")
    
    dataset_test = torch.load(git_ignore_folder + "dataset_test")
    
    
    delta_data_ids = torch.load(git_ignore_folder + "delta_data_ids")
    
    learning_rate_all_epochs = torch.load(git_ignore_folder + 'learning_rate_all_epochs')
    
#     generate_random_id_add(git_ignore_folder, dataset_train, num_epochs)
    
    random_ids_all_epochs = torch.load(git_ignore_folder + 'random_ids_multi_epochs')
    
    
    sorted_ids_multi_epochs = torch.load(git_ignore_folder + 'sorted_ids_multi_epochs')
    
    mini_batch_num = int((len(dataset_train) - 1)/batch_size) + 1
    
    
    para_list_all_epochs = torch.load(git_ignore_folder + 'para_list_all_epochs')
    
    gradient_list_all_epochs = torch.load(git_ignore_folder + 'gradient_list_all_epochs')
    
#     data_train_loader = torch.load(git_ignore_folder + "data_train_loader")
#     
#     data_test_loader = torch.load(git_ignore_folder + "data_test_loader")
    
    
    

    dim = [len(dataset_train), len(dataset_train[0][0])]

    
    origin_train_data_size = len(dataset_train)
    
    
    
    
    num_class = get_data_class_num_by_name(data_preparer, dataset_name)
    
    hyper_para_function=getattr(Data_preparer, "get_hyperparameters_" + dataset_name)
    
    model = model_class(dim[1], num_class)
    
    
    init_para_list = list(torch.load(git_ignore_folder + 'init_para'))
    
    
    init_model(model,init_para_list)
    
    print('data dimension::',dim)
    
    if is_GPU:
        model.to(device)
    
#     init_model_params = list(model.parameters())
    
    
    criterion, optimizer = hyper_para_function(data_preparer, model.parameters(), lr_lists[0], regularization_coeff)
    
#     hyper_params = [criterion, optimizer]
    
    if method == baseline_method:
        
        
        t1 = time.time()
                
        updated_model = model_update_standard_lib(num_epochs, dataset_train, model, random_ids_all_epochs, sorted_ids_multi_epochs, delta_data_ids, batch_size, learning_rate_all_epochs, criterion, optimizer, is_GPU, device)
    
        t2 = time.time()
            
        process = psutil.Process(os.getpid())

        print('memory usage::', process.memory_info().rss)
        
        
        print('time_baseline::', t2 - t1)
    
        origin_model = torch.load(git_ignore_folder + 'origin_model')
        
        compute_model_para_diff(list(origin_model.parameters()), list(updated_model.parameters()))
    
    
        torch.save(updated_model, git_ignore_folder + 'model_base_line')
        
#         torch.save(exp_para_list, git_ignore_folder + 'exp_para_list')
#         
#         torch.save(exp_grad_list, git_ignore_folder + 'exp_grad_list')    
        
    
    else:
        if method == deltagrad_method:
            
#             added_random_ids_multi_super_iteration = torch.load(git_ignore_folder + 'added_random_ids_multi_super_iteration')
#             
#             dataset_train.data = torch.cat([dataset_train.data, X_to_add], 0)
#     
#             dataset_train.labels = torch.cat([dataset_train.labels, Y_to_add], 0)
#             
#             exp_para_list = torch.load(git_ignore_folder + 'exp_para_list')
#         
#             exp_grad_list = torch.load(git_ignore_folder + 'exp_grad_list')
            
            period = args.period
            
            init_epochs = args.init
            
            m = args.m
            
            cached_size = args.cached_size
            
            grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor = cache_grad_para_history(git_ignore_folder, cached_size, is_GPU, device)
            
#             model_update_provenance_test3(period, 1, init_epochs, dataset_train, model, grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, max_epoch, 2, learning_rate_all_epochs, random_ids_multi_epochs, sorted_ids_multi_epochs, batch_size, dim, added_random_ids_multi_super_iteration, X_to_add, Y_to_add, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device)
            
            t1 = time.time()
            
            updated_model = model_update_deltagrad(num_epochs, period, 1, init_epochs, dataset_train, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, delta_data_ids, m, learning_rate_all_epochs, random_ids_all_epochs, sorted_ids_multi_epochs, batch_size, dim, criterion, optimizer, regularization_coeff, is_GPU, device)
            
#             updated_model = model_update_deltagrad(exp_para_list, exp_grad_list, period, 1, init_epochs, dataset_train, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, m, learning_rate_all_epochs, random_ids_all_epochs, batch_size, dim, added_random_ids_multi_super_iteration, criterion, optimizer, regularization_coeff, is_GPU, device)
            
            t2 = time.time()
            
            process = psutil.Process(os.getpid())
    
            print('memory usage::', process.memory_info().rss)
            
            
            print('time_deltagrad::', t2 - t1)
            
            
            model_base_line = torch.load(git_ignore_folder + 'model_base_line')
            
            compute_model_para_diff(list(model_base_line.parameters()), list(updated_model.parameters()))
            
            torch.save(updated_model, git_ignore_folder + 'model_deltagrad')    


def generate_random_id_del(git_ignore_folder, dataset_train, epochs):
    
    
#     delta_data_ids = torch.load(git_ignore_folder + 'delta_data_ids')
#         
#     torch.save(dataset_train.data[delta_data_ids], git_ignore_folder + 'X_to_add')
#     
#     torch.save(dataset_train.labels[delta_data_ids], git_ignore_folder + 'Y_to_add')
#     
#     
#     selected_rows = get_subset_training_data(dataset_train.data.shape[0], delta_data_ids)        
#     
#     dataset_train.data = dataset_train.data[selected_rows]
#     
#     dataset_train.labels = dataset_train.labels[selected_rows]
    
    generate_random_ids_list(dataset_train, epochs, git_ignore_folder)


def main_del(args, lr_lists):
    
    
    model_name = args.model
    
    git_ignore_folder = args.repo
    
    dataset_name = args.dataset
    
    num_epochs = args.epochs
    
    batch_size = args.bz
    
    is_GPU = args.GPU
    
    
    regularization_coeff = args.wd
    
    if not is_GPU:
        device = torch.device("cpu")
    else:    
        GPU_ID = int(args.GID)
        device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    
    
    model_class = getattr(sys.modules[__name__], model_name)
    
    
    dataset_train = torch.load(git_ignore_folder + "dataset_train")
    
    dataset_test = torch.load(git_ignore_folder + "dataset_test")
    
#     data_train_loader = torch.load(git_ignore_folder + "data_train_loader")
#     
#     data_test_loader = torch.load(git_ignore_folder + "data_test_loader")
    
    
    data_preparer = Data_preparer()
    
    
#     dataset_train.data = data_preparer.normalize(dataset_train.data)
#     
#     dataset_test.data = data_preparer.normalize(dataset_test.data)
#     
#     print(dataset_train.data.shape)
#     
#     
#     dataset_train, dataset_test, data_train_loader, data_test_loader = get_train_test_data_loader_by_name_lr(data_preparer, model_class, dataset_name, batch_size)
    
    dim = [len(dataset_train), len(dataset_train[0][0])]
    
    num_class = get_data_class_num_by_name(data_preparer, dataset_name)
    
    hyper_para_function=getattr(Data_preparer, "get_hyperparameters_" + dataset_name)
    
    if model_name == 'Logistic_regression':
        model = model_class(dim[1], num_class)
    else:
        model = model_class()
    
    if is_GPU:
        model.to(device)
    
    init_model_params = list(model.parameters())
    
    
    criterion, optimizer = hyper_para_function(data_preparer, model.parameters(), lr_lists[0], regularization_coeff)
    
    hyper_params = [criterion, optimizer]
    
    
    
    
#     lrs = ast.literal_eval(input)#map(float, input.strip('[]').split(','))
#     [2.0, 3.0, 4.0, 5.0]
    
#     model, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs, all_ids_list_all_epochs = model_training_skipnet(num_epochs, model, data_train_loader, data_test_loader, len(dataset_train), len(dataset_test), optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs)

# net, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs, X_theta_prod_seq, X_theta_prod_softmax_seq, random_ids_multi_epochs

    generate_random_id_del(git_ignore_folder, dataset_train, num_epochs)

    random_ids_all_epochs = torch.load(git_ignore_folder + 'random_ids_multi_epochs')
        
#     sorted_random_ids_all_epochs = torch.load(git_ignore_folder + 'sorted_ids_multi_epochs_' + str(repetition))



    t1 = time.time()
        
    model, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs = model_training_lr_test(random_ids_all_epochs, num_epochs, model, dataset_train, len(dataset_train), optimizer, criterion, batch_size, is_GPU, device, lr_lists)

    
    t2 = time.time()


#     data_train_loader = update_data_train_loader(len(dataset_train), dataset_train, random_ids_multi_epochs, batch_size)
    
    
    t3 = time.time()
    
#     capture_provenance(git_ignore_folder, data_train_loader, len(dataset_train), dim, num_epochs, num_class, batch_size, int((dim[0] - 1)/batch_size) + 1, torch.stack(random_ids_multi_epochs), X_theta_prod_softmax_seq, X_theta_prod_seq)

#     data_train_loader.batch_sampler.reset_ids()
    

#     x_sum_by_class_by_batch = compute_x_sum_by_class_by_batch(data_train_loader, len(dataset_train), batch_size, num_class, random_ids_multi_epochs)
    
    
#     data_train_loader.batch_sampler.reset_ids()
    
    t4 = time.time()
    
    
    print("training time full::", t2 - t1)
    
    print("provenance prepare time::", t4 - t3)
    
    
    
    torch.save(dataset_train, git_ignore_folder + "dataset_train")
    
    torch.save(dataset_test, git_ignore_folder + "dataset_test")
    
    
#     torch.save(x_sum_by_class_by_batch, git_ignore_folder+'x_sum_by_class')
#     torch.save(all_ids_list_all_epochs, git_ignore_folder + "all_ids_list_all_epochs")
    
#     torch.save(ids2_list_all_epochs, git_ignore_folder + "ids2_list_all_epochs")
    
#     torch.save(delta_data_ids, git_ignore_folder + "delta_data_ids")
    
    
    torch.save(gradient_list_all_epochs, git_ignore_folder + 'gradient_list_all_epochs')
    
    torch.save(para_list_all_epochs, git_ignore_folder + 'para_list_all_epochs')
    
    torch.save(learning_rate_all_epochs, git_ignore_folder + 'learning_rate_all_epochs')


#     torch.save(random_ids_multi_epochs, git_ignore_folder + 'random_ids_multi_epochs')
                  
    torch.save(num_epochs, git_ignore_folder+'epoch')    
    
    torch.save(hyper_params, git_ignore_folder + 'hyper_params')
    
    save_random_id_orders(git_ignore_folder, random_ids_all_epochs)
    
    torch.save(para_list_all_epochs[0], git_ignore_folder + 'init_para')
    
    torch.save(model, git_ignore_folder + 'origin_model')
    
    torch.save(model_class, git_ignore_folder + 'model_class')
    
#     torch.save(data_train_loader, git_ignore_folder + 'data_train_loader')
#     
#     torch.save(data_test_loader, git_ignore_folder + 'data_test_loader')
    
    torch.save(regularization_coeff, git_ignore_folder + 'beta')
    
    torch.save(dataset_name, git_ignore_folder + 'dataset_name')
    
    torch.save(batch_size, git_ignore_folder + 'batch_size')

    torch.save(device, git_ignore_folder + 'device')

    torch.save(is_GPU, git_ignore_folder + 'is_GPU')
    
#     torch.save(noise_rate, git_ignore_folder + 'noise_rate')
    
    print("here")
    
    test(model, dataset_test, batch_size, criterion, len(dataset_test), is_GPU, device)