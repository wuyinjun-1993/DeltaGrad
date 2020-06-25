'''
Created on Mar 15, 2019


'''
'''
Created on Mar 15, 2019


'''
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import psutil

try:
    from sensitivity_analysis_SGD.logistic_regression.incremental_updates_logistic_regression import *
    from data_IO.Load_data import *
except ImportError:
    from incremental_updates_logistic_regression import * 
    from Load_data import *
def get_relative_change(tensor1, tensor2):
    print('relative magnitude change::', torch.max(torch.pow(tensor1 - tensor2, 2)))


def load_svd(directory):
    u_list = []
 
#     s_list = []
     
    v_s_list = []
    
    directory = directory + 'svd_folder/'
    
    term1_len = torch.load(directory + 'len')[0]
    
    for i in range(term1_len):
        u_list.append(torch.from_numpy(np.load(directory + '/u_' + str(i) + '.npy')))
        v_s_list.append(torch.from_numpy(np.load(directory + '/v_' + str(i) + '.npy')))
    
    
    
    return u_list, v_s_list

if __name__ == '__main__':
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    print(git_ignore_folder)
    
    X = torch.load(git_ignore_folder+'noise_X')
    
    Y = torch.load(git_ignore_folder+'noise_Y')
    
    dim = X.shape
    
    sys_args = sys.argv
    
    
    alpha = torch.load(git_ignore_folder + 'alpha')
    
    beta = torch.load(git_ignore_folder + 'beta')
    random_ids_multi_super_iterations = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations')
    sorted_ids_multi_super_iterations = torch.load(git_ignore_folder + 'sorted_ids_multi_super_iterations')


    theta_list = torch.load(git_ignore_folder + 'expected_theta_list')
    
    gradient_list = torch.load(git_ignore_folder + 'expected_gradient_list')

    
#     opt = bool(int(sys.argv[3]))
    M = torch.load(git_ignore_folder + 'eigen_vectors')
    
    s = torch.load(git_ignore_folder + 'eigen_values')    
#     print(s)
#     
#     print(torch.sort(s, descending = True))
    
    M_inverse = torch.load(git_ignore_folder + 'eigen_vectors_inverse')
    
    
    M = M.type(torch.double)
    
    M_inverse = M_inverse.type(torch.double)
    
    s = s.type(torch.double)
    
    opt = bool(int(sys_args[1]))
    
    
    cut_off_epoch = torch.load(git_ignore_folder + 'cut_off_epoch')


    print('cut_off_epoch', cut_off_epoch)
    
    
#     print(X)
#     
#     print(Y)
    origin_theta_list = torch.load(git_ignore_folder + 'origin_theta_list')
        
    origin_grad_list = torch.load(git_ignore_folder + 'origin_grad_list')
        
    w_seq = torch.load(git_ignore_folder+'w_seq')
    
    b_seq = torch.load(git_ignore_folder+'b_seq')
    
#     term1 = torch.load(git_ignore_folder+'term1')
    
    term2 = torch.load(git_ignore_folder+'term2')

    batch_size = torch.load(git_ignore_folder + 'batch_size')
    
    min_batch_num_per_epoch = int((X.shape[0] - 1)/batch_size) + 1

    
#     avg_term1 = torch.mean(torch.stack(term1[-min_batch_num_per_epoch:],0), 0)
    avg_term2 = torch.mean(torch.stack(term2[-min_batch_num_per_epoch:],0), 0)

#     avg_term2 = torch.mean(term2[-min_batch_num_per_epoch:], 0)
    
    X_Y_mult = torch.load(git_ignore_folder + 'X_Y_mult')
    
    mini_batch_epoch = torch.load(git_ignore_folder + 'mini_batch_epoch')
    
#     X_product = torch.load(git_ignore_folder+'X_product')
    
#     x_sum_by_class = torch.load(git_ignore_folder+'x_sum_by_class')
    
    max_epoch = torch.load(git_ignore_folder+'epoch')
    
    print('max_epoch', max_epoch)
    
    mini_batch_epoch = torch.load(git_ignore_folder + 'mini_batch_epoch')
    
    num_class = torch.unique(Y).shape[0]
    
    delta_data_ids = torch.load(git_ignore_folder+'noise_data_ids')
    
#     delta_data_ids = torch.load(git_ignore_folder+'delta_data_ids')
    
    delta_data_ids = delta_data_ids.type(torch.LongTensor)
    
    print(delta_data_ids)
    
    u_list, v_s_list = load_svd(git_ignore_folder)
    
    
    
    if opt:
        
        cut_off_super_epochs = cut_off_epoch/min_batch_num_per_epoch
        
        curr_weight = w_seq[0:int(cut_off_super_epochs*X.shape[0])]
        
        curr_offset = b_seq[0:int(cut_off_super_epochs*X.shape[0])]


        curr_term2 = term2[0:cut_off_epoch]
        
        del w_seq, b_seq, term2
        
        
        w_seq = curr_weight
        
        b_seq = curr_offset
        
        term2 = curr_term2
    
    
    
    
    
#     delta_batch_ids = (delta_data_ids/batch_size)
#     
#     delta_batch_ids = torch.unique(delta_batch_ids.type(torch.IntTensor))
#     
#     
#     
    batch_num = int(dim[0]/batch_size)
#     
#     
#     delta_w_seq_ids = torch.tensor(list(range(w_seq.shape[0]/batch_num))).view(1,-1) + delta_batch_ids.view(-1,1)
    
    
    
    
#     delta_data_ids = torch.load(git_ignore_folder+'delta_data_ids')
    
    
    print(delta_data_ids.shape[0])
    
    update_X, selected_rows = get_subset_training_data(X, X.shape, delta_data_ids)
    
    update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
    
    
    
    
    
#     w_delta_data_ids = selected_rows.view(-1,1)+ X.shape[0]*torch.tensor(range(int(w_seq.shape[0]/X.shape[0]) + 2)).view(1,-1)
#     
# #     print(w_delta_data_ids)
#     
#     w_delta_data_ids = torch.reshape(w_delta_data_ids,[-1])
#     
#     print(w_delta_data_ids)
#     
#     print(w_seq.shape)
#     
#     sub_w_seq = w_seq[w_delta_data_ids[torch.nonzero(w_delta_data_ids < w_seq.shape[0]).view(-1)]]
#     
#     sub_b_seq = b_seq[w_delta_data_ids[torch.nonzero(w_delta_data_ids < b_seq.shape[0]).view(-1)]]
#     
#     end_pos = int(sub_w_seq.shape[0]/update_X.shape[0])*update_X.shape[0] + int((sub_w_seq.shape[0] -int(sub_w_seq.shape[0]/update_X.shape[0])*update_X.shape[0])/batch_size)*batch_size
#     
#     sub_w_seq = sub_w_seq[0:end_pos]
#     
#     sub_b_seq = sub_b_seq[0:end_pos]
    
    
    
    
    
    
    
    
    
    
    
    t1 = time.time()
    
    res3 = None
    
    if len(delta_data_ids) < (X.shape[0])/2:

#         sub_w_seq = w_seq[delta_w_seq_ids]#get_subset_parameter_list(selected_rows, delta_data_ids, w_seq, dim, 1)
#           
#         sub_b_seq = b_seq[delta_w_seq_ids]#get_subset_parameter_list(b_seq, delta_data_ids, b_seq, dim, 1)
         
#         delta_X = X[delta_data_ids]
#          
#         delta_Y = Y[delta_data_ids]
#          
#         delta_X_product = torch.bmm(delta_X.view(delta_X.shape[0], X.shape[1], 1), delta_X.view(delta_X.shape[0], 1, X.shape[1]))
#          
# #         delta_X_product = X_product[delta_data_ids]
#          
#         delta_X_Y_prod = delta_X.mul(delta_Y) 

#         delta_X_Y_prod = X_Y_mult[delta_data_ids]
        
#         sub_term_1, sub_term_2, A, B = prepare_term_1_batch2_delta(alpha, beta, term1, term2, delta_X_product, delta_X_Y_prod, w_seq, b_seq, dim, batch_num, batch_size, delta_data_ids, mini_batch_epoch, max_epoch, cut_off_epoch)
         
         
#         sub_term_1, sub_term_2, A, B, cut_off_epoch = prepare_term_1_batch2_delta2(alpha, beta, update_X, update_Y, sub_w_seq, sub_b_seq, update_X.shape, batch_num, batch_size, mini_batch_epoch, max_epoch)
#         sub_term_2 = prepare_term_2_batch2_delta(, b_seq, dim, batch_num, batch_size, mini_batch_epoch, max_epoch, delta_data_ids, cut_off_epoch)#(delta_X_Y_prod, sub_b_seq, delta_X.shape)     
          
        init_theta = Variable(initialize(update_X).theta)
        
#         res3 = compute_model_parameter_by_approx_incremental_2(term1 - sub_term_1, term2 - sub_term_2, X.shape, init_theta, max_epoch)
        
#         if not opt:

#         print(term1.shape)
#         
#         print(sub_term_1.shape)
#         
#         print(term2.shape)
#         
#         
#         print(sub_term_2.shape)

        if not opt:
#             print('here')
            res3 = compute_model_parameter_by_provenance(mini_batch_epoch, theta_list, gradient_list, X, Y, X_Y_mult, w_seq, b_seq, delta_data_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, None, term2, dim, init_theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, None, avg_term2, origin_theta_list, origin_grad_list, u_list, v_s_list)
            
        else:
            
            
            res3 = compute_model_parameter_by_provenance_2(mini_batch_epoch, [], M, M_inverse, s, X, Y, X_Y_mult, w_seq, b_seq, delta_data_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, None, term2, dim, init_theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, None, avg_term2, u_list, v_s_list)
            
#         else:
#             res3 = compute_model_parameter_by_approx_incremental_4(term1 - sub_term_1, term2 - sub_term_2, X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta)
    
#     else:
#         t2_5 = time.time()
#         
#         sub_w_seq = w_seq[selected_rows]
#         
# #         sub_w_seq = torch.index_select(w_seq, 0, selected_rows)#get_subset_parameter_list(selected_rows, delta_data_ids, w_seq, dim, 1)
#         sub_b_seq = b_seq[selected_rows]
#         
# #         sub_b_seq = torch.index_select(b_seq, 0, selected_rows)#get_subset_parameter_list(b_seq, delta_data_ids, b_seq, dim, 1)
#          
#         delta_X = X[selected_rows]
#          
#         delta_Y = Y[selected_rows]
#          
#         delta_X_product = torch.bmm(delta_X.view(delta_X.shape[0], X.shape[1], 1), delta_X.view(delta_X.shape[0], 1, X.shape[1]))
# #          
# #         delta_X_Y_prod = delta_X.mul(delta_Y) 
# 
# #         delta_X_product = X_product[selected_rows]
#          
# #         delta_X_Y_prod = delta_X.mul(delta_Y) 
# 
#         delta_X_Y_prod = X_Y_mult[selected_rows]
#         
#         
#         t2_6 = time.time()
#         
#         t2_3 = time.time()
#         
#         sub_term_1 = prepare_term_1_batch2(delta_X_product, sub_w_seq, delta_X.shape)
#          
#         sub_term_2 = prepare_term_2_batch2(delta_X_Y_prod, sub_b_seq, delta_X.shape)     
#                 
#                 
#         t2_4 = time.time()        
#         
#         init_theta = Variable(initialize(update_X).theta)
#          
# #         res3 = compute_model_parameter_by_approx_incremental_2(sub_term_1, sub_term_2, X.shape, init_theta, max_epoch)
# 
#         t2_1 = time.time()
# 
#         if opt:
#             res3 = compute_model_parameter_by_approx_incremental_3(sub_term_1, sub_term_2, X.shape, init_theta, max_epoch, cut_off_epoch)
#         else:
#             res3 = compute_model_parameter_by_approx_incremental_4(sub_term_1, sub_term_2, X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta)
#         
#         
#         t2_2 = time.time()
    
    t2 = time.time()
    
    process = psutil.Process(os.getpid())
    
    print('memory usage::', process.memory_info().rss)
    
    torch.save(res3, git_ignore_folder+'model_provenance')
    
    
    print('training_time_provenance::', t2 - t1)
    
    
#     print(t2_2 - t2_1)
#     
#     print(t2_4 - t2_3)
#     
#     print(t2_6 - t2_5)
    
    model_origin = torch.load(git_ignore_folder+'model_origin')
    
    model_standard_lib = torch.load(git_ignore_folder+'model_standard_lib')
    
    model_iteration = torch.load(git_ignore_folder+'model_iteration')
    
#     res3 = torch.load(git_ignore_folder+'model_provenance')
    
    expect_updates = model_origin - model_standard_lib
    
    real_updates = model_origin - res3
    
    
    error = torch.norm(expect_updates - real_updates)/torch.norm(model_standard_lib)
    
    print('model_origin::', model_origin.view(1,-1))
    
    print('model_standard_lib::', model_standard_lib)
    
    print('model_prov::', res3)
    
    print('absolute_error::', torch.norm(res3 - model_standard_lib))
    
    
    print('angle::', torch.dot(res3.view(-1), model_iteration.view(-1))/(torch.norm(res3.view(-1))*torch.norm(model_iteration.view(-1))))
    
    print('absolute_error2::', torch.norm(res3 - model_iteration))
    
    print('expect_updates::', torch.norm(expect_updates))
    
    print('relative_error::', error)
    
    test_X = torch.load(git_ignore_folder + 'test_X')
    
    test_Y = torch.load(git_ignore_folder + 'test_Y')
    
    print('training_accuracy::', compute_accuracy2(update_X, update_Y, res3))
    
    print('test_accuracy::', compute_accuracy2(test_X, test_Y, res3))
#     print('test_accuracy::', compute_accuracy2(test_X, test_Y, model_standard_lib))

    print(res3.shape)
    
    print(torch.nonzero(res3*model_iteration > 0).shape)
    
    get_relative_change(res3, model_iteration)
    
    
    