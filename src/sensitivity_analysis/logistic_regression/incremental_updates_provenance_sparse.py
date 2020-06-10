'''
Created on Mar 15, 2019

'''
'''
Created on Mar 15, 2019

'''
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import psutil

try:
    from sensitivity_analysis.logistic_regression.incremental_updates_logistic_regression import *
    from data_IO.Load_data import *
except ImportError:
    from incremental_updates_logistic_regression import * 
    from Load_data import *  
      
def get_relative_change(tensor1, tensor2):
    print('relative magnitude change::', torch.max(torch.pow(tensor1 - tensor2, 2)/torch.pow(tensor1, 2)))


def convert_coo_matrix2_dense_tensor(Y_coo):
    
#     indices = np.vstack((Y_coo.row, Y_coo.col))
#     values = Y_coo.data
#     
#     i = torch.LongTensor(indices)
#     v = torch.DoubleTensor(values)
#     shape = Y_coo.shape
    
#     print(Y_coo)
    Y = torch.from_numpy(Y_coo.todense()).type(torch.DoubleTensor)
#     Y = torch.sparse.DoubleTensor(i, v, torch.Size(shape)).to_dense()
    
    return Y


def check_and_convert_to_sparse_tensor(res):
    non_zero_entries = torch.nonzero(res)
    final_res_values = res[res != 0]
        
    final_res = coo_matrix((final_res_values.detach().numpy(), (non_zero_entries[:,0].detach().numpy(), non_zero_entries[:,1].detach().numpy())), shape=list(res.shape))
        
    return final_res.tocsr()

if __name__ == '__main__':
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    print(git_ignore_folder)
    
    Y = torch.load(git_ignore_folder+'noise_Y')
    
    sys_args = sys.argv
        
    X = scipy.sparse.load_npz(git_ignore_folder + 'noise_X.npz')
#     X = convert_coo_matrix2_dense_tensor(sparse_X)
    
    
    alpha = torch.load(git_ignore_folder + 'alpha')
    
    beta = torch.load(git_ignore_folder + 'beta')
    
    
#     opt = bool(int(sys_args[1]))
    
    
    cut_off_epoch = torch.load(git_ignore_folder + 'cut_off_epoch')


    print('cut_off_epoch', cut_off_epoch)
    
#     print(X)
#     
#     print(Y)
    
#     M = torch.load(git_ignore_folder + 'eigen_vectors')
#     
#     s = torch.load(git_ignore_folder + 'eigen_values')
#     
# #     expected_A = torch.load(git_ignore_folder + 'expected_A')
#     
# #     print(s)
# #     
# #     print(torch.sort(s, descending = True))
#     
#     M_inverse = torch.load(git_ignore_folder + 'eigen_vectors_inverse')
#     
#     
#     M = M.type(torch.double)
#     
#     M_inverse = M_inverse.type(torch.double)
#     
#     s = s.type(torch.double)
    
    
    w_seq = torch.load(git_ignore_folder+'w_seq')
    
    b_seq = torch.load(git_ignore_folder+'b_seq')

    term1 = load_term1(git_ignore_folder + 'term1_folder')
    
    term2 = torch.load(git_ignore_folder+'term2')
    
    term2 = check_and_convert_to_sparse_tensor(term2)
    
    X_Y_mult = scipy.sparse.load_npz(git_ignore_folder + 'X_Y_mult.npz')
    
#     X_product = torch.load(git_ignore_folder+'X_product')
    
#     x_sum_by_class = torch.load(git_ignore_folder+'x_sum_by_class')
    
    max_epoch = torch.load(git_ignore_folder+'epoch')
    
    num_class = torch.unique(Y).shape[0]
    
    delta_data_ids = torch.load(git_ignore_folder+'noise_data_ids')
    
#     delta_data_ids = torch.load(git_ignore_folder+'delta_data_ids')
    
    
    print(delta_data_ids.shape[0])
    
    update_X, selected_rows = get_subset_training_data_sparse(X, X.shape, delta_data_ids)
    
    update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
    delta_X_Y_prod = X_Y_mult[delta_data_ids]

    t1 = time.time()
    
    res3 = None
    
#     if len(delta_data_ids) < (X.shape[0])/2:

    sub_w_seq = w_seq[delta_data_ids]#get_subset_parameter_list(selected_rows, delta_data_ids, w_seq, dim, 1)
      
    sub_b_seq = b_seq[delta_data_ids]#get_subset_parameter_list(b_seq, delta_data_ids, b_seq, dim, 1)
     
    delta_X = X[delta_data_ids]
     
    delta_Y = Y[delta_data_ids]
     
#         delta_X_product = torch.bmm(delta_X.view(delta_X.shape[0], X.shape[1], 1), delta_X.view(delta_X.shape[0], 1, X.shape[1]))
     
#         delta_X_product = X_product[delta_data_ids]
     
#         delta_X_Y_prod = delta_X.mul(delta_Y) 
    t3 = time.time()
    sub_term_2 = prepare_term_2_batch2_sparse(delta_X_Y_prod, sub_b_seq, delta_X.shape)
    t4 = time.time()
#         sub_term_1 = prepare_term_1_batch2(delta_X_product, sub_w_seq, delta_X.shape)
     
    init_theta = Variable(initialize_by_size(delta_X.get_shape()).theta)
    
#         res3 = compute_model_parameter_by_approx_incremental_2(term1 - sub_term_1, term2 - sub_term_2, X.shape, init_theta, max_epoch)
    
#         if not opt:
     

#             res3 = compute_model_parameter_by_approx_incremental_3(term1 - sub_term_1, term2 - sub_term_2, update_X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta)
#             sub_term_1 = prepare_term_1_batch2(delta_X_product, sub_w_seq, delta_X.shape)
        
    res3 = compute_model_parameter_by_approx_incremental_3_2_sparse(term1, delta_X, term2 - sub_term_2, [X.get_shape()[0] - delta_X.get_shape()[0], X.get_shape()[1]], init_theta, max_epoch, cut_off_epoch, alpha, beta, sub_w_seq)
#         else:

#             sub_term_1 = prepare_term_1_batch2(delta_X_product, sub_w_seq, delta_X.shape)
                        
#             res3 = compute_model_parameter_by_approx_incremental_4(s, M, M_inverse, expected_A, term1 - sub_term_1, term2 - sub_term_2, update_X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta)
            
#             res3 = compute_model_parameter_by_approx_incremental_4_2(s, M, M_inverse, term1, delta_X, term2 - sub_term_2, update_X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta, sub_w_seq)
    
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
# #         if opt:
#         res3 = compute_model_parameter_by_approx_incremental_3(sub_term_1, sub_term_2, update_X.shape, init_theta, max_epoch, cut_off_epoch)
# #         else:
# #             res3 = compute_model_parameter_by_approx_incremental_4(sub_term_1, sub_term_2, update_X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta)
#         
#         
#         t2_2 = time.time()
    
    t2 = time.time()
    process = psutil.Process(os.getpid())
    
    print('memory usage::', process.memory_info().rss)
    
    torch.save(res3, git_ignore_folder+'model_provenance')
    
    
    print('training_time_provenance::', t2 - t1)
    
    print('sub_term2 time::', t4 - t3)
#     print(t2_2 - t2_1)
#     
#     print(t2_4 - t2_3)
#     
#     print(t2_6 - t2_5)
    
    model_origin = torch.load(git_ignore_folder+'model_origin')
    
    model_standard_lib = torch.load(git_ignore_folder+'model_standard_lib')
    
    model_iteration = torch.load(git_ignore_folder+'model_iteration')
    
    model_provenance = torch.load(git_ignore_folder+'model_provenance')
    
    expect_updates = model_origin - model_standard_lib
    
    real_updates = model_origin - model_provenance
    
    
    error = torch.norm(expect_updates - real_updates)/torch.norm(model_standard_lib)
    
    print('model_origin::', model_origin.view(1,-1))
    
    print('model_standard_lib::', model_standard_lib)
    
    print('model_prov::', model_provenance)
    
    print('absolute_error::', torch.norm(model_provenance - model_standard_lib))
    
    print('absolute_error2::', torch.norm(model_provenance - model_iteration))
    
    print('expect_updates::', torch.norm(expect_updates))
    
    print('angle::', torch.dot(model_provenance.view(-1), model_standard_lib.view(-1))/(torch.norm(model_provenance.view(-1))*torch.norm(model_standard_lib.view(-1))))
    
    print('relative_error::', error)
    
#     test_X = torch.load(git_ignore_folder + 'test_X')

    test_X = scipy.sparse.load_npz(git_ignore_folder + 'test_X.npz')
    
    test_Y = torch.load(git_ignore_folder + 'test_Y')
    
    print('training_accuracy::', compute_accuracy2_sparse(update_X, update_Y, res3))
    
    print('test_accuracy::', compute_accuracy2_sparse(test_X, test_Y, res3))
    
    print(res3.shape)
    
    print(torch.nonzero(res3*model_iteration > 0).shape)
    
    get_relative_change(res3, model_iteration)
    
    
    
    
        
    
    