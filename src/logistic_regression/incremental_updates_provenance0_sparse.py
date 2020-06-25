'''
Created on Mar 15, 2019


'''
# from sensitivity_analysis.logistic_regression.Logistic_regression import initialize_by_size
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


def check_and_convert_to_sparse_tensor(res):
    non_zero_entries = torch.nonzero(res)
    final_res_values = res[res != 0]
        
    final_res = coo_matrix((final_res_values.detach().numpy(), (non_zero_entries[:,0].detach().numpy(), non_zero_entries[:,1].detach().numpy())), shape=list(res.shape))
        
    return final_res.tocsr()


def compute_X_weight_product(weights, offsets, X, X_Y_mult):
    
    end = False
    
    X_weight_prod = []
    
#     X_offset_prod = []
    
    print(random_ids_multi_super_iterations.shape[0])
    
    print('weights.shape::', weights.shape[0])
    
    print('x shape::', dim[0])
    
    for k in range(random_ids_multi_super_iterations.shape[0]):

        random_ids = random_ids_multi_super_iterations[k]
        
        print(k)
        
        super_iter_id = k
        
#         if k > cut_off_super_iteration:
#             super_iter_id = cut_off_super_iteration
        
        end_id_super_iteration = (super_iter_id + 1)*dim[0]
        
        
        if end_id_super_iteration >= weights.shape[0]:
            end_id_super_iteration = weights.shape[0]
            end = True
        
        t1 = time.time()    
        
        weights_this_super_iteration = weights[super_iter_id*dim[0]:end_id_super_iteration]
        
        offsets_this_super_iteration = offsets[super_iter_id*dim[0]:end_id_super_iteration]
        
        
#         print(X.shape)
#         
#         print(weights_this_super_iteration.view(-1).numpy().shape)
        
        curr_X_weight_prod = scipy.sparse.csr_matrix(np.multiply(X_Y_mult, weights_this_super_iteration))

        print(X_Y_mult.shape)
        
        print(offsets_this_super_iteration.shape)
#         curr_X_offset_prod = scipy.sparse.csr_matrix(np.multiply(X_Y_mult, offsets_this_super_iteration))
        
        X_weight_prod.append(curr_X_weight_prod)
        
#         X_offset_prod.append(curr_X_offset_prod)
        
        
        if end == True:
            break
        
    del X, X_Y_mult
        
#     return X_offset_prod
    return X_weight_prod
        

if __name__ == '__main__':
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    print(git_ignore_folder)
    
#     X = torch.load(git_ignore_folder+'noise_X')
    X = scipy.sparse.load_npz(git_ignore_folder + 'noise_X.npz')
    
    
    Y = torch.load(git_ignore_folder+'noise_Y')
    
    dim = X.shape
    
    sys_args = sys.argv
    
    sorted_ids_multi_super_iterations = torch.load(git_ignore_folder + 'sorted_ids_multi_super_iterations')

    batch_size = torch.load(git_ignore_folder + 'batch_size').item()

    
    min_batch_num_per_epoch = int((X.shape[0] - 1)/batch_size) + 1

    
    
    
    alpha = torch.load(git_ignore_folder + 'alpha')
    
    beta = torch.load(git_ignore_folder + 'beta')
    
    
    random_ids_multi_super_iterations = np.load(git_ignore_folder + 'random_ids_multi_super_iterations.npy')


#     theta_list = torch.load(git_ignore_folder + 'expected_theta_list')
#     
#     gradient_list = torch.load(git_ignore_folder + 'expected_gradient_list')

    
#     opt = bool(int(sys.argv[3]))
#     M = torch.load(git_ignore_folder + 'eigen_vectors')
#     
#     s = torch.load(git_ignore_folder + 'eigen_values')    
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
#     
#     opt = bool(int(sys_args[1]))
    
    
    cut_off_epoch = torch.load(git_ignore_folder + 'cut_off_epoch')

#     term1 = load_term1(git_ignore_folder + 'term1_folder', 'term1')
#     
#     term2 = load_term1(git_ignore_folder + 'term2_folder', 'term2')
#     
#     avg_term1 = None
#     
    
#     
#     avg_term1 = scipy.sparse.csr_matrix(avg_term1/min_batch_num_per_epoch)
#     
    avg_term2 =  None#scipy.sparse.load_npz(git_ignore_folder + 'avg_term2.npz')    
#     avg_term1 = scipy.sparse.vstack(term1)[-min_batch_num_per_epoch:].mean(axis = 0)
#     avg_term2 = scipy.sparse.vstack(term2)[-min_batch_num_per_epoch:].mean(axis = 0)
    print('cut_off_epoch', cut_off_epoch)
    
#     print(X)
#     
#     print(Y)
    
    w_seq = torch.load(git_ignore_folder+'w_seq').view(-1,1).numpy()
    
    b_seq = torch.load(git_ignore_folder+'b_seq').view(-1,1).numpy()
    
#     term1 = torch.load(git_ignore_folder+'term1')
    
#     term2 = torch.load(git_ignore_folder+'term2')

#     min_batch_num_per_epoch = int((X.shape[0] - 1)/batch_size) + 1

    
#     avg_term1 = torch.mean(torch.stack(term1[-min_batch_num_per_epoch:-1],0), 0)
    
#     avg_term2 = torch.mean(term2[-min_batch_num_per_epoch:-1], 0)
    
#     avg_term2 = check_and_convert_to_sparse_tensor(avg_term2)
    
#     X_Y_mult = torch.load(git_ignore_folder + 'X_Y_mult')
    
    X_Y_mult = scipy.sparse.load_npz(git_ignore_folder + 'X_Y_mult.npz')
    
#     mini_batch_epoch = torch.load(git_ignore_folder + 'mini_batch_epoch')
    
#     X_product = torch.load(git_ignore_folder+'X_product')
    
#     x_sum_by_class = torch.load(git_ignore_folder+'x_sum_by_class')
    
#     max_epoch = torch.load(git_ignore_folder+'epoch')
    
    mini_batch_epoch = torch.load(git_ignore_folder + 'mini_batch_epoch')
    
    num_class = torch.unique(Y).shape[0]
    
    delta_data_ids = torch.load(git_ignore_folder+'noise_data_ids')
    
#     delta_data_ids = torch.load(git_ignore_folder+'delta_data_ids')
    
    delta_data_ids = delta_data_ids.type(torch.LongTensor)
    
    print(delta_data_ids)
    
    batch_num = int(dim[0]/batch_size)

    print(delta_data_ids.shape[0])
    
    update_X, selected_rows = get_subset_training_data_sparse(X, X.shape, delta_data_ids)
    
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
    
    
    
    
    
    
#     theta_list = torch.load(git_ignore_folder + 'expected_theta_list')
#     
#     gradient_list = torch.load(git_ignore_folder + 'expected_gradient_list')
    
#     X_w_prod = compute_X_weight_product(w_seq, b_seq,  X.todense(), X_Y_mult.todense())
#     
#     torch.save(X_w_prod, git_ignore_folder + 'X_w_prod')
#     X_w_prod = torch.load(git_ignore_folder + 'X_w_prod')
#     time.sleep(5)
    init_theta = Variable(initialize_by_size(update_X.shape).theta)


#     X_cat_X_Y_mult = scipy.sparse.hstack([X, X_Y_mult], format = 'csr')
#     X = X.tocsc()
#     
#     X_Y_mult = X_Y_mult.tocsc()
    print(X.shape)
    
    print(X_Y_mult.shape)
    
#     print(X_cat_X_Y_mult.shape)

    
    t1 = time.time()
    
    res3 = None
    
#     if len(delta_data_ids) < (X.shape[0])/2:

    
#     res3 = compute_model_parameter_by_provenance_sparse(theta_list, gradient_list, X, Y, X_Y_mult, w_seq, b_seq, delta_data_ids, random_ids_multi_super_iterations, term1, term2, dim, init_theta, mini_batch_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, avg_term1, avg_term2)
    res3 = compute_model_parameter_by_provenance_sparse7([], [], [], [], X, Y, X_Y_mult, w_seq, b_seq, s_rows, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, dim, init_theta, mini_batch_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, None, avg_term2)
#     res3 = compute_model_parameter_by_provenance_sparse4([], [], [], X_cat_X_Y_mult, X, Y, X_Y_mult, w_seq, b_seq, s_rows, random_ids_multi_super_iterations, dim, init_theta, mini_batch_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, term2, avg_term2)
#     res3 = compute_model_parameter_by_provenance_sparse3(X_w_prod, [], [], [], X, Y, X_Y_mult, w_seq, b_seq, s_rows, random_ids_multi_super_iterations, dim, init_theta, mini_batch_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, term2, avg_term2)
            
    t2 = time.time()
    
    process = psutil.Process(os.getpid())
    
    print('memory usage::', process.memory_info().rss)
    
    torch.save(res3, git_ignore_folder+'model_provenance')
    
    
    print('training_time_provenance::', t2 - t1)
    
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
    
    
    print('angle::', torch.dot(model_provenance.view(-1), model_iteration.view(-1))/(torch.norm(model_provenance.view(-1))*torch.norm(model_iteration.view(-1))))
    
    print('absolute_error2::', torch.norm(model_provenance - model_iteration))
    
    print('expect_updates::', torch.norm(expect_updates))
    
    print('relative_error::', error)
    
#     test_X = torch.load(git_ignore_folder + 'test_X')
    test_X = scipy.sparse.load_npz(git_ignore_folder + 'test_X.npz')

    
    test_Y = torch.load(git_ignore_folder + 'test_Y')
    
    print('training_accuracy::', compute_accuracy2_sparse(update_X, update_Y, res3))
    
    print('test_accuracy::', compute_accuracy2_sparse(test_X, test_Y, res3))
    
    print(res3.shape)
    
    print(torch.nonzero(res3*model_iteration >= 0).shape)
    
#     print(torch.nonzero(torch.nonzero((res3 == 0)) == torch.nonzero((model_iteration == 0))).shape)
    
    get_relative_change(res3, model_iteration)
    
    
    