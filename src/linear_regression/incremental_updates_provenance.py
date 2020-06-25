'''
Created on Mar 15, 2019


'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time


import torch
try:
    from sensitivity_analysis_SGD.linear_regression.incremental_updates_linear_regression import *
    from data_IO.Load_data import *
    from sensitivity_analysis_SGD.linear_regression.evaluating_test_samples import *
except ImportError:
    from incremental_updates_linear_regression import *
    from Load_data import * 
    from evaluating_test_samples import *

import psutil


def load_svd(directory):
    u_list = []
 
#     s_list = []
     
    v_s_list = []
    
    directory = directory + 'svd_folder/'
    
    term1_len = torch.load(directory + 'len')[0]
    
    for i in range(term1_len):
        u_list.append(torch.from_numpy(np.load(directory + '/u_' + str(i) + '.npy')))
        v_s_list.append(torch.from_numpy(np.load(directory + '/v_' + str(i) + '.npy')))
    
    print('len::', term1_len)
    
    return u_list, v_s_list


if __name__ == '__main__':
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    X = torch.load(git_ignore_folder+'noise_X')
    
    Y = torch.load(git_ignore_folder+'noise_Y')
    
#     mini_batch_epoch = torch.load(git_ignore_folder + 'mini_batch_epoch')
    theta_list = torch.load(git_ignore_folder + 'theta_list')

    sys_args = sys.argv

    opt = bool(int(sys_args[1]))

#     batch_size = int(sys_args[1])

    batch_size = torch.load(git_ignore_folder + 'batch_size')
    
    alpha = torch.load(git_ignore_folder + 'alpha')
    
    beta = torch.load(git_ignore_folder + 'beta')
    
    
    max_epoch = torch.load(git_ignore_folder+'epoch')

    u_list = None

    v_s_list = None
    
    X_Y_prod_list = None
    
    X_prod_list = None
    
    if not opt:
        
        
        if X.shape[1] >= min_feature_num:
            u_list, v_s_list = load_svd(git_ignore_folder)
#             u_list = torch.load(git_ignore_folder + 'u_list')
#      
#             v_s_list = torch.load(git_ignore_folder + 'v_s_list')
            
        else:
            X_prod_list = torch.load(git_ignore_folder + 'X_prod_list')
        
#         u_list, v_s_list = load_svd(git_ignore_folder)
        
        
        
        X_Y_prod_list = torch.load(git_ignore_folder+'X_Y_prod_list')
    
        
    else:
        X_prod = torch.load(git_ignore_folder + 'X_prod')
        X_Y_mult = torch.load(git_ignore_folder+'X_Y_mult')
    
    M = torch.load(git_ignore_folder + 'eigen_vectors')
    
    M_inverse = torch.load(git_ignore_folder + 'eigen_vectors_inverse')
    
    s = torch.load(git_ignore_folder + 'eigen_values')
    
    random_ids_multi_super_iterations = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations')
    sorted_ids_multi_super_iterations = torch.load(git_ignore_folder + 'sorted_ids_multi_super_iterations')

    
    print(max_epoch)
    
    
    print('alpha::', alpha)
    
    print('beta::', beta)
    
    
    delta_data_ids = torch.load(git_ignore_folder+'noise_data_ids')
    
#     X_Y_mult = torch.load(git_ignore_folder+'X_Y_mult')

#     delta_data_ids = torch.load(git_ignore_folder+'delta_data_ids')
    
    print(delta_data_ids[:10])
    
    print(delta_data_ids.shape)
    
    update_X, selected_rows = get_subset_training_data(X, X.shape, delta_data_ids)
#     
    update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
    init_theta = Variable(initialize(update_X.shape, update_Y.shape[1]).theta)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
    
#     batch_X_list, batch_Y_list, batch_X_prod_list, batch_X_Y_mult_list = get_id_mappings_per_batch(X, Y, selected_rows, batch_size)
#     update_X_Y_mult = X_Y_mult[selected_rows]
    
#     update_X_product = torch.mm(torch.t(update_X), update_X)
#     X_product = torch.mm(torch.t(X), X)
    
    
#     X_Y_prod = torch.mm(torch.t(X), Y)

    delta_X = X[delta_data_ids]
    
    delta_Y = Y[delta_data_ids]


    M_inverse_times_theta = torch.mm(M_inverse, init_theta)

    t1 = time.time()
    
    
#     update_x_sum_by_class = compute_x_sum_by_class(update_X, update_Y, num_class, update_X.shape)
#     update_X_Y_mult = update_X.mul(update_Y)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)dim, theta,  X, Y, X_sum_by_class, num_class
#     res2, total_time = compute_model_parameter_by_iteration2(update_X_Y_mult.shape, init_theta, update_X_Y_mult, max_epoch, alpha, beta, mini_batch_epoch, selected_rows, batch_size)
    if opt:
#         res2 = linear_regression_provenance(X_prod - torch.mm(torch.t(delta_X), delta_X), X_Y_mult - torch.mm(torch.t(delta_X), delta_Y), update_X.shape, init_theta, max_epoch, alpha, beta)
#         res2, expected_s, exp_res1, exp_res2 = linear_regression_provenance_opt(s, M, M_inverse, X_prod,X, X_prod - torch.mm(torch.t(delta_X), delta_X), X_Y_mult - torch.mm(torch.t(delta_X), delta_Y), update_X.shape, init_theta, max_epoch, alpha, beta)
        
        res2 = linear_regression_provenance_opt2(s, M, M_inverse, X_prod, X, delta_X, X_Y_mult - torch.mm(torch.t(delta_X), delta_Y), update_X.shape, init_theta, max_epoch, alpha, beta, M_inverse_times_theta)
    else:
        res2, total_time = compute_model_parameter_by_iteration3(theta_list, delta_data_ids, X_prod_list, X_Y_prod_list, X, Y, X.shape, init_theta, max_epoch, alpha, beta, batch_size, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, u_list, v_s_list)

    
    t2 = time.time()
    
    model_standard_lib = torch.load(git_ignore_folder+'model_standard_lib')
    
    
    process = psutil.Process(os.getpid())
    
    print('memory usage::', process.memory_info().rss)
#     model_provenance = torch.load(git_ignore_folder + 'model_provenance')
    
    torch.save(res2, git_ignore_folder+'model_provenance')
    
    
    print('training_time_provenance::', t2 - t1)
    
    print('absolute_error::', torch.norm(model_standard_lib - res2))
    
    
    test_X = torch.load(git_ignore_folder+'test_X')
    
    test_Y = torch.load(git_ignore_folder+'test_Y')
    
    print('training_accuracy::', compute_accuracy2(update_X, update_Y, res2))
    
    print('test_accuracy::', compute_accuracy2(test_X, test_Y, res2))
    
#     print(res2)
    print('angle::', torch.dot(res2.view(-1), model_standard_lib.view(-1))/(torch.norm(res2.view(-1))*torch.norm(model_standard_lib.view(-1))))

    print(res2.shape)
    
    print(torch.nonzero(res2*model_standard_lib >= 0).shape)
    
    get_relative_change(res2, model_standard_lib)
    
#     print(model_standard_lib)
    
    
    