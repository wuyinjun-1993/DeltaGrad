'''
Created on Mar 15, 2019


'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time

import psutil
import torch
try:
    from sensitivity_analysis_SGD.logistic_regression.incremental_updates_logistic_regression import *
    from data_IO.Load_data import *
    from sensitivity_analysis_SGD.logistic_regression.evaluating_test_samples import *
except ImportError:
    from incremental_updates_logistic_regression import *
    from Load_data import * 
    from evaluating_test_samples import *

if __name__ == '__main__':
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    X = scipy.sparse.load_npz(git_ignore_folder + 'noise_X.npz')
    
    Y = torch.load(git_ignore_folder+'noise_Y')
    
    mini_batch_epoch = torch.load(git_ignore_folder + 'mini_batch_epoch')
    
    batch_size = torch.load(git_ignore_folder + 'batch_size')

#     random_ids_multi_super_iterations = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations')
    random_ids_multi_super_iterations = np.load(git_ignore_folder + 'random_ids_multi_super_iterations.npy')
    sorted_ids_multi_super_iterations = torch.load(git_ignore_folder + 'sorted_ids_multi_super_iterations')

    print(random_ids_multi_super_iterations.shape)
    
    sys_args = sys.argv
    
    
    alpha = torch.load(git_ignore_folder + 'alpha')
    
    beta = torch.load(git_ignore_folder + 'beta')
    
    
    max_epoch = torch.load(git_ignore_folder+'epoch')
    
    print(max_epoch)
    
    delta_data_ids = torch.load(git_ignore_folder+'noise_data_ids')
    
    X_Y_mult = scipy.sparse.load_npz(git_ignore_folder + 'X_Y_mult.npz')

#     delta_data_ids = torch.load(git_ignore_folder+'delta_data_ids')
    
    print(delta_data_ids[:100])
    
    print(delta_data_ids.shape)
    
    update_X, selected_rows = get_subset_training_data_sparse(X, X.shape, delta_data_ids)
#     
    
    update_Y, selected_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
#     update_X = X[selected_rows]
    
    init_theta = Variable(initialize_by_size(X.shape).theta)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
    print(X_Y_mult.shape)
    
    print(X.shape)
    t1 = time.time()
    

    
#     update_X_Y_mult = X_Y_mult[selected_rows]
    
#     update_x_sum_by_class = compute_x_sum_by_class(update_X, update_Y, num_class, update_X.shape)
#     update_X_Y_mult = update_X.mul(update_Y)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)dim, theta,  X, Y, X_sum_by_class, num_class
#     res2, total_time = compute_model_parameter_by_iteration2(update_X_Y_mult.shape, init_theta, update_X_Y_mult, max_epoch, alpha, beta, mini_batch_epoch, selected_rows, batch_size)
    
    res2, total_time, theta_list, gradient_list = compute_model_parameter_by_iteration_2_sparse(X, Y, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, X_Y_mult.shape, init_theta, X_Y_mult, max_epoch, alpha, beta, mini_batch_epoch, selected_rows, batch_size)

    
    t2 = time.time()
    
    process = psutil.Process(os.getpid())
    
    print('memory usage::', process.memory_info().rss)
    
    model_standard_lib = torch.load(git_ignore_folder+'model_standard_lib')
    
    
    torch.save(res2, git_ignore_folder+'model_iteration')
    
    torch.save(theta_list, git_ignore_folder + 'expected_theta_list')
    
    torch.save(gradient_list, git_ignore_folder + 'expected_gradient_list')
    
    
    print('training_time_iteration::', t2 - t1)
    
    print(torch.norm(model_standard_lib - res2))
    
    
    test_X = scipy.sparse.load_npz(git_ignore_folder + 'test_X.npz')

    test_Y = torch.load(git_ignore_folder + 'test_Y')
    
    print('training_accuracy::', compute_accuracy2_sparse(update_X, update_Y, res2))
    
    print('test_accuracy::', compute_accuracy2_sparse(test_X, test_Y, res2))
    
    print(res2)
    
    
    