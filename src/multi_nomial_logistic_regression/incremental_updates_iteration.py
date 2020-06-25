'''
Created on Mar 15, 2019


'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import torch
try:
    from sensitivity_analysis_SGD.multi_nomial_logistic_regression.incremental_updates_logistic_regression_multi_dim import *
except ImportError:
    from incremental_updates_logistic_regression_multi_dim import *

if __name__ == '__main__':
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    X = torch.load(git_ignore_folder + 'noise_X')
        
    Y = torch.load(git_ignore_folder + 'noise_Y')
    
    sys_args = sys.argv
    
    alpha = torch.load(git_ignore_folder + 'alpha')
    
    beta = torch.load(git_ignore_folder + 'beta')
    
    max_epoch = torch.load(git_ignore_folder+'epoch')
    
    num_class = torch.unique(Y).shape[0]
    
    
    batch_size = torch.load(git_ignore_folder + 'batch_size')
#     delta_data_ids = torch.load(git_ignore_folder+'delta_data_ids')
    
    delta_data_ids = torch.load(git_ignore_folder+'noise_data_ids')
    
    print(delta_data_ids[:100])
    
    print(delta_data_ids.shape)
    
    print(max_epoch)
    
    update_X, selected_rows = get_subset_training_data(X, X.shape, delta_data_ids)
    
    update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
    print(update_X.shape)
    
    init_theta = Variable(initialize(update_X, num_class).theta)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
    batch_X_list, batch_Y_list = get_id_mappings_per_batch(X, Y, selected_rows, batch_size)
    
    
    t1 = time.time()
    
    x_sum_by_class_list = compute_x_sum_by_class_by_batch1(batch_X_list, batch_Y_list, num_class)#(update_X, update_Y, num_class, update_X.shape)
    #     update_X_Y_mult = update_X.mul(update_Y)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)dim, theta,  X, Y, X_sum_by_class, num_class
    res2, total_time = compute_model_parameter_by_iteration(X.shape, init_theta, update_X, update_Y, x_sum_by_class_list, num_class, max_epoch, alpha, beta, batch_X_list, batch_Y_list)

    
    t2 = time.time()
    
    
    torch.save(res2, git_ignore_folder+'model_iteration')
    
    model_standard_lib = torch.load(git_ignore_folder + 'model_standard_lib')
    
    torch.save(curr_selected_data_ids_list, git_ignore_folder + 'selected_data_ids_list')
    
    
    print(res2 - model_standard_lib)
    
    
    print('training_time_iteration::', t2 - t1)
    
    print(torch.norm(res2 - model_standard_lib))
    
    test_X = torch.load(git_ignore_folder+'test_X')
    
    test_Y = torch.load(git_ignore_folder+'test_Y')
    
    print('training_accuracy::', compute_accuracy2(update_X, update_Y.type(torch.DoubleTensor), res2))
    
    print('test_accuracy::', compute_accuracy2(test_X, test_Y, res2))
    
    