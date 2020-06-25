'''
Created on Mar 15, 2019


'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')

try:
    from sensitivity_analysis_SGD.multi_nomial_logistic_regression.incremental_updates_logistic_regression_multi_dim import *
except ImportError:
    from incremental_updates_logistic_regression_multi_dim import *
    

import torch
 
import psutil

if __name__ == '__main__':
    
    
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    X = torch.load(git_ignore_folder + 'noise_X')
        
    Y = torch.load(git_ignore_folder + 'noise_Y')
    
    sys_args = sys.argv

    
    alpha = torch.load(git_ignore_folder + 'alpha')
    
    beta = torch.load(git_ignore_folder + 'beta')
    
    random_ids_multi_super_iterations = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations')

    batch_size = torch.load(git_ignore_folder + 'batch_size')
    
#     delta_data_ids = random_generate_subset_ids(X.shape, delta_num)
#      
#     print(delta_data_ids)
#      
#     torch.save(delta_data_ids, git_ignore_folder+'delta_data_ids')

    delta_data_ids = torch.load(git_ignore_folder+'noise_data_ids')
    
    print(delta_data_ids.shape)
    
    max_epoch = torch.load(git_ignore_folder+'epoch')
    
    print('max_epoch::',max_epoch)
    
    num_class = torch.unique(Y).shape[0]
    
#     delta_data_ids = torch.load(git_ignore_folder+'delta_data_ids')
    
    update_X, selected_rows = get_subset_training_data(X, X.shape, delta_data_ids)
    
    update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
    
#     batch_X_list, batch_Y_list = get_id_mappings_per_batch(X, Y, selected_rows, batch_size)
    
    t1 = time.time()
    
    lr = initialize(update_X, num_class)
#     update_x_sum_by_class = compute_x_sum_by_class(update_X, update_Y, num_class, update_X.shape)
    #     update_X_Y_mult = update_X.mul(update_Y)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)dim, theta,  X, Y, X_sum_by_class, num_class
    res2, theta_list, grad_list = logistic_regression_by_standard_library(random_ids_multi_super_iterations, selected_rows, X, Y, lr, X.shape, max_epoch, alpha, beta, batch_size)

    
    t2 = time.time()
    process = psutil.Process(os.getpid())
    
    print('memory usage::', process.memory_info().rss)
    
    torch.save(res2, git_ignore_folder+'model_standard_lib')
    
    print(res2)
    
    torch.save(theta_list, git_ignore_folder+'theta_list')
    torch.save(grad_list, git_ignore_folder+'grad_list')
    
    print('training_time_standard_lib::', t2 - t1)
    
    test_X = torch.load(git_ignore_folder+'test_X')
    
    test_Y = torch.load(git_ignore_folder+'test_Y')
    
    print('training_accuracy::', compute_accuracy2(update_X, update_Y.type(torch.DoubleTensor), res2))
    
    print('test_accuracy::', compute_accuracy2(test_X, test_Y, res2))
    
    
    
    
    
    
    
    
    
    