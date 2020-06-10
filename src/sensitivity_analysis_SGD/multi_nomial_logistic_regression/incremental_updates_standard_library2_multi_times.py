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
    
    X = torch.load(git_ignore_folder + 'X')
        
    Y = torch.load(git_ignore_folder + 'Y')
    
    sys_args = sys.argv

    
    alpha = torch.load(git_ignore_folder + 'alpha')
    
    beta = torch.load(git_ignore_folder + 'beta')
    
    random_ids_multi_super_iterations = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations')
    sorted_ids_multi_super_iterations = torch.load(git_ignore_folder + 'sorted_ids_multi_super_iterations')


    delta_id_list = torch.load(git_ignore_folder + 'delta_id_list')
    
    selected_id_list = torch.load(git_ignore_folder + 'selected_id_list')

    batch_size = torch.load(git_ignore_folder + 'batch_size')
    
    max_epoch = torch.load(git_ignore_folder+'epoch')
    
    print('max_epoch::',max_epoch)
    
    print('X_shape::', X.shape)
    
    num_class = torch.unique(Y).shape[0]

    
#     delta_data_ids = random_generate_subset_ids(X.shape, delta_num)
#      
#     print(delta_data_ids)
#      
#     torch.save(delta_data_ids, git_ignore_folder+'delta_data_ids')

#     delta_data_ids = torch.load(git_ignore_folder+'noise_data_ids')
#     
#     print(delta_data_ids.shape)
    
    updated_model_list = []
    
    t1 = time.time()

    for ii in range(len(delta_id_list)):
#     for ii in range(4,6):
        
        lr = initialize(X, num_class)
        
        selected_rows = selected_id_list[ii]
        
        print(ii, selected_rows.shape)
#     update_x_sum_by_class = compute_x_sum_by_class(update_X, update_Y, num_class, update_X.shape)
    #     update_X_Y_mult = update_X.mul(update_Y)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)dim, theta,  X, Y, X_sum_by_class, num_class
        updated_model, theta_list, grad_list = logistic_regression_by_standard_library(random_ids_multi_super_iterations, selected_rows, X, Y, lr, X.shape, max_epoch, alpha, beta, batch_size)

        updated_model_list.append(updated_model)
    
    t2 = time.time()
    
    print('training_time_standard_lib::', t2 - t1)

    process = psutil.Process(os.getpid())
    
    print('memory usage::', process.memory_info().rss)
    test_X = torch.load(git_ignore_folder+'test_X')
    
    test_Y = torch.load(git_ignore_folder+'test_Y')
    
    for ii in range(len(updated_model_list)):
        
        selected_rows = selected_id_list[ii]
    
        update_X = X[selected_rows]# = get_subset_training_data(X, X.shape, delta_data_ids)
    
        update_Y = Y[selected_rows]#, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
        updated_model = updated_model_list[ii]
    
        print('training_accuracy::', compute_accuracy2(update_X, update_Y.type(torch.DoubleTensor), updated_model))
    
        print('test_accuracy::', compute_accuracy2(test_X, test_Y, updated_model))
    
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
    
#     batch_X_list, batch_Y_list = get_id_mappings_per_batch(X, Y, selected_rows, batch_size)
    
    
    
    torch.save(updated_model_list, git_ignore_folder+'model_standard_lib_list')
    
#     print(res2)
    
#     torch.save(theta_list, git_ignore_folder+'theta_list')
#     torch.save(grad_list, git_ignore_folder+'grad_list')
    
    
    
    
    
    
    
    
    