'''
Created on Mar 15, 2019


'''


import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import psutil
import torch
 
try:
    from sensitivity_analysis_SGD.logistic_regression.Logistic_regression import *
    from data_IO.Load_data import *
except ImportError:
    from incremental_updates_logistic_regression import *
    from Load_data import *    



if __name__ == '__main__':
    
    
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']

    
#     X = torch.load(git_ignore_folder+'noise_X')
    
    Y = torch.load(git_ignore_folder+'noise_Y')
#     print(Y.)
    
    batch_size = torch.load(git_ignore_folder + 'batch_size')
    
    
    mini_batch_epoch = torch.load(git_ignore_folder + 'mini_batch_epoch')
    
    
    print('mini_batch_epoch::', mini_batch_epoch)
    
    Y = Y.type(torch.DoubleTensor)
    
    sys_args = sys.argv
    
    
    run_rc1 = bool(int(sys_args[1]))
    
    if run_rc1:
        sparse_X = scipy.sparse.load_npz(git_ignore_folder + 'noise_X.npz')
        X = convert_coo_matrix2_dense_tensor(sparse_X)
        random_ids_multi_super_iterations = torch.from_numpy(np.load(git_ignore_folder + 'random_ids_multi_super_iterations.npy')).type(torch.IntTensor)

    else:
        X = torch.load(git_ignore_folder+'noise_X')
        random_ids_multi_super_iterations = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations')
   
    print(X.shape)

    
    alpha = torch.load(git_ignore_folder + 'alpha')
    
    beta = torch.load(git_ignore_folder + 'beta')
    
# 
#     
#     delta_num = 30000
#  
#      
#       
#     delta_data_ids = random_generate_subset_ids(X.shape, delta_num)     
#     torch.save(delta_data_ids, git_ignore_folder+'delta_data_ids')
 
    delta_data_ids = torch.load(git_ignore_folder + 'noise_data_ids')
    
    
    max_epoch = torch.load(git_ignore_folder+'epoch')
    
    print(max_epoch)
#     num_class = torch.unique(Y).shape[0]
    
#     delta_data_ids = torch.load(git_ignore_folder+'delta_data_ids')
    print(delta_data_ids[:100])
    
    print(delta_data_ids.shape)

    update_X, selected_rows = get_subset_training_data(X, X.shape, delta_data_ids)
    
    print(selected_rows.shape)
    
    print(update_X.shape)
    
    update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
    
    t1 = time.time()
    
    lr = initialize(update_X)
#     update_x_sum_by_class = compute_x_sum_by_class(update_X, update_Y, num_class, update_X.shape)
    #     update_X_Y_mult = update_X.mul(update_Y)
#         res1 = update_model_parameters_from_the_scratch(update_X, update_Y)dim, theta,  X, Y, X_sum_by_class, num_class
#     res2 = logistic_regression_by_standard_library(update_X, update_Y, lr, X.shape, max_epoch, alpha, beta, batch_size, mini_batch_epoch)
    
    res2 = logistic_regression_by_standard_library2(X, Y, lr, X.shape, max_epoch, alpha, beta, batch_size, selected_rows, mini_batch_epoch, random_ids_multi_super_iterations)

    
    t2 = time.time()
    
    print(res2)

    process = psutil.Process(os.getpid())
    
    print('memory usage::', process.memory_info().rss)
    
    torch.save(res2, git_ignore_folder+'model_standard_lib')
    
    
#     test_X = torch.load(git_ignore_folder + 'test_X')
    test_Y = torch.load(git_ignore_folder + 'test_Y')
    
    if run_rc1:
        sparse_X = scipy.sparse.load_npz(git_ignore_folder + 'test_X.npz')
        test_X = convert_coo_matrix2_dense_tensor(sparse_X)
        
        
#         print('training_accuracy::', compute_accuracy2_sparse(update_X, update_Y, res2))
#     
#         print('test_accuracy::', compute_accuracy2_sparse(test_X, test_Y, res2))
        
    else:
        test_X = torch.load(git_ignore_folder+'test_X')
    
    
    print(update_X.shape)
    
    print(update_Y.shape)
    
        
    print('training_accuracy::', compute_accuracy2(update_X, update_Y, res2))

    print('test_accuracy::', compute_accuracy2(test_X, test_Y, res2))
        
        
    
    print('training_time_standard_lib::', t2 - t1)
    
    
    
    
    
    