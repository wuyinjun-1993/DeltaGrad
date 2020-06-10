'''
Created on Mar 15, 2019

'''

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



import torch
try:
    from sensitivity_analysis.logistic_regression.incremental_updates_logistic_regression import *
    from data_IO.Load_data import *
except ImportError:
    from incremental_updates_logistic_regression import *
    from Load_data import * 

import psutil

if __name__ == '__main__':
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
#     X = torch.load(git_ignore_folder+'noise_X')
    sys_args = sys.argv

    run_rc1 = bool(int(sys_args[1]))

    if run_rc1:
        sparse_X = scipy.sparse.load_npz(git_ignore_folder + 'noise_X.npz')
        X = convert_coo_matrix2_dense_tensor(sparse_X)
    else:
        X = torch.load(git_ignore_folder+'noise_X')
    
    Y = torch.load(git_ignore_folder+'noise_Y')
    
    
    
    alpha = torch.load(git_ignore_folder + 'alpha')
    
    beta = torch.load(git_ignore_folder + 'beta')
    
    
    max_epoch = torch.load(git_ignore_folder+'epoch')
    
    delta_data_ids = torch.load(git_ignore_folder+'noise_data_ids')
    
    
    if run_rc1:
        sparse_X_Y_mult = scipy.sparse.load_npz(git_ignore_folder + 'X_Y_mult.npz')
        X_Y_mult = convert_coo_matrix2_dense_tensor(sparse_X_Y_mult)
    else:
        X_Y_mult = torch.load(git_ignore_folder+'X_Y_mult')
    
#     X_Y_mult = torch.load(git_ignore_folder+'X_Y_mult')

#     delta_data_ids = torch.load(git_ignore_folder+'delta_data_ids')
    
    print(delta_data_ids.shape[0])
    
    update_X, selected_rows = get_subset_training_data(X, X.shape, delta_data_ids)
    
    update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
    init_theta = Variable(initialize(update_X).theta)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
    
    t1 = time.time()
    
    update_X_Y_mult = X_Y_mult[selected_rows]
    
#     update_x_sum_by_class = compute_x_sum_by_class(update_X, update_Y, num_class, update_X.shape)
#     update_X_Y_mult = update_X.mul(update_Y)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)dim, theta,  X, Y, X_sum_by_class, num_class
    res2, total_time = compute_model_parameter_by_iteration(update_X.shape, init_theta, update_X_Y_mult, max_epoch, alpha, beta)

    process = psutil.Process(os.getpid())
    
    print('memory usage::', process.memory_info().rss)
    
    t2 = time.time()
    
    model_standard_lib = torch.load(git_ignore_folder+'model_standard_lib')
    
    
    torch.save(res2, git_ignore_folder+'model_iteration')
    
    
    print('training_time_iteration::', t2 - t1)
    
    print(torch.norm(model_standard_lib - res2))
    
    if run_rc1:
        test_parse_X = scipy.sparse.load_npz(git_ignore_folder + 'test_X.npz')
        test_X = convert_coo_matrix2_dense_tensor(test_parse_X)
    else:
        test_X = torch.load(git_ignore_folder+'test_X')
#     test_X = torch.load(git_ignore_folder+'test_X')
    
    test_Y = torch.load(git_ignore_folder+'test_Y')
    
    print('training_accuracy::', compute_accuracy2(update_X, update_Y, res2))
    
    print('test_accuracy::', compute_accuracy2(test_X, test_Y, res2))
    
    print(res2)
    
    