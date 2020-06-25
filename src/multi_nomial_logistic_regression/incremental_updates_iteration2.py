'''
Created on Mar 15, 2019


'''

import sys, os
from sensitivity_analysis.DNN.DNN import get_all_vectorized_parameters
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')

import psutil

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
#     theta_list = torch.load(git_ignore_folder+'theta_list')
#     grad_list = torch.load(git_ignore_folder+'grad_list')
    
#     exp_selected_data_ids_list = torch.load(git_ignore_folder + 'selected_data_ids_list')
    
    batch_size = torch.load(git_ignore_folder + 'batch_size')
    delta_data_ids = torch.load(git_ignore_folder+'delta_data_ids')
    
#     delta_data_ids = torch.load(git_ignore_folder+'noise_data_ids')
    
    random_ids_multi_super_iterations = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations')
    
    sorted_ids_multi_super_iterations = torch.load(git_ignore_folder + 'sorted_ids_multi_super_iterations')
    
    learning_rate_all_epochs = torch.load(git_ignore_folder + 'learning_rate_all_epochs')
#     for i in range(random_ids_multi_super_iterations.shape[0]):
#         sorted_ids_multi_super_iterations.append(random_ids_multi_super_iterations[i].numpy().argsort())
    
    
    exp_gradient_list_all_epochs = torch.load(git_ignore_folder + 'expected_gradient_list_all_epochs')
      
    exp_para_list_all_epochs = torch.load(git_ignore_folder + 'expected_para_list_all_epochs')
    
    print(delta_data_ids[:100])
    
    print(delta_data_ids.shape)
    
    print(max_epoch)
    
    update_X, selected_rows = get_subset_training_data(X, X.shape, delta_data_ids)
    
    update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
    print(update_X.shape)
    
    init_para_list = list(torch.load(git_ignore_folder + 'init_para'))
    
    lr = initialize(update_X, num_class)
    
    lr.theta = init_para_list[0].T
    
    init_theta = Variable(lr.theta)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
#     batch_X_list, batch_Y_list = get_id_mappings_per_batch(X, Y, selected_rows, batch_size)
    
    
    t1 = time.time()
    
#     x_sum_by_class_list = compute_x_sum_by_class_by_batch1(batch_X_list, batch_Y_list, num_class)#(update_X, update_Y, num_class, update_X.shape)
    #     update_X_Y_mult = update_X.mul(update_Y)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)dim, theta,  X, Y, X_sum_by_class, num_class
    
#     res1, theta_list, grad_list = logistic_regression_by_standard_library(random_ids_multi_super_iterations, selected_rows, X, Y, lr, X.shape, max_epoch, alpha, beta, batch_size)
    res2, grad_list, theta_list = compute_model_parameter_by_iteration3(batch_size, exp_gradient_list_all_epochs, exp_para_list_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, X.shape, init_theta, X, Y, selected_rows, num_class, max_epoch, learning_rate_all_epochs, beta)

    
    t2 = time.time()
    
    process = psutil.Process(os.getpid())
    
    print('memory usage::', process.memory_info().rss)
    
    origin_model = torch.load(git_ignore_folder + 'model_base_line')
    
    
    torch.save(res2, git_ignore_folder+'model_iteration')
    
    model_standard_lib = torch.load(git_ignore_folder + 'model_standard_lib')
    
    torch.save(theta_list, git_ignore_folder + 'theta_list')
#     
#     
    torch.save(grad_list, git_ignore_folder + 'grad_list')
    print(torch.norm(res2 - list(origin_model.parameters())[0].T))
    
    
    print('training_time_iteration::', t2 - t1)
    
    print(torch.norm(res2 - model_standard_lib))
    
    test_X = torch.load(git_ignore_folder+'test_X')
    
    test_Y = torch.load(git_ignore_folder+'test_Y')
    
    print('training_accuracy::', compute_accuracy2(update_X, update_Y.type(torch.DoubleTensor), res2))
    
    print('test_accuracy::', compute_accuracy2(test_X, test_Y, res2))
    
    