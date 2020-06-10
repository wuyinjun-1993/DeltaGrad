'''
Created on Mar 15, 2019


'''

import torch

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from sensitivity_analysis_SGD.linear_regression.Linear_regression import * 
    from data_IO.Load_data import *
    from sensitivity_analysis_SGD.linear_regression.utils import *
    from sensitivity_analysis_SGD.linear_regression.evaluating_test_samples import *
except ImportError:
    from Linear_regression import * 
    from Load_data import *
    from utils import *
    from evaluating_test_samples import *

if __name__ == '__main__':
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    sys_args = sys.argv

    
    alpha = torch.load(git_ignore_folder + 'alpha')
    
    beta = torch.load(git_ignore_folder + 'beta')
    
    
    X = torch.load(git_ignore_folder+'noise_X')
    
    Y = torch.load(git_ignore_folder+'noise_Y')
    
    
    dim = X.shape
    
    num_of_output = Y.shape[1]
    
    max_epoch = torch.load(git_ignore_folder+'epoch')
    
    delta_data_ids = torch.load(git_ignore_folder+'noise_data_ids')
    
    X_Y_mult = torch.load(git_ignore_folder+'X_Y_mult')

#     delta_data_ids = torch.load(git_ignore_folder+'delta_data_ids')
    
    print(delta_data_ids.shape[0])
    
    update_X, selected_rows = get_subset_training_data(X, X.shape, delta_data_ids)
    
    update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
    init_theta = initialize(update_X.shape, num_of_output)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
    
    t1 = time.time()
    
    
    
#     update_x_sum_by_class = compute_x_sum_by_class(update_X, update_Y, num_class, update_X.shape)
#     update_X_Y_mult = update_X.mul(update_Y)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)dim, theta,  X, Y, X_sum_by_class, num_class
#     res2 = linear_regression_iteration(torch.mm(torch.t(update_X), update_X), torch.mm(torch.t(update_X), update_Y), update_X.shape, init_theta, max_epoch, alpha, beta)#(X.shape, init_theta, update_X_Y_mult, max_epoch)
    res2 = linear_regression_closed_form(torch.mm(torch.t(update_X), update_X), torch.mm(torch.t(update_X), update_Y), update_X.shape)

    
    t2 = time.time()
    
    model_standard_lib = torch.load(git_ignore_folder+'model_standard_lib')
    
    
    torch.save(res2, git_ignore_folder+'model_closed_form')
    
#     torch.save(X_prod_inverse, git_ignore_folder + 'exp_X_prod_inverse')
    
    
    print('training_time_closed_form::', t2 - t1)
    
    print(torch.norm(model_standard_lib - res2))
    
    
    test_X = torch.load(git_ignore_folder+'test_X')
    
    test_Y = torch.load(git_ignore_folder+'test_Y')
    
    print('training_accuracy::', compute_accuracy2(update_X, update_Y, res2))
    
    print('test_accuracy::', compute_accuracy2(test_X, test_Y, res2))
    
    print('angle::', torch.dot(res2.view(-1), model_standard_lib.view(-1))/(torch.norm(res2.view(-1))*torch.norm(model_standard_lib.view(-1))))
    
#     print(res2)

    error = torch.norm(res2 - model_standard_lib)/torch.norm(model_standard_lib)
#     cos_sim = torch.dot(torch.reshape(res2, [-1]), torch.reshape(model_standard_lib, [-1]))/(torch.norm(torch.reshape(res2, [-1]))*torch.norm(torch.reshape(model_standard_lib, [-1])))
    
    print('absolute_error::', torch.norm(res2 - model_standard_lib))
    
    print('relative_error::', error)
    
    print(res2.shape)
    
    print(torch.nonzero(res2*model_standard_lib > 0).shape)
    
    get_relative_change(res2, model_standard_lib)
    
    
    
    