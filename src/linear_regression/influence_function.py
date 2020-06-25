'''
Created on Mar 16, 2019


'''

import torch
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import time


try:
    from data_IO.Load_data import *
    from sensitivity_analysis_SGD.linear_regression.utils import *
    from sensitivity_analysis_SGD.linear_regression.Linear_regression import *
    from sensitivity_analysis_SGD.linear_regression.evaluating_test_samples import *
except ImportError:
    from Load_data import *
    from utils import *
    from Linear_regression import *
    from evaluating_test_samples import *

if __name__ == '__main__':
    
    
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    sys_args = sys.argv

    
    
    X = torch.load(git_ignore_folder+'noise_X')
    
    Y = torch.load(git_ignore_folder+'noise_Y')
    
    
    X_Y_mult = torch.load(git_ignore_folder + 'X_Y_mult')
    
    X_prod = torch.load(git_ignore_folder + 'X_prod')
    
    Hessian_inverse = torch.load(git_ignore_folder + 'Hessian_inverse')
#     delta_data_ids = torch.load(git_ignore_folder + 'delta_data_ids')
    delta_data_ids = torch.load(git_ignore_folder + 'noise_data_ids')
    
    delta_data_ids = delta_data_ids.type(torch.long)
    
    
    print(delta_data_ids.shape[0])
    
    print(X.shape)
    
    
    update_X, selected_rows = get_subset_training_data(X, X.shape, delta_data_ids)
    
    update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
    delta_X = torch.index_select(X, 0, delta_data_ids)
         
    delta_Y = torch.index_select(Y, 0, delta_data_ids)
    
    
    dim = X.shape
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
#     
#     X_Y_mult = X.mul(Y)
    
    model_origin = Variable(torch.load(git_ignore_folder + 'model_origin'))
        
#     Hessin_matrix = compute_hessian_matrix(X, X_Y_mult, model_origin, dim, X_product)

#     Hessian_inverse = torch.inverse(Hessin_matrix)
    
    
#     update_X_Y_mult = torch.index_select(X_Y_mult, 0, selected_rows)
# #     
#     update_X_product = torch.index_select(X_product, 0, selected_rows)

#     Hessin_matrix = compute_hessian_matrix(X_prod, model_origin, dim)
    
#     Hessin_matrix2 = compute_hessian_matrix_3(X, X_Y_mult, res, dim)
    
#     print('Hessian::', Hessin_matrix)
#     
#     print(Hessin_matrix - Hessin_matrix2)
    

#     Hessian_inverse = torch.inverse(Hessin_matrix)

    
    t1 = time.time()
    
    curr_X_Y_mult = torch.mm(torch.t(delta_X), delta_Y)
    
    
    
    
    first_derivative = compute_first_derivative(update_X, curr_X_Y_mult, model_origin, dim)


#     print('first_derivative::', first_derivative)
#     curr_X_Y_mult = torch.index_select(X_Y_mult, 0, delta_data_ids)
    
#     compute_first_derivative_single_data2(torch.index_select(X_Y_mult, 0, delta_data_ids), Variable(model_origin), dim)
    
    
    delta_theta = torch.mm(Hessian_inverse, first_derivative)/dim[0]
    
    updated_theta = delta_theta + model_origin
    
    
    t2 = time.time()
    
    expected_theta = torch.load(git_ignore_folder + 'model_standard_lib')
    
    
#     print('gap::', compute_first_derivative_single_data(X_Y_mult, selected_rows, expected_theta))
    
    
    print(delta_theta)
    
    print(model_origin)
    
    print(expected_theta)
    
    print(updated_theta)
    
    
    
    print('absolute_errors::', torch.norm(expected_theta - updated_theta))
    
    error = torch.norm(expected_theta - updated_theta)/torch.norm(expected_theta)
    
    print('relative_errors::', error)
    
    print(torch.norm(model_origin - expected_theta))
    
    print(torch.norm(delta_theta))
    
    print('training_time_influence_function::', t2 - t1)
    
    test_X = torch.load(git_ignore_folder + 'test_X')
    
    test_Y = torch.load(git_ignore_folder + 'test_Y')
    
    print('training_accuracy::', compute_accuracy2(update_X, update_Y, updated_theta))
    
    print('test_accuracy::', compute_accuracy2(test_X, test_Y, updated_theta))
    
    
    print('angle::', torch.dot(updated_theta.view(-1), expected_theta.view(-1))/(torch.norm(updated_theta.view(-1))*torch.norm(expected_theta.view(-1))))
#     print('test_accuracy::', compute_accuracy2(test_X, test_Y, updated_theta))
    
#     print('accuracy::', compute_accuracy('../../../data/toxic/test_labels.csv', updated_theta))
    
#     gap = torch.mm(Hessin_matrix, delta_theta) - first_derivative/dim[0]
#     
#     print(gap)
#     delta_X_product = torch.index_select(X_product, 0, delta_data_ids)
#               
#     delta_X_Y_mult = torch.index_select(X_Y_mult, 0, delta_data_ids)
#     
#     delta_Hessin_matrix = compute_hessian_matrix(X, X_Y_mult, model_origin, dim, X_product)
    
    
#     print(torch.norm(Hessin_matrix))