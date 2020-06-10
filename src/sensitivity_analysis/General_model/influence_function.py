'''
Created on Mar 23, 2019

'''

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



import torch


try:
    from sensitivity_analysis.DNN.DNN import *
except ImportError:
    from DNN import *


if __name__ == '__main__':
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    X = torch.load(git_ignore_folder + 'noise_X')
        
    Y = torch.load(git_ignore_folder + 'noise_Y')
    
    test_X = torch.load(git_ignore_folder + 'test_X')
        
    test_Y = torch.load(git_ignore_folder + 'test_Y')
    
    num_class = torch.unique(Y).shape[0]
    
    origin_model = torch.load(git_ignore_folder + 'model_without_noise')
    
    
    para_list = torch.load(git_ignore_folder + 'hessian_para_list')
        
    gradient_list = torch.load(git_ignore_folder + 'hessian_gradient_list')
    
    
    
    
#     X_Y_mult = torch.load(git_ignore_folder + 'X_Y_mult')
    
#     X_product = torch.load(git_ignore_folder + 'X_product')
    
    model_baseline = torch.load(git_ignore_folder + 'model_base_line')
    
    Hessian_inverse = torch.load(git_ignore_folder + 'Hessian_inverse')
#     delta_data_ids = torch.load(git_ignore_folder + 'delta_data_ids')
    delta_data_ids = torch.load(git_ignore_folder + 'noise_data_ids')
    
    
    hidden_dim = torch.load(git_ignore_folder + 'hidden_dims')
    
    
    input_dim = X.shape[1]
    
    num_class = torch.unique(Y).shape[0]
    
    output_dim = num_class
    
    print(delta_data_ids)
    
    
    update_X, updated_Y, selected_rows = get_subset_training_data(X, Y, X.shape, delta_data_ids)
    
#     update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
    delta_X = X[delta_data_ids]
         
    delta_Y = Y[delta_data_ids]
    
    
    dim = X.shape
    
#     X_product = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
#     
#     X_Y_mult = X.mul(Y)
    
    model_origin = torch.load(git_ignore_folder + 'model_without_noise')
    
    init_model(model_origin, para_list)
    
#     Hessin_matrix = compute_hessian_matrix(model_origin, X, dim, num_class, X_product)
# 
#     torch.save(torch.inverse(Hessin_matrix), git_ignore_folder + 'Hessian_inverse')
    
#     Hessin_matrix = compute_hessian_matrix(X, X_Y_mult, model_origin, dim, X_product)

#     Hessian_inverse = torch.inverse(Hessin_matrix)
    
    
#     update_X_Y_mult = torch.index_select(X_Y_mult, 0, selected_rows)
#     
#     update_X_product = torch.index_select(X_product, 0, selected_rows)

    '''theta, X, dim, num_class, X_product'''

#     Hessin_matrix2 = compute_hessian_matrix_3(X, X_Y_mult, res, dim)
    
#     print('Hessian::', Hessin_matrix)
#     
#     print(Hessin_matrix - Hessin_matrix2)
    

#     Hessian_inverse = torch.inverse(Hessin_matrix)

    
    t1 = time.time()
    
#     curr_X_Y_mult = X_Y_mult[delta_data_ids]

    error = nn.CrossEntropyLoss()

    first_derivative = compute_first_derivative(model_origin, delta_X, delta_Y, error)#compute_first_derivative(delta_X, delta_Y, dim, model_origin, num_class)

    print('first_derivative::', first_derivative.shape)
    

#     curr_X_Y_mult = torch.index_select(X_Y_mult, 0, delta_data_ids)
    
#     compute_first_derivative_single_data2(torch.index_select(X_Y_mult, 0, delta_data_ids), Variable(model_origin), dim)
    
    
    delta_theta = torch.t((torch.mm(Hessian_inverse, first_derivative.view(-1,1)*delta_data_ids.shape[0] - get_all_vectorized_parameters(gradient_list).view(-1,1))/dim[0]))
    
    model_origin_params = get_all_vectorized_parameters(list(model_origin.parameters()))
    
    updated_model_params = model_origin_params + delta_theta
    
    model_expected_params = get_all_vectorized_parameters(list(model_baseline.parameters()))
    
    
    print("model diff:", torch.norm(updated_model_params - model_expected_params))
    
    init_model(origin_model, get_devectorized_parameters(updated_model_params, input_dim, hidden_dim, output_dim))
    
    compute_test_acc(origin_model, test_X, test_Y)
    
#     print(delta_theta.shape)
#     
#     print(model_origin.shape)
#     
#     updated_theta = delta_theta + model_origin_params
#     
#     
#     t2 = time.time()
#     
#     expected_theta = torch.load(git_ignore_folder + 'model_standard_lib')
#     
#     
# #     print('gap::', compute_first_derivative_single_data(X_Y_mult, selected_rows, expected_theta))
#     
#     
#     print(delta_theta)
#     
#     print(model_origin)
#     
#     print(expected_theta)
#     
#     print(updated_theta)
#     
#     
#     
#     print('absolute_errors::', torch.norm(expected_theta - updated_theta))
#     
#     error = torch.norm(expected_theta - updated_theta)/torch.norm(expected_theta)
#     
#     print('relative_errors::', error)
#     
#     print('angle::', torch.dot(torch.reshape(updated_theta, [-1]), torch.reshape(expected_theta, [-1]))/(torch.norm(torch.reshape(updated_theta, [-1]))*torch.norm(torch.reshape(expected_theta, [-1]))))
#     
#     print(torch.norm(model_origin - expected_theta))
#     
#     print(torch.norm(delta_theta))
#     
#     print(t2 - t1)
#     
#     test_X = torch.load(git_ignore_folder + 'test_X')
#     
#     test_Y = torch.load(git_ignore_folder + 'test_Y')
#     
#     print('training_accuracy::', compute_accuracy2(update_X, update_Y.type(torch.DoubleTensor), updated_theta))
#     
#     print('test_accuracy::', compute_accuracy2(test_X, test_Y, updated_theta))
#     
    
    
    
#     print('test_accuracy::', compute_accuracy2(test_X, test_Y, updated_theta))