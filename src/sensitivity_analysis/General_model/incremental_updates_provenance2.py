'''
Created on Mar 15, 2019

'''
from sensitivity_analysis.DNN.DNN import model_update_provenance_cp0_stochastic
'''
Created on Mar 15, 2019

'''
import sys, os
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
    
    origin_model = torch.load(git_ignore_folder + 'model_without_noise')
    
    
    alpha = torch.load(git_ignore_folder + 'alpha')
    
#     beta = torch.load(git_ignore_folder + 'beta')
    
#     max_epoch = torch.load(git_ignore_folder+'epoch')
    
    
    max_epoch = torch.load(git_ignore_folder + 'update_max_epochs')
    
#     X = torch.load(git_ignore_folder+'noise_X')
    
    hessian_matrix = torch.load(git_ignore_folder + 'hessian_matrix')
    
    origin_X = torch.load(git_ignore_folder + 'noise_X')
    
    origin_Y = torch.load(git_ignore_folder + 'noise_Y')
        
#     Y = torch.load(git_ignore_folder+'noise_Y')
    
    test_X = torch.load(git_ignore_folder + 'test_X')
    
    test_Y = torch.load(git_ignore_folder + 'test_Y')
    
    hidden_dims = torch.load(git_ignore_folder + 'hidden_dims')
    
    output_list = torch.load(git_ignore_folder + 'output_list')
    
#     A_list_all_epochs = torch.load(git_ignore_folder + 'A_list')
#     
#     B_list_all_epochs = torch.load(git_ignore_folder + 'B_list')
#     
#     w_delta_prod_list = torch.load(git_ignore_folder + 'w_delta_prod_list')
#     
#     b_delta_prod_list = torch.load(git_ignore_folder + 'b_delta_prod_list')
    
    
    output_list_all_epochs = torch.load(git_ignore_folder + 'output_list')
    
    init_para_list = list(torch.load(git_ignore_folder + 'init_para'))
    
    
    gradient_list = torch.load(git_ignore_folder + 'hessian_gradient_list')
    
    expected_gradient_list_all_epochs = torch.load(git_ignore_folder + 'expected_gradient_list_all_epochs')
    
    epxected_para_list_all_epochs_all_epochs = torch.load(git_ignore_folder + 'expected_para_list_all_epochs')
    
    
    w_list = torch.load(git_ignore_folder + 'w_seq')
      
    b_list = torch.load(git_ignore_folder + 'b_seq')
    
    
    input_dim = origin_X.shape[1]
    
    num_class = torch.unique(origin_Y).shape[0]
    
    output_dim = num_class
    
    model = DNNModel(input_dim, hidden_dims, output_dim)
    
    
    hessian_para_list = torch.load(git_ignore_folder + 'hessian_para_list')
    
    batch_num = torch.load(git_ignore_folder + 'batch_num')
    
    init_model(model, hessian_para_list)

#     init_model(model, init_para_list)
#     init_model(model, init_para_list)
    
#     selected_rows = torch.load(git_ignore_folder + 'selected_rows')

#     delta_data_ids = torch.load(git_ignore_folder + 'delta_data_ids')
    delta_data_ids = torch.load(git_ignore_folder + 'noise_data_ids')
    
    batch_size = int(delta_data_ids.shape[0]/batch_num)
    
    print("detal_data_ids::", delta_data_ids.shape)

    update_X, update_Y, selected_rows = get_subset_training_data(origin_X, origin_Y, origin_X.shape, delta_data_ids)

    error = nn.CrossEntropyLoss()
    
    compute_derivative_one_more_step(origin_model, error, update_X, update_Y)
     
#     gradient_list = get_model_gradient(origin_model)

    
    model_base_line = torch.load(git_ignore_folder + 'model_base_line')

    
    dim = origin_X.shape
    
#     max_epoch = 150
    
    print("learning rate::", alpha)
    
    
#     compute_derivative_one_more_step(model, error, origin_X, origin_Y)
#     
#     expected_full_gradient_list2 = get_all_gradient(model)
#         
#     expected_full_gradient_list, expected_full_para_list = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, origin_X, origin_Y, gradient_list)
#     
#     print(get_all_vectorized_parameters(expected_full_gradient_list) - get_all_vectorized_parameters(gradient_list))
#     model_para_list = model_update_provenance(alpha, max_epoch, model, dim, output_list, A_list, B_list, w_delta_prod_list, b_delta_prod_list, input_dim, hidden_dims, output_dim, delta_data_ids, gradient_list)


#     model_para_list = model_update_provenance1_2(alpha, init_para_list, selected_rows, model, origin_X, origin_Y, w_list_all_epochs, b_list_all_epochs, num_class, input_dim, hidden_dims, output_dim, expected_gradient_list_all_epochs)

#     model_para_list = model_update_provenance1(init_para_list, alpha, model, dim, output_list_all_epochs, A_list_all_epochs, B_list_all_epochs, w_delta_prod_list_all_epochs, b_delta_prod_list_all_epochs, input_dim, hidden_dims, output_dim, selected_rows, expected_gradient_list_all_epochs)
    
#     for i in range(len(gradient_list)):
#         gradient_list[i] = gradient_list[i]*origin_X.shape[0]
    
    
    
    
#     _, count, dual_para_list = model_update_standard_lib_with_recording_parameters(max_epoch, origin_X[delta_data_ids], origin_Y[delta_data_ids], alpha, error, model)
#     
#     
#     init_model(model, hessian_para_list)

#     model_para_list = model_update_provenance_by_dual(alpha, dim, dual_para_list, hessian_matrix, gradient_list, get_all_vectorized_parameters(list(model.parameters())), max_epoch, input_dim, hidden_dims, output_dim, delta_data_ids, expected_gradient_list_all_epochs, epxected_para_list_all_epochs_all_epochs)
#     hessian_matrix = torch.diag(torch.diag(hessian_matrix))
    
#     model_para_list = model_update_provenance_cp0_stochastic(batch_size, alpha, origin_X, origin_Y, hessian_matrix, gradient_list, get_all_vectorized_parameters(list(model.parameters())), max_epoch, model, dim, w_list, b_list, input_dim, hidden_dims, output_dim, delta_data_ids, expected_gradient_list_all_epochs, epxected_para_list_all_epochs_all_epochs, selected_rows)
    model_para_list = model_update_provenance_cp0(alpha, origin_X, origin_Y, hessian_matrix, gradient_list, get_all_vectorized_parameters(list(model.parameters())), int((max_epoch)), model, dim, w_list, b_list, input_dim, hidden_dims, output_dim, delta_data_ids, expected_gradient_list_all_epochs, epxected_para_list_all_epochs_all_epochs, selected_rows)
#     model_para_list = model_update_provenance3(origin_X, origin_Y, alpha, model, dim, output_list_all_epochs, A_list_all_epochs, B_list_all_epochs, w_delta_prod_list_all_epochs, b_delta_prod_list_all_epochs, input_dim, hidden_dims, output_dim, delta_data_ids, expected_gradient_list_all_epochs)

    compute_model_para_diff(list(model_base_line.parameters()), model_para_list)
    
    compute_model_para_diff(list(origin_model.parameters()), model_para_list)

    compute_model_para_diff(list(origin_model.parameters()), list(model_base_line.parameters()))
    
    compute_test_acc(model, test_X, test_Y)
#     A_list, B_list, w_delta_prod_list, b_delta_prod_list, output_list = capture_provenance(model, origin_X, origin_Y, w_list, b_list, dim, output_dim, input_dim, hidden_dims, output_dim)
# 
#     model_para_list = model_update_provenance(alpha, max_epoch, model, dim, output_list, A_list, B_list, w_delta_prod_list, b_delta_prod_list, input_dim, hidden_dims, output_dim, delta_data_ids, gradient_list)
# 
#     compute_model_para_diff(list(model_base_line.parameters()), model_para_list)
#     
#     compute_model_para_diff(list(origin_model.parameters()), model_para_list)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     sys_args = sys.argv
#     
#     
#     alpha = torch.load(git_ignore_folder + 'alpha')
#     
#     beta = torch.load(git_ignore_folder + 'beta')
#     
#     
#     opt = bool(int(sys_args[1]))
#     
#     
#     cut_off_epoch = torch.load(git_ignore_folder + 'cut_off_epoch')
# 
# 
#     print('cut_off_epoch', cut_off_epoch)
#     
# #     print(X)
# #     
# #     print(Y)
#     
#     M = torch.load(git_ignore_folder + 'eigen_vectors')
#     
#     s = torch.load(git_ignore_folder + 'eigen_values')
#     
#     expected_A = torch.load(git_ignore_folder + 'expected_A')
#     
# #     print(s)
# #     
# #     print(torch.sort(s, descending = True))
#     
#     M_inverse = torch.load(git_ignore_folder + 'eigen_vectors_inverse')
#     
#     
#     M = M.type(torch.double)
#     
#     M_inverse = M_inverse.type(torch.double)
#     
#     s = s.type(torch.double)
#     
#     
#     w_seq = torch.load(git_ignore_folder+'w_seq')
#     
#     b_seq = torch.load(git_ignore_folder+'b_seq')
#     
#     term1 = torch.load(git_ignore_folder+'term1')
#     
#     term2 = torch.load(git_ignore_folder+'term2')
#     
#     X_Y_mult = torch.load(git_ignore_folder + 'X_Y_mult')
#     
# #     X_product = torch.load(git_ignore_folder+'X_product')
#     
# #     x_sum_by_class = torch.load(git_ignore_folder+'x_sum_by_class')
#     
#     max_epoch = torch.load(git_ignore_folder+'epoch')
#     
#     num_class = torch.unique(Y).shape[0]
#     
#     delta_data_ids = torch.load(git_ignore_folder+'noise_data_ids')
#     
# #     delta_data_ids = torch.load(git_ignore_folder+'delta_data_ids')
#     
#     
#     print(delta_data_ids.shape[0])
#     
#     update_X, selected_rows = get_subset_training_data(X, X.shape, delta_data_ids)
#     
#     update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
#     
#     #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
#     
#     t1 = time.time()
#     
#     res3 = None
#     
#     if len(delta_data_ids) < (X.shape[0])/2:
# 
#         sub_w_seq = w_seq[delta_data_ids]#get_subset_parameter_list(selected_rows, delta_data_ids, w_seq, dim, 1)
#           
#         sub_b_seq = b_seq[delta_data_ids]#get_subset_parameter_list(b_seq, delta_data_ids, b_seq, dim, 1)
#          
#         delta_X = X[delta_data_ids]
#          
#         delta_Y = Y[delta_data_ids]
#          
#         delta_X_product = torch.bmm(delta_X.view(delta_X.shape[0], X.shape[1], 1), delta_X.view(delta_X.shape[0], 1, X.shape[1]))
#          
# #         delta_X_product = X_product[delta_data_ids]
#          
#         delta_X_Y_prod = delta_X.mul(delta_Y) 
# 
# #         delta_X_Y_prod = X_Y_mult[delta_data_ids]
#         
#         sub_term_1 = prepare_term_1_batch2(delta_X_product, sub_w_seq, delta_X.shape)
#          
#         sub_term_2 = prepare_term_2_batch2(delta_X_Y_prod, sub_b_seq, delta_X.shape)     
#           
#         init_theta = Variable(initialize(update_X).theta)
#         
# #         res3 = compute_model_parameter_by_approx_incremental_2(term1 - sub_term_1, term2 - sub_term_2, X.shape, init_theta, max_epoch)
#         
#         if not opt:
#             res3 = compute_model_parameter_by_approx_incremental_3(term1 - sub_term_1, term2 - sub_term_2, update_X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta)
#         else:
#             res3 = compute_model_parameter_by_approx_incremental_4(s, M, M_inverse, expected_A, term1 - sub_term_1, term2 - sub_term_2, update_X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta)
#     
#     else:
#         t2_5 = time.time()
#         
#         sub_w_seq = w_seq[selected_rows]
#         
# #         sub_w_seq = torch.index_select(w_seq, 0, selected_rows)#get_subset_parameter_list(selected_rows, delta_data_ids, w_seq, dim, 1)
#         sub_b_seq = b_seq[selected_rows]
#         
# #         sub_b_seq = torch.index_select(b_seq, 0, selected_rows)#get_subset_parameter_list(b_seq, delta_data_ids, b_seq, dim, 1)
#          
#         delta_X = X[selected_rows]
#          
#         delta_Y = Y[selected_rows]
#          
#         delta_X_product = torch.bmm(delta_X.view(delta_X.shape[0], X.shape[1], 1), delta_X.view(delta_X.shape[0], 1, X.shape[1]))
# #          
# #         delta_X_Y_prod = delta_X.mul(delta_Y) 
# 
# #         delta_X_product = X_product[selected_rows]
#          
# #         delta_X_Y_prod = delta_X.mul(delta_Y) 
# 
#         delta_X_Y_prod = X_Y_mult[selected_rows]
#         
#         
#         t2_6 = time.time()
#         
#         t2_3 = time.time()
#         
#         sub_term_1 = prepare_term_1_batch2(delta_X_product, sub_w_seq, delta_X.shape)
#          
#         sub_term_2 = prepare_term_2_batch2(delta_X_Y_prod, sub_b_seq, delta_X.shape)     
#                 
#                 
#         t2_4 = time.time()        
#         
#         init_theta = Variable(initialize(update_X).theta)
#          
# #         res3 = compute_model_parameter_by_approx_incremental_2(sub_term_1, sub_term_2, X.shape, init_theta, max_epoch)
# 
#         t2_1 = time.time()
# 
#         if opt:
#             res3 = compute_model_parameter_by_approx_incremental_3(sub_term_1, sub_term_2, update_X.shape, init_theta, max_epoch, cut_off_epoch)
#         else:
#             res3 = compute_model_parameter_by_approx_incremental_4(sub_term_1, sub_term_2, update_X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta)
#         
#         
#         t2_2 = time.time()
#     
#     t2 = time.time()
#     
#     
#     torch.save(res3, git_ignore_folder+'model_provenance')
#     
#     
#     print('training_time_provenance::', t2 - t1)
#     
#     
# #     print(t2_2 - t2_1)
# #     
# #     print(t2_4 - t2_3)
# #     
# #     print(t2_6 - t2_5)
#     
#     model_origin = torch.load(git_ignore_folder+'model_origin')
#     
#     model_standard_lib = torch.load(git_ignore_folder+'model_standard_lib')
#     
#     model_iteration = torch.load(git_ignore_folder+'model_iteration')
#     
#     model_provenance = torch.load(git_ignore_folder+'model_provenance')
#     
#     expect_updates = model_origin - model_standard_lib
#     
#     real_updates = model_origin - model_provenance
#     
#     
#     error = torch.norm(expect_updates - real_updates)/torch.norm(model_standard_lib)
#     
#     print('model_origin::', model_origin.view(1,-1))
#     
#     print('model_standard_lib::', model_standard_lib)
#     
#     print('model_prov::', model_provenance)
#     
#     print('absolute_error::', torch.norm(model_provenance - model_standard_lib))
#     
#     print('absolute_error2::', torch.norm(model_provenance - model_iteration))
#     
#     print('expect_updates::', torch.norm(expect_updates))
#     
#     print('angle::', torch.dot(model_provenance.view(-1), model_standard_lib.view(-1))/(torch.norm(model_provenance.view(-1))*torch.norm(model_standard_lib.view(-1))))
#     
#     print('relative_error::', error)
#     
#     test_X = torch.load(git_ignore_folder + 'test_X')
#     
#     test_Y = torch.load(git_ignore_folder + 'test_Y')
#     
#     print('training_accuracy::', compute_accuracy2(update_X, update_Y, res3))
#     
#     print('test_accuracy::', compute_accuracy2(test_X, test_Y, res3))
    
    
    