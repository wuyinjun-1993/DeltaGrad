'''
Created on Mar 15, 2019

'''
'''
Created on Mar 15, 2019

'''
import sys, os


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')

# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch

try:
    from benchmark_exp import *
    from data_IO.Load_data import *
    from Models.Data_preparer import *

except ImportError:
    from benchmark_exp import *
    from Load_data import *
    from Data_preparer import *



# def compute_curr_S_K_Y_K_list(s_k_list, Y_k_list, exp_para_list_all_epochs, exp_gradient_list_all_epochs, para_list_all_epochs, gradient_list_all_epochs):
#     
#     
#     num_paras = get_all_vectorized_parameters(exp_para_list_all_epochs[0]).shape[1]
#     
#     curr_S_k_list = torch.zeros([num_paras, len(exp_para_list_all_epochs)], dtype = torch.double)
#     
#     curr_Y_k_list = torch.zeros([num_paras, len(exp_para_list_all_epochs)], dtype = torch.double)
#     
#     for i in range(len(exp_para_list_all_epochs)):
#         curr_S_k_list[:,i] = (get_all_vectorized_parameters(exp_para_list_all_epochs[i]) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
#         
#         curr_Y_k_list[:,i] = (get_all_vectorized_parameters(exp_gradient_list_all_epochs[i]) - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
#         
#     
#     
#     
#     
#     
#     
#     
#     curr_S_k_list2 = torch.zeros([num_paras, len(exp_para_list_all_epochs) - 1], dtype = torch.double)
#     
#     curr_Y_k_list2 = torch.zeros([num_paras, len(exp_para_list_all_epochs) - 1], dtype = torch.double)
#     
#     for i in range(len(exp_para_list_all_epochs)-1):
#         curr_S_k_list2[:,i] = (get_all_vectorized_parameters(exp_para_list_all_epochs[i+1]) - get_all_vectorized_parameters(exp_para_list_all_epochs[i])).view(-1)
#         
#         curr_Y_k_list2[:,i] = (get_all_vectorized_parameters(exp_gradient_list_all_epochs[i+1]) - get_all_vectorized_parameters(exp_gradient_list_all_epochs[i])).view(-1)
#     
#     
#     
#     return curr_S_k_list, curr_Y_k_list
    
    
# def extend_s_k_y_k_list(S_k_list, Y_k_list, model, para_list_all_epochs, gradient_list_all_epochs, input_dim, hidden_dims, output_dim, X, Y, beta, error):
#     
#     s,v,d = torch.svd(S_k_list)
#     
#     ids = (v > 1e-5)
#     
#     truncted_v = v[ids]
#     
#     truncted_s = s[:,ids]
#     
#     truncted_d = d[:,ids]
#     
# #     extended_S_k_list = torch.zeros([len(para_list_all_epochs), ids.shape[0]], dtype = torch.double)
#     
#     extended_Y_k_list = torch.zeros([len(para_list_all_epochs), S_k_list.shape[0], torch.nonzero(ids).shape[0]], dtype = torch.double)
# #     truncted_loadings = torch.mm(torch.diag(truncted_v), torch.t(truncted_d))
#     small_const = 0.1
#     
#     for i in range(len(para_list_all_epochs)):
#          
#         print("iteration::", i)
#          
#         for j in range(torch.nonzero(ids).shape[0]):
#             curr_para = get_devectorized_parameters(get_all_vectorized_parameters(para_list_all_epochs[i]) + small_const*truncted_s[:,j].view(1,-1), input_dim, hidden_dims, output_dim)
#             
#             init_model(model, curr_para)
#             
#             compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             curr_gradient = get_all_vectorized_parameters(get_all_gradient(model))
#             
#             extended_Y_k_list[i,:, j] = curr_gradient - get_all_vectorized_parameters(gradient_list_all_epochs[i])
#             
#             
#         
#     
#     return small_const*truncted_s, extended_Y_k_list
#     
    
    
    
    


if __name__ == '__main__':
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
#     origin_model = torch.load(git_ignore_folder + 'origin_model')
    
    origin_model = torch.load(git_ignore_folder + 'origin_model', map_location=torch.device('cpu'))
    
    
    
    exp_gradient_list_all_epochs = torch.load(git_ignore_folder + 'expected_gradient_list_all_epochs')
      
    exp_para_list_all_epochs = torch.load(git_ignore_folder + 'expected_para_list_all_epochs')
    
    
    
#     alpha = torch.load(git_ignore_folder + 'alpha')
#     
#     beta = torch.load(git_ignore_folder + 'beta')
#     
#     max_epoch = torch.load(git_ignore_folder+'epoch')
# 
# 
# 
# #     S_k_list = torch.load(git_ignore_folder + 'S_k_list')
# #     
# #     Y_k_list = torch.load(git_ignore_folder + 'Y_k_list')
# 
# 
# 
#     gradient_list_all_epochs = torch.load(git_ignore_folder + 'gradient_list_all_epochs')
#         
#     para_list_all_epochs = torch.load(git_ignore_folder + 'para_list_all_epochs')
#     
#     learning_rate_all_epochs = torch.load(git_ignore_folder + 'learning_rate_all_epochs')
#     
#     batch_size = torch.load(git_ignore_folder + 'batch_size')
#     
#     random_ids_multi_super_iterations = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations')
# 
#     sorted_ids_multi_super_iterations = torch.load(git_ignore_folder + 'sorted_ids_multi_super_iterations')
# #     curr_S_k_list, curr_Y_k_list = compute_curr_S_K_Y_K_list(S_k_list, Y_k_list, exp_para_list_all_epochs, exp_gradient_list_all_epochs, para_list_all_epochs, gradient_list_all_epochs)
#     
#     
#     
#     
#     
#     
#     
#     
#     
# 
# 
# 
#     
#     
# #     max_epoch = torch.load(git_ignore_folder + 'update_max_epochs')
#     
# #     X = torch.load(git_ignore_folder+'noise_X')
#     
# #     hessian_matrix = torch.load(git_ignore_folder + 'hessian_matrix')
#     
#     origin_X = torch.load(git_ignore_folder + 'noise_X')
#     
#     origin_Y = torch.load(git_ignore_folder + 'noise_Y')
#         
# #     Y = torch.load(git_ignore_folder+'noise_Y')
#     
#     test_X = torch.load(git_ignore_folder + 'test_X')
#     
#     test_Y = torch.load(git_ignore_folder + 'test_Y')
#     
#     hidden_dims = torch.load(git_ignore_folder + 'hidden_dims')
#     
# #     output_list = torch.load(git_ignore_folder + 'output_list')
#     
# #     A_list_all_epochs = torch.load(git_ignore_folder + 'A_list')
# #     
# #     B_list_all_epochs = torch.load(git_ignore_folder + 'B_list')
# #     
# #     w_delta_prod_list = torch.load(git_ignore_folder + 'w_delta_prod_list')
# #     
# #     b_delta_prod_list = torch.load(git_ignore_folder + 'b_delta_prod_list')
#     
#     
# #     output_list_all_epochs = torch.load(git_ignore_folder + 'output_list')
#     
#     init_para_list = list(torch.load(git_ignore_folder + 'init_para'))
#     
#     
#     
#     
#     
#     
#     
#     
#     
# #     gradient_list = torch.load(git_ignore_folder + 'hessian_gradient_list')
# #     
# #     expected_gradient_list_all_epochs = torch.load(git_ignore_folder + 'expected_gradient_list_all_epochs')
# #     
# #     epxected_para_list_all_epochs_all_epochs = torch.load(git_ignore_folder + 'expected_para_list_all_epochs')
#     
#     
# #     w_list = torch.load(git_ignore_folder + 'w_seq')
# #       
# #     b_list = torch.load(git_ignore_folder + 'b_seq')
#     
#     
#     input_dim = origin_X.shape[1]
#     
#     num_class = torch.unique(origin_Y).shape[0]
#     
#     output_dim = num_class
#     
#     model = DNNModel(input_dim, hidden_dims, output_dim)
#     
#     error = nn.CrossEntropyLoss()
#     
# #     truncted_s0, extended_Y_k_list = extend_s_k_y_k_list(S_k_list, Y_k_list, model, para_list_all_epochs, gradient_list_all_epochs, input_dim, hidden_dims, output_dim, origin_X, origin_Y, beta, error)
#     
#     
# #     s,v,d = torch.svd(curr_S_k_list[:,1:truncted_s0.shape[1]+1])
# #     
# #     truncted_s = s[:, 0:truncted_s0.shape[1]] 
#     
#     
#     
#     
#     
#     
# #     hessian_para_list = torch.load(git_ignore_folder + 'hessian_para_list')
# #     
# #     batch_num = torch.load(git_ignore_folder + 'batch_num')
#     
# #     init_model(model, hessian_para_list)
# 
#     init_model(model, init_para_list)
# #     init_model(model, init_para_list)
#     
# #     selected_rows = torch.load(git_ignore_folder + 'selected_rows')
# 
# #     delta_data_ids = torch.load(git_ignore_folder + 'delta_data_ids')
#     delta_data_ids = torch.load(git_ignore_folder + 'noise_data_ids')
#     
# #     batch_size = int(delta_data_ids.shape[0]/batch_num)
#     
#     print("detal_data_ids::", delta_data_ids.shape)
# 
#     update_X, update_Y, selected_rows = get_subset_training_data(origin_X, origin_Y, origin_X.shape, delta_data_ids)
# 
#     error = nn.CrossEntropyLoss()
# #     
# #     compute_derivative_one_more_step(origin_model, error, update_X, update_Y)
#      
# #     gradient_list = get_model_gradient(origin_model)
# 
#     print(git_ignore_folder)
#     model_base_line = torch.load(git_ignore_folder + 'model_base_line')
# 
# #     compute_model_para_diff(list(model_base_line.parameters()), exp_para_list_all_epochs[-1])
#     dim = origin_X.shape
#     
# #     max_epoch = 150
#     
#     print("learning rate::", alpha)
#     
#     
# #     compute_derivative_one_more_step(model, error, origin_X, origin_Y)
# #     
# #     expected_full_gradient_list2 = get_all_gradient(model)
# #         
# #     expected_full_gradient_list, expected_full_para_list = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, origin_X, origin_Y, gradient_list)
# #     
# #     print(get_all_vectorized_parameters(expected_full_gradient_list) - get_all_vectorized_parameters(gradient_list))
# #     model_para_list = model_update_provenance(alpha, max_epoch, model, dim, output_list, A_list, B_list, w_delta_prod_list, b_delta_prod_list, input_dim, hidden_dims, output_dim, delta_data_ids, gradient_list)
# 
# 
# #     model_para_list = model_update_provenance1_2(alpha, init_para_list, selected_rows, model, origin_X, origin_Y, w_list_all_epochs, b_list_all_epochs, num_class, input_dim, hidden_dims, output_dim, expected_gradient_list_all_epochs)
# 
# #     model_para_list = model_update_provenance1(init_para_list, alpha, model, dim, output_list_all_epochs, A_list_all_epochs, B_list_all_epochs, w_delta_prod_list_all_epochs, b_delta_prod_list_all_epochs, input_dim, hidden_dims, output_dim, selected_rows, expected_gradient_list_all_epochs)
#     
# #     for i in range(len(gradient_list)):
# #         gradient_list[i] = gradient_list[i]*origin_X.shape[0]
#     
#     
#     
#     
# #     _, count, dual_para_list = model_update_standard_lib_with_recording_parameters(max_epoch, origin_X[delta_data_ids], origin_Y[delta_data_ids], alpha, error, model)
# #     
# #     
# #     init_model(model, hessian_para_list)
# 
# #     model_para_list = model_update_provenance_by_dual(alpha, dim, dual_para_list, hessian_matrix, gradient_list, get_all_vectorized_parameters(list(model.parameters())), max_epoch, input_dim, hidden_dims, output_dim, delta_data_ids, expected_gradient_list_all_epochs, epxected_para_list_all_epochs_all_epochs)
# #     hessian_matrix = torch.diag(torch.diag(hessian_matrix))
#     
# #     model_para_list = model_update_provenance_cp0_stochastic(batch_size, alpha, origin_X, origin_Y, hessian_matrix, gradient_list, get_all_vectorized_parameters(list(model.parameters())), max_epoch, model, dim, w_list, b_list, input_dim, hidden_dims, output_dim, delta_data_ids, expected_gradient_list_all_epochs, epxected_para_list_all_epochs_all_epochs, selected_rows)
# #     model_para_list = model_update_provenance3(origin_X, origin_Y, alpha, model, dim, output_list_all_epochs, A_list_all_epochs, B_list_all_epochs, w_delta_prod_list_all_epochs, b_delta_prod_list_all_epochs, input_dim, hidden_dims, output_dim, delta_data_ids, expected_gradient_list_all_epochs)
# 
#     update_X = origin_X[selected_rows]
#     
#     update_Y = origin_Y[selected_rows]
# 
#     delta_X = origin_X[delta_data_ids]
#     
#     delta_Y = origin_Y[delta_data_ids]
    sorted_ids_multi_super_iterations = torch.load(git_ignore_folder + 'sorted_ids_multi_super_iterations')
#     model_base_line = torch.load(git_ignore_folder + 'model_base_line')
    
    model_base_line = torch.load(git_ignore_folder + 'model_base_line', map_location=torch.device('cpu'))
    dataset_train = torch.load(git_ignore_folder + "dataset_train")
    
    dataset_test = torch.load(git_ignore_folder + "test_data")
    
    delta_data_ids = torch.load(git_ignore_folder + "delta_data_ids")


    learning_rate = torch.load(git_ignore_folder + 'alpha')

    regularization_coeff = torch.load(git_ignore_folder + 'beta')

#     hyper_params = torch.load(git_ignore_folder + 'hyper_params')
    dataset_name = torch.load(git_ignore_folder + 'dataset_name')

#     origin_model = torch.load(git_ignore_folder + 'origin_model')
    
    
    gradient_list_all_epochs = torch.load(git_ignore_folder + 'gradient_list_all_epochs')
    
    para_list_all_epochs = torch.load(git_ignore_folder + 'para_list_all_epochs')
    
    learning_rate_all_epochs = torch.load(git_ignore_folder + 'learning_rate_all_epochs')
    
    device = 'cpu'#torch.load(git_ignore_folder + 'device')
    
    is_GPU = False#torch.load(git_ignore_folder + 'is_GPU')
    
#     [criterion, optimizer, lr_scheduler] = hyper_params

    
#     alpha = torch.load(git_ignore_folder + 'alpha')
#     
#     beta = torch.load(git_ignore_folder + 'beta')
    
    
    init_para_list = list(torch.load(git_ignore_folder + 'init_para'))
    
    random_ids_multi_super_iterations = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations')
    
    model_class = torch.load(git_ignore_folder + 'model_class')
    
#     data_train_loader = torch.load(git_ignore_folder + 'data_train_loader')
    
    
    data_test_loader = torch.load(git_ignore_folder + 'data_test_loader')
    
    
    learning_rate_all_epochs = torch.load(git_ignore_folder + 'learning_rate_all_epochs')
    
    
#     delta_gradient_all_epochs = torch.load(git_ignore_folder + 'delta_gradient_all_epochs')
    
#     beta = torch.load(git_ignore_folder + 'beta')
    
#     hessian_matrix = torch.load(git_ignore_folder + 'hessian_matrix')
    
#     gradient_list = torch.load(git_ignore_folder + 'gradient_list')
    
    max_epoch = torch.load(git_ignore_folder+'epoch')
    
    print("max_epoch::", max_epoch)
    
#     X = torch.load(git_ignore_folder+'noise_X')
#     
#     Y = torch.load(git_ignore_folder+'noise_Y')
    
    
#     delta_data_ids = torch.load(git_ignore_folder + 'noise_data_ids')
    dim = dataset_train.data.data.shape

#     delta_size = int(dim[0]*0.1)
#     
#     print("delta_size::", delta_size)

    print("delta_size::", delta_data_ids.shape[0])
    
    
#     delta_data_ids = random_generate_subset_ids(dim, delta_size)
    selected_rows = get_subset_training_data(len(dataset_train), delta_data_ids)
#     update_X, update_Y, selected_rows = get_subset_training_data(X, Y, dim, delta_data_ids)

#     torch.save(delta_data_ids, git_ignore_folder + 'delta_data_ids')
    
    
    
#     update_X, update_Y, selected_rows = get_subset_training_data(X, Y, X.shape, delta_data_ids)
    
#     test_X = torch.load(git_ignore_folder + 'test_X')
#     
#     test_Y = torch.load(git_ignore_folder + 'test_Y')
#     
#     hidden_dim = torch.load(git_ignore_folder + 'hidden_dims')
    
#     delta_gradient_all_epochs = torch.load(git_ignore_folder + 'delta_gradient_all_epochs')
    
#     delta_all_epochs = torch.load(git_ignore_folder + 'delta_all_epochs')
    
#     old_para_list_all_epochs = torch.load(git_ignore_folder + "old_para_list")
    
    
    
#     input_dim = X.shape[1]
#     
#     num_class = torch.unique(Y).shape[0]
#     
#     output_dim = num_class
    
    
    
    model = model_class()# DNNModel(input_dim, hidden_dim, output_dim)
    
    
    if is_GPU:
        model.to(device)
    
    data_preparer = Data_preparer()
    
    
    hyper_para_function=getattr(Data_preparer, "get_hyperparameters_" + dataset_name)

    
    criterion, optimizer, lr_scheduler = hyper_para_function(data_preparer, model.parameters(), learning_rate, regularization_coeff)
    
    init_model(model,init_para_list)
    
    
#     print_model_para(model)
#     init_model(model, list(origin_model.parameters()))

#     hessian_para_list = torch.load(git_ignore_folder + 'hessian_para_list')
    
#     init_model(model, hessian_para_list)
    
#     error = nn.CrossEntropyLoss()
# 
# 
#     learning_rate = 0.1
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    
#     print("learning rate::", alpha)
    
    print("max_epoch::", max_epoch)
    
    batch_size = torch.load(git_ignore_folder + 'batch_size')
    
    sys_args = sys.argv
    init_epochs = int(sys_args[1])
    
    period = int(sys_args[2])

    t1 = time.time()

#     model_para_list = model_update_provenance_test(None, None, exp_gradient_list_all_epochs, exp_para_list_all_epochs, origin_X, origin_Y, model, S_k_list, Y_k_list, gradient_list_all_epochs, para_list_all_epochs, max_epoch, delta_data_ids, input_dim, hidden_dims, output_dim, 50, alpha, beta, selected_rows, error)
    
    
    
    
    model_para_list = model_update_provenance_test1_2(period, 1, init_epochs, None, None, exp_gradient_list_all_epochs, exp_para_list_all_epochs, dataset_train, model, gradient_list_all_epochs, para_list_all_epochs, max_epoch, delta_data_ids, 2, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device, 0.2)
#     model_para_list = model_update_provenance_test2(30, 1, init_epochs, None, None, exp_gradient_list_all_epochs, exp_para_list_all_epochs, origin_X, origin_Y, model, gradient_list_all_epochs, para_list_all_epochs, max_epoch, delta_data_ids, input_dim, hidden_dims, output_dim, 2, alpha, beta, selected_rows, error, delta_X, delta_Y, update_X, update_Y, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim)

#     model_para_list = model_update_provenance_test1_advanced(1, 20, 1, init_epochs, None, None, exp_gradient_list_all_epochs, exp_para_list_all_epochs, origin_X, origin_Y, model, gradient_list_all_epochs, para_list_all_epochs, max_epoch, delta_data_ids, input_dim, hidden_dims, output_dim, 3, alpha, beta, selected_rows, error, delta_X, delta_Y, update_X, update_Y, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim)

    t2 = time.time()
    print('time_provenance::', t2 - t1)


    compute_model_para_diff(list(model_base_line.parameters()), model_para_list)
    
    compute_model_para_diff(list(origin_model.parameters()), model_para_list)

#     compute_model_para_diff(list(origin_model.parameters()), list(model_base_line.parameters()))
    
    test(model, data_test_loader, criterion, len(dataset_test), is_GPU, device)
    
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
    
    
    