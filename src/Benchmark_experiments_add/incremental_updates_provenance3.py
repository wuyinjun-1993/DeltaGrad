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
import time

try:
    from benchmark_exp import *
    from data_IO.Load_data import *
    from Models.Data_preparer import *

except ImportError:
    from benchmark_exp import *
    from Load_data import *
    from Data_preparer import *




if __name__ == '__main__':
#     configs = load_config_data(config_file)
    
    git_ignore_folder = '../../.gitignore/'# configs['git_ignore_folder']
    
#     origin_model = torch.load(git_ignore_folder + 'origin_model')
    
    origin_model = torch.load(git_ignore_folder + 'origin_model')
    
    
    
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
#     random_ids_multi_epochs = torch.load(git_ignore_folder + 'random_ids_multi_epochs')
# 
#     sorted_ids_multi_epochs = torch.load(git_ignore_folder + 'sorted_ids_multi_epochs')
#     curr_S_k_list, curr_Y_k_list = compute_curr_S_K_Y_K_list(S_k_list, Y_k_list, exp_para_list_all_epochs, exp_gradient_list_all_epochs, para_list_all_epochs, gradient_list_all_epochs)
    
    
    
    
    added_random_ids_multi_super_iteration = torch.load(git_ignore_folder + 'added_random_ids_multi_super_iteration')
    
    added_batch_size = torch.load(git_ignore_folder + 'added_batch_size')

    X_to_add = torch.load(git_ignore_folder + 'X_to_add')
            
    Y_to_add = torch.load(git_ignore_folder + 'Y_to_add')
    
    
    



    
    
#     max_epoch = torch.load(git_ignore_folder + 'update_max_epochs')
    
#     X = torch.load(git_ignore_folder+'noise_X')
    
#     hessian_matrix = torch.load(git_ignore_folder + 'hessian_matrix')
    
    sorted_ids_multi_epochs = torch.load(git_ignore_folder + 'sorted_ids_multi_epochs')
#     model_base_line = torch.load(git_ignore_folder + 'model_base_line')
    
    model_base_line = torch.load(git_ignore_folder + 'model_base_line')
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
    
    device = torch.load(git_ignore_folder + 'device')
    
    is_GPU = torch.load(git_ignore_folder + 'is_GPU')
    
#     [criterion, optimizer, lr_scheduler] = hyper_params

    
#     alpha = torch.load(git_ignore_folder + 'alpha')
#     
#     beta = torch.load(git_ignore_folder + 'beta')
    
    
    init_para_list = list(torch.load(git_ignore_folder + 'init_para'))
    
    random_ids_multi_epochs = torch.load(git_ignore_folder + 'random_ids_multi_epochs')
    
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
    selected_rows = get_subset_training_data0(len(dataset_train), delta_data_ids)
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
    
    dim = [len(dataset_train), len(dataset_train[0][0])]
    
    data_preparer = Data_preparer()
    
    num_class = get_data_class_num_by_name(data_preparer, dataset_name)
    
    model = model_class(dim[1], num_class)# DNNModel(input_dim, hidden_dim, output_dim)
    
    
    if is_GPU:
        model.to(device)
    
    
    
    
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

    repetition_time = int(sys_args[3])
    
    deletion_rate = sys_args[4]

    t1 = time.time()

#     model_para_list = model_update_provenance_test(None, None, exp_gradient_list_all_epochs, exp_para_list_all_epochs, origin_X, origin_Y, model, S_k_list, Y_k_list, gradient_list_all_epochs, para_list_all_epochs, max_epoch, delta_data_ids, input_dim, hidden_dims, output_dim, 50, alpha, beta, selected_rows, error)
    
    
    
    
    model_para_list = model_update_provenance_test1(init_epochs, 1, init_epochs, None, None, None, None, dataset_train, model, gradient_list_all_epochs, para_list_all_epochs, max_epoch, period, learning_rate_all_epochs, random_ids_multi_epochs, sorted_ids_multi_epochs, batch_size, dim, added_random_ids_multi_super_iteration, added_batch_size, X_to_add, Y_to_add, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device)
#     model_para_list = model_update_provenance_test2(30, 1, init_epochs, None, None, exp_gradient_list_all_epochs, exp_para_list_all_epochs, origin_X, origin_Y, model, gradient_list_all_epochs, para_list_all_epochs, max_epoch, delta_data_ids, input_dim, hidden_dims, output_dim, 2, alpha, beta, selected_rows, error, delta_X, delta_Y, update_X, update_Y, random_ids_multi_epochs, sorted_ids_multi_epochs, batch_size, dim)

#     model_para_list = model_update_provenance_test1_advanced(1, 20, 1, init_epochs, None, None, exp_gradient_list_all_epochs, exp_para_list_all_epochs, origin_X, origin_Y, model, gradient_list_all_epochs, para_list_all_epochs, max_epoch, delta_data_ids, input_dim, hidden_dims, output_dim, 3, alpha, beta, selected_rows, error, delta_X, delta_Y, update_X, update_Y, random_ids_multi_epochs, sorted_ids_multi_epochs, batch_size, dim)

    t2 = time.time()
    print('time_provenance::', t2 - t1)


    compute_model_para_diff(list(model_base_line.parameters()), model_para_list)
    
    compute_model_para_diff(list(origin_model.parameters()), model_para_list)

#     compute_model_para_diff(list(origin_model.parameters()), list(model_base_line.parameters()))
    
    init_model(model, model_para_list)

#     compute_model_para_diff(list(origin_model.parameters()), list(model_base_line.parameters()))
    
#     test(model, data_test_loader, criterion, len(dataset_test), is_GPU, device)
    test(model, dataset_test, batch_size, criterion, len(dataset_test), is_GPU, device)
    
#     torch.save(model, git_ignore_folder + 'incremental_provenance_' + str(init_epochs) + "_" + str(period) + "_" + str(repetition))

    torch.save(model, git_ignore_folder + 'incremental_provenance_' + str(init_epochs) + "_" + str(period) + "_" + str(repetition_time) + "_" + deletion_rate + "_" + str(batch_size))
    
    torch.save(model_base_line, git_ignore_folder + 'model_base_line_' + str(init_epochs) + "_" + str(period) + "_" + str(repetition_time) + "_" + deletion_rate + "_" + str(batch_size))
    

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
    
    
    