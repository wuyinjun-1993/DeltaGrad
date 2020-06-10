'''
Created on Mar 15, 2019

'''
'''
Created on Mar 15, 2019

'''
import sys, os


import psutil

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
    
def get_perturbed_para_by_id(i, j, k, para, delta):

    updated_para = []
    
    for m in range(len(para)):
        if not m == i:
            updated_para.append(para[m].clone())
        else:
            curr_update_para = para[m].clone()
            
            if k < 0:
                curr_update_para[j] += delta
            else:
                curr_update_para[j][k] += delta
            
            updated_para.append(curr_update_para)


    return updated_para

def get_exp_grad_delta(para_delta, perturbed_para_all, perturbed_grad_all):
    
    scales = torch.mm(para_delta.view(1,-1), torch.inverse(perturbed_para_all))
#     for i in range(perturbed_para_all.shape[0]):
        
        
    res_grad_delta = torch.mm(scales.view(1,-1), perturbed_grad_all)
    
    return res_grad_delta
    

def populate_perturbed_para_grad_table(perturbed_para, first_para, first_grad, perturbed_para_all, perturbed_grad_all, X, Y, selected_rows, criterion, optimizer, model, id):
    
    init_model(model, perturbed_para)
        
#         compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
        
    compute_derivative_one_more_step(model, X[selected_rows], Y[selected_rows], criterion, optimizer)
    
    perturbed_grad = get_all_vectorized_parameters(model.get_all_gradient())

    perturbed_para_all[id] = get_all_vectorized_parameters(perturbed_para) - get_all_vectorized_parameters(first_para)
    
    perturbed_grad_all[id] = perturbed_grad - get_all_vectorized_parameters(first_grad)









def estimate_gradients(gradient_list_all_epochs, para_list_all_epochs, exp_gradient_list_all_epochs, exp_para_list_all_epochs, model, random_ids_all_epochs, batch_size, X, Y, criterion, optimizer, delta):
    
    
    
    para_ids = [8, 9, 10]
    
    
    perturbed_grad_all_diff_epochs = []
    
    perturbed_para_all_diff_epochs = []
    
    
    for para_id in para_ids:
    
    
#         para_id = 10
        
        first_para = para_list_all_epochs[para_id]
        
        first_grad = gradient_list_all_epochs[para_id]
        
        
        curr_random_ids = random_ids_all_epochs[0]
        
        selected_rows = curr_random_ids[batch_size*para_id: batch_size*(para_id + 1)]   
        
        init_model(model, first_para)
        
        compute_derivative_one_more_step(model, X[selected_rows], Y[selected_rows], criterion, optimizer)
        
        first_grad_2 = get_all_vectorized_parameters(model.get_all_gradient())
        
        delta_first_grad = get_all_vectorized_parameters(first_grad) - first_grad_2
        
        print(torch.norm(delta_first_grad))
        
        full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(first_para)
        
        perturbed_para_all = torch.zeros([total_shape_size, total_shape_size], dtype = torch.double)
        
        perturbed_grad_all = torch.zeros([total_shape_size, total_shape_size], dtype = torch.double)
        
        id = 0
        
        print(total_shape_size)
        
        for i in range(len(first_para)):
            
            print(id)
            
            curr_shape = first_para[i].shape
            
            if len(curr_shape) <= 1:
                for j in range(len(curr_shape)):
                    perturbed_para = get_perturbed_para_by_id(i, j, -1, first_para, delta)
                    
                    populate_perturbed_para_grad_table(perturbed_para, first_para, first_grad, perturbed_para_all, perturbed_grad_all, X, Y, selected_rows, criterion, optimizer, model, id)
                    
                    id += 1
                    
                    
            else:
                for j in range(curr_shape[0]):
                    for k in range(curr_shape[1]):
                        perturbed_para = get_perturbed_para_by_id(i, j, k, first_para, delta)
                        
                        
                        populate_perturbed_para_grad_table(perturbed_para, first_para, first_grad, perturbed_para_all, perturbed_grad_all, X, Y, selected_rows, criterion, optimizer, model, id)
        
                        id += 1
            
            
        
        exp_para = exp_para_list_all_epochs[para_id]
        
        para_delta = get_all_vectorized_parameters(exp_para) - get_all_vectorized_parameters(first_para)
        
        U, S, V =torch.svd(perturbed_grad_all)
        
        gradients_count = 100
        
        random_ids = torch.randperm(perturbed_grad_all.shape[0])
        
        approx_U, approx_S, approx_V =torch.svd(perturbed_grad_all[random_ids[0:gradients_count]])
        
        
        
        upper_bound = 100
        
        sub_s = S[0:upper_bound]
        
    
        sub_u = U[:,0:upper_bound]
         
        
         
        sub_v = V[:,0:upper_bound]
        
#         approx_perturbed_grad_all = torch.mm(sub_u*sub_s, torch.t(sub_v))
        
        approx_perturbed_grad_all = torch.mm(approx_V*approx_S, torch.t(approx_V))
        
        
        print(S[0:upper_bound])
        
        print(torch.norm(approx_perturbed_grad_all - perturbed_grad_all))
        
        
        
        
        res_delta_grad = get_exp_grad_delta(para_delta, perturbed_para_all, approx_perturbed_grad_all)
        
        res_grad = get_all_vectorized_parameters(first_grad).view(-1) + res_delta_grad.view(-1)
        
    #     exp_grad = get_all_vectorized_parameters(exp_gradient_list_all_epochs[para_id])
    
        init_model(model, exp_para)
        
        compute_derivative_one_more_step(model, X[selected_rows], Y[selected_rows], criterion, optimizer)
        
        exp_grad = get_all_vectorized_parameters(model.get_all_gradient())
        
        print(torch.norm(res_grad - exp_grad))
        
        perturbed_grad_all_diff_epochs.append(perturbed_grad_all)
    
    
    for i in range(len(perturbed_grad_all_diff_epochs) - 1):
        
        prev_perturbed_grad_all = perturbed_grad_all_diff_epochs[i]
        
        prev_U, prev_S, prev_V =torch.svd(prev_perturbed_grad_all)
        
        
        
        curr_perturbed_grad_all = perturbed_grad_all_diff_epochs[i + 1]
        
        curr_U, curr_S, curr_V =torch.svd(curr_perturbed_grad_all)
        
        delta_perturbed_grad = curr_perturbed_grad_all - prev_perturbed_grad_all 
        
        exp_curr_S = prev_S.view(-1) + torch.diag(torch.mm(torch.mm(torch.t(prev_U), delta_perturbed_grad), prev_V)).view(-1)
        
         
        print('curr_gap::', torch.max(torch.abs(exp_curr_S - curr_S)))
        
        print(torch.norm(prev_perturbed_grad_all - curr_perturbed_grad_all))
        
        print('here')
        
        
        
        


        
    print('here')
        
        
        
        
        
        
    
     
    
    
    


if __name__ == '__main__':
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
#     origin_model = torch.load(git_ignore_folder + 'origin_model')
    
    origin_model = torch.load(git_ignore_folder + 'origin_model')
    
    
    
    exp_gradient_list_all_epochs = torch.load(git_ignore_folder + 'expected_gradient_list_all_epochs')
      
    exp_para_list_all_epochs = torch.load(git_ignore_folder + 'expected_para_list_all_epochs')
    
    sub_v = torch.load(git_ignore_folder + 'sub_v')
    
    
    perturbed_para_all = torch.load(git_ignore_folder + 'perturbed_para_all')
    
    
    delta_para_sub_U_list = torch.load(git_ignore_folder + 'delta_para_sub_U_list')
    
    delta_para_sub_V_list = torch.load(git_ignore_folder + 'delta_para_sub_V_list')
    
    sorted_ids_multi_super_iterations = torch.load(git_ignore_folder + 'sorted_ids_multi_super_iterations')
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
    
#     estimate_gradients(gradient_list_all_epochs, para_list_all_epochs, exp_gradient_list_all_epochs, exp_para_list_all_epochs, model, random_ids_multi_super_iterations, batch_size, dataset_train.data, dataset_train.labels, criterion, optimizer, delta_data_ids.shape[0]*1.0*0.1/dataset_train.data.shape[0])

    
    sys_args = sys.argv
#     init_epochs = int(sys_args[1])
#     
#     period = int(sys_args[2])

#     repetition_time = int(sys_args[3])
    
#     deletion_rate = sys_args[4]
    
    cached_size = int(sys_args[1])

    para_list_all_epoch_tensor, grad_list_all_epoch_tensor = post_processing_gradien_para_list_all_epochs(para_list_all_epochs, gradient_list_all_epochs)

    end_cached_id = cached_size
    
    if end_cached_id > len(para_list_all_epochs):
        end_cached_id =  len(para_list_all_epochs)
    

    para_list_GPU_tensor = para_list_all_epoch_tensor[0:cached_size]
    
    grad_list_GPU_tensor = grad_list_all_epoch_tensor[0:cached_size]

    if is_GPU:
        para_list_GPU_tensor = para_list_GPU_tensor.to(device)
        
        grad_list_GPU_tensor = grad_list_GPU_tensor.to(device) 

    
    
    
    perturbed_para_all_inverse = torch.mm(torch.t(perturbed_para_all), torch.inverse(torch.mm(perturbed_para_all, torch.t(perturbed_para_all))))


    if is_GPU:
        perturbed_para_all_inverse = perturbed_para_all_inverse.to(device)

    t1 = time.time()
    
#     model_para_list = model_update_provenance_test(None, None, exp_gradient_list_all_epochs, exp_para_list_all_epochs, origin_X, origin_Y, model, S_k_list, Y_k_list, gradient_list_all_epochs, para_list_all_epochs, max_epoch, delta_data_ids, input_dim, hidden_dims, output_dim, 50, alpha, beta, selected_rows, error)
    
    model_para_list, res_para_list, res_grad_list = model_update_provenance_test4(10, 10, 2, perturbed_para_all, perturbed_para_all_inverse, sub_v, delta_para_sub_V_list, delta_para_sub_U_list, max_epoch, None, None, exp_gradient_list_all_epochs, exp_para_list_all_epochs, dataset_train, model, grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, delta_data_ids, 2, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device)
    
    
#     model_para_list = model_update_provenance_test1(period, 1, init_epochs, None, None, exp_gradient_list_all_epochs, exp_para_list_all_epochs, dataset_train, model, gradient_list_all_epochs, para_list_all_epochs, max_epoch, delta_data_ids, 2, learning_rate_all_epochs, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device)
#     model_para_list = model_update_provenance_test2(30, 1, init_epochs, None, None, exp_gradient_list_all_epochs, exp_para_list_all_epochs, origin_X, origin_Y, model, gradient_list_all_epochs, para_list_all_epochs, max_epoch, delta_data_ids, input_dim, hidden_dims, output_dim, 2, alpha, beta, selected_rows, error, delta_X, delta_Y, update_X, update_Y, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim)

#     model_para_list = model_update_provenance_test1_advanced(1, 20, 1, init_epochs, None, None, exp_gradient_list_all_epochs, exp_para_list_all_epochs, origin_X, origin_Y, model, gradient_list_all_epochs, para_list_all_epochs, max_epoch, delta_data_ids, input_dim, hidden_dims, output_dim, 3, alpha, beta, selected_rows, error, delta_X, delta_Y, update_X, update_Y, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim)

    t2 = time.time()
    
    process = psutil.Process(os.getpid())
    
    print('memory usage::', process.memory_info().rss)
    
    
    print('time_provenance::', t2 - t1)


    compute_model_para_diff(list(model_base_line.parameters()), model_para_list)
    
    compute_model_para_diff(list(origin_model.parameters()), model_para_list)

    init_model(model, model_para_list)

#     compute_model_para_diff(list(origin_model.parameters()), list(model_base_line.parameters()))
    
#     test(model, data_test_loader, criterion, len(dataset_test), is_GPU, device)
    test(model, dataset_test, batch_size, criterion, len(dataset_test), is_GPU, device)
    
#     torch.save(model, git_ignore_folder + 'incremental_provenance_' + str(init_epochs) + "_" + str(period) + "_" + str(repetition))

#     torch.save(model, git_ignore_folder + 'incremental_provenance_' + str(init_epochs) + "_" + str(period) + "_" + str(repetition_time) + "_" + deletion_rate + "_" + str(batch_size))
#     
#     torch.save(model_base_line, git_ignore_folder + 'model_base_line_' + str(init_epochs) + "_" + str(period) + "_" + str(repetition_time) + "_" + deletion_rate + "_" + str(batch_size))
    
    
    torch.save(res_para_list, git_ignore_folder + 'res_para_list')
    
    torch.save(res_grad_list, git_ignore_folder + 'res_grad_list')
    
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
    
    
    