'''
Created on Mar 15, 2019


'''
'''
Created on Mar 15, 2019


'''
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import psutil
import torch
try:
    from sensitivity_analysis_SGD.multi_nomial_logistic_regression.incremental_updates_logistic_regression_multi_dim import *
except ImportError:
    from incremental_updates_logistic_regression_multi_dim import *


# def load_svd():
# 
# 
# #     u_list = torch.load()
# # 
# # #     s_list = []
# #     
# #     v_s_list = []
#         
#     u_list = torch.load(git_ignore_folder + 'u_list')
#     
# #     torch.save(s_list, git_ignore_folder + 's_list')
#     
#     v_s_list = torch.load(git_ignore_folder + 'v_s_list')
#     
#     return u_list, v_s_list


def load_svd(directory):
    u_list = []
 
#     s_list = []
     
    v_s_list = []
    
    directory = directory + 'svd_folder/'
    
    term1_len = torch.load(directory + 'len')[0]
    
    for i in range(term1_len):
        u_list.append(torch.from_numpy(np.load(directory + '/u_' + str(i) + '.npy')))
        v_s_list.append(torch.from_numpy(np.load(directory + '/v_' + str(i) + '.npy')))
    
    
    
    return u_list, v_s_list


if __name__ == '__main__':
    
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    X = torch.load(git_ignore_folder + 'noise_X')
        
    Y = torch.load(git_ignore_folder + 'noise_Y')
    
    t1 = time.time()
    
#     X_product = torch.load(git_ignore_folder + 'X_product')

    random_ids_multi_super_iterations = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations')


    sorted_ids_multi_super_iterations = torch.load(git_ignore_folder + 'sorted_ids_multi_super_iterations')
    
    
#     for i in range(random_ids_multi_super_iterations.shape[0]):
#         sorted_ids_multi_super_iterations.append(random_ids_multi_super_iterations[i].numpy().argsort())

    sys_args = sys.argv

    alpha = torch.load(git_ignore_folder + 'alpha')
    
    beta = torch.load(git_ignore_folder + 'beta')

    num_class = torch.unique(Y).shape[0]

    
    if X.shape[1]*num_class < 10000:
        M = torch.load(git_ignore_folder + 'eigen_vectors')
         
        s = torch.load(git_ignore_folder + 'eigen_values')
    #     s = s[:,0].view(-1)
         
        M_inverse = torch.load(git_ignore_folder + 'eigen_vectors_inverse')
    
    opt = bool(int(sys_args[1]))
    
    t2 = time.time()
    
    print(t2 - t1)
    
    learning_rate_all_epochs = torch.load(git_ignore_folder + 'learning_rate_all_epochs')
    
#     mini_epochs_per_super_iteration = int((X.shape[0] - 1)/batch_size) + 1
#     
#     super_iteration = (int((X.shape[0] - 1)/mini_epochs_per_super_iteration) + 1)
#     cut_off_super_iteration = int(super_iteration*theta_record_threshold)#(int((cut_off_epoch - 1)/mini_epochs_per_super_iteration) + 1)
    
#     
#     cut_off_epoch = cut_off_super_iteration*mini_epochs_per_super_iteration
    
    
    
    cut_off_epoch = torch.load(git_ignore_folder + 'cut_off_epoch')
    
    cut_off_epoch = int(cut_off_epoch*1.0/0.8*0.7)
    
    print('cut_off_epoch::', cut_off_epoch)
    
    weights = torch.load(git_ignore_folder+'weights')
    
    offsets = torch.load(git_ignore_folder+'offsets')
    
    batch_size = torch.load(git_ignore_folder + 'batch_size')
    
#     term1 = torch.load(git_ignore_folder+'term1')
     
    term2 = torch.load(git_ignore_folder+'term2')
    
    min_batch_num_per_epoch = int((X.shape[0] - 1)/batch_size) + 1


    if opt:
        
        cut_off_super_epochs = int(cut_off_epoch/min_batch_num_per_epoch)
        
        curr_weight = weights[0:int(cut_off_super_epochs*X.shape[0])]
        
        curr_offset = offsets[0:int(cut_off_super_epochs*X.shape[0])]


        curr_term2 = term2[0:cut_off_epoch]
        
        del weights, offsets, term2
        
        
        weights = curr_weight
        
        offsets = curr_offset
        
        term2 = curr_term2
        
        



#     avg_u = torch.load(git_ignore_folder + 'avg_u')
#     
#     avg_v_s = torch.load(git_ignore_folder + 'avg_v_s')
#     
#     
#     avg_term1 = torch.load(git_ignore_folder + 'avg_term1')
#     avg_term1 = torch.mean(term1[-min_batch_num_per_epoch:], 0)
    
    
    
     
    avg_term2 = torch.mean(term2[-min_batch_num_per_epoch:], 0)
#     
    x_sum_by_class_list = torch.load(git_ignore_folder+'x_sum_by_class')
    
    
    x_sum_by_class_list_copy = []
    
    for i in range(x_sum_by_class_list.shape[0]):
        x_sum_by_class_list_copy.append(x_sum_by_class_list[i])
        
    del x_sum_by_class_list
    
    x_sum_by_class_list = x_sum_by_class_list_copy
    
    
    max_epoch = torch.load(git_ignore_folder+'epoch')
    
    print('max_epoch::', max_epoch)
    
#     epoch_record_epoch_seq = torch.load(git_ignore_folder + 'epoch_record_epoch_seq')
    
    
    grad_list = torch.load(git_ignore_folder + 'grad_list')
    
    
    weights = weights.view(weights.shape[0], num_class, num_class)
    
    theta_list = torch.load(git_ignore_folder+'theta_list')
#     grad_list = torch.load(git_ignore_folder+'grad_list')    
#     X_theta_prod_softmax_seq_tensor = torch.load(git_ignore_folder + 'X_theta_prod_softmax_seq_tensor')
    
    delta_data_ids = torch.load(git_ignore_folder+'delta_data_ids')
    
#     delta_data_ids = torch.load(git_ignore_folder+'noise_data_ids')
    
    delta_data_ids = delta_data_ids.type(torch.LongTensor)

    print(delta_data_ids.shape)
    
    print(delta_data_ids[:100])
    
    delta_X = X[delta_data_ids]
    
    update_X, selected_rows = get_subset_training_data(X, X.shape, delta_data_ids)
    
    update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
#     batch_X_list, batch_Y_list = get_id_mappings_per_batch(X, Y, selected_rows, batch_size)
    u_list = None
    
    v_s_list = None
    
    term1 = None

#     if batch_size < num_class*X.shape[1]:
    u_list, v_s_list = load_svd(git_ignore_folder)
#     else:
#         term1 = torch.load(git_ignore_folder + 'term1')
# 
# 
# 
#         if opt:
#             curr_term1 = term1[0:cut_off_epoch]
#         
#             del term1
#         
#             term1 = curr_term1
#     delta_X = X[delta_ids]
            
    t1 = time.time()
    
    res3 = None
    
    if len(delta_data_ids) < (X.shape[0])/2:

#         t3_1 = time.time()


#         sub_weights = weights[delta_data_ids]#get_subset_parameter_list(selected_rows, delta_data_ids, w_seq, dim, 1)
#           
#         sub_offsets = offsets[delta_data_ids]#get_subset_parameter_list(b_seq, delta_data_ids, b_seq, dim, 1)
         
#         delta_X = X[delta_data_ids]
#          
#         delta_Y = Y[delta_data_ids]
         
#         delta_X_product = torch.bmm(delta_X.view(delta_X.shape[0], X.shape[1], 1), delta_X.view(delta_X.shape[0], 1, X.shape[1]))
         
         
#         delta_x_sum_by_class_list = compute_x_sum_by_class_by_batch2(X, Y, delta_data_ids, num_class, batch_size) 
        
#         print('delta_X_product_shape::', delta_X_product) 
        
#         delta_X_product = X_product[delta_data_ids] 
        
#         t3_2 = time.time()  
        
#         t3_5 = time.time()  
        
#         sub_term_1 = prepare_term_1_batch2_0(delta_X, sub_weights, delta_X.shape, max_epoch, num_class)#
         
#         sub_term_1 = prepare_term_1_batch2_0_delta(delta_X_product, sub_weights, delta_X.shape, max_epoch, num_class, x_sum_by_class_list, delta_x_sum_by_class_list)
#         init_theta = Variable(initialize(update_X, num_class).theta)

        print('rand_ids_multi_super_iter::', random_ids_multi_super_iterations[0].shape)

#         delta_term1, delta_term2, A, B = prepare_term_1_batch2_1_delta(random_ids_multi_super_iterations, alpha, beta, X, weights, offsets, X.shape, max_epoch, num_class, cut_off_epoch, batch_size, delta_data_ids, term1, term2, x_sum_by_class_list, delta_x_sum_by_class_list)
#         print(torch.norm(sub_term_1 - sub_term_1_0))
        
#         sub_term_2 = prepare_term_2_batch2(delta_X, sub_offsets, delta_X.shape, max_epoch, num_class)     
          
#         sub_x_sum_by_class = compute_x_sum_by_class(delta_X, delta_Y, num_class, delta_X.shape) 
         
#         t3_6 = time.time() 
        
#         init_theta = Variable(initialize(update_X, num_class).theta)
        
        
        init_para_list = list(torch.load(git_ignore_folder + 'init_para'))
    
#         lr = initialize(update_X, num_class)
        
        init_theta = Variable(init_para_list[0].T)
        
        
        if not opt:
            
#             res3 = compute_model_parameter_by_approx_incremental_1(A, B, delta_term1, delta_term2, X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class)
#             res2, total_time,theta_list, grad_list, output_list, exp_x_sum_by_class_list = compute_model_parameter_by_iteration2(batch_size, theta_list, grad_list, random_ids_multi_super_iterations, X.shape, init_theta, X, Y, selected_rows, num_class, max_epoch, alpha, beta)
#             res3, theta_list = compute_model_parameter_by_approx_incremental_4_4(s, M, M_inverse, [], X, Y, weights, offsets, delta_data_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, term1, term2, X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, x_sum_by_class_list, avg_term2, u_list, v_s_list)
            
            res3 = compute_model_parameter_by_approx_incremental_1_2(cut_off_epoch, [], [], theta_list, grad_list,  X, Y, weights, offsets, delta_data_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, term1, term2, X.shape, init_theta, max_epoch, learning_rate_all_epochs, beta, batch_size, num_class, x_sum_by_class_list, avg_term2, u_list, v_s_list)
            
#             if X.shape[1]*num_class > batch_size:
#                 res3 = compute_model_parameter_by_approx_incremental_1_3(weights, offsets, batch_size, [], [], random_ids_multi_super_iterations, X.shape, init_theta, X, Y, selected_rows, num_class, max_epoch, alpha, beta, x_sum_by_class_list)
#             else:
#                     res3 = compute_model_parameter_by_approx_incremental_3(delta_term1, delta_term2, A, B,  X.shape, init_theta, num_class, max_epoch, cut_off_epoch, epoch_record_epoch_seq, alpha, beta)
#                 res3 = compute_model_parameter_by_approx_incremental_1_2([], [], [], [], X, Y, weights, offsets, delta_data_ids, random_ids_multi_super_iterations, term1, term2, X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, x_sum_by_class_list, avg_term1, avg_term2)
#         t3_3 = time.time()
        else:
            
            '''s, M, M_inverse, theta_list, output_list, sub_term2_list, x_sum_by_list, sub_term_1_theta_list, origin_X, origin_Y, weights, offsets, delta_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, term1, term2, dim, theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, x_sum_by_class_list, avg_term2, u_list, v_s_list'''
            
            res3 = compute_model_parameter_by_approx_incremental_4_4(delta_X, s, M, M_inverse, [], X, Y, weights, offsets, delta_data_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, term1, term2, X.shape, init_theta, max_epoch, cut_off_epoch, learning_rate_all_epochs, beta, batch_size, num_class, x_sum_by_class_list, avg_term2, u_list, v_s_list)
            
            
#             res3 = compute_model_parameter_by_approx_incremental_4_3(delta_X, cut_off_epoch, term2, M, M_inverse, s, weights, offsets, batch_size, [], [], random_ids_multi_super_iterations, X.shape, init_theta, X, Y, selected_rows, num_class, max_epoch, alpha, beta)
            
#             res3 = compute_model_parameter_by_approx_incremental_4_2(term1 - sub_term_1, term2 - sub_term_2, x_sum_by_class - sub_x_sum_by_class, X.shape, init_theta, num_class, max_epoch, cut_off_epoch, epoch_record_epoch_seq, alpha, beta)
    
#         t3_4 = time.time()
    
    else:
        
        t3_1 = time.time()
        
        sub_weights = weights[selected_rows]#get_subset_parameter_list(selected_rows, delta_data_ids, w_seq, dim, 1)
          
        sub_offsets = offsets[selected_rows]#get_subset_parameter_list(b_seq, delta_data_ids, b_seq, dim, 1)
         
        delta_X = X[selected_rows]
         
        delta_Y = Y[selected_rows]
        
#         delta_X_product = X_product[selected_rows] 
         
#         delta_X_product = torch.bmm(delta_X.view(delta_X.shape[0], X.shape[1], 1), delta_X.view(delta_X.shape[0], 1, X.shape[1]))
        t3_2 = time.time()  
        
        
        t3_5 = time.time()  
        
        sub_term_1 = prepare_term_1_batch2_0(delta_X, sub_weights, delta_X.shape, max_epoch, num_class)
         
        sub_term_2 = prepare_term_2_batch2(delta_X, sub_offsets, delta_X.shape, max_epoch, num_class)     
          
        t3_6 = time.time() 
          
        sub_x_sum_by_class = compute_x_sum_by_class(delta_X, delta_Y, num_class, delta_X.shape) 
         
        
        
        init_theta = Variable(initialize(update_X, num_class).theta)
         
#         res3 = compute_model_parameter_by_approx_incremental_2(sub_term_1, sub_term_2, sub_x_sum_by_class, X.shape, init_theta, num_class, max_epoch)
        t3_3 = time.time()
        
        
        if not opt:
            res3 = compute_model_parameter_by_approx_incremental_3(sub_term_1, sub_term_2, sub_x_sum_by_class, X.shape, init_theta, num_class, max_epoch, cut_off_epoch, epoch_record_epoch_seq, alpha, beta)
        
        else:
            res3 = compute_model_parameter_by_approx_incremental_4(sub_term_1, sub_term_2, sub_x_sum_by_class, X.shape, init_theta, num_class, max_epoch, cut_off_epoch, epoch_record_epoch_seq, alpha, beta)

        t3_4 = time.time()
    
    t2 = time.time()
    
    
    process = psutil.Process(os.getpid())
    
    print('memory usage::', process.memory_info().rss)
    
    torch.save(res3, git_ignore_folder+'model_provenance')
    
    
    print('training_time_provenance::', t2 - t1)
    
#     print(t3_2 - t3_1)
#      
#     print(t3_4 - t3_3)
#      
#     print(t3_6 - t3_5)
    
    
#     model_origin = torch.load(git_ignore_folder+'model_origin')
    
    model_standard_lib = torch.load(git_ignore_folder+'model_standard_lib')
    
    model_iteration = torch.load(git_ignore_folder+'model_iteration')
    
    model_provenance = torch.load(git_ignore_folder+'model_provenance')
    
#     expect_updates = model_origin - model_iteration
#     
#     real_updates = model_origin - model_provenance
    
    
    error = torch.norm(model_provenance - model_iteration)/torch.norm(model_iteration)
    
#     print('model_standard_lib::', model_standard_lib)
#     
#     print('model_iteration::', model_iteration)
#     
#     print('model_prov::', model_provenance)
    
#     print('absolute_error::', torch.norm(model_provenance - model_standard_lib))
    
    print('absolute_error2::', torch.norm(model_provenance - model_iteration))
    
#     print('expect_updates::', torch.norm(model_provenance))
    
    print('angle::', torch.dot(torch.reshape(model_provenance, [-1]), torch.reshape(model_iteration, [-1]))/(torch.norm(torch.reshape(model_provenance, [-1]))*torch.norm(torch.reshape(model_iteration, [-1]))))

    
    print('relative_error::', error)
    
    test_X = torch.load(git_ignore_folder+'test_X')
    
    test_Y = torch.load(git_ignore_folder+'test_Y')
    
    print('training_accuracy::', compute_accuracy2(update_X, update_Y.type(torch.DoubleTensor), res3))
    
    print('test_accuracy::', compute_accuracy2(test_X, test_Y, res3))
    
#     print(res3*model_iteration > 0)
    
    print(res3.shape[0]*res3.shape[1])
    
    print(torch.nonzero(res3*model_iteration >= 0).shape)
    
    get_relative_change(res3, model_iteration)
    
    