'''
Created on Mar 15, 2019


'''
'''
Created on Mar 15, 2019


'''
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    
    t1 = time.time()
    
#     X_product = torch.load(git_ignore_folder + 'X_product')

    sys_args = sys.argv

    alpha = torch.load(git_ignore_folder + 'alpha')
    
    beta = torch.load(git_ignore_folder + 'beta')
    
    opt = bool(int(0))
    
    t2 = time.time()
    
    print(t2 - t1)
    
    
    cut_off_epoch = torch.load(git_ignore_folder + 'cut_off_epoch')
    
    
    print('cut_off_epoch::', cut_off_epoch)
    
    weights = torch.load(git_ignore_folder+'weights')
    
    offsets = torch.load(git_ignore_folder+'offsets')
    
    batch_size = torch.load(git_ignore_folder + 'batch_size')
    
    term1 = torch.load(git_ignore_folder+'term1')
    
    term2 = torch.load(git_ignore_folder+'term2')
    
    x_sum_by_class_list = torch.load(git_ignore_folder+'x_sum_by_class')
    
    max_epoch = torch.load(git_ignore_folder+'epoch')
    
    epoch_record_epoch_seq = torch.load(git_ignore_folder + 'epoch_record_epoch_seq')
    
    num_class = torch.unique(Y).shape[0]
    
#     delta_data_ids = torch.load(git_ignore_folder+'delta_data_ids')
    
    delta_data_ids = torch.load(git_ignore_folder+'noise_data_ids')
    
    delta_data_ids = delta_data_ids.type(torch.LongTensor)

    print(delta_data_ids.shape)
    
    print(delta_data_ids[:100])
    
    update_X, selected_rows = get_subset_training_data(X, X.shape, delta_data_ids)
    
    update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
#     batch_X_list, batch_Y_list = get_id_mappings_per_batch(X, Y, selected_rows, batch_size)

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
         
         
        delta_x_sum_by_class_list = compute_x_sum_by_class_by_batch2(X, Y, delta_data_ids, num_class, batch_size) 
        
#         print('delta_X_product_shape::', delta_X_product) 
        
#         delta_X_product = X_product[delta_data_ids] 
        
#         t3_2 = time.time()  
        
#         t3_5 = time.time()  
        
#         sub_term_1 = prepare_term_1_batch2_0(delta_X, sub_weights, delta_X.shape, max_epoch, num_class)#
         
#         sub_term_1 = prepare_term_1_batch2_0_delta(delta_X_product, sub_weights, delta_X.shape, max_epoch, num_class, x_sum_by_class_list, delta_x_sum_by_class_list)
#         init_theta = Variable(initialize(update_X, num_class).theta)

        delta_term1, delta_term2, A, B = prepare_term_1_batch2_0_delta(alpha, beta, X, weights, offsets, X.shape, max_epoch, num_class, cut_off_epoch, batch_size, delta_data_ids, term1, term2, x_sum_by_class_list, delta_x_sum_by_class_list)
#         print(torch.norm(sub_term_1 - sub_term_1_0))
        
#         sub_term_2 = prepare_term_2_batch2(delta_X, sub_offsets, delta_X.shape, max_epoch, num_class)     
          
#         sub_x_sum_by_class = compute_x_sum_by_class(delta_X, delta_Y, num_class, delta_X.shape) 
         
#         t3_6 = time.time() 
        
        init_theta = Variable(initialize(update_X, num_class).theta)
         
         
        if not opt:
            
            res3 = compute_model_parameter_by_approx_incremental_1(A, B, delta_term1, delta_term2, X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class)
            
#             res3 = compute_model_parameter_by_approx_incremental_3(delta_term1, delta_term2, A, B,  X.shape, init_theta, num_class, max_epoch, cut_off_epoch, epoch_record_epoch_seq, alpha, beta)
        
#         t3_3 = time.time()
        else:
            res3 = compute_model_parameter_by_approx_incremental_4(term1 - sub_term_1, term2 - sub_term_2, x_sum_by_class - sub_x_sum_by_class, X.shape, init_theta, num_class, max_epoch, cut_off_epoch, epoch_record_epoch_seq, alpha, beta)
    
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
    
    
    torch.save(res3, git_ignore_folder+'model_provenance')
    
    
    print('training_time_provenance::', t2 - t1)
    
#     print(t3_2 - t3_1)
#      
#     print(t3_4 - t3_3)
#      
#     print(t3_6 - t3_5)
    
    
    model_origin = torch.load(git_ignore_folder+'model_origin')
    
    model_standard_lib = torch.load(git_ignore_folder+'model_standard_lib')
    
    model_iteration = torch.load(git_ignore_folder+'model_iteration')
    
    model_provenance = torch.load(git_ignore_folder+'model_provenance')
    
    expect_updates = model_origin - model_standard_lib
    
    real_updates = model_origin - model_provenance
    
    
    error = torch.norm(model_provenance - model_standard_lib)/torch.norm(model_standard_lib)
    
    print('model_standard_lib::', model_standard_lib)
    
    print('model_iteration::', model_iteration)
    
    print('model_prov::', model_provenance)
    
    print('absolute_error::', torch.norm(expect_updates - real_updates))
    
    print('absolute_error2::', torch.norm(model_provenance - model_iteration))
    
    print('expect_updates::', torch.norm(expect_updates))
    
    print('angle::', torch.dot(torch.reshape(model_provenance, [-1]), torch.reshape(model_standard_lib, [-1]))/(torch.norm(torch.reshape(model_provenance, [-1]))*torch.norm(torch.reshape(model_standard_lib, [-1]))))

    
    print('relative_error::', error)
    
    test_X = torch.load(git_ignore_folder+'test_X')
    
    test_Y = torch.load(git_ignore_folder+'test_Y')
    
    print('training_accuracy::', compute_accuracy2(update_X, update_Y.type(torch.DoubleTensor), res3))
    
    print('test_accuracy::', compute_accuracy2(test_X, test_Y, res3))
    
    