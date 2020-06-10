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
    
    X = torch.load(git_ignore_folder + 'X')
        
    Y = torch.load(git_ignore_folder + 'Y')
    
    t1 = time.time()
    
    delta_id_list = torch.load(git_ignore_folder + 'delta_id_list')
    
    selected_id_list = torch.load(git_ignore_folder + 'selected_id_list')
    
#     X_product = torch.load(git_ignore_folder + 'X_product')

    random_ids_multi_super_iterations = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations')


    sorted_ids_multi_super_iterations = torch.load(git_ignore_folder + 'sorted_ids_multi_super_iterations')
    
    
#     for i in range(random_ids_multi_super_iterations.shape[0]):
#         sorted_ids_multi_super_iterations.append(random_ids_multi_super_iterations[i].numpy().argsort())

    sys_args = sys.argv

    alpha = torch.load(git_ignore_folder + 'alpha')
    
    beta = torch.load(git_ignore_folder + 'beta')
    
    M = torch.load(git_ignore_folder + 'eigen_vectors')
     
    s = torch.load(git_ignore_folder + 'eigen_values')
#     s = s[:,0].view(-1)
     
    M_inverse = torch.load(git_ignore_folder + 'eigen_vectors_inverse')
    
    opt = bool(int(sys_args[1]))
    
    t2 = time.time()
    
    print(t2 - t1)
    
    
    cut_off_epoch = torch.load(git_ignore_folder + 'cut_off_epoch')
    
#     cut_off_epoch = int(cut_off_epoch*1.0/0.8*0.7)
#     
#     print('cut_off_epoch::', cut_off_epoch)
    
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
    
#     theta_list = torch.load(git_ignore_folder + 'theta_list')
    
    num_class = torch.unique(Y).shape[0]
    
    
    
    weights = weights.view(weights.shape[0], num_class, num_class)
    
#     theta_list = torch.load(git_ignore_folder+'theta_list')
#     grad_list = torch.load(git_ignore_folder+'grad_list')    
    
    u_list = None
    
    v_s_list = None
    
    term1 = None

#     if batch_size < num_class*X.shape[1]:
    u_list, v_s_list = load_svd(git_ignore_folder)

    
    updated_model_list = []
    
    t1 = time.time()
    for ii in range(len(delta_id_list)):
#     for ii in range(3, 5):
    
        delta_data_ids = delta_id_list[ii]
    
        delta_data_ids = delta_data_ids.type(torch.LongTensor)

        print(ii, delta_data_ids.shape)
    
        init_theta = Variable(initialize(X, num_class).theta)
             
        if not opt:
            
#             updated_theta1 = compute_model_parameter_by_approx_incremental_4_4(s, M, M_inverse, [], X, Y, weights, offsets, delta_data_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, term1, term2, X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, x_sum_by_class_list, avg_term2, u_list, v_s_list)
#             
#             print(updated_theta1)
#             
#             print('x_sum_by_class::', x_sum_by_class_list[0].view(-1))
#             
#             print('u_list::', u_list[0])
#             
#             print('v_list::', v_s_list[0])
#             
#             print('weights::', weights[0].view(-1))
#             
#             print('offsets::', offsets[0].view(-1))
#             
#             print('delta_data_ids::', delta_data_ids[:10])
#             
#             print('random_ids::', random_ids_multi_super_iterations[0][0:100])
#             
#             print('term2::', term2[0].view(-1))
            
            updated_theta = compute_model_parameter_by_approx_incremental_1_2(cut_off_epoch, [], [], [], [],  X, Y, weights, offsets, delta_data_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, term1, term2, X.shape, init_theta, max_epoch, alpha, beta, batch_size, num_class, x_sum_by_class_list, avg_term2, u_list, v_s_list)
            
        else:
            delta_X = X[delta_data_ids]
            '''s, M, M_inverse, theta_list, origin_X, origin_Y, weights, offsets, delta_ids, random_ids_multi_super_iterations, term1, term2, dim, theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, x_sum_by_class_list'''
            
            updated_theta = compute_model_parameter_by_approx_incremental_4_4(delta_X, s, M, M_inverse, [], X, Y, weights, offsets, delta_data_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, term1, term2, X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, x_sum_by_class_list, avg_term2, u_list, v_s_list)
    
#             print(updated_theta)
    
        updated_model_list.append(updated_theta)
    
    
    t2 = time.time()
    
    
    print('training_time_provenance::', t2 - t1)
    
    process = psutil.Process(os.getpid())
    
    print('memory usage::', process.memory_info().rss)
    
#     torch.save(res3, git_ignore_folder+'model_provenance')
    
    model_iteration_list = torch.load(git_ignore_folder+'model_iteration_list')
    
    model_standard_lib_list = torch.load(git_ignore_folder+'model_standard_lib_list')

    test_X = torch.load(git_ignore_folder+'test_X')
        
    test_Y = torch.load(git_ignore_folder+'test_Y')
    
    
    for ii in range(len(updated_model_list)):
        
        delta_data_ids = delta_id_list[ii]
        
        selected_rows = selected_id_list[ii]
    
        update_X = X[selected_rows]# = get_subset_training_data(X, X.shape, delta_data_ids)
        
        update_Y = Y[selected_rows]#, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
        
        model_provenance = updated_model_list[ii]
    
#         model_origin = torch.load(git_ignore_folder+'model_origin')
    
#         #torch.load(git_ignore_folder+'model_standard_lib')
        
        model_iteration = model_iteration_list[ii]#torch.load(git_ignore_folder+'model_iteration')
        
#         model_provenance = torch.load(git_ignore_folder+'model_provenance')
#         
#         expect_updates = model_origin - model_iteration
#         
#         real_updates = model_origin - model_provenance
        
        
        error = torch.norm(model_provenance - model_iteration)/torch.norm(model_iteration)
        
    #     print('model_standard_lib::', model_standard_lib)
    #     
    #     print('model_iteration::', model_iteration)
    #     
    #     print('model_prov::', model_provenance)
        
#         print('absolute_error::', torch.norm(expect_updates - real_updates))
#         model_standard_lib = model_standard_lib_list[ii]
#         print('absolute_error::', torch.norm(model_provenance - model_standard_lib))
        
        print('absolute_error2::', torch.norm(model_provenance - model_iteration))
        
#         print('expect_updates::', torch.norm(expect_updates))
        
        print('angle::', torch.dot(torch.reshape(model_provenance, [-1]), torch.reshape(model_iteration, [-1]))/(torch.norm(torch.reshape(model_provenance, [-1]))*torch.norm(torch.reshape(model_iteration, [-1]))))
    
        
        print('relative_error::', error)
        
        
        print('training_accuracy::', compute_accuracy2(update_X, update_Y.type(torch.DoubleTensor), model_provenance))
        
        print('test_accuracy::', compute_accuracy2(test_X, test_Y, model_provenance))
        
    #     print(res3*model_iteration > 0)
        
        print(model_provenance.shape[0]*model_provenance.shape[1])
        
        print(torch.nonzero(model_provenance*model_iteration >= 0).shape)
        
        get_relative_change(model_provenance, model_iteration)

    
    
    
    
    
    
    