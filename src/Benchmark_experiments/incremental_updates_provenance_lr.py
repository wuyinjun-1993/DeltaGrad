'''
Created on Mar 15, 2019

'''
import torch
import psutil
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from data_IO.Load_data import *
# from sensitivity_analysis.logistic_regression.Logistic_regression import test_X
    from utils import *
#     from sensitivity_analysis.linear_regression.evaluating_test_samples import *
    from benchmark_exp import *

    from Models.Data_preparer import *

except ImportError:
    from Load_data import *
# from sensitivity_analysis.logistic_regression.Logistic_regression import test_X
    from utils import *
    from Models.Data_preparer import *
#     from evaluating_test_samples import *
    from benchmark_exp import *



def get_provenance_info(git_ignore_folder, dataset_train_len, batch_size, num_class, opt):
    
    cut_off_epoch = torch.load(git_ignore_folder + 'cut_off_epoch')
    
#     cut_off_epoch = int(cut_off_epoch*1.0/0.8*0.7)
    
    print('cut_off_epoch::', cut_off_epoch)
    
    weights = torch.load(git_ignore_folder+'weights')
    
    offsets = torch.load(git_ignore_folder+'offsets')
    
#     batch_size = torch.load(git_ignore_folder + 'batch_size')
    
#     term1 = torch.load(git_ignore_folder+'term1')
     
    term2 = torch.load(git_ignore_folder+'term2')
    
    min_batch_num_per_epoch = int((dataset_train_len - 1)/batch_size) + 1


    if opt:
        
        cut_off_super_epochs = int(cut_off_epoch/min_batch_num_per_epoch)
        
        curr_weight = weights[0:int(cut_off_super_epochs*dataset_train_len)]
        
        curr_offset = offsets[0:int(cut_off_super_epochs*dataset_train_len)]


        curr_term2 = term2[0:cut_off_epoch]
        
        del weights, offsets, term2
        
        
        weights = curr_weight
        
        offsets = curr_offset
        
        term2 = curr_term2


    avg_term2 = torch.mean(term2[-min_batch_num_per_epoch:], 0)
#     
    x_sum_by_class_list = torch.load(git_ignore_folder+'x_sum_by_class')
    
    
    x_sum_by_class_list_copy = []
    
    for i in range(x_sum_by_class_list.shape[0]):
        x_sum_by_class_list_copy.append(x_sum_by_class_list[i])
        
    del x_sum_by_class_list
    
    x_sum_by_class_list = x_sum_by_class_list_copy
    
    weights = weights.view(weights.shape[0], num_class, num_class)

    return weights, offsets, x_sum_by_class_list, term2, avg_term2


if __name__ == '__main__':
    
    
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    sys_args = sys.argv
    
    opt = bool(int(sys_args[1]))

    dataset_train = torch.load(git_ignore_folder + "dataset_train")
    
    dataset_test = torch.load(git_ignore_folder + "test_data")
    

    learning_rate = torch.load(git_ignore_folder + 'alpha')

    regularization_coeff = torch.load(git_ignore_folder + 'beta')

#     hyper_params = torch.load(git_ignore_folder + 'hyper_params')
    dataset_name = torch.load(git_ignore_folder + 'dataset_name')

    origin_model = torch.load(git_ignore_folder + 'origin_model')
    
    
    gradient_list_all_epochs = torch.load(git_ignore_folder + 'gradient_list_all_epochs')
    
    para_list_all_epochs = torch.load(git_ignore_folder + 'para_list_all_epochs')
    
    learning_rate_all_epochs = torch.load(git_ignore_folder + 'learning_rate_all_epochs')
    
    device = torch.load(git_ignore_folder + 'device')
    
    is_GPU = torch.load(git_ignore_folder + 'is_GPU')
    
    
    exp_gradient_list_all_epochs = torch.load(git_ignore_folder + 'expected_gradient_list_all_epochs')
      
    exp_para_list_all_epochs = torch.load(git_ignore_folder + 'expected_para_list_all_epochs')
    
    
    
#     noise_rate = float(sys_args[1])#torch.load(git_ignore_folder + 'noise_rate')
    
    data_preparer = Data_preparer()
    dim = [len(dataset_train), len(dataset_train[0][0])]
    
    num_class = get_data_class_num_by_name(data_preparer, dataset_name)
    
    sorted_ids_multi_super_iterations = torch.load(git_ignore_folder + 'sorted_ids_multi_super_iterations')
    
#     record_params = bool(int(sys_args[2]))
    
#     delta_data_ids = random_deletion(len(dataset_train), int(len(dataset_train)*noise_rate))
    
#     delta_data_ids = random_deletion(len(dataset_train), 1)
    
    delta_data_ids = torch.load(git_ignore_folder + "delta_data_ids")
    
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
    
    
    
    model = model_class(dim[1], num_class)# DNNModel(input_dim, hidden_dim, output_dim)
    
    if is_GPU:
        model.to(device)
        
    
    batch_size = torch.load(git_ignore_folder + 'batch_size')
    
    weights, offsets, x_sum_by_class_list, term2, avg_term2 = get_provenance_info(git_ignore_folder, len(dataset_train), batch_size, num_class, opt)
    
#     weights = weights.view(weights.shape[0], num_class, num_class)
    
    
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
    
    
    
#     random.seed(random_seed)
#     os.environ['PYTHONHASHSEED'] = str(random_seed)
#     np.random.seed(random_seed)
#     torch.manual_seed(random_seed)
#     torch.cuda.manual_seed(random_seed)
#     torch.cuda.manual_seed_all(random_seed) # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = False
    print("batch_size::", batch_size)
    
#     model, gradient_list, res_list, count = model_training(max_epoch, update_X, update_Y, test_X, test_Y, alpha, error, model)
    
    
    
#     max_epoch = 150
    
#     model, count = model_update_standard_lib_stochastic(batch_size, max_epoch, update_X, update_Y, alpha, error, model)

    dim = [len(dataset_train), len(dataset_train[0][0])]
    
    u_list, v_s_list = load_svd(git_ignore_folder)
    
    t1 = time.time()
    
    
    
#     if not opt:
            
#             res3 = compute_model_parameter_by_approx_incremental_1(A, B, delta_term1, delta_term2, X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class)
#             res2, total_time,theta_list, grad_list, output_list, exp_x_sum_by_class_list = compute_model_parameter_by_iteration2(batch_size, theta_list, grad_list, random_ids_multi_super_iterations, X.shape, init_theta, X, Y, selected_rows, num_class, max_epoch, alpha, beta)
#             res3, theta_list = compute_model_parameter_by_approx_incremental_4_4(s, M, M_inverse, [], X, Y, weights, offsets, delta_data_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, term1, term2, X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, x_sum_by_class_list, avg_term2, u_list, v_s_list)
            
    res3 = compute_model_parameter_by_approx_incremental_1_2(cut_off_epoch, exp_para_list_all_epochs, exp_gradient_list_all_epochs, None, [],  dataset_train, weights, offsets, delta_data_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, term2, dim, init_para_list[0].data.T, max_epoch, learning_rate_all_epochs, regularization_coeff, batch_size, num_class, x_sum_by_class_list, avg_term2, u_list, v_s_list, is_GPU, device)
            
#             if X.shape[1]*num_class > batch_size:
#                 res3 = compute_model_parameter_by_approx_incremental_1_3(weights, offsets, batch_size, [], [], random_ids_multi_super_iterations, X.shape, init_theta, X, Y, selected_rows, num_class, max_epoch, alpha, beta, x_sum_by_class_list)
#             else:
#                     res3 = compute_model_parameter_by_approx_incremental_3(delta_term1, delta_term2, A, B,  X.shape, init_theta, num_class, max_epoch, cut_off_epoch, epoch_record_epoch_seq, alpha, beta)
#                 res3 = compute_model_parameter_by_approx_incremental_1_2([], [], [], [], X, Y, weights, offsets, delta_data_ids, random_ids_multi_super_iterations, term1, term2, X.shape, init_theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, x_sum_by_class_list, avg_term1, avg_term2)
#         t3_3 = time.time()
#     else:
#         
#         '''s, M, M_inverse, theta_list, output_list, sub_term2_list, x_sum_by_list, sub_term_1_theta_list, origin_X, origin_Y, weights, offsets, delta_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, term1, term2, dim, theta, max_epoch, cut_off_epoch, alpha, beta, batch_size, num_class, x_sum_by_class_list, avg_term2, u_list, v_s_list'''
#         
#         res3 = compute_model_parameter_by_approx_incremental_4_4(delta_X, s, M, M_inverse, [], X, Y, weights, offsets, delta_data_ids, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, term2, X.shape, init_theta, max_epoch, cut_off_epoch, learning_rate_all_epochs, regularization_coeff, batch_size, num_class, x_sum_by_class_list, avg_term2, u_list, v_s_list)

    
    
    
    
#     model, count, exp_para_list_all_epochs, exp_gradient_list_all_epochs = model_update_standard_lib(max_epoch, dataset_train, model, random_ids_multi_super_iterations, selected_rows, batch_size, learning_rate_all_epochs, criterion, optimizer, para_list_all_epochs, gradient_list_all_epochs, is_GPU, device, record_params)

    t2 = time.time()
    
    process = psutil.Process(os.getpid())
    
    print('memory usage::', process.memory_info().rss)
    
    print('time_provenance0::', t2 - t1)
    cut_off_epoch = max_epoch

    model_base_line = torch.load(git_ignore_folder + 'model_base_line')    

#     model, gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, epoch = model_training(max_epoch, update_X, update_Y, test_X, test_Y, alpha, beta, error, model, True, batch_size, dim)

#     compute_model_parameter_iteration(max_epoch, model, update_X, update_Y, alpha, error, update_X.shape, num_class, input_dim, hidden_dim, output_dim, old_para_list_all_epochs, delta_gradient_all_epochs, delta_all_epochs)
    
#     print_model_para(model)
    
    
    
    model_provenance = res3.T
#     compute_model_para_diff(list(model_base_line.parameters()), list(model.parameters()))
    model_standard_lib = list(model_base_line.parameters())[0].data

    print('absolute_error::', torch.norm(res3.T - model_standard_lib))
    
#     print('absolute_error2::', torch.norm(model_provenance - model_iteration))
    
#     print('expect_updates::', torch.norm(model_provenance))
    
    print('angle::', torch.dot(torch.reshape(model_provenance, [-1]), torch.reshape(model_standard_lib, [-1]))/(torch.norm(torch.reshape(model_provenance, [-1]))*torch.norm(torch.reshape(model_standard_lib, [-1]))))





    
    
#     torch.save(model, git_ignore_folder + 'model_base_line')    
    
#     compute_derivative_one_more_step(model, error, X, Y)
     
#     origin_gradient_list = torch.load(git_ignore_folder + 'gradient_list')
    
    
#     loss = torch.load(git_ignore_folder + 'loss')
    
    
#     torch.save(exp_gradient_list_all_epochs, git_ignore_folder + 'expected_gradient_list_all_epochs')
#       
#     torch.save(exp_para_list_all_epochs, git_ignore_folder + 'expected_para_list_all_epochs')
#     
#     torch.save(max_epoch, git_ignore_folder + 'update_max_epochs')
    
    
#     torch.save(batch_size, git_ignore_folder + 'batch_size')
    
#     batch_num = update_X.shape[0]/batch_size
    
#     torch.save(batch_num, git_ignore_folder + 'batch_num')
    
    init_model(model, list([res3.T]))
    
    test(model, data_test_loader, criterion, len(dataset_test), is_GPU, device)
    
#     compute_test_acc(model, test_X, test_Y)
    
#     compute_derivative_one_more_step(model, error, X, Y)
#     
#     gradient_list2, para_list2 = compute_gradient_iteration(model, input_dim, hidden_dim, output_dim, X, Y)
#     
#     
#     
#     hessian_para_list = torch.load(git_ignore_folder + 'hessian_para_list')
#     
#     hessian_gradient_list = torch.load(git_ignore_folder + 'hessian_gradient_list')
#     
#     
#     
#     updated_gradient = torch.mm(get_all_vectorized_parameters(para_list2) - get_all_vectorized_parameters(hessian_para_list), hessian_matrix) + get_all_vectorized_parameters(hessian_gradient_list)
#     
#     print(torch.norm((updated_gradient - get_all_vectorized_parameters(gradient_list2))/X.shape[0]))
    
    
    
    
    
#     num_of_output = Y.shape[1]
#     
#     dim = X.shape
#     
#     print(X.shape)
#     
#     Y = Y.type(torch.DoubleTensor)
#     
#     sys_args = sys.argv
# 
#     
#     delta_num = int(10000)
# 
#     
#      
# #     delta_data_ids = random_generate_subset_ids(X.shape, delta_num)     
# #     torch.save(delta_data_ids, git_ignore_folder+'delta_data_ids')
# 
#     delta_data_ids = torch.load(git_ignore_folder + 'noise_data_ids')
#     
#     print(delta_data_ids.shape[0])
#     
#     max_epoch = torch.load(git_ignore_folder+'epoch')
#     
#     print(max_epoch)
# #     num_class = torch.unique(Y).shape[0]
#     
# #     delta_data_ids = torch.load(git_ignore_folder+'delta_data_ids')
#     
#     update_X, selected_rows = get_subset_training_data(X, X.shape, delta_data_ids)
#     
#     print(selected_rows.shape)
#     
#     update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
#     
#     #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
#     
#     t1 = time.time()
#     
#     lr = initialize(update_X.shape, num_of_output)
# #     update_x_sum_by_class = compute_x_sum_by_class(update_X, update_Y, num_class, update_X.shape)
#     #     update_X_Y_mult = update_X.mul(update_Y)
#     #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)dim, theta,  X, Y, X_sum_by_class, num_class
#     res2 = linear_regression_standard_library(update_X, update_Y, lr, update_X.shape, max_epoch, alpha, beta)
# 
#     
#     t2 = time.time()
#     
#     
#     torch.save(res2, git_ignore_folder+'model_standard_lib')
#     
#     
#     test_X = torch.load(git_ignore_folder + 'test_X')
#     
#     test_Y = torch.load(git_ignore_folder + 'test_Y')
#     
#     print('training_accuracy::', compute_accuracy2(update_X, update_Y, res2))
#     
#     print('test_accuracy::', compute_accuracy2(test_X, test_Y, res2))
#     
#     print('training_time_standard_lib::', t2 - t1)
#     
#     print(res2)
    
    
    
    
    
    