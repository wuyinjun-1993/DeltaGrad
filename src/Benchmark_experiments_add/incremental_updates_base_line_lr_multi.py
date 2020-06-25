'''
Created on Mar 15, 2019

'''
import torch

import sys, os, time

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


def get_sampling_each_iteration(random_ids_multi_super_iterations, add_num):


    added_random_ids_multi_super_iteration = []

    for i in range(len(random_ids_multi_super_iterations)):
        
        
        random_ids = torch.randperm(add_num)
        
        added_random_ids_multi_super_iteration.append(random_ids)


    return added_random_ids_multi_super_iteration




if __name__ == '__main__':
    
    
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    sys_args = sys.argv

    dataset_train = torch.load(git_ignore_folder + "dataset_train")
    
    dataset_test = torch.load(git_ignore_folder + "dataset_test")
    

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
    
#     noise_rate = float(sys_args[1])#torch.load(git_ignore_folder + 'noise_rate')
    
    
    record_params = bool(int(sys_args[1]))
    
    max_epoch = torch.load(git_ignore_folder+'epoch')
    
    print("max_epoch::", max_epoch)
    
    
#     delta_gradient_all_epochs = torch.load(git_ignore_folder + 'delta_gradient_all_epochs')
    
#     beta = torch.load(git_ignore_folder + 'beta')
    
#     hessian_matrix = torch.load(git_ignore_folder + 'hessian_matrix')
    
#     gradient_list = torch.load(git_ignore_folder + 'gradient_list')
    
#     max_epoch = torch.load(git_ignore_folder+'epoch')
#     
#     print("max_epoch::", max_epoch)
#     
#     X = torch.load(git_ignore_folder+'noise_X')
#     
#     Y = torch.load(git_ignore_folder+'noise_Y')
    init_para_list = list(torch.load(git_ignore_folder + 'init_para'))
    
    random_ids_multi_super_iterations = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations')
    
    model_class = torch.load(git_ignore_folder + 'model_class')
    
#     data_train_loader = torch.load(git_ignore_folder + 'data_train_loader')
    
    
    data_test_loader = torch.load(git_ignore_folder + 'data_test_loader')
    batch_size = torch.load(git_ignore_folder + 'batch_size')
    
    learning_rate_all_epochs = torch.load(git_ignore_folder + 'learning_rate_all_epochs')
    
    
    X_to_add = torch.load(git_ignore_folder + 'X_to_add')
            
    Y_to_add = torch.load(git_ignore_folder + 'Y_to_add')

#     added_random_ids_multi_super_iteration = get_sampling_each_iteration(random_ids_multi_super_iterations, 1)
    
    
    added_batch_size = int(1.0/len(dataset_train)*batch_size + 0.5)
    
    
    if added_batch_size < 1:
        added_batch_size = 1
    
#     torch.save(added_random_ids_multi_super_iteration, git_ignore_folder + 'added_random_ids_multi_super_iteration')
#     
#     torch.save(added_batch_size, git_ignore_folder + 'added_batch_size')
    
    
#     delta_data_ids = torch.load(git_ignore_folder + 'noise_data_ids')
#     dim = X.shape


    dim = dataset_train.data.data.shape
    

    
    data_preparer = Data_preparer()
    num_class = get_data_class_num_by_name(data_preparer, dataset_name)
    
    model = model_class(dim[1], num_class)# DNNModel(input_dim, hidden_dim, output_dim)

    if is_GPU:
        model.to(device)
    hyper_para_function=getattr(Data_preparer, "get_hyperparameters_" + dataset_name)

    
    criterion, optimizer, lr_scheduler = hyper_para_function(data_preparer, model.parameters(), learning_rate, regularization_coeff)
    
    init_model(model,init_para_list)

#     delta_size = int(dim[0]*0.1)
#     
#     print("delta_size::", delta_size)

#     print("delta_size::", delta_data_ids.shape[0])
    
    
#     delta_data_ids = random_generate_subset_ids(dim, delta_size)
    
#     update_X, update_Y, selected_rows = get_subset_training_data(X, Y, dim, delta_data_ids)
# 
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
#     
#     model = DNNModel(input_dim, hidden_dim, output_dim)
#     
#     init_model(model,init_para_list)
    
#     init_model(model, list(origin_model.parameters()))

#     hessian_para_list = torch.load(git_ignore_folder + 'hessian_para_list')
    
#     init_model(model, hessian_para_list)
    
#     error = nn.CrossEntropyLoss()
# # 
# # 
# #     learning_rate = 0.1
# #     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# 
#     
#     print("learning rate::", alpha)
#     
#     print("max_epoch::", max_epoch)
    
    
    mini_batch_num = int((len(dataset_train) - 1)/batch_size) + 1
    
    
    
    
    all_added_random_ids_list_all_samples = generate_added_random_ids_all_epochs(len(dataset_train), X_to_add, mini_batch_num, random_ids_multi_super_iterations)
    
    
    origin_train_data_size = len(dataset_train)
     
    dataset_train.data = torch.cat([dataset_train.data, X_to_add], 0)
     
    dataset_train.labels = torch.cat([dataset_train.labels, Y_to_add], 0)
    
    
    
#     model, gradient_list, res_list, count = model_training(max_epoch, update_X, update_Y, test_X, test_Y, alpha, error, model)
    
#     all_added_random_ids_list_all_samples = torch.load(git_ignore_folder + 'all_added_random_ids_list_all_samples')
    
#     max_epoch = 150
    
#     model, count = model_update_standard_lib_stochastic(batch_size, max_epoch, update_X, update_Y, alpha, error, model)
    t1 = time.time()
    
#     num_epochs, dataset_train, model, random_ids_multi_super_iterations, selected_rows, batch_size, learning_rate_all_epochs, criterion, optimizer, para_list_all_epochs, gradient_list_all_epochs, is_GPU, device, record_params, all_ids_list_all_epochs
#     model, count, exp_para_list_all_epochs, exp_gradient_list_all_epochs, all_added_random_ids_list_all_samples, all_res = model_update_standard_lib_multi2(origin_model, max_epoch, dataset_train, dim, model, random_ids_multi_super_iterations, batch_size, learning_rate_all_epochs, added_random_ids_multi_super_iteration, added_batch_size, X_to_add, Y_to_add, criterion, optimizer, para_list_all_epochs, gradient_list_all_epochs, is_GPU, device, record_params, regularization_coeff, mini_batch_num, all_added_random_ids_list_all_samples)
    model, count, exp_para_list_all_epochs, exp_gradient_list_all_epochs, all_res = model_update_standard_lib_multi0(all_added_random_ids_list_all_samples, origin_model, max_epoch, dataset_train, dim, model, random_ids_multi_super_iterations, batch_size, learning_rate_all_epochs, added_batch_size, X_to_add, Y_to_add, criterion, optimizer, para_list_all_epochs, gradient_list_all_epochs, is_GPU, device, record_params, regularization_coeff, mini_batch_num, origin_train_data_size)
    
    
#     model, count, exp_para_list_all_epochs, exp_gradient_list_all_epochs, all_res = model_update_standard_lib_multi(all_added_random_ids_list_all_samples, origin_model, max_epoch, dataset_train, dim, model, random_ids_multi_super_iterations, batch_size, learning_rate_all_epochs, added_batch_size, X_to_add, Y_to_add, criterion, optimizer, para_list_all_epochs, gradient_list_all_epochs, is_GPU, device, record_params, regularization_coeff, mini_batch_num)

    t2 = time.time()
    
    print('time_baseline::', t2 - t1)
    cut_off_epoch = max_epoch

    

#     model, gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, epoch = model_training(max_epoch, update_X, update_Y, test_X, test_Y, alpha, beta, error, model, True, batch_size, dim)

#     compute_model_parameter_iteration(max_epoch, model, update_X, update_Y, alpha, error, update_X.shape, num_class, input_dim, hidden_dim, output_dim, old_para_list_all_epochs, delta_gradient_all_epochs, delta_all_epochs)
    
#     print_model_para(model)
    
    
    compute_model_para_diff(list(origin_model.parameters()), list(model.parameters()))
    
    
    torch.save(model, git_ignore_folder + 'model_base_line')    
    
#     compute_derivative_one_more_step(model, error, X, Y)
     
#     origin_gradient_list = torch.load(git_ignore_folder + 'gradient_list')
    
    
#     loss = torch.load(git_ignore_folder + 'loss')
    
    torch.save(all_added_random_ids_list_all_samples, git_ignore_folder + 'all_added_random_ids_list_all_samples')
    
    torch.save(exp_gradient_list_all_epochs, git_ignore_folder + 'expected_gradient_list_all_epochs')
      
    torch.save(exp_para_list_all_epochs, git_ignore_folder + 'expected_para_list_all_epochs')
    
    torch.save(all_res, git_ignore_folder + 'all_res')
    
#     test(model, data_test_loader, criterion, len(dataset_test), is_GPU, device)
    test(model, dataset_test, batch_size, criterion, len(dataset_test), is_GPU, device)
    
#     torch.save(max_epoch, git_ignore_folder + 'update_max_epochs')
    
    
#     torch.save(batch_size, git_ignore_folder + 'batch_size')
    
#     batch_num = update_X.shape[0]/batch_size
    
#     torch.save(batch_num, git_ignore_folder + 'batch_num')
    
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
    
    
    
    
    
    