'''
Created on Mar 15, 2019

'''
import torch

from sklearn.utils.extmath import *


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

def populate_perturbed_para_grad_table(is_GPU, device, perturbed_para, first_para, first_grad, perturbed_grad_all, X, Y, selected_rows, criterion, optimizer, model, id):
    
    init_model(model, perturbed_para)

    if is_GPU:
        compute_derivative_one_more_step(model, X[selected_rows].to(device), Y[selected_rows].to(device), criterion, optimizer)
    else:
        compute_derivative_one_more_step(model, X[selected_rows], Y[selected_rows], criterion, optimizer)
    
    perturbed_grad = get_all_vectorized_parameters(model.get_all_gradient())

    perturbed_grad_all[id] = perturbed_grad - get_all_vectorized_parameters(first_grad)


def get_exp_grad_delta(para_delta, perturbed_para_all, perturbed_grad_all):
    
    
    scales = torch.mm(torch.mm(para_delta.view(1,-1), torch.t(perturbed_para_all)), torch.inverse(torch.mm(perturbed_para_all, torch.t(perturbed_para_all))))
    
    res_grad_delta = torch.mm(scales.view(1,-1), perturbed_grad_all)
    
    return res_grad_delta

def compute_para_gap_subspace(exp_para_list_all_epochs, para_list_all_epochs, exp_grad_list_all_epochs, grad_list_all_epochs, upper_bound):
    
        
    delta_para_list = []
    
    delta_grad_list = []
    
    for m in range(len(exp_para_list_all_epochs)):
    
        delta_para_list.append(get_all_vectorized_parameters(exp_para_list_all_epochs[m]) - get_all_vectorized_parameters(para_list_all_epochs[m]))
        
        delta_grad_list.append(get_all_vectorized_parameters(exp_grad_list_all_epochs[m]) - get_all_vectorized_parameters(grad_list_all_epochs[m]))
        
        
    delta_para_tensor = torch.cat(delta_para_list, 0)
    
    
    if is_GPU:
        delta_para_tensor = delta_para_tensor.to('cpu')
        
    U_np, S_np, V_np = randomized_svd(delta_para_tensor.numpy(), n_components = upper_bound)
    
    U = torch.from_numpy(U_np).double().to(device)
    
    S = torch.from_numpy(S_np).double().to(device)
    
    V = torch.t(torch.from_numpy(V_np).double().to(device))
    
    sub_s = S
     
    sub_u = U
      
    sub_v = V
    
    approx_delta_para = torch.mm(sub_u*sub_s, torch.t(sub_v))
    
    approx_delta_para2 = torch.mm(sub_u, torch.t(sub_v*sub_s))
    
    if is_GPU:
        delta_para_tensor = delta_para_tensor.to(device)
    
    print("approx_errors::", torch.norm(delta_para_tensor -approx_delta_para))
    
    print("approx_errors2::", torch.norm(delta_para_tensor -approx_delta_para2))

    return sub_v.to('cpu')

def compute_delta_para_sub_space(is_GPU, device, max_epoch, upper_bound, sub_v, para_list_all_epochs, gradient_list_all_epochs, exp_para_list_all_epochs, exp_grad_list_all_epochs, model, delta, X, Y, random_ids_multi_super_iterations, batch_size, criterion, optimizer):
    
    
    delta_para_sub_U_list = []
    
    delta_para_sub_V_list = []
    
    
    l = 0
    
    perturbed_para_all = torch.t(sub_v)*delta
    
    
    perturbed_grad_all_all_epochs = []
    
#     for k in range(len(random_ids_multi_super_iterations)):
    for k in range(max_epoch):
        
        random_ids = random_ids_multi_super_iterations[k]
        
#         for r in range(len(random_ids_list)):
        for r in range(0, X.shape[0], batch_size):
#             random_ids = random_ids_list[r]
        
            end_id = r + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            
            
            curr_rand_ids = random_ids[r: end_id]
    
    
    
#     for l in range(len(para_list_all_epochs)):
            if k == 0 and r == 0:
                print(curr_rand_ids[0:50])
        
    
            print('iteration id::', l)
    
            first_para = para_list_all_epochs[l]
                
            first_grad = gradient_list_all_epochs[l]
            
            
    #         curr_random_ids = random_ids_all_epochs[0]
            
            selected_rows = curr_rand_ids#delta_ids_all_epochs[l]#curr_random_ids[batch_size*l: batch_size*(l + 1)]   
            
            init_model(model, first_para)
            
            if is_GPU:
                compute_derivative_one_more_step(model, X[selected_rows].to(device), Y[selected_rows].to(device), criterion, optimizer)
            else:
                compute_derivative_one_more_step(model, X[selected_rows], Y[selected_rows], criterion, optimizer)
            
            first_grad_2 = get_all_vectorized_parameters(model.get_all_gradient())
            
            delta_first_grad = get_all_vectorized_parameters(first_grad) - first_grad_2
            
#             print(torch.norm(delta_first_grad))
            
            full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(first_para)
            
            
            
            perturbed_grad_all = torch.zeros([upper_bound, total_shape_size], dtype = torch.double)
            
            id = 0
            
#             print(total_shape_size)
            
            
            for i in range(upper_bound):
                
#                 print(id)
                
        #             curr_shape = first_para[i].shape
#                 if is_GPU:
#                     first_para = first_para.to(device)
                
                perturbed_para = get_devectorized_parameters(get_all_vectorized_parameters1(first_para) + delta*sub_v[:,i].view(1,-1), full_shape_list, shape_list)
                        
                populate_perturbed_para_grad_table(is_GPU, device, perturbed_para, first_para, first_grad, perturbed_grad_all, X, Y, selected_rows, criterion, optimizer, model, id)
                
                id += 1
            
            
            perturbed_grad_all_all_epochs.append(perturbed_grad_all)
    
    
    
            
    
#         stacked_perturbed_grad_all_all_epochs = torch.cat(perturbed_grad_all_all_epochs)
#         
#         U, S, V =torch.svd(stacked_perturbed_grad_all_all_epochs)
#         
#         
#         grad_dim_count = 250
#                  
#         delta_grad_sub_s = S[0:grad_dim_count]
#          
#      
#         delta_grad_sub_u = U[:,0:grad_dim_count]
#           
#          
#           
#         delta_grad_sub_v = V[:,0:grad_dim_count]
#         
#         delta_para_sub_U_list.append(delta_grad_sub_u*delta_grad_sub_s)
#         
#         delta_para_sub_V_list.append(delta_grad_sub_v)
#         
#     #         
#         approx_perturbed_grad_all = torch.mm(delta_grad_sub_u*delta_grad_sub_s, torch.t(delta_grad_sub_v))
#     #         
#     #         approx_perturbed_grad_all = torch.mm(approx_V*approx_S, torch.t(approx_V))
#     #         
#     #         
#     #         print(S[0:upper_bound])
#     #         
#         print(torch.norm(approx_perturbed_grad_all - stacked_perturbed_grad_all_all_epochs))
#         
#         print('here')
#     
#     
#     for l in range(len(exp_para_list_all_epochs)):
    
    
            exp_para = exp_para_list_all_epochs[l]
            
            para_delta = get_all_vectorized_parameters1(exp_para) - get_all_vectorized_parameters1(first_para)
            
            
            
            
#             if is_GPU:
#                 perturbed_grad_all = perturbed_grad_all.to('cpu')
                
            U_np, S_np, V_np = randomized_svd(perturbed_grad_all.numpy(), n_components = upper_bound)
            
            
            U = torch.from_numpy(U_np).double()
    
            S = torch.from_numpy(S_np).double()
            
            V = torch.t(torch.from_numpy(V_np).double())
            
#             sub_s = S
#              
#          
#             sub_u = U
#               
#              
#               
#             sub_v = V
            
#             U, S, V =torch.svd(perturbed_grad_all)
             
    #         U_list.append(U)
    #          
    #         V_list.append(V)
    #          
    #         S_list.append(S)
            
    #         gradients_count = 100
    #         
    #         random_ids = torch.randperm(perturbed_grad_all.shape[0])
    #         
    #         approx_U, approx_S, approx_V =torch.svd(perturbed_grad_all[random_ids[0:gradients_count]])
    #         
    #         
    #         
            grad_dim_count = 200
             
            delta_grad_sub_s = S[0:grad_dim_count]
             
         
            delta_grad_sub_u = U[:,0:grad_dim_count]
              
             
              
            delta_grad_sub_v = V[:,0:grad_dim_count]
            
            delta_para_sub_U_list.append(delta_grad_sub_u*delta_grad_sub_s)
            
            delta_para_sub_V_list.append(delta_grad_sub_v)
            
    #         
            approx_perturbed_grad_all = torch.mm(delta_grad_sub_u*delta_grad_sub_s, torch.t(delta_grad_sub_v))
    #         
    #         approx_perturbed_grad_all = torch.mm(approx_V*approx_S, torch.t(approx_V))
    #         
    #         
    #         print(S[0:upper_bound])
    #         
            print(torch.norm(approx_perturbed_grad_all - perturbed_grad_all))
            
            
            
            
            res_delta_grad2 = get_exp_grad_delta(para_delta, perturbed_para_all, approx_perturbed_grad_all)
            
            res_delta_grad = get_exp_grad_delta(para_delta, perturbed_para_all, perturbed_grad_all)
            
            print('product_delta::', torch.norm(res_delta_grad - res_delta_grad2))
            
            res_grad = get_all_vectorized_parameters(first_grad).view(-1) + res_delta_grad.view(-1)
            
        #     exp_grad = get_all_vectorized_parameters(exp_gradient_list_all_epochs[para_id])
        
            init_model(model, exp_para)
            
            if is_GPU:
                compute_derivative_one_more_step(model, X[selected_rows].to(device), Y[selected_rows].to(device), criterion, optimizer)
            else:
                compute_derivative_one_more_step(model, X[selected_rows], Y[selected_rows], criterion, optimizer)
            
            exp_grad = get_all_vectorized_parameters(model.get_all_gradient())
            
            print(torch.norm(res_grad - exp_grad))
            
            l += 1
            
            
        
        
    return delta_para_sub_U_list, delta_para_sub_V_list, perturbed_para_all



if __name__ == '__main__':
    
    
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    sys_args = sys.argv

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
    
#     noise_rate = float(sys_args[1])#torch.load(git_ignore_folder + 'noise_rate')
    
    
    record_params = bool(int(sys_args[1]))
    
#     delta_data_ids = random_deletion(len(dataset_train), int(len(dataset_train)*noise_rate))
#     delta_data_ids = torch.load(git_ignore_folder + "delta_data_ids")



#     delta_data_ids = random_deletion(len(dataset_train), 1)
    
#     torch.save(delta_data_ids, git_ignore_folder + "delta_data_ids")
    
#     [criterion, optimizer, lr_scheduler] = hyper_params

    
#     alpha = torch.load(git_ignore_folder + 'alpha')
#     
#     beta = torch.load(git_ignore_folder + 'beta')
    
    
    init_para_list = list(torch.load(git_ignore_folder + 'init_para'))
    
    random_ids_multi_super_iterations = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations')
    
    sorted_ids_multi_super_iterations = torch.load(git_ignore_folder + 'sorted_ids_multi_super_iterations')
    
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
    
    full_ids_list = list(range(dim[0]))
    
    delta_data_ids = random_generate_subset_ids2(int(dim[0]*0.001), full_ids_list)

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
    
    
#     random.seed(random_seed)
#     os.environ['PYTHONHASHSEED'] = str(random_seed)
#     np.random.seed(random_seed)
#     torch.manual_seed(random_seed)
#     torch.cuda.manual_seed(random_seed)
#     torch.cuda.manual_seed_all(random_seed) # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = False
#     print("batch_size::", batch_size)
    
    
#     model, gradient_list, res_list, count = model_training(max_epoch, update_X, update_Y, test_X, test_Y, alpha, error, model)
    
    
    
#     max_epoch = 150
    
#     model, count = model_update_standard_lib_stochastic(batch_size, max_epoch, update_X, update_Y, alpha, error, model)

    '''num_epochs, dataset_train, model, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, delta_data_ids, batch_size, learning_rate_all_epochs, criterion, optimizer, para_list_all_epochs, gradient_list_all_epochs, is_GPU, device, record_params'''


    t1 = time.time()
    model, count, exp_para_list_all_epochs, exp_gradient_list_all_epochs, _ = model_update_standard_lib(max_epoch, dataset_train, model, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, delta_data_ids, batch_size, learning_rate_all_epochs, criterion, optimizer, para_list_all_epochs, gradient_list_all_epochs, is_GPU, device, record_params)

    t2 = time.time()
    
    print('time_baseline::', t2 - t1)
    cut_off_epoch = max_epoch

    
    upper_bound = 200
    
    delta = delta_data_ids.shape[0]*1.0/dataset_train.data.shape[0]

    sub_v = compute_para_gap_subspace(exp_para_list_all_epochs, para_list_all_epochs, exp_gradient_list_all_epochs, gradient_list_all_epochs, upper_bound)
    
    
    if upper_bound >= sub_v.shape[1]:
        upper_bound = sub_v.shape[1]
    
    
    
    delta_para_sub_U_list, delta_para_sub_V_list, perturbed_para_all = compute_delta_para_sub_space(is_GPU, device, max_epoch, upper_bound, sub_v, para_list_all_epochs, gradient_list_all_epochs, exp_para_list_all_epochs,exp_gradient_list_all_epochs, model, delta, dataset_train.data, dataset_train.labels, random_ids_multi_super_iterations, batch_size, criterion, optimizer)
    
    
    
    
    print('save sub_v::')
    
    torch.save(sub_v, git_ignore_folder + 'sub_v')
    
    print(perturbed_para_all.shape)
    
    torch.save(perturbed_para_all, git_ignore_folder + 'perturbed_para_all')
    
    torch.save(delta_para_sub_U_list, git_ignore_folder + 'delta_para_sub_U_list')
    
    torch.save(delta_para_sub_V_list, git_ignore_folder + 'delta_para_sub_V_list')

#     model, gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, epoch = model_training(max_epoch, update_X, update_Y, test_X, test_Y, alpha, beta, error, model, True, batch_size, dim)

#     compute_model_parameter_iteration(max_epoch, model, update_X, update_Y, alpha, error, update_X.shape, num_class, input_dim, hidden_dim, output_dim, old_para_list_all_epochs, delta_gradient_all_epochs, delta_all_epochs)
    
#     print_model_para(model)
    
    
    compute_model_para_diff(list(origin_model.parameters()), list(model.parameters()))
    
    
    torch.save(model, git_ignore_folder + 'model_base_line')    
    
#     compute_derivative_one_more_step(model, error, X, Y)
     
#     origin_gradient_list = torch.load(git_ignore_folder + 'gradient_list')
    
    
#     loss = torch.load(git_ignore_folder + 'loss')
    
    
    torch.save(exp_gradient_list_all_epochs, git_ignore_folder + 'expected_gradient_list_all_epochs')
      
    torch.save(exp_para_list_all_epochs, git_ignore_folder + 'expected_para_list_all_epochs')
    
#     torch.save(max_epoch, git_ignore_folder + 'update_max_epochs')
    
    
#     torch.save(batch_size, git_ignore_folder + 'batch_size')
    
#     batch_num = update_X.shape[0]/batch_size
    
#     torch.save(batch_num, git_ignore_folder + 'batch_num')
    
    test(model, dataset_test, batch_size, criterion, len(dataset_test), is_GPU, device)
    
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
    
    
    
    
    
    