'''
Created on Jun 24, 2020

'''
import sys, os
import torch
import time

import psutil

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/data_IO')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Interpolation')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Models')


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.abspath(__file__))




from Models.Data_preparer import *

from utils import *

from model_train import *




try:
    from data_IO.Load_data import *
    from utils import *
    from Models.DNN import DNNModel
    from Models.Data_preparer import *
    from Models.DNN_single import *
    from Models.DNN2 import DNNModel2
    from Models.DNN3 import DNNModel3
    from Models.ResNet import *
    from Models.Pretrained_models import *

except ImportError:
    from Load_data import *
    from utils import *
    from Models.DNN import DNNModel
    from Models.DNN2 import DNNModel2
    from Models.DNN3 import DNNModel3
    from Models.Data_preparer import *
    from Models.DNN_single import *
    from Models.ResNet import *
    from Models.Pretrained_models import *


#     parser.add_argument('--add',  action='store_true', help="The flag for incrementally adding training samples, otherwise for incrementally deleting training samples")
#     
#     
#     parser.add_argument('--ratio',  type=float, help="delete rate or add rate")
#     
#     parser.add_argument('--bz',  type=int, help="batch size in SGD")
#         
#     parser.add_argument('--epochs',  type=int, help="number of epochs in SGD")
#     
#     parser.add_argument('--model',  help="name of models to be used")
#     
#     parser.add_argument('--dataset',  help="dataset to be used")
#     
#     parser.add_argument('--wd', type = float, help="l2 regularization")
#     
#     parser.add_argument('--lr', nargs='+', type = float, help="learning rates")
#     
#     parser.add_argument('--lrlen', nargs='+', type = int, help="The epochs to use some learning rate, used for the case with decayed learning rates")
#     
#     parser.add_argument('--GPU', action='store_true', help="whether the experiments run on GPU")
#     
#     parser.add_argument('--GID', type = int, help="Device ID of the GPU")
#     
#     parser.add_argument('--train', action='store_true', help = 'Train phase over the full training datasets')
#     
#     
#     parser.add_argument('--dataset',  help="name of dataset used in the experiments")







def model_update_add(args, method, lr_lists):
    
    
    model_name = args.model
    
    git_ignore_folder = args.repo
    
    dataset_name = args.dataset
    
    num_epochs = args.epochs
    
    batch_size = args.bz
    
    is_GPU = args.GPU
    
#     args.ratio


    regularization_coeff = args.wd
    
    if not is_GPU:
        device = torch.device("cpu")
    else:    
        GPU_ID = int(args.GID)
        device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    
    model_class = getattr(sys.modules[__name__], model_name)
    
    
    data_preparer = Data_preparer()
    
    
    dataset_train = torch.load(git_ignore_folder + "dataset_train")
    
    dataset_test = torch.load(git_ignore_folder + "dataset_test")
    
    
    X_to_add = torch.load(git_ignore_folder + 'X_to_add')
            
    Y_to_add = torch.load(git_ignore_folder + 'Y_to_add')
    
    learning_rate_all_epochs = torch.load(git_ignore_folder + 'learning_rate_all_epochs')
    
#     generate_random_id_add(git_ignore_folder, dataset_train, num_epochs)
    
    random_ids_all_epochs = torch.load(git_ignore_folder + 'random_ids_multi_epochs')
    
    mini_batch_num = int((len(dataset_train) - 1)/batch_size) + 1
    
    
    para_list_all_epochs = torch.load(git_ignore_folder + 'para_list_all_epochs')
    
    gradient_list_all_epochs = torch.load(git_ignore_folder + 'gradient_list_all_epochs')
    
#     data_train_loader = torch.load(git_ignore_folder + "data_train_loader")
#     
#     data_test_loader = torch.load(git_ignore_folder + "data_test_loader")
    
    
    

    dim = [len(dataset_train), len(dataset_train[0][0])]

    
    origin_train_data_size = len(dataset_train)
    
    
    
    
    num_class = get_data_class_num_by_name(data_preparer, dataset_name)
    
    hyper_para_function=getattr(Data_preparer, "get_hyperparameters_" + dataset_name)
    
    model = model_class(dim[1], num_class)
    
    
    init_para_list = list(torch.load(git_ignore_folder + 'init_para'))
    
    
    init_model(model,init_para_list)
    
    print('data dimension::',dim)
    
    if is_GPU:
        model.to(device)
    
#     init_model_params = list(model.parameters())
    
    
    criterion, optimizer = hyper_para_function(data_preparer, model.parameters(), lr_lists[0], regularization_coeff)
    
#     hyper_params = [criterion, optimizer]
    
    if method == baseline_method:
        
        
        added_random_ids_multi_epochs = get_sampling_each_iteration0(random_ids_all_epochs, X_to_add.shape[0], mini_batch_num, len(dataset_train))
     
        print("delta data size::", X_to_add.shape[0])
         
        torch.save(added_random_ids_multi_epochs, git_ignore_folder + 'added_random_ids_multi_epochs')

#         added_random_ids_multi_epochs = torch.load(git_ignore_folder + 'added_random_ids_multi_epochs')

        dataset_train.data = torch.cat([dataset_train.data, X_to_add], 0)
    
        dataset_train.labels = torch.cat([dataset_train.labels, Y_to_add], 0)

        
        t1 = time.time()
        
        updated_model, exp_para_list, exp_grad_list = model_update_standard_lib_add(num_epochs, dataset_train, dim, model, random_ids_all_epochs, batch_size, learning_rate_all_epochs, added_random_ids_multi_epochs, criterion, optimizer, is_GPU, device, regularization_coeff)
    
        t2 = time.time()
            
        process = psutil.Process(os.getpid())

        print('memory usage::', process.memory_info().rss)
        
        
        print('time_baseline::', t2 - t1)
    
        origin_model = torch.load(git_ignore_folder + 'origin_model')
        
        compute_model_para_diff(list(origin_model.parameters()), list(updated_model.parameters()))
    
    
        torch.save(updated_model, git_ignore_folder + 'model_base_line')
        
        # torch.save(exp_para_list, git_ignore_folder + 'exp_para_list')
        #
        # torch.save(exp_grad_list, git_ignore_folder + 'exp_grad_list')    
        
    
    else:
        if method == deltagrad_method:
            
            added_random_ids_multi_epochs = torch.load(git_ignore_folder + 'added_random_ids_multi_epochs')
            
            dataset_train.data = torch.cat([dataset_train.data, X_to_add], 0)
    
            dataset_train.labels = torch.cat([dataset_train.labels, Y_to_add], 0)
            
            exp_para_list = None#torch.load(git_ignore_folder + 'exp_para_list')
        
            exp_grad_list = None#torch.load(git_ignore_folder + 'exp_grad_list')
            
            period = args.period
            
            init_epochs = args.init
            
            m = args.m
            
            cached_size = args.cached_size
            
            grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor = cache_grad_para_history(git_ignore_folder, cached_size, is_GPU, device)
            
#             model_update_provenance_test3(period, 1, init_epochs, dataset_train, model, grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, max_epoch, 2, learning_rate_all_epochs, random_ids_multi_epochss, sorted_ids_multi_epochss, batch_size, dim, added_random_ids_multi_epochs, X_to_add, Y_to_add, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device)
            
            t1 = time.time()
            
            updated_model = model_update_delta_grad_add(exp_para_list, exp_grad_list, period, 1, init_epochs, dataset_train, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, m, learning_rate_all_epochs, random_ids_all_epochs, batch_size, dim, added_random_ids_multi_epochs, criterion, optimizer, regularization_coeff, is_GPU, device)
            
            t2 = time.time()
            
            process = psutil.Process(os.getpid())
    
            print('memory usage::', process.memory_info().rss)
            
            
            print('time_deltagrad::', t2 - t1)
            
            
            model_base_line = torch.load(git_ignore_folder + 'model_base_line')
            
            compute_model_para_diff(list(model_base_line.parameters()), list(updated_model.parameters()))
            
            torch.save(updated_model, git_ignore_folder + 'model_deltagrad')    
            
            

def prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, m, k, is_GPU, device):
 
 
    zero_mat_dim = k#ids.shape[0]
    

    curr_S_k = torch.cat(list(S_k_list), dim = 0)
          
    curr_Y_k = torch.cat(list(Y_k_list), dim = 0)
    
    S_k_time_Y_k = torch.mm(curr_S_k, torch.t(curr_Y_k))
    
    
    S_k_time_S_k = torch.mm(curr_S_k, torch.t(curr_S_k))
    
    
    if is_GPU:
        R_k = np.triu(S_k_time_Y_k.to('cpu').numpy())
        L_k = S_k_time_Y_k - torch.from_numpy(R_k).to(device)
    else:
        R_k = np.triu(S_k_time_Y_k.numpy())
        L_k = S_k_time_Y_k - torch.from_numpy(R_k)
    
    
    D_k_diag = torch.diag(S_k_time_Y_k)
    
    
    sigma_k = torch.mm(Y_k_list[-1],torch.t(S_k_list[-1]))/(torch.mm(S_k_list[-1], torch.t(S_k_list[-1])))
    
    upper_mat = torch.cat([-torch.diag(D_k_diag), torch.t(L_k)], dim = 1)
    
    lower_mat = torch.cat([L_k, sigma_k*S_k_time_S_k], dim = 1)
    
    mat = torch.cat([upper_mat, lower_mat], dim = 0)
    
    
    return zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat


def compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, v_vec, is_GPU, device):
    
    if is_GPU:
        p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double, device =device)
    else:
        p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double)
     
    torch.mm(curr_Y_k, v_vec, out = p_mat[0:zero_mat_dim])
    
    torch.mm(curr_S_k, v_vec*sigma_k, out = p_mat[zero_mat_dim:zero_mat_dim*2])

    p_mat = torch.mm(mat, p_mat)
    
    approx_prod = sigma_k*v_vec
    
    approx_prod -= (torch.mm(torch.t(curr_Y_k), p_mat[0:zero_mat_dim]) + torch.mm(sigma_k*torch.t(curr_S_k), p_mat[zero_mat_dim:zero_mat_dim*2]))
    
    return approx_prod

def cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, i, m, k, v_vec, period, is_GPU, device):
 
    period_num = int(i/period)
    
    
    ids = torch.tensor(range(m)).view(-1)
    
    if period_num > 0:
        ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
    ids = ids - 1
    
    ids = ids[ids >= 0]
    
    if ids.shape[0] > k:
        ids = ids[-k:]
    
    zero_mat_dim = k#ids.shape[0]
    
    curr_S_k = torch.t(torch.cat(list(S_k_list), dim = 0))
          
    curr_Y_k = torch.t(torch.cat(list(Y_k_list), dim = 0))
    
    S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
    
    
    S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)
    
    
    if is_GPU:
        
        R_k = np.triu(S_k_time_Y_k.to('cpu').numpy())
        
        L_k = S_k_time_Y_k - torch.from_numpy(R_k).to(device)
        
    else:
        R_k = np.triu(S_k_time_Y_k.numpy())
    
        L_k = S_k_time_Y_k - torch.from_numpy(R_k)
    
    D_k_diag = torch.diag(S_k_time_Y_k)
    
    
    sigma_k = torch.mm(Y_k_list[-1],torch.t(S_k_list[-1]))/(torch.mm(S_k_list[-1], torch.t(S_k_list[-1])))
    if is_GPU:
        p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double, device = device)
    else:
        p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double)
    
    tmp = torch.mm(torch.t(curr_Y_k), v_vec)
    
    p_mat[0:zero_mat_dim] = tmp
    
    p_mat[zero_mat_dim:zero_mat_dim*2] = torch.mm(torch.t(curr_S_k), v_vec)*sigma_k
    
    upper_mat = torch.cat([-torch.diag(D_k_diag), torch.t(L_k)], dim = 1)
    
    lower_mat = torch.cat([L_k, sigma_k*S_k_time_S_k], dim = 1)
    
    mat = torch.cat([upper_mat, lower_mat], dim = 0)
    

    mat = np.linalg.inv(mat.cpu().numpy())
        
    inv_mat = torch.from_numpy(mat)
    
    if is_GPU:
        
        inv_mat = inv_mat.to(device)
        
        
    
    p_mat = torch.mm(inv_mat, p_mat)
    
    
    approx_prod = sigma_k*v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    return approx_prod,zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat


def compute_grad_final3(para, hessian_para_prod, gradient_dual, grad_list_tensor, para_list_tensor, size1, size2, alpha, beta, is_GPU, device):
    
    gradients = None
    
    if gradient_dual is not None:
        
        hessian_para_prod += grad_list_tensor 
        
        hessian_para_prod += beta*para_list_tensor 
        
        gradients = hessian_para_prod*size1
        
        gradients += (gradient_dual + beta*para_list_tensor)*size2
        
        gradients /= (size1 + size2)
            
    else:
        
        hessian_para_prod += (grad_list_tensor + beta*para_list_tensor)
        
        gradients = hessian_para_prod
        
    return True, gradients


def update_para_final2(vec_para, gradient_list, alpha):
    
    vec_para -= alpha*gradient_list

    return vec_para


def get_expect_full_gradient(dataset_train, curr_rand_ids, is_GPU, device, model, para, criterion, optimizer):
    batch_remaining_X = dataset_train.data[curr_rand_ids]
    
    batch_remaining_Y = dataset_train.labels[curr_rand_ids]
    
    if is_GPU:
        batch_remaining_X = batch_remaining_X.to(device)
        
        batch_remaining_Y = batch_remaining_Y.to(device)
    
    init_model(model, para)
    
    compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
    
    
    expect_gradients = get_all_vectorized_parameters1(model.get_all_gradient())
    
    return expect_gradients

def model_update_delta_grad_add(exp_para_list, exp_grad_list, period, length, init_epochs, dataset_train, model, gradient_list_all_epochs_tensor, para_list_all_epochs_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, m, learning_rate_all_epochs, random_ids_multi_epochss, batch_size, dim, added_random_ids_multi_epochs, criterion, optimizer, regularization_coeff, is_GPU, device):
    
    
    para = list(model.parameters())
    
    use_standard_way = False
    
    recorded = 0
    
    overhead = 0
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    if not is_GPU:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    else:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double, device=device)
    
    i = 0
    
    S_k_list = deque()
    
    Y_k_list = deque()
    
    overhead2 = 0
    
    overhead3 = 0
    
    overhead4 = 0
    
    overhead5 = 0
    
    old_lr = 0
    
    cached_id = 0
    
    batch_id = 1
    
    res_para = []
    
    res_grad = []
    
    for k in range(len(random_ids_multi_epochss)):
    
        random_ids = random_ids_multi_epochss[k]
        
        added_to_random_ids = added_random_ids_multi_epochs[k]
        
        j = 0
        
        to_add = True
        
        curr_init_epochs = init_epochs
            
        for jj in range(len(added_to_random_ids)):
        
            end_id = j + batch_size
            
            curr_added_random_ids = added_to_random_ids[jj]
            
            if end_id > len(dataset_train):
                end_id = len(dataset_train)
            
            if curr_added_random_ids.shape[0] <= 0:
                to_add = False
            else:
                to_add = True

            curr_added_size = 0

            

            if to_add:
                
                batch_delta_X = dataset_train.data[curr_added_random_ids]
                
                batch_delta_Y = dataset_train.labels[curr_added_random_ids]
            
                curr_added_size = curr_added_random_ids.shape[0]
                
                
                if is_GPU:
                    batch_delta_X = batch_delta_X.to(device)
                    
                    batch_delta_Y = batch_delta_Y.to(device)
                
            
            
            learning_rate = learning_rate_all_epochs[i]
            
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate    
                
                      
            if (i-curr_init_epochs)%period == 0:
                
                recorded = 0
                
                use_standard_way = True
                
                
            if i< curr_init_epochs or use_standard_way == True:
                t7 = time.time()
                
                curr_rand_ids = random_ids[j:end_id]
            
            
                batch_remaining_X = dataset_train.data[curr_rand_ids]
                
                batch_remaining_Y = dataset_train.labels[curr_rand_ids]
                
                if is_GPU:
                    batch_remaining_X = batch_remaining_X.to(device)
                    
                    batch_remaining_Y = batch_remaining_Y.to(device)
                
                
                
                t8 = time.time()
            
                overhead4 += (t8 - t7)
                
                
                t5 = time.time()
                
                init_model(model, para)
                
                compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
                
                
                expect_gradients = get_all_vectorized_parameters1(model.get_all_gradient())
                
                t6 = time.time()

                overhead3 += (t6 - t5)
                
                gradient_remaining = 0
#                 if curr_matched_ids_size > 0:
                if to_add:
                    
                    t3 = time.time()
                    
                    clear_gradients(model.parameters())
                        
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                
                
                    gradient_remaining = get_all_vectorized_parameters1(model.get_all_gradient())     
                    
                    
                    t4 = time.time()
                
                
                    overhead2 += (t4  -t3)
                
                with torch.no_grad():
                               
                
                    curr_para = get_all_vectorized_parameters1(para)
                
                    if k > 0 or (k == 0 and jj > 0):
                        
                        prev_para = para_list_GPU_tensor[cached_id]
                        
                        curr_s_list = (curr_para - prev_para)+ 1e-16
                        
                        S_k_list.append(curr_s_list)
                        if len(S_k_list) > m:
                            removed_s_k = S_k_list.popleft()
                            
                            del removed_s_k
                        
                    gradient_full = (expect_gradients*curr_rand_ids.shape[0] + gradient_remaining*curr_added_size)/(curr_rand_ids.shape[0] + curr_added_size)
                    
                    if k > 0 or (k == 0 and jj > 0):
                        
                        Y_k_list.append((expect_gradients - grad_list_GPU_tensor[cached_id] + regularization_coeff*curr_s_list)+ 1e-16)
                        
                        if len(Y_k_list) > m:
                            removed_y_k = Y_k_list.popleft()
                            
                            del removed_y_k
                    
                    alpha = learning_rate_all_epochs[i]
                    
#                     res_para.append(curr_para)
#                     
#                     res_grad.append(gradient_full)
                    
                    
                    # compute_model_para_diff(exp_para_list[i], para)
                
                    # print('gradient diff::', torch.norm(get_all_vectorized_parameters1(exp_grad_list[i]) - gradient_full))
                    
                    para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_para - learning_rate*gradient_full, full_shape_list, shape_list)
                    
                    recorded += 1

                    del gradient_full
                    
                    del gradient_remaining
                    
                    del expect_gradients
                    
                    del batch_remaining_X
                    
                    del batch_remaining_Y
                    
                    if to_add:
                        
                        del batch_delta_X
                        
                        del batch_delta_Y
                    
                    if k > 0 or (k == 0 and jj > 0):
                        del prev_para
                    
                        del curr_para
                    
                    if recorded >= length:
                        use_standard_way = False
                
                
            else:
                
                gradient_dual = None
    
                if to_add:
                
                    init_model(model, para)
                    
                    compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                    
                    gradient_dual = model.get_all_gradient()
                
                curr_rand_ids = random_ids[j:end_id]
                    
                expect_full_gradients = get_all_vectorized_parameters1(get_expect_full_gradient(dataset_train, curr_rand_ids, is_GPU, device, model, para, criterion, optimizer))

                
                with torch.no_grad():
                
                    curr_vec_para = get_all_vectorized_parameters1(para)

                    vec_para_diff = torch.t((curr_vec_para - para_list_GPU_tensor[cached_id]))
                    
                    if (i-curr_init_epochs)/period >= 1:
                        if (i-curr_init_epochs) % period == 1:
                            zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, is_GPU, device)
#                             zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_1, mat_2 = prepare_hessian_vec_prod0(S_k_list, Y_k_list, i, init_epochs, m, period)
                            
                            
                            
                            mat = np.linalg.inv(mat_prime.cpu().numpy())
                            mat = torch.from_numpy(mat)
                            if is_GPU:
                                
                                
                                mat = mat.to(device)
                            
                    
                        hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff, is_GPU, device)
                        
                    else:
                        
                        hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period, is_GPU, device)
                    exp_gradient, exp_param = None, None
                    
                    delta_const = 0
                    
                    alpha = learning_rate_all_epochs[i]
                    
                    if gradient_dual is not None:
                        is_positive, final_gradient_list = compute_grad_final3(curr_vec_para, torch.t(hessian_para_prod), get_all_vectorized_parameters1(gradient_dual), grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_added_size, learning_rate, regularization_coeff, is_GPU, device)
                        
                    else:
                        is_positive, final_gradient_list = compute_grad_final3(curr_vec_para, torch.t(hessian_para_prod), None, grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_added_size, learning_rate, regularization_coeff, is_GPU, device)
                    
                    # compute_model_para_diff(exp_para_list[i], para)
                
                    # print('gradient diff::', torch.norm(get_all_vectorized_parameters1(exp_grad_list[i]) - final_gradient_list))

                    
                    vec_para = update_para_final2(curr_vec_para, final_gradient_list, learning_rate)
                    
                    
#                     res_para.append(curr_vec_para)
#                     
#                     res_grad.append(final_gradient_list - regularization_coeff*curr_vec_para)
                    
                    para = get_devectorized_parameters(vec_para, full_shape_list, shape_list)
                 
                
            i = i + 1
            
            
            cached_id += 1
            
            if cached_id%cached_size == 0:
                
                GPU_tensor_end_id = (batch_id + 1)*cached_size
                
                if GPU_tensor_end_id > para_list_all_epochs_tensor.shape[0]:
                    GPU_tensor_end_id = para_list_all_epochs_tensor.shape[0] 
                print("end_tensor_id::", GPU_tensor_end_id)
                
                para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(para_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(gradient_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                batch_id += 1
                
                cached_id = 0
            
            j += batch_size
            
            
    print('overhead::', overhead)
    
    print('overhead2::', overhead2)
    
    print('overhead3::', overhead3)
    
    print('overhead4::', overhead4)
    
    print('overhead5::', overhead5)
    
    init_model(model, para)
    
    return model



def model_update_standard_lib_add(max_epoch, dataset_train, dim, model, random_ids_multi_epochs, batch_size, learning_rate_all_epochs, added_random_ids_multi_epochs, criterion, optimizer, is_GPU, device, regularization_coeff):
    count = 0

    elapse_time = 0

    overhead = 0
    
    overhead2 = 0

    t1 = time.time()
    
    para = list(model.parameters())
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    old_lr = -1
    
    exp_para_list = []
    
    exp_grad_list = []

    for k in range(max_epoch):
        
        random_ids = random_ids_multi_epochs[k]
        
        added_random_ids = added_random_ids_multi_epochs[k]
        
        id_start = 0
        
        id_end = 0

        j = 0
        
        to_add = True
        
        

        for i in range(0, dim[0], batch_size):
            
            end_id = i + batch_size
            
#             added_end_id = j + added_batch_size
            
            curr_to_add_rand_ids = added_random_ids[j]
            
            
            if curr_to_add_rand_ids.shape[0] > 0:
                full_random_ids = torch.cat([curr_to_add_rand_ids, random_ids[i:end_id]], 0)
            else:
                full_random_ids = random_ids[i:end_id]
            
#             print(j, curr_to_add_rand_ids.shape[0])
            
            if end_id > dim[0]:
                end_id = dim[0]
            if count == 11:
                print(curr_to_add_rand_ids, i, end_id)
                print('here')

            learning_rate = learning_rate_all_epochs[count]
        
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            
            old_lr = learning_rate
            init_model(model, para)
            
            batch_X = dataset_train.data[full_random_ids]
            
            batch_Y = dataset_train.labels[full_random_ids]
            
            if is_GPU:
                batch_X = batch_X.to(device)
                
                batch_Y = batch_Y.to(device)
            
            compute_derivative_one_more_step(model, batch_X, batch_Y, criterion, optimizer)

#             exp_para_list.append(list(model.parameters()))
#             
#             exp_grad_list.append(list(model.get_all_gradient()))

            append_gradient_list(exp_grad_list, None, exp_para_list, model, None, is_GPU, device)

            grad_full = get_all_vectorized_parameters1(model.get_all_gradient())
            
            
            
            para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*get_all_vectorized_parameters1(para) - learning_rate*grad_full, full_shape_list, shape_list)
            
            count += 1
            
            j += 1
             
#             print("loss::", loss)
    init_model(model, para)
    
    
    t2 = time.time()
        
    elapse_time += (t2 - t1)  

    print("training time is", elapse_time)
    
    print("overhead::", overhead)
    
    print("overhead2::", overhead2)
    
    return model, exp_para_list, exp_grad_list



def model_training_test(random_ids_multi_epochs, epoch, net, dataset_train, dataset_test, data_train_size, data_test_size, optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs):
#     global cur_batch_win
    net.train()
    
    gradient_list_all_epochs = []
    
    para_list_all_epochs = []
    
#     output_list_all_epochs = []
    
    learning_rate_all_epochs = []
    
    
    
    loss_list, batch_list = [], []
    
    t1 = time.time()
    
    
#     test(net, data_test_loader, criterion, data_test_size, is_GPU, device)

    for j in range(epoch):
        
#         random_ids = torch.zeros([data_train_size], dtype = torch.long)
    
    
#         random_ids = torch.randperm(data_train_size)
        random_ids = random_ids_multi_epochs[j]
    
#         k = 0
        
        learning_rate = lrs[j]
        update_learning_rate(optimizer, learning_rate)
        
#         item0 = data_train_loader.dataset.data[100]
    
#         for i, items in enumerate(data_train_loader):
        i = 0

        for k in range(0, data_train_size, batch_size):
            
            
            
            
#             random.seed(random_seed)
#             os.environ['PYTHONHASHSEED'] = str(random_seed)
#             np.random.seed(random_seed)
#             torch.manual_seed(random_seed)
#             torch.cuda.manual_seed(random_seed)
#             torch.cuda.manual_seed_all(random_seed) # if you are using multi-GPU.
#             torch.backends.cudnn.benchmark = False
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.enabled = False
            
            end_id = k + batch_size
            
            curr_rand_ids = random_ids[k:end_id]
            
            if end_id > data_train_size:
                end_id = data_train_size
            if not is_GPU:
                images, labels =  dataset_train.data[curr_rand_ids], dataset_train.labels[curr_rand_ids]
            else:
                images, labels =  dataset_train.data[curr_rand_ids].to(device), dataset_train.labels[curr_rand_ids].to(device)
            

            
#             print(k, end_id)
#             random_ids[k:end_id] = ids
            
            
#             k = k + batch_size
            
            optimizer.zero_grad()
    
            output = net(images)
            
#             print(output[0])
#              
#             print(torch.sort(ids)[0])
    
            loss = criterion(output, labels)
    
    
#             loss_list.append(loss.detach().cpu().item())
#             batch_list.append(i+1)
    
            if i % 10 == 0:
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (j, i, loss.detach().cpu().item()))
            
#             if i % 10 == 0:
#                 lr_scheduler.step()
                 
    
            loss.backward()
    
            append_gradient_list(gradient_list_all_epochs, None, para_list_all_epochs, net, None, is_GPU, device)
    
    
            i += 1
            # Update Visualization
    #         if viz.check_connection():
    #             cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
    #                                      win=cur_batch_win, name='current_batch_loss',
    #                                      update=(None if cur_batch_win is None else 'replace'),
    #                                      opts=cur_batch_win_opts)
#             learning_rate = list(optimizer.param_groups)[0]['lr']
            
#             regularization_rate = list(optimizer.param_groups)[0]['weight_decay']
            
#             exp_model_param = update_model(net, learning_rate, regularization_rate)
            
            
            optimizer.step()
            
#             print('parameter comparison::')
#             
#             compute_model_para_diff(list(net.parameters()), exp_model_param)
            
            
            learning_rate_all_epochs.append(learning_rate)
        
        
        
#         item1 = data_train_loader.dataset.data[100]
#         print(torch.norm(item0[0] - item1[0]))
        
#         random_ids_multi_epochs.append(random_ids)
    test(net, dataset_test, batch_size, criterion, data_test_size, is_GPU, device)
#     test(net, data_test_loader, criterion, data_test_size, is_GPU, device)
        
    
    t2 = time.time()
    
    print("training_time::", (t2 - t1))
    
    return net, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs



def generate_random_id_add(git_ignore_folder, dataset_train, epochs):
    
    
    delta_data_ids = torch.load(git_ignore_folder + 'delta_data_ids')
        
    torch.save(dataset_train.data[delta_data_ids], git_ignore_folder + 'X_to_add')
    
    torch.save(dataset_train.labels[delta_data_ids], git_ignore_folder + 'Y_to_add')
    
    
    selected_rows = get_subset_training_data(dataset_train.data.shape[0], delta_data_ids)        
    
    dataset_train.data = dataset_train.data[selected_rows]
    
    dataset_train.labels = dataset_train.labels[selected_rows]
    
    generate_random_ids_list(dataset_train, epochs, git_ignore_folder)

def main_add(args, lr_lists):
    model_name = args.model
    
    git_ignore_folder = args.repo
    
    dataset_name = args.dataset
    
    num_epochs = args.epochs
    
    batch_size = args.bz
    
    is_GPU = args.GPU
    
#     args.ratio


    regularization_coeff = args.wd
    
    if not is_GPU:
        device = torch.device("cpu")
    else:    
        GPU_ID = int(args.GID)
        device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    
    model_class = getattr(sys.modules[__name__], model_name)
    
    
    data_preparer = Data_preparer()
    
    
    dataset_train = torch.load(git_ignore_folder + "dataset_train")
    
    dataset_test = torch.load(git_ignore_folder + "dataset_test")
    
    
    generate_random_id_add(git_ignore_folder, dataset_train, num_epochs)
    
    random_ids_all_epochs = torch.load(git_ignore_folder + 'random_ids_multi_epochs')
    
#     data_train_loader = torch.load(git_ignore_folder + "data_train_loader")
#     
#     data_test_loader = torch.load(git_ignore_folder + "data_test_loader")
    
    
    
    
    dim = [len(dataset_train), len(dataset_train[0][0])]
    
    num_class = get_data_class_num_by_name(data_preparer, dataset_name)
    
    hyper_para_function=getattr(Data_preparer, "get_hyperparameters_" + dataset_name)
    
    if model_name == 'Logistic_regression':
        model = model_class(dim[1], num_class)
    else:
        model = model_class()
    
#     model = model_class(dim[1], num_class)
    
    print('data dimension::',dim)
    
    if is_GPU:
        model.to(device)
    
    init_model_params = list(model.parameters())
    
    
    criterion, optimizer = hyper_para_function(data_preparer, model.parameters(), lr_lists[0], regularization_coeff)
    
    hyper_params = [criterion, optimizer]
    
    
#     lrs = ast.literal_eval(input)#map(float, input.strip('[]').split(','))
#     [2.0, 3.0, 4.0, 5.0]
    
#     model, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs, all_ids_list_all_epochs = model_training_skipnet(num_epochs, model, data_train_loader, data_test_loader, len(dataset_train), len(dataset_test), optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs)

# net, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs, X_theta_prod_seq, X_theta_prod_softmax_seq, random_ids_multi_epochs
    
    t1 = time.time()
    
#     model, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs, X_theta_prod_seq, X_theta_prod_softmax_seq, random_ids_multi_epochs = model_training_lr(num_epochs, model, dataset_train, data_test_loader, len(dataset_train), len(dataset_test), optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs)

    model, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs = model_training_lr_test(random_ids_all_epochs, num_epochs, model, dataset_train, len(dataset_train), optimizer, criterion, batch_size, is_GPU, device, lr_lists)
    
    t2 = time.time()
    
    
    t3 = time.time()
    
#     capture_provenance(git_ignore_folder, data_train_loader, len(dataset_train), dim, num_epochs, num_class, batch_size, int((dim[0] - 1)/batch_size) + 1, torch.stack(random_ids_multi_epochs), X_theta_prod_softmax_seq, X_theta_prod_seq)

#     data_train_loader.batch_sampler.reset_ids()
    

#     x_sum_by_class_by_batch = compute_x_sum_by_class_by_batch(data_train_loader, len(dataset_train), batch_size, num_class, random_ids_multi_epochs)
    
    
#     data_train_loader.batch_sampler.reset_ids()
    
    t4 = time.time()
    
    
    print("training time full::", t2 - t1)
    
    print("provenance prepare time::", t4 - t3)    
    
    torch.save(dataset_train, git_ignore_folder + "dataset_train")
    
    torch.save(gradient_list_all_epochs, git_ignore_folder + 'gradient_list_all_epochs')
    
    torch.save(para_list_all_epochs, git_ignore_folder + 'para_list_all_epochs')
    
    torch.save(learning_rate_all_epochs, git_ignore_folder + 'learning_rate_all_epochs')
                  
    torch.save(num_epochs, git_ignore_folder+'epoch')    
    
    torch.save(hyper_params, git_ignore_folder + 'hyper_params')
    
    save_random_id_orders(git_ignore_folder, random_ids_all_epochs)
    
    torch.save(para_list_all_epochs[0], git_ignore_folder + 'init_para')
    
    torch.save(model, git_ignore_folder + 'origin_model')
    
    torch.save(model_class, git_ignore_folder + 'model_class')
    
    torch.save(regularization_coeff, git_ignore_folder + 'beta')
    
    torch.save(dataset_name, git_ignore_folder + 'dataset_name')
    
    torch.save(batch_size, git_ignore_folder + 'batch_size')

    torch.save(device, git_ignore_folder + 'device')

    torch.save(is_GPU, git_ignore_folder + 'is_GPU')
        
    test(model, dataset_test, batch_size, criterion, len(dataset_test), is_GPU, device)
        
    
    
    