'''
Created on Dec 18, 2019


'''




import sys, os


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')

# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch

try:
    from DNN import *
    from data_IO.Load_data import *
except ImportError:
    from DNN import *
    from Load_data import *




if __name__ == '__main__':
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    origin_model = torch.load(git_ignore_folder + 'model_without_noise')
    
    
    
#     exp_gradient_list_all_epochs = torch.load(git_ignore_folder + 'expected_gradient_list_all_epochs')
     
#     exp_para_list_all_epochs = torch.load(git_ignore_folder + 'expected_para_list_all_epochs')
    
    random_theta_list_all_epochs = torch.load(git_ignore_folder + 'random_theta_list_all_epochs')
    
    alpha = torch.load(git_ignore_folder + 'alpha')
    
    beta = torch.load(git_ignore_folder + 'beta')
    
    max_epoch = torch.load(git_ignore_folder+'epoch')


    epsilon = torch.load(git_ignore_folder + 'epsilon')

    gradient_list_all_epochs = torch.load(git_ignore_folder + 'gradient_list_all_epochs')
        
    para_list_all_epochs = torch.load(git_ignore_folder + 'para_list_all_epochs')
    
    batch_size = torch.load(git_ignore_folder + 'batch_size')
    
    random_ids_multi_super_iterations = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations')

    sorted_ids_multi_super_iterations = torch.load(git_ignore_folder + 'sorted_ids_multi_super_iterations')

    origin_X = torch.load(git_ignore_folder + 'noise_X')
    
    origin_Y = torch.load(git_ignore_folder + 'noise_Y')
    
    test_X = torch.load(git_ignore_folder + 'test_X')
    
    test_Y = torch.load(git_ignore_folder + 'test_Y')
    
    hidden_dims = torch.load(git_ignore_folder + 'hidden_dims')
    
    init_para_list = list(torch.load(git_ignore_folder + 'init_para'))
    
    input_dim = origin_X.shape[1]
    
    num_class = torch.unique(origin_Y).shape[0]
    
    output_dim = num_class
    
    model = DNNModel(input_dim, hidden_dims, output_dim)
    
    error = nn.CrossEntropyLoss()
    
    init_model(model, init_para_list)

    delta_data_ids = torch.load(git_ignore_folder + 'noise_data_ids')
    
    print("detal_data_ids::", delta_data_ids.shape)

    update_X, update_Y, selected_rows = get_subset_training_data(origin_X, origin_Y, origin_X.shape, delta_data_ids)

    error = nn.CrossEntropyLoss()

    print(git_ignore_folder)
    model_base_line = torch.load(git_ignore_folder + 'model_base_line')

    dim = origin_X.shape
    
    print("learning rate::", alpha)
    
    update_X = origin_X[selected_rows]
    
    update_Y = origin_Y[selected_rows]

    delta_X = origin_X[delta_data_ids]
    
    delta_Y = origin_Y[delta_data_ids]
    
    init_epochs = 10
    
    t1 = time.time()
    
    retrain_or_not = model_training_quantized_incremental(origin_model, epsilon, origin_X, origin_Y, model, gradient_list_all_epochs, para_list_all_epochs, max_epoch, delta_data_ids, input_dim, hidden_dims, output_dim, alpha, beta, selected_rows, error, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, batch_size, dim, random_theta_list_all_epochs)
    
    t2  =time.time()
    
    print(retrain_or_not)
    
    compute_model_para_diff3(list(origin_model.parameters()), list(model.parameters()))
    
    print("quantized_update_time:", (t2 - t1))
    
    
