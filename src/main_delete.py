'''
Created on Jun 24, 2020

'''


import sys, os
import torch


def main_del(args):
    model_class = getattr(sys.modules[__name__], model_name)
    
    
    data_preparer = Data_preparer()
    
    
    dataset_train = torch.load(git_ignore_folder + "dataset_train")
    
    dataset_test = torch.load(git_ignore_folder + "dataset_test")
    
    data_train_loader = torch.load(git_ignore_folder + "data_train_loader")
    
    data_test_loader = torch.load(git_ignore_folder + "data_test_loader")
    
#     dataset_train, dataset_test, data_train_loader, data_test_loader = get_train_test_data_loader_by_name_lr(data_preparer, model_class, dataset_name, batch_size)
     
    hyper_para_function=getattr(Data_preparer, "get_hyperparameters_" + dataset_name)
    
    model = model_class()
    
    if is_GPU:
        model.to(device)
    
    init_model_params = list(model.parameters())
    
    
    criterion, optimizer, lr_scheduler = hyper_para_function(data_preparer, model.parameters(), learning_rate, regularization_coeff)
    
    hyper_params = [criterion, optimizer, lr_scheduler]
    
    random_ids_all_epochs = torch.load(git_ignore_folder + 'random_ids_multi_super_iterations_' + str(repetition))
        
#     sorted_random_ids_all_epochs = torch.load(git_ignore_folder + 'sorted_ids_multi_super_iterations_' + str(repetition))

    
    model, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs = model_training_test(random_ids_all_epochs, num_epochs, model, dataset_train, dataset_test, len(dataset_train), len(dataset_test), optimizer, criterion, lr_scheduler, batch_size, is_GPU, device, lrs)


#     capture_provenance(X, Y, dim, epoch, num_class, batch_size, mini_epochs_per_super_iteration, random_ids_multi_super_iterations_tensors)

    torch.save(random_ids_all_epochs, git_ignore_folder + 'random_ids_multi_super_iterations')
    
    torch.save(dataset_train, git_ignore_folder + "dataset_train")
    
    torch.save(dataset_test, git_ignore_folder + "test_data")
    
#     torch.save(delta_data_ids, git_ignore_folder + "delta_data_ids")
    
    
    torch.save(gradient_list_all_epochs, git_ignore_folder + 'gradient_list_all_epochs')
    
    torch.save(para_list_all_epochs, git_ignore_folder + 'para_list_all_epochs')
    
    torch.save(learning_rate_all_epochs, git_ignore_folder + 'learning_rate_all_epochs')


#     torch.save(random_ids_multi_super_iterations, git_ignore_folder + 'random_ids_multi_super_iterations')
                  
    torch.save(num_epochs, git_ignore_folder+'epoch')    
    
    torch.save(hyper_params, git_ignore_folder + 'hyper_params')
    
    save_random_id_orders(git_ignore_folder, random_ids_all_epochs)
    
    torch.save(para_list_all_epochs[0], git_ignore_folder + 'init_para')
    
    torch.save(model, git_ignore_folder + 'origin_model')
    
    torch.save(model_class, git_ignore_folder + 'model_class')
    
    torch.save(data_train_loader, git_ignore_folder + 'data_train_loader')
    
    torch.save(data_test_loader, git_ignore_folder + 'data_test_loader')
    
    torch.save(learning_rate, git_ignore_folder + 'alpha')

    torch.save(regularization_coeff, git_ignore_folder + 'beta')
    
    torch.save(dataset_name, git_ignore_folder + 'dataset_name')
    
    torch.save(batch_size, git_ignore_folder + 'batch_size')

    torch.save(device, git_ignore_folder + 'device')

    torch.save(is_GPU, git_ignore_folder + 'is_GPU')
    
    torch.save(noise_rate, git_ignore_folder + 'noise_rate')

    test(model, dataset_test, batch_size, criterion, len(dataset_test), is_GPU, device)
