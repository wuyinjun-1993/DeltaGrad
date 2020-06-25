'''
Created on Jan 13, 2020

'''
import torch

import sys, os

import argparse


sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/data_IO')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Interpolation')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Models')


sys.path.append(os.path.abspath(__file__))

from utils import *

# try:
# # from sensitivity_analysis.logistic_regression.Logistic_regression import test_X
#     from utils import *
# #     from sensitivity_analysis.linear_regression.evaluating_test_samples import *
# #     from Models.Data_preparer import *
#     
#     from generate_noise import *
# 
# except ImportError:
#     from Load_data import *
# # from sensitivity_analysis.logistic_regression.Logistic_regression import test_X
#     from utils import *
# #     from Models.Data_preparer import *
# #     from evaluating_test_samples import *
#     from generate_noise import *

if __name__ == '__main__':
    
#     sys_args = sys.argv

    parser = argparse.ArgumentParser('generate_rand_ids')

    
    parser.add_argument('--dataset',  help="dataset to be used")
    
    parser.add_argument('--ratio',  type=float, help="delete rate or add rate")
    
    parser.add_argument('--restart',action='store_true',  help="whether to append the deleted or added samples or reconstruct the added or deleted samples")

    parser.add_argument('--repo', default = gitignore_repo, help = 'repository to store the data and the intermediate results')

#     parser.add_argument('--add',  action='store_true', help="The flag for incrementally adding training samples, otherwise for incrementally deleting training samples")

#     parser.add_argument('--epochs', type = int, help="number of epochs used in SGD")

#     configs = load_config_data(config_file)
    
#     git_ignore_folder = configs['git_ignore_folder']

    args = parser.parse_args()

    git_ignore_folder = args.repo
    
    noise_rate = args.ratio
    
    
    dataset_name = args.dataset
    
#     epochs = args.epochs
    
    start = args.restart    
    
    data_preparer = Data_preparer()
    
    
    function=getattr(Data_preparer, "prepare_" + dataset_name)
    
    
    
    
#     dataset_train, dataset_test, data_train_loader, data_test_loader = get_train_test_data_loader_by_name(data_preparer, model_class, dataset_name, batch_size)

    

    if start:
        
#         training_data, trainin_labels, test_data, test_labels = function(data_preparer)
        dataset_train = torch.load(git_ignore_folder + "dataset_train")
        
        print(dataset_train.data.shape)
        
        full_ids_list = list(range(len(dataset_train.data)))
        
        delta_data_ids = random_generate_subset_ids2(int(len(dataset_train.data)*noise_rate), full_ids_list)
            
        train_data_len = len(dataset_train.data)
        
        torch.save(train_data_len, git_ignore_folder + 'train_data_len')
        
    else:
        old_delta_ids = torch.load(git_ignore_folder + "delta_data_ids")
    
        train_data_len = torch.load(git_ignore_folder + 'train_data_len')
    
        full_ids_list = list(range(train_data_len))
    
        remaining_size = int(train_data_len*noise_rate) - old_delta_ids.shape[0]
        
        remaining_full_ids_list = list(set(full_ids_list).difference(set(old_delta_ids.tolist())))
        
        if remaining_size > 0:
            curr_delta_data_ids = random_generate_subset_ids2(remaining_size, remaining_full_ids_list)
        
            delta_data_ids = torch.tensor(list(set(old_delta_ids.tolist()).union(set(curr_delta_data_ids.tolist()))))
        else:
            delta_data_ids = old_delta_ids
        
    
#     if not args.add:

#     else:
        
        
        
#     delta_data_ids = random_deletion(len(dataset_train), 1)
    
    print(delta_data_ids)
    
    torch.save(delta_data_ids, git_ignore_folder + "delta_data_ids")
    
    
    
    
    
    
    