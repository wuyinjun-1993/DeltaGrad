'''
Created on Jan 13, 2020

'''
import torch

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
    
    from generate_noise import *

except ImportError:
    from Load_data import *
# from sensitivity_analysis.logistic_regression.Logistic_regression import test_X
    from utils import *
    from Models.Data_preparer import *
#     from evaluating_test_samples import *
    from benchmark_exp import *
    
    from generate_noise import *

if __name__ == '__main__':
    
    sys_args = sys.argv
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    noise_rate = float(sys_args[1])
    
    
    dataset_name = sys_args[2]
    
    
    start = bool(int(sys_args[3]))    
    
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
        
        
    
#     delta_data_ids = random_deletion(len(dataset_train), 1)
    
    print(delta_data_ids)
    
    torch.save(delta_data_ids, git_ignore_folder + "delta_data_ids")
    
    
    
    
    
    
    