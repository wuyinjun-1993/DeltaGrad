'''
Created on Apr 12, 2020

'''

import torch
import numpy as np

import sys,os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_IO.Load_data import *
    from Models.DNN_single import *
    from Benchmark_experiments.generate_dataset_train_test import *
except ImportError:
    from Load_data import *
    from DNN_single import *
    from Benchmark_experiments.generate_dataset_train_test import *

if __name__ == '__main__':
    
    configs = load_config_data(config_file)
    
#     print(configs)
    global git_ignore_folder
    git_ignore_folder = configs['git_ignore_folder']
    
    sys_argv = sys.argv 
    
    feature_num = int(sys_argv[1]) 
    
    sample_num = int(sys_argv[2])
    
    epochs = int(sys_argv[3])
    
    repetition = int(sys_argv[4])
    
    loc1 = np.random.rand(feature_num)
    
    loc2 = np.random.rand(feature_num)
    
    distance = np.sqrt(np.power(loc1 - loc2, 2))*10
#     distance = np.random.rand(feature_num)
    
    print(distance.shape)
    
    print(loc1.shape)
    
    dataset1 = np.random.normal(loc = loc1, scale = distance, size =[sample_num, feature_num])
    
    dataset2 = np.random.normal(loc = loc2, scale = distance, size = [sample_num, feature_num])
    
    print(dataset1.shape)
    
    print(dataset2.shape)
    
#     print(loc1)
#     
#     print(loc2)
#     
#     print(np.average(dataset1, 0))
#     
#     print(np.average(dataset2, 0))
    
    
    data_tensor1 = torch.from_numpy(dataset1).type(torch.DoubleTensor)
    
    label_tensor1 = torch.ones(dataset1.shape[0], dtype = torch.long)
    
    
    data_tensor2 = torch.from_numpy(dataset2).type(torch.DoubleTensor)
    
    
    label_tensor2 = torch.zeros(dataset2.shape[0], dtype = torch.long)
    
    data_array = [data_tensor1, data_tensor2]
    
    
    train_X = torch.cat(data_array, 0)
    
    
    train_Y = torch.cat([label_tensor1, label_tensor2], 0) 
    
    random_ids = torch.randperm(train_X.shape[0])
    
    train_X = train_X[random_ids]
    
    train_Y = train_Y[random_ids]
    
    lr_model = Logistic_regression(feature_num, 2)
    train_dataset = lr_model.MyDataset(train_X, train_Y)
    
    
    generate_random_ids_list(train_dataset, epochs, repetition)
    
    
    
    
    torch.save(train_dataset, git_ignore_folder + 'gaussian_train_dataset')
    
#     torch.save(train_Y, git_ignore_folder + 'gaussian_train_X')

