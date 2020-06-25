'''
Created on Jan 27, 2020

'''


import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import time
from torchvision import datasets, transforms
from torch import nn, optim
import os
from collections import deque 
import random
import sys, ast

import argparse
import torch


from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader



sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/data_IO')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Interpolation')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Models')


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.abspath(__file__))





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

config_file = 'train_data_meta_info.ini'




if __name__ == '__main__':

    parser = argparse.ArgumentParser('DeltaGrad_prepare_data')

    parser.add_argument('--model',  help="name of models used in the experiments")
    
    parser.add_argument('--dataset',  help="name of dataset used in the experiments")
    
#     parser.add_argument('--bz', type = int, help="minibatch size used in SGD")
    

    parser.add_argument('--repo', default = gitignore_repo, help = 'repository to store the data and the intermediate results')

    args = parser.parse_args()
    
    git_ignore_folder = args.repo
    
#     configs = load_config_data(config_file)
    
#     print(configs)
#     global git_ignore_folder
#     git_ignore_folder = configs['git_ignore_folder']
    
    model_name = args.model
    
    dataset_name = args.dataset
    
#     batch_size = args.bz
    
    model_class = getattr(sys.modules[__name__], model_name)

    data_preparer = Data_preparer()
        
    dataset_train, dataset_test = get_train_test_data_loader_by_name_lr(data_preparer, model_class, dataset_name, git_ignore_folder)

    

    torch.save(dataset_train, git_ignore_folder + "dataset_train")
    
#     torch.save(data_train_loader, git_ignore_folder + "data_train_loader")
#     
#     torch.save(data_test_loader, git_ignore_folder + "data_test_loader")
    
    torch.save(dataset_test, git_ignore_folder + "dataset_test")


