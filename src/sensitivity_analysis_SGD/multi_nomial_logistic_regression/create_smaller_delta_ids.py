'''
Created on Oct 31, 2019


'''
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')

import psutil

import torch
try:
    from sensitivity_analysis_SGD.multi_nomial_logistic_regression.incremental_updates_logistic_regression_multi_dim import *
except ImportError:
    from incremental_updates_logistic_regression_multi_dim import *


X = torch.load(git_ignore_folder + 'noise_X')
        
Y = torch.load(git_ignore_folder + 'noise_Y')


delta_data_ids = torch.load(git_ignore_folder+'noise_data_ids')


len = delta_data_ids.shape[0]

sub_delta_data_ids = delta_data_ids[0: int(len*0.01)]

print(sub_delta_data_ids)

print(len)

torch.save(sub_delta_data_ids, git_ignore_folder+'sub_noise_data_ids')

