'''
Created on Jan 4, 2020

'''

import torch

from torch import utils

class Batch_sampler(torch.utils.data.Sampler):
    '''
    classdocs
    '''
    def __init__(self, sample_ids_list_all_epochs):
#         self.ids = ids
        
        self.sample_ids_list_index = 0
        
        self.sample_ids_list_all_epochs = sample_ids_list_all_epochs
        
        self.num_samples = len(sample_ids_list_all_epochs[self.sample_ids_list_index])

    def __iter__(self):
        print ('\tcalling Sampler:__iter__')
        
        
        
        return iter(self.sample_ids_list_all_epochs[self.sample_ids_list_index])
#         return iter([list(range(self.num_samples))])
        

    def __len__(self):
        print ('\tcalling Sampler:__len__')
        self.num_samples = len(self.sample_ids_list_all_epochs[self.sample_ids_list_index])
        return self.num_samples
    
    def increm_ids(self):
        self.sample_ids_list_index += 1
    
    
    def reset_ids(self):
        self.sample_ids_list_index = 0
    
    
    
    