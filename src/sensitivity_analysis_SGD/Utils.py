'''
Created on Apr 12, 2019


'''

import torch

def random_shuffle_data(X, Y, dim, noise_data_ids):
         
    random_ids = torch.randperm(dim[0])
     
    X = X[random_ids]
     
     
    Y = Y[random_ids]
    
    
    shuffled_noise_data_ids = torch.zeros(noise_data_ids.shape)
    
    for i in range(noise_data_ids.shape[0]):
        
        shuffled_id = torch.nonzero(random_ids == noise_data_ids[i])
        
#             print(shuffled_id)
        
        shuffled_noise_data_ids[i] = shuffled_id 
        
        
    return X, Y, shuffled_noise_data_ids