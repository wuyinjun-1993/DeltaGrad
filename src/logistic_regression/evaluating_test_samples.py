'''
Created on Mar 20, 2019


'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')


try:
    from data_IO.Load_data import *
    from sensitivity_analysis_SGD.logistic_regression.Logistic_regression import *
except ImportError:
    from Load_data import *
    from Logistic_regression import * 
import torch


# def get_model_parameters(name):
#     dataset = torch.load(git_ignore_folder + name)
#     
#     return dataset


def compute_accuracy(file_name, model):
    [X, Y] = load_data(True, file_name)

#     [X, Y] = clean_sensor_data(file_name)
    
    X = extended_by_constant_terms(X)
    
    
            
            
#     rids = torch.concat(((Y == 1), (Y ==0)), dim=0 ).nonzero()

    rids = ((Y.view(-1) == 1) + (Y.view(-1) == 0)).nonzero() 

#     rids1 = (Y.view(-1) == 1).nonzero()
#     
#     rids2 = (Y.view(-1) == 0).nonzero()
#     
#     
#     rids = torch.cat((rids1, rids2), dim= 0)
    Y = Y.view(-1,1)
    
    Y = torch.index_select(Y, 0, rids.view(-1))
    
    X = torch.index_select(X, 0, rids.view(-1))
    
    
    
#     rids2 = (Y == 0).nonzero()
    
    
    
    min_label = torch.min(Y)
            
    
    if min_label == 0:
        Y = 2*Y-1
    Y = Y.view(-1,1)
    
    
    
    res = torch.mm(X, model)
    
    res[res >= 0] = 1
    
    res[res < 0] = -1
    
    accuracy = torch.sum(res == Y).numpy()*1.0/Y.shape[0]
    
    return accuracy
    
def compute_accuracy2(X, Y, model):
#     [X, Y] = load_data(True, file_name)

#     [X, Y] = clean_sensor_data(file_name)
    
#     X = extended_by_constant_terms(X)
    
    min_label = torch.min(Y)
            
    if min_label == 0:
        Y = 2*Y-1
    Y = Y.view(-1,1)
    
    
    
    res = torch.mm(X, model)
    
#     res2 = torch.mm(X, model_expect)
    
    res[res >= 0] = 1
    
#     res2[res2 >= 0] = 1
    
    res[res < 0] = -1
    
#     res2[res2 < 0] = -1
    
    accuracy = torch.sum(res == Y).numpy()*1.0/Y.shape[0]
    
    return accuracy   
    
    
    

def compute_accuracy2_sparse(X, Y, model):
#     [X, Y] = load_data(True, file_name)

#     [X, Y] = clean_sensor_data(file_name)
    
#     X = extended_by_constant_terms(X)
    
    min_label = torch.min(Y)
            
    if min_label == 0:
        Y = 2*Y-1
    Y = Y.view(-1,1)
    Y = Y.numpy()
    
    
    res = X.dot(model.detach().numpy())
    
#     res2 = torch.mm(X, model_expect)
    
    res[res >= 0] = 1
    
#     res2[res2 >= 0] = 1
    
    res[res < 0] = -1
    
#     res2[res2 < 0] = -1
    
    accuracy = np.sum(res == Y)*1.0/Y.shape[0]
    
    return accuracy      