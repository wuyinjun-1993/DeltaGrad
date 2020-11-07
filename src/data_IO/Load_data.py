'''
Created on Feb 4, 2019

'''

import csv
import numpy as np
from numpy import linalg as LA, average
import torch
from torch.autograd import Variable
import pandas as pd
import configparser
import json

from scipy.sparse import coo_matrix
from sklearn.datasets import fetch_rcv1

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 
try:
    from generate_config_files import *
except ImportError:
    from data_IO.generate_config_files import *

git_ignore_folder = '../../../.gitignore/'


# file_name = '../../data/szeged-weather/weatherHistory.csv'
# file_name = '../../data/candy/candy-data.csv'
# file_name = '../../data/adult.csv'
# file_name = '../../data/train.csv'

file_name = '../../../data/creditcard.csv'

# file_name = '../../data/minist.csv'
# 
# y_cols = [0]
# 
# x_cols = list(range(784))
# 
# x_cols = [x+1 for x in x_cols]


#  
# x_cols = list(range(29))
#   
# x_cols = [x+1 for x in x_cols]
#   
# y_cols = [30]

# x_cols = [3, 4, 5, 6, 7]
#       
# y_cols = [2]



# x_cols = [0, 4, 10, 11, 12]
#      
# y_cols = [14]

# x_cols = [2,3,4,5,6,7,8,9,10,11,12]
#  
# y_cols = [1]

# x_cols = [5,6,7,8]
# 
# y_cols = [3,4]



# config_file = '../train_data_meta_info.ini'

def get_relative_change(tensor1, tensor2):
    
#     non_zero_id = (tensor1 != 0)
    
    print('relative magnitude change::', torch.max(torch.pow(tensor1 - tensor2, 2)))

def load_config_data(path):
    with open(path) as f:
        data = json.load(f)
    return data

def extended_by_constant_terms(X, extend_more_columns):
    X = torch.cat((X, torch.ones([X.shape[0], 1], dtype=torch.double)), 1)
    
    if extend_more_columns:
        X = torch.cat((X, torch.rand([X.shape[0], 500], dtype = torch.double)/100), 1)
    
    return X


def extended_by_constant_terms_numpy(X):
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
    return X

def split_train_test_rcv1(X, Y, ratio):
    
    Y_labels = torch.unique(Y)
        
    all_selected_rows = []
    
    for Y_label in Y_labels:
        rids = (Y.view(-1) == Y_label).nonzero()
    
        curr_selected_num = int(rids.shape[0]*ratio)
        
        if curr_selected_num == 0:
            continue
        
        rid_rids = torch.tensor(list(np.random.choice(list(range(rids.shape[0])), curr_selected_num, replace = False)))

        all_selected_rows.append(rids[rid_rids])

    selected_rows = torch.cat(all_selected_rows, 0).view(-1)
    
    train_set_rids = torch.tensor(list(set(range(X.shape[0])) - set(selected_rows.numpy())))
    
    test_X = torch.zeros(selected_rows.shape[0], X.shape[1])
    
    test_Y = torch.zeros(selected_rows.shape[0], Y.shape[1])
    
    for i in range(selected_rows.shape[0]):
        
        rid = selected_rows[i]
        
        test_X[i] = X[rid]
        
        test_Y[i] = Y[rid]


    train_X = torch.zeros(train_set_rids.shape[0], X.shape[1])
    
    train_Y = torch.zeros(train_set_rids.shape[0], Y.shape[1])
    
    for i in range(train_set_rids.shape[0]):
        
        rid = train_set_rids[i]
        
        train_X[i] = X[rid]
        
        train_Y[i] = Y[rid]


    return train_X, train_Y, test_X, test_Y
        
    
    
    
    
    

def split_train_test_data(X, Y, ratio, is_classification):
    
    
    if is_classification:
    
        Y_labels = torch.unique(Y)
        
        all_selected_rows = []
        
        for Y_label in Y_labels:
            rids = (Y.view(-1) == Y_label).nonzero()
        
            curr_selected_num = int(rids.shape[0]*ratio)
            
            if curr_selected_num == 0:
                continue
            
            rid_rids = torch.tensor(list(np.random.choice(list(range(rids.shape[0])), curr_selected_num, replace = False)))
    
            all_selected_rows.append(rids[rid_rids])
    
        selected_rows = torch.cat(all_selected_rows, 0).view(-1)
        
        
    else:
        
        rids = list(range(Y.shape[0]))
        
        selected_rows = torch.tensor(list(np.random.choice(list(range(Y.shape[0])), int(Y.shape[0]*ratio), replace = False)))
    
        
        
        
#     positive_rids = (Y.view(-1) == 1).nonzero()
#     
#     negative_rids = (Y.view(-1) == -1).nonzero()
#     
#     
#     
#     selected_num1 = int(positive_rids.shape[0]*ratio)
#     
#     
#     selected_num2 = int(negative_rids.shape[0]*ratio)
#     
#     
#     positive_rid_rids = torch.tensor(list(np.random.choice(list(range(positive_rids.shape[0])), selected_num1, replace = False)))
#     
#     selected_rows1 = positive_rids[positive_rid_rids]
#     
#     
#     negative_rid_rids = torch.tensor(list(np.random.choice(list(range(negative_rids.shape[0])), selected_num2, replace = False)))
#     
#     selected_rows2 = negative_rids[negative_rid_rids]
    
    
    
    
    test_X = torch.index_select(X, 0, selected_rows)
    
    test_Y = torch.index_select(Y, 0, selected_rows)
    
    train_set_rids = torch.tensor(list(set(range(X.shape[0])) - set(selected_rows.numpy())))
    
    
    train_X = torch.index_select(X, 0, train_set_rids)
    
    train_Y = torch.index_select(Y, 0, train_set_rids)
    
    return train_X, train_Y, test_X, test_Y
    
def load_data2(is_classification, file_name):
    
#     x_data = np.array()
#     
#     y_data = np.array()

    print('start loading data...')
    
    
    configs = load_config_data(config_file)
    
    from_csv = configs[file_name]['from_csv']
    
    if not from_csv:
        return clean_sensor_data(file_name, is_classification)
    
    train_data = pd.read_csv(file_name)
    
    
    x_cols = configs[file_name]['x_cols']
    
    y_cols = configs[file_name]['y_cols']
    
    x_data = train_data.iloc[:,x_cols].get_values()
    
    y_data = train_data.iloc[:,y_cols].get_values()
    
#     with open(file_name, 'r') as f:
#         reader = csv.reader(f)
#         line_count = 0
#         for row in reader:
#             if line_count == 0:
# #                 print(f'Column names are {", ".join(row)}')
#                 line_count += 1
#             else:
#                 
#                 if line_count == 1:
#                     x_data = np.array(cleaning(row, x_cols), dtype=np.float64)
#                      
#                     y_data = np.array(cleaning(row, y_cols), dtype=np.float64)
#                 else:
#                 
#                     x_data = np.vstack([x_data, np.array(cleaning(row, x_cols), dtype=np.float64)])
#                     
#                     y_data = np.vstack([y_data, np.array(cleaning(row, y_cols), dtype=np.float64)])
#                 
#                 line_count += 1
        
    
    
    x_data = normalize(x_data)
    
    
    if not is_classification:
        y_data = normalize(y_data)
    
#     print(x_[0:10, :])
    
    x_train = torch.from_numpy(x_data)

    y_train = torch.from_numpy(y_data)
    
    
    x_train,y_train=x_train.type(torch.DoubleTensor),y_train.type(torch.DoubleTensor)
    
#     print(x_train[0:10, :])

#     print('sum', torch.sum(x_train, dim=0))
    
    X = Variable(x_train)
    
    Y = Variable(y_train)
    
    if is_classification:
        min_label = torch.min(Y)
            
        if min_label == 0:
            Y = 2*Y-1
    
#     print(Y)
    
    print('X_max::', torch.max(X))
    
    print('X_min::', torch.min(X))
    
    print('Y_max::', torch.max(Y))
    
    print('Y_min::', torch.min(Y))
    
    print('x_shape::', X.shape)
    
    print('loading data done...')
    
    print('X_norm::', torch.norm(torch.mm(torch.t(X), X), p=2))
    
    return X, Y#split_train_test_data(X, Y, 0.1, is_classification)  
    
def convert_coo_matrix2tensor(Y_coo):
    
    
    Y = Y_coo.todense()
    
    
    res = torch.from_numpy(Y).type(torch.DoubleTensor).to_sparse()
    
    
#     indices = np.vstack((Y_coo.row, Y_coo.col))
#     values = Y_coo.data
#     
#     i = torch.LongTensor(indices)
#     v = torch.DoubleTensor(values)
#     shape = Y_coo.shape
#     
# #     print(Y_coo)
#     
#     Y = torch.sparse.DoubleTensor(i, v, torch.Size(shape))
    
    return res


def convert_coo_matrix2_dense_tensor(Y_coo):
    
    
    Y = Y_coo.todense()
    
    
    res = torch.from_numpy(Y).type(torch.DoubleTensor)
    
    
#     indices = np.vstack((Y_coo.row, Y_coo.col))
#     values = Y_coo.data
#     
#     i = torch.LongTensor(indices)
#     v = torch.DoubleTensor(values)
#     shape = Y_coo.shape
#     
# #     print(Y_coo)
#     
#     Y = torch.sparse.DoubleTensor(i, v, torch.Size(shape)).to_dense()
    
    return res

def load_data(is_classification, file_name):
    
#     x_data = np.array()
#     
#     y_data = np.array()

    print('start loading data...')
    
    
    configs = load_config_data(config_file)
    
    from_csv = configs[file_name]['from_csv']
    
    if not from_csv:
        return clean_sensor_data(file_name, is_classification)
    
    train_data = pd.read_csv(file_name)
    
    
    x_cols = configs[file_name]['x_cols']
    
    y_cols = configs[file_name]['y_cols']
    
    x_data = train_data.iloc[:,x_cols].get_values()
    
    y_data = train_data.iloc[:,y_cols].get_values()
    
#     with open(file_name, 'r') as f:
#         reader = csv.reader(f)
#         line_count = 0
#         for row in reader:
#             if line_count == 0:
# #                 print(f'Column names are {", ".join(row)}')
#                 line_count += 1
#             else:
#                 
#                 if line_count == 1:
#                     x_data = np.array(cleaning(row, x_cols), dtype=np.float64)
#                      
#                     y_data = np.array(cleaning(row, y_cols), dtype=np.float64)
#                 else:
#                 
#                     x_data = np.vstack([x_data, np.array(cleaning(row, x_cols), dtype=np.float64)])
#                     
#                     y_data = np.vstack([y_data, np.array(cleaning(row, y_cols), dtype=np.float64)])
#                 
#                 line_count += 1
        
    
    
    x_data = normalize(x_data)
    
    
    if not is_classification:
        y_data = normalize(y_data)
    
#     print(x_[0:10, :])
    
    x_train = torch.from_numpy(x_data)

    y_train = torch.from_numpy(y_data)
    
    
    x_train,y_train=x_train.type(torch.DoubleTensor),y_train.type(torch.DoubleTensor)
    
#     print(x_train[0:10, :])

#     print('sum', torch.sum(x_train, dim=0))
    
    X = Variable(x_train)
    
    Y = Variable(y_train)
    
    if is_classification:
        min_label = torch.min(Y)
            
        if min_label == 0:
            Y = 2*Y-1
    
#     print(Y)
    
    print('X_max::', torch.max(X))
    
    print('X_min::', torch.min(X))
    
    print('Y_max::', torch.max(Y))
    
    print('Y_min::', torch.min(Y))
    
    print('x_shape::', X.shape)
    
    print('loading data done...')
    
    print('X_norm::', torch.norm(torch.mm(torch.t(X), X), p=2))
    
    return split_train_test_data(X, Y, 0.1, is_classification)


def load_data_numpy(is_classification, file_name):
    
#     x_data = np.array()
#     
#     y_data = np.array()

    print('start loading data...')
    
    
    train_data = pd.read_csv(file_name)
    
    configs = load_config_data(config_file)
    
    x_cols = configs[file_name]['x_cols']
    
    y_cols = configs[file_name]['y_cols']
    
    x_data = train_data.iloc[:,x_cols].get_values()
    
    y_data = train_data.iloc[:,y_cols].get_values()
    
#     with open(file_name, 'r') as f:
#         reader = csv.reader(f)
#         line_count = 0
#         for row in reader:
#             if line_count == 0:
# #                 print(f'Column names are {", ".join(row)}')
#                 line_count += 1
#             else:
#                 
#                 if line_count == 1:
#                     x_data = np.array(cleaning(row, x_cols), dtype=np.float64)
#                      
#                     y_data = np.array(cleaning(row, y_cols), dtype=np.float64)
#                 else:
#                 
#                     x_data = np.vstack([x_data, np.array(cleaning(row, x_cols), dtype=np.float64)])
#                     
#                     y_data = np.vstack([y_data, np.array(cleaning(row, y_cols), dtype=np.float64)])
#                 
#                 line_count += 1
        
    
    
    x_data = normalize(x_data)
    
    
    if not is_classification:
        y_data = normalize(y_data)
    
#     print(x_[0:10, :])
    
#     x_train = torch.from_numpy(x_data)
# 
#     y_train = torch.from_numpy(y_data)
#     
#     
#     x_train,y_train=x_train.type(torch.DoubleTensor),y_train.type(torch.DoubleTensor)
#     
# #     print(x_train[0:10, :])
# 
# #     print('sum', torch.sum(x_train, dim=0))
#     
#     X = Variable(x_train)
#     
#     Y = Variable(y_train)
    
    if is_classification:
        min_label = np.min(y_data)
            
        if min_label == 0:
            y_data = 2*y_data-1
    
#     print(Y)
    
#     print('X_max::', torch.max(X))
#     
#     print('X_min::', torch.min(X))
#     
#     print('Y_max::', torch.max(Y))
#     
#     print('Y_min::', torch.min(Y))
#     
#     print('x_shape::', X.shape)
#     
#     print('loading data done...')
#     
#     print('X_norm::', torch.norm(torch.mm(torch.t(X), X), p=2))
#     
#     return split_train_test_data(X, Y, 0.3)

    return x_data, y_data



def load_data_multi_classes(is_classification, file_name, split_id = None):
    
#     x_data = np.array()
#     
#     y_data = np.array()

    print('start loading data...')
    
    
    configs = load_config_data(config_file_name)
    
    from_csv = configs[file_name]['from_csv']
    
    if not from_csv:
        return clean_sensor_data(file_name, is_classification, split_id)
    
    train_data = pd.read_csv(file_name)

    print('loading data done')
    
    x_cols = configs[file_name]['x_cols']
    
    y_cols = configs[file_name]['y_cols']
    
    
    train_data.fillna(0)
    
    x_data = train_data.iloc[:,x_cols].get_values()
    
    y_data = train_data.iloc[:,y_cols].get_values()
    
    
    
    
    
#     with open(file_name, 'r') as f:
#         reader = csv.reader(f)
#         line_count = 0
#         for row in reader:
#             if line_count == 0:
# #                 print(f'Column names are {", ".join(row)}')
#                 line_count += 1
#             else:
#                 
#                 if line_count == 1:
#                     x_data = np.array(cleaning(row, x_cols), dtype=np.float64)
#                      
#                     y_data = np.array(cleaning(row, y_cols), dtype=np.float64)
#                 else:
#                 
#                     x_data = np.vstack([x_data, np.array(cleaning(row, x_cols), dtype=np.float64)])
#                     
#                     y_data = np.vstack([y_data, np.array(cleaning(row, y_cols), dtype=np.float64)])
#                 
#                 line_count += 1
        
    
    
    x_data = normalize(x_data)
#     x_data = normalize_with_known_range(x_data, 255, 0)
    
    
    if not is_classification:
        y_data = normalize(y_data)
    
    print('normalized done!!')
    
#     print(x_[0:10, :])
    
    x_train = torch.from_numpy(x_data)

    y_train = torch.from_numpy(y_data)
    
    
#     x_train,y_train=x_train.type(torch.FloatTensor),y_train.type(torch.FloatTensor)
    
    x_train,y_train=x_train.type(torch.DoubleTensor),y_train.type(torch.DoubleTensor)
    
#     print(x_train[0:10, :])

#     print('sum', torch.sum(x_train, dim=0))
    
    X = Variable(x_train)
    
    Y = Variable(y_train)
    
    if is_classification:
        
        Y_uniques = torch.unique(Y)
        
        if not (set(Y_uniques.numpy()) == set(range(Y_uniques.shape[0]))):
#         print(Y_uniques)
            Y_copy = torch.zeros(Y.shape)

            
            for k in range(Y_uniques.shape[0]):
    #             print((Y==Y_uniques[k]).nonzero()[:, 0])
                
                Y_copy[(Y==Y_uniques[k]).nonzero()[:, 0]] = k
                
            Y = Y_copy 

        
        
        
#         min_label = torch.min(Y)
#            
#         if min_label == 0:
#             Y = 2*Y-1
    
#     print(Y)
    
    print('X_max::', torch.max(X))
    
    print('X_min::', torch.min(X))
    
    print('Y_max::', torch.max(Y))
    
    print('Y_min::', torch.min(Y))
    
    print('x_shape::', X.shape)
    
    print('loading data done...')
    
#     print('X_norm::', torch.norm(torch.mm(torch.t(X), X), p=2))
    
    if split_id is None:
        return split_train_test_data(X, Y, 0.1, is_classification)
    else:
        return X[0:split_id], Y[0:split_id], X[split_id:], Y[split_id:]





def load_data_multi_classes_single(is_classification, file_name):
    
#     x_data = np.array()
#     
#     y_data = np.array()

    print('start loading data...')
    
    
    configs = load_config_data(config_file)
    
    from_csv = configs[file_name]['from_csv']
    
    if not from_csv:
        return clean_sensor_data_single(file_name, is_classification)
    
    train_data = pd.read_csv(file_name)

    print('loading data done')
    
    x_cols = configs[file_name]['x_cols']
    
    y_cols = configs[file_name]['y_cols']
    
    
    train_data.fillna(0)
    
    x_data = train_data.iloc[:,x_cols].get_values()
    
    y_data = train_data.iloc[:,y_cols].get_values()
    
    
    
    
    
#     with open(file_name, 'r') as f:
#         reader = csv.reader(f)
#         line_count = 0
#         for row in reader:
#             if line_count == 0:
# #                 print(f'Column names are {", ".join(row)}')
#                 line_count += 1
#             else:
#                 
#                 if line_count == 1:
#                     x_data = np.array(cleaning(row, x_cols), dtype=np.float64)
#                      
#                     y_data = np.array(cleaning(row, y_cols), dtype=np.float64)
#                 else:
#                 
#                     x_data = np.vstack([x_data, np.array(cleaning(row, x_cols), dtype=np.float64)])
#                     
#                     y_data = np.vstack([y_data, np.array(cleaning(row, y_cols), dtype=np.float64)])
#                 
#                 line_count += 1
        
    
    
    x_data = normalize(x_data)
#     x_data = normalize_with_known_range(x_data, 255, 0)
    
    
    if not is_classification:
        y_data = normalize(y_data)
    
    print('normalized done!!')
    
#     print(x_[0:10, :])
    
    x_train = torch.from_numpy(x_data)

    y_train = torch.from_numpy(y_data)
    
    
#     x_train,y_train=x_train.type(torch.FloatTensor),y_train.type(torch.FloatTensor)
    
    x_train,y_train=x_train.type(torch.DoubleTensor),y_train.type(torch.DoubleTensor)
    
#     print(x_train[0:10, :])

#     print('sum', torch.sum(x_train, dim=0))
    
    X = Variable(x_train)
    
    Y = Variable(y_train)
    
    if is_classification:
        
        Y_uniques = torch.unique(Y)
        
        if not (set(Y_uniques.numpy()) == set(range(Y_uniques.shape[0]))):
#         print(Y_uniques)
            Y_copy = torch.zeros(Y.shape)

            
            for k in range(Y_uniques.shape[0]):
    #             print((Y==Y_uniques[k]).nonzero()[:, 0])
                
                Y_copy[(Y==Y_uniques[k]).nonzero()[:, 0]] = k
                
            Y = Y_copy 

        
        
        
#         min_label = torch.min(Y)
#            
#         if min_label == 0:
#             Y = 2*Y-1
    
#     print(Y)
    
    print('X_max::', torch.max(X))
    
    print('X_min::', torch.min(X))
    
    print('Y_max::', torch.max(Y))
    
    print('Y_min::', torch.min(Y))
    
    print('x_shape::', X.shape)
    
    print('loading data done...')
    
#     print('X_norm::', torch.norm(torch.mm(torch.t(X), X), p=2))
    
    return X, Y


def load_data_rcv1_test():
    
    rcv1 = fetch_rcv1()
    
    X_coo = rcv1.data[23149:].tocoo()#coo_matrix(([3,4,5], ([0,1,1], [2,0,2])), shape=(2,3))
    
    Y_coo = rcv1.target[23149:].tocoo()
    
    
    values = X_coo.data
#     print(X_coo)
    indices = np.vstack((X_coo.row, X_coo.col))
    
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    shape = X_coo.shape
    
    X_sparse = torch.sparse.DoubleTensor(i, v, torch.Size(shape))
    
    
    indices = np.vstack((Y_coo.row, Y_coo.col))
    values = Y_coo.data
    
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    shape = Y_coo.shape
    
    
    Y_sparse = torch.sparse.DoubleTensor(i, v, torch.Size(shape))
    
    return X_sparse, Y_sparse
    
    

def load_data_multi_classes_rcv1():
    
#     x_data = np.array()
#     
#     y_data = np.array()

    print('start loading data...')
    
    rcv1 = fetch_rcv1()



    X_coo = rcv1.data[0:23149].tocoo()#coo_matrix(([3,4,5], ([0,1,1], [2,0,2])), shape=(2,3))
    
    Y_coo = rcv1.target[0:23149].tocoo()
    
    
    values = X_coo.data
#     print(X_coo)
    indices = np.vstack((X_coo.row, X_coo.col))
    
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    shape = X_coo.shape
    
    X = torch.sparse.DoubleTensor(i, v, torch.Size(shape)).to_dense()
    
    
    indices = np.vstack((Y_coo.row, Y_coo.col))
    values = Y_coo.data
    
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    shape = Y_coo.shape
    
#     print(Y_coo)
    
    Y = torch.sparse.DoubleTensor(i, v, torch.Size(shape)).to_dense()






    
#     configs = load_config_data(config_file)
#     
#     from_csv = configs[file_name]['from_csv']
#     
#     if not from_csv:
#         return clean_sensor_data(file_name, is_classification)
#     
#     train_data = pd.read_csv(file_name)
# 
#     
#     
#     x_cols = configs[file_name]['x_cols']
#     
#     y_cols = configs[file_name]['y_cols']
#     
#     
#     
#     
#     x_data = train_data.iloc[:,x_cols].get_values()
#     
#     y_data = train_data.iloc[:,y_cols].get_values()
#     
# #     with open(file_name, 'r') as f:
# #         reader = csv.reader(f)
# #         line_count = 0
# #         for row in reader:
# #             if line_count == 0:
# # #                 print(f'Column names are {", ".join(row)}')
# #                 line_count += 1
# #             else:
# #                 
# #                 if line_count == 1:
# #                     x_data = np.array(cleaning(row, x_cols), dtype=np.float64)
# #                      
# #                     y_data = np.array(cleaning(row, y_cols), dtype=np.float64)
# #                 else:
# #                 
# #                     x_data = np.vstack([x_data, np.array(cleaning(row, x_cols), dtype=np.float64)])
# #                     
# #                     y_data = np.vstack([y_data, np.array(cleaning(row, y_cols), dtype=np.float64)])
# #                 
# #                 line_count += 1
#         
#     
#     
#     x_data = normalize(x_data)
# #     x_data = normalize_with_known_range(x_data, 255, 0)
#     
#     
#     if not is_classification:
#         y_data = normalize(y_data)
#     
# #     print(x_[0:10, :])
#     
#     x_train = torch.from_numpy(x_data)
# 
#     y_train = torch.from_numpy(y_data)
#     
#     
# #     x_train,y_train=x_train.type(torch.FloatTensor),y_train.type(torch.FloatTensor)
#     
#     x_train,y_train=x_train.type(torch.FloatTensor),y_train.type(torch.FloatTensor)
#     
# #     print(x_train[0:10, :])
# 
# #     print('sum', torch.sum(x_train, dim=0))
#     
#     X = Variable(x_train)
#     
#     Y = Variable(y_train)
    
#     if is_classification:
        
    Y_uniques = torch.unique(Y)
    
    if not (set(Y_uniques.numpy()) == set(range(Y_uniques.shape[0]))):
#         print(Y_uniques)
        Y_copy = torch.zeros(Y.shape)

        
        for k in range(Y_uniques.shape[0]):
#             print((Y==Y_uniques[k]).nonzero()[:, 0])
            
            Y_copy[(Y==Y_uniques[k]).nonzero()[:, 0]] = k
            
        Y = Y_copy 

        
        
        
#         min_label = torch.min(Y)
#            
#         if min_label == 0:
#             Y = 2*Y-1
    
#     print(Y)
    
    print('X_max::', torch.max(X))
    
    print('X_min::', torch.min(X))
    
    print('Y_max::', torch.max(Y))
    
    print('Y_min::', torch.min(Y))
    
    print('x_shape::', X.shape)
    
    print('y_shape::', Y[:,0].shape)
    
    print('loading data done...')
    
#     print('X_norm::', torch.norm(torch.mm(torch.t(X), X), p=2))
    
    return split_train_test_data(X, Y[:,0], 0.1, True)

def normalize_with_known_range(data, x_max, x_min):
    
#     average_value = np.mean(data, axis = 0)
#     
#     std_value = np.std(data, axis = 0)
    
    
#     x_max = np.amax(data, axis = 0)
#     
#     x_min = np.amin(data, axis = 0)
    
    range = x_max - x_min
    
#     print(average_value.shape)
#     
#     print(data)
#     
#     print(average_value)
#     
#     print(std_value)
    
    data = (data - x_min)/range
    
#     data = data /std_value
    
    return data


def normalize(data):
    
#     average_value = np.mean(data, axis = 0)
#     
#     std_value = np.std(data, axis = 0)
    print('normalization start!!')
    
    x_max = np.amax(data, axis = 0)
    
    x_min = np.amin(data, axis = 0)
    
    range = x_max - x_min
    
    update_data = data[:,range != 0] 
    
    
#     print(average_value.shape)
#     
#     print(data)
#     
#     print(average_value)
#     
#     print(std_value)
    
    data = (update_data - x_min[range!=0])/range[range!=0]
    
#     data = data /std_value
    
    return data
    


def cleaning(data, cols):
    
    converted_data = []
    
    i = 0
    
    for col in cols:
#         print(data[col])
        converted_data.append(float(data[col]))
        i = i + 1
    
    return converted_data

def check_correctness(data):
#     print(data)
    print(data.shape)
    res = LA.norm(data[:,3])*LA.norm(data[:,3])
    print('res', res)
    
def clean_sensor_data0(file_name, is_classification, num_features, split_id = None):

    Y_data = []
        
    X_data = []

#     configs = load_config_data(config_file)
# 
#     num_features = int(configs[file_name]['feature_num'])

    with open(file_name) as fp:  
        line = fp.readline()
        cnt = 1
        
        while line:
#             print("Line {}".format(cnt))
            
            contents = line.split(' ')
            
            if ':' not in contents[-1]:
                contents.pop()
            
            if '\n' in contents[-1]:
                contents[-1] = contents[-1][:-1]
            
            
            Y_data.append(float(contents[0]))
            
            data_map = {}
            
            for i in range(len(contents)-1):
                id = contents[i+1].split(':')[0]
                
                curr_content = float(contents[i+1].split(':')[1])
                
                data_map[id] = curr_content
            
            curr_X_data = []
                
                
            for i in range(num_features):
                if str(i+1) in data_map:
                    curr_X_data.append(data_map[str(i+1)])
                else:
                    curr_X_data.append(0.0)
                
            
#             print(cnt, curr_X_data)
            
            cnt = cnt+1
            
            X_data.append(curr_X_data)
            
            line = fp.readline()
#             
#             curr_X_data = []
#             
#             if len(contents) < num_features + 1:
#                 continue
#             
#             for i in range(num_features):
#                 
#                 id = contents[i+1].split(':')[0]
#                 
#                 if int(id) != i+1:
#                     break
#                 
#                 curr_X_data.append(float(contents[i+1].split(':')[1]))
#             
#             
#             if len(curr_X_data) < num_features:
#                 continue
#             
#             X_data.append(curr_X_data)
#             
#             
#             cnt += 1
    
    X_data = normalize(np.array(X_data))
    
    
    train_X_data = torch.tensor(X_data, dtype = torch.double)
    
    train_Y_data = torch.tensor(Y_data, dtype = torch.double)
    

    if torch.min(train_Y_data) != 0:
        train_Y_data = train_Y_data - 1
    
#     print('unique_Y::', torch.unique(train_Y_data))
    
    train_Y_data = train_Y_data.view(-1,1)
    
#     print('Y_dim::', train_Y_data.shape)
    
    
    if split_id is None:
        return split_train_test_data(train_X_data, train_Y_data, 0.1, is_classification)
    else:
        return train_X_data[0:split_id], train_Y_data[0:split_id], train_X_data[split_id:], train_Y_data[split_id:]    
    
def clean_sensor_data(file_name, is_classification, split_id = None):

    Y_data = []
        
    X_data = []

    configs = load_config_data(config_file)

    num_features = int(configs[file_name]['feature_num'])

    with open(file_name) as fp:  
        line = fp.readline()
        cnt = 1
        
        while line:
#             print("Line {}".format(cnt))
            
            contents = line.split(' ')
            
            if ':' not in contents[-1]:
                contents.pop()
            
            if '\n' in contents[-1]:
                contents[-1] = contents[-1][:-1]
            
            
            Y_data.append(float(contents[0]))
            
            data_map = {}
            
            for i in range(len(contents)-1):
                id = contents[i+1].split(':')[0]
                
                curr_content = float(contents[i+1].split(':')[1])
                
                data_map[id] = curr_content
            
            curr_X_data = []
                
                
            for i in range(num_features):
                if str(i+1) in data_map:
                    curr_X_data.append(data_map[str(i+1)])
                else:
                    curr_X_data.append(0.0)
                
            
#             print(cnt, curr_X_data)
            
            cnt = cnt+1
            
            X_data.append(curr_X_data)
            
            line = fp.readline()
#             
#             curr_X_data = []
#             
#             if len(contents) < num_features + 1:
#                 continue
#             
#             for i in range(num_features):
#                 
#                 id = contents[i+1].split(':')[0]
#                 
#                 if int(id) != i+1:
#                     break
#                 
#                 curr_X_data.append(float(contents[i+1].split(':')[1]))
#             
#             
#             if len(curr_X_data) < num_features:
#                 continue
#             
#             X_data.append(curr_X_data)
#             
#             
#             cnt += 1
    
    X_data = normalize(np.array(X_data))
    
    
    train_X_data = torch.tensor(X_data, dtype = torch.double)
    
    train_Y_data = torch.tensor(Y_data, dtype = torch.double)
    

    if torch.min(train_Y_data) != 0:
        train_Y_data = train_Y_data - 1
    
#     print('unique_Y::', torch.unique(train_Y_data))
    
    train_Y_data = train_Y_data.view(-1,1)
    
#     print('Y_dim::', train_Y_data.shape)
    
    
    if split_id is None:
        return split_train_test_data(train_X_data, train_Y_data, 0.1, is_classification)
    else:
        return train_X_data[0:split_id], train_Y_data[0:split_id], train_X_data[split_id:], train_Y_data[split_id:]
    
#     return split_train_test_data(train_X_data, train_Y_data, 0.1, is_classification)    



def clean_sensor_data_single(file_name, is_classification):

    Y_data = []
        
    X_data = []

    configs = load_config_data(config_file)

    num_features = int(configs[file_name]['feature_num'])

    with open(file_name) as fp:  
        line = fp.readline()
        cnt = 1
        
        while line:
#             print("Line {}".format(cnt))
            
            contents = line.split(' ')
            
            if ':' not in contents[-1]:
                contents.pop()
            
            if '\n' in contents[-1]:
                contents[-1] = contents[-1][:-1]
            
            
            Y_data.append(float(contents[0]))
            
            data_map = {}
            
            for i in range(len(contents)-1):
                id = contents[i+1].split(':')[0]
                
                curr_content = float(contents[i+1].split(':')[1])
                
                data_map[id] = curr_content
            
            curr_X_data = []
                
                
            for i in range(num_features):
                if str(i+1) in data_map:
                    curr_X_data.append(data_map[str(i+1)])
                else:
                    curr_X_data.append(0.0)
                
            
#             print(cnt, curr_X_data)
            
            cnt = cnt+1
            
            X_data.append(curr_X_data)
            
            line = fp.readline()
#             
#             curr_X_data = []
#             
#             if len(contents) < num_features + 1:
#                 continue
#             
#             for i in range(num_features):
#                 
#                 id = contents[i+1].split(':')[0]
#                 
#                 if int(id) != i+1:
#                     break
#                 
#                 curr_X_data.append(float(contents[i+1].split(':')[1]))
#             
#             
#             if len(curr_X_data) < num_features:
#                 continue
#             
#             X_data.append(curr_X_data)
#             
#             
#             cnt += 1
    
    X_data = normalize(np.array(X_data))
    
    
    train_X_data = torch.tensor(X_data, dtype = torch.double)
    
    train_Y_data = torch.tensor(Y_data, dtype = torch.double)
    

    if torch.min(train_Y_data) != 0:
        train_Y_data = train_Y_data - 1
    
#     print('unique_Y::', torch.unique(train_Y_data))
    
    train_Y_data = train_Y_data.view(-1,1)
    
#     print('Y_dim::', train_Y_data.shape)
    
    return train_X_data, train_Y_data    






    
    
    
    
    

if __name__ == '__main__':
    
    
    [x_data, y_data] = load_data()
    check_correctness(x_data)
