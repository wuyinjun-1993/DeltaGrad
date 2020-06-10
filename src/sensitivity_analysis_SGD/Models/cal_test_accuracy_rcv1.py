
import sys



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
import ast


from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader



sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Benchmark_experiments')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Models')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/multi_nomial_logistic_regression')


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.datasets import load_svmlight_file
from sklearn.datasets import fetch_rcv1


try:
    from data_IO.Load_data import *
    from utils import *
    from Interpolation.piecewise_linear_interpolation_2D import *
    from Models.DNN import DNNModel
    from Models.Lenet5 import LeNet5
    from Models.Lenet5_cifar import LeNet5_cifar
    from Models.Data_preparer import *
    from DNN_single import *
    from Models.DNN2 import DNNModel2
    from Models.DNN3 import DNNModel3
    from Models.ResNet import *
    from Models.Skipnet import *
    from Models.CNN import *
    from Models.Pretrained_models import *
    from multi_nomial_logistic_regression.Multi_logistic_regression import *
    from multi_nomial_logistic_regression.incremental_updates_logistic_regression_multi_dim import *
    from Benchmark_experiments.benchmark_exp import *
    
except ImportError:
    from Load_data import *
    from utils import *
    from piecewise_linear_interpolation_2D import *
    from Models.DNN import DNNModel
    from Models.DNN2 import DNNModel2
    from Models.DNN3 import DNNModel3
    from Models.Lenet5 import LeNet5
    from Models.Lenet5_cifar import LeNet5_cifar
    from Models.Data_preparer import *
    from DNN_single import *
    from Models.ResNet import *
    from Models.Skipnet import *
    from Models.CNN import *
    from Models.Pretrained_models import *
    from multi_nomial_logistic_regression.Multi_logistic_regression import *
    from multi_nomial_logistic_regression.incremental_updates_logistic_regression_multi_dim import *
    from Benchmark_experiments.benchmark_exp import *



def get_dense_tensor(X_coo):
    
    values = X_coo.data
#     print(X_coo)
    indices = np.vstack((X_coo.row, X_coo.col))
    
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    shape = X_coo.shape
    
    X = torch.sparse.DoubleTensor(i, v, torch.Size(shape)).to_dense()
    
    return X

def transform_Y(Y):
    
    Y_uniques = torch.unique(Y)
    
    if not (set(Y_uniques.numpy()) == set(range(Y_uniques.shape[0]))):
#         print(Y_uniques)
        Y_copy = torch.zeros(Y.shape)

        
        for k in range(Y_uniques.shape[0]):
#             print((Y==Y_uniques[k]).nonzero()[:, 0])
            
            Y_copy[(Y==Y_uniques[k]).nonzero()[:, 0]] = k
            
        Y = Y_copy 
        
    return Y[:,0]

    
if __name__ == '__main__':


    configs = load_config_data(config_file)
    
#     print(configs)
    git_ignore_folder = configs['git_ignore_folder']
    
    directory_name = configs['directory']

#     X_train, y_train = load_svmlight_file(directory_name + "rcv1_train.multiclass")
    X_test, Y_test = load_svmlight_file(directory_name + "rcv1_test.binary")

    
    sys_args = sys.argv
    
    model_name = sys_args[1]

    print("Model_name::", model_name)

    model = torch.load(git_ignore_folder + model_name, map_location=torch.device('cpu'))

    out = X_test.dot(list(model.parameters())[0].T.detach().numpy())


    softmax_layer = nn.LogSoftmax()
    
    out = softmax_layer(torch.from_numpy(out))

    pred = out.detach().max(1)[1]

    test_Y_batch = torch.from_numpy(Y_test).type(torch.LongTensor)
    
    test_Y_batch += 1
    
    test_Y_batch /= 2
    
#     predict_Y = torch.from_numpy()

#     avg_loss = criterion(pred, test_Y_batch).sum()
    
    
    if len(test_Y_batch.shape) > 1 and test_Y_batch.shape[1] > 1:
        test_Y_batch = torch.nonzero(test_Y_batch)[:,1]
    
    total_correct = pred.eq(test_Y_batch.view_as(pred)).sum()
    
    print(total_correct)
    
    print("RCV1 Test Avg. Accuracy::", total_correct.item()*1.0/X_test.shape[0])


#     dataset_name = "rcv1"
# 
# 
# #     test_X_sparse, test_Y_sparse = load_data_rcv1_test()
#     
#     
# #     num_feature = test_X_sparse.shape[1]
#     
#     model_name = "Logistic_regression"
#     
#     model_class = getattr(sys.modules[__name__], model_name)
# 
#     data_preparer = Data_preparer()
#     
#     num_class = get_data_class_num_by_name(data_preparer, dataset_name)
#     
#     model = model_class(X_test.shape[1], num_class)# DNNModel(input_dim, hidden_dim, output_dim)
#     
#     
#     
#     hyper_para_function=getattr(Data_preparer, "get_hyperparameters_" + dataset_name)
# 
#     
#     criterion, optimizer, lr_scheduler = hyper_para_function(data_preparer, model.parameters(), 0.1, 0.1)
# 
# 
# 
# #     rcv1 = fetch_rcv1()
# 
# #     start_id = 23149
# 
#     
#     
#     
#     
#     batch_size = 10000
#     
#     model.eval()
#     
#     total_correct = 0
#     avg_loss = 0.0
#     
#     for j in range(0, X_test.shape[0], batch_size):
#         
#         print(j)
#         
#         end_id = j + batch_size
#         
#         if end_id >= X_test.shape[0]:
#             end_id = X_test.shape[0]
#         
#         test_X_batch = torch.from_numpy(X_test[j:end_id].todense()).type(torch.DoubleTensor)
#         test_Y_batch = torch.from_numpy(Y_test[j:end_id]).type(torch.LongTensor)
#         
#         test_Y_batch += 1
#         
#         test_Y_batch /= 2
#         
# #         test_X_batch = get_dense_tensor(X_test[j:end_id].tocoo())
# #         
# #         test_Y_batch = get_dense_tensor(X_test[j:end_id].tocoo()).type(torch.LongTensor)
# #         
# #         
# #         test_Y_batch = transform_Y(test_Y_batch)
# #         test_X_batch = test_X_sparse[j:end_id].to_dense()
#         
# #         test_Y_batch = test_Y_sparse[j:end_id].to_dense()
#         
#     
#         output = model(test_X_batch)
# #         print(test_X_batch.shape, test_Y_batch.shape)
#         avg_loss += criterion(output, test_Y_batch).sum()
#         pred = output.detach().max(1)[1]
#         
#         if len(test_Y_batch.shape) > 1 and test_Y_batch.shape[1] > 1:
#             test_Y_batch = torch.nonzero(test_Y_batch)[:,1]
#         
#         total_correct += pred.eq(test_Y_batch.view_as(pred)).sum()
#         
#         print(total_correct)
#         
#         del test_X_batch, test_Y_batch
# 
#     avg_loss /= Y_test.shape[0]
#         
#     avg_accuracy = float(total_correct) / Y_test.shape[0]
#     
#     print("loss::", avg_loss)
#     
#     print("accuracy::", avg_accuracy)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
        