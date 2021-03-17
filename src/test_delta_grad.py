'''
Created on Jan 12, 2021

'''
import sys, os
import torch
import time


import psutil

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/data_IO')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Interpolation')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Models')


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.abspath(__file__))




from Models.Data_preparer import *

from main_delete import *
from main_add import *
from utils import *


from model_train import *




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
    
    
import argparse

    
    
    
    
if __name__ == '__main__':


    parser = argparse.ArgumentParser('DeltaGrad')
    parser.add_argument('--add',  action='store_true', help="The flag for incrementally adding training samples, otherwise for incrementally deleting training samples")
    
    
#     parser.add_argument('--ratio',  type=float, help="delete rate or add rate")
    
    parser.add_argument('--bz',  type=int, help="batch size in SGD")
        
    parser.add_argument('--epochs',  type=int, help="number of epochs in SGD")
    
    parser.add_argument('--model',  help="name of models to be used")
    
    parser.add_argument('--dataset',  help="dataset to be used")
    
    parser.add_argument('--wd', type = float, help="l2 regularization")
    
    parser.add_argument('--lr', nargs='+', type = float, help="learning rates")
    
    parser.add_argument('--lrlen', nargs='+', type = int, help="The epochs to use some learning rate, used for the case with decayed learning rates")
    
    parser.add_argument('--GPU', action='store_true', help="whether the experiments run on GPU")
    
    parser.add_argument('--GID', type = int, help="Device ID of the GPU")
    
    parser.add_argument('--train', action='store_true', help = 'Train phase over the full training datasets')
    
    parser.add_argument('--repo', default = gitignore_repo, help = 'repository to store the data and the intermediate results')
        
    parser.add_argument('--method', default = baseline_method, help = 'methods to update the models')
    
    parser.add_argument('--period', type = int, help = 'period used in deltagrad')
    
    parser.add_argument('--init', type = int, help = 'initial epochs used in deltagrad')
    
    parser.add_argument('-m', type = int, help = 'history size used in deltagrad')
    
    parser.add_argument('--cached_size', type = int, default = 1000, help = 'size of gradients and parameters cached in GPU in deltagrad')    

    args = parser.parse_args()



    model_name = args.model
    
    git_ignore_folder = args.repo
    
    dataset_name = args.dataset
    
    num_epochs = args.epochs
    
    batch_size = args.bz
    
    is_GPU = args.GPU
    
#     args.ratio


    regularization_coeff = args.wd
    
    if not is_GPU:
        device = torch.device("cpu")
    else:    
        GPU_ID = int(args.GID)
        device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    
    model_class = getattr(sys.modules[__name__], model_name)
    
    
    data_preparer = Data_preparer()


    num_epochs = args.epochs

    num_class = get_data_class_num_by_name(data_preparer, dataset_name)

    dataset_train = torch.load(git_ignore_folder + "dataset_train")
    
    generate_random_ids_list(dataset_train, args.epochs, git_ignore_folder)

    
    random_ids_all_epochs = torch.load(git_ignore_folder + 'random_ids_multi_epochs')
    
    sorted_ids_multi_epochs = torch.load(git_ignore_folder + 'sorted_ids_multi_epochs')
    
#     learning_rate_all_epochs = torch.load(git_ignore_folder + 'learning_rate_all_epochs')
    
    delta_data_ids = torch.load(git_ignore_folder + "delta_data_ids")

    dim = [len(dataset_train), len(dataset_train[0][0])]


    model = model_class(dim[1], num_class)
    
    print(num_class, dim[1])
    
    lrs = args.lr
    
    lrlens = args.lrlen
    
    
    lr_lists = get_lr_list(lrs, lrlens)
    
    hyper_para_function=getattr(Data_preparer, "get_hyperparameters_" + dataset_name)
    
    criterion, optimizer = hyper_para_function(data_preparer, model.parameters(), lr_lists[0], regularization_coeff)
    
    criterion = model.get_loss_function()
    
    hyper_params = [criterion, optimizer]
    
    model, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs = model_training_lr_test(random_ids_all_epochs, num_epochs, model, dataset_train, len(dataset_train), optimizer, criterion, batch_size, is_GPU, device, lr_lists)
    
    torch.save(dataset_train, git_ignore_folder + "dataset_train")
    
    torch.save(gradient_list_all_epochs, git_ignore_folder + 'gradient_list_all_epochs')
    
    torch.save(para_list_all_epochs, git_ignore_folder + 'para_list_all_epochs')
    
    torch.save(learning_rate_all_epochs, git_ignore_folder + 'learning_rate_all_epochs')
    
    t1 = time.time()
    
    init_model(model,para_list_all_epochs[0])
                
    updated_model, _, exp_para_list_all_epochs, exp_gradient_list_all_epochs, _ = model_update_standard_lib(num_epochs, dataset_train, model, random_ids_all_epochs, sorted_ids_multi_epochs, delta_data_ids, batch_size, learning_rate_all_epochs, criterion, optimizer, is_GPU, device, record_params = True)

    t2 = time.time()
    
    
    period = 5
    
    init_epochs = 20
    
    m = 10
    
    cached_size = 10000
    
    init_model(model,para_list_all_epochs[0])
    
    grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor = cache_grad_para_history(git_ignore_folder, cached_size, is_GPU, device)
    
    updated_model = model_update_deltagrad(num_epochs, period, 1, init_epochs, dataset_train, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, delta_data_ids, m, learning_rate_all_epochs, random_ids_all_epochs, sorted_ids_multi_epochs, batch_size, dim, criterion, optimizer, regularization_coeff, is_GPU, device, exp_para_list_all_epochs, exp_gradient_list_all_epochs)
    
    
    
    
    
    
    