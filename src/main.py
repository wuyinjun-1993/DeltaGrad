'''
Created on Jun 16, 2020

'''
import argparse
import torch
import sys, os


sys.path.append(os.path.abspath(__file__))


from main_delete import *
from main_add import *
from utils import *
# try:
#     from main_delete import *
#     from utils import *
# 
# except ImportError:
#     from Load_data import *
#     from utils import *



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
    
    
    lrs = args.lr
    
    lrlens = args.lrlen
    
    
    lr_lists = get_lr_list(lrs, lrlens)
    
    is_GPU =args.GPU
    
    device_id = args.GID
    
    
    is_training = args.train
    
    
    if not os.path.exists(args.repo):
        os.makedirs(args.repo)
    
    
    if not is_GPU:
        device = torch.device("cpu")
    else:    
        GPU_ID = int(device_id)
        device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    
    
    if is_training:
        if args.add:
            
            main_add(args, lr_lists)
        else:
            main_del(args, lr_lists)
    
    
    else:
        if args.add:
            model_update_add(args, args.method, lr_lists)
            
        else:
            model_update_del(args, args.method, lr_lists)
    



