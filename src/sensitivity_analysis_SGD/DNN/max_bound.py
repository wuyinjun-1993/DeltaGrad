'''
Created on Nov 18, 2019

'''
import sys, os


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')

# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch

try:
    from DNN import *
    from data_IO.Load_data import *
except ImportError:
    from DNN import *
    from Load_data import *
    

default_epoch_num = 1


default_batch_size = 10

    
    
if __name__ == '__main__':
    sys_argv = sys.argv
    
    batch_size = int(sys_argv[1])
    
#     total_num_iterations= 1000
    # Create ANN
    learning_rate = 0.1
    
    regularization_coeff = 0.05
    
    X = torch.load(git_ignore_folder + 'X')
        
    Y = torch.load(git_ignore_folder + 'Y')
    
    print(X.shape)
    
    print(Y.shape)
    
#         batch_size = X.shape[0]
    
    test_X = torch.load(git_ignore_folder + 'test_X')
    
    test_Y = torch.load(git_ignore_folder + 'test_Y')

    input_dim = X.shape[1]
#         hidden_dim = [10] #hidden layer dim is one of the hyper parameter and it should be chosen and tuned. For now I only say 150 there is no reason.
    
    num_class = torch.unique(Y).shape[0]
    
    output_dim = num_class
    
    
    
    if batch_size < default_batch_size:
        num_epochs = 1
    else:
        num_epochs = int((batch_size/default_batch_size)*default_epoch_num)
    
    
    hidden_dim = [10]
    
    model = DNNModel(input_dim, hidden_dim, output_dim)
        
#     init_model(model, init_para_list)

    error = nn.CrossEntropyLoss()


#     compute_linearized_coefficient([], [], model)


#         optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    cut_off_epoch = num_epochs
    
    print('epoch_num::', num_epochs)

    t1 = time.time()
    
    model, gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, epoch = model_training(num_epochs, X, Y, test_X, test_Y, learning_rate, regularization_coeff, error, model, False, batch_size, X.shape)

    t2 = time.time()
    
    print('time::', (t2 - t1))
    
    
    
    