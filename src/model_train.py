'''
Created on Jun 24, 2020

'''

import torch
import time

import sys,os

sys.path.append(os.path.abspath(__file__))


from utils import *



def model_training_lr_test(random_ids_multi_epochs, epoch, net, dataset_train, data_train_size, optimizer, criterion, batch_size, is_GPU, device, lrs):
#     global cur_batch_win
    net.train()
    
    gradient_list_all_epochs = []
    
    para_list_all_epochs = []
    
#     output_list_all_epochs = []
    
    learning_rate_all_epochs = []
    
    
#     X_theta_prod_seq, X_theta_prod_softmax_seq = [], []
#     loss_list, batch_list = [], []
    
#     random_ids_all_iterations = []
    
    t1 = time.time()
    
    
#     test(net, data_test_loader, criterion, data_test_size, is_GPU, device)

    for j in range(epoch):
        
#         random_ids = torch.zeros([data_train_size], dtype = torch.long)
#         random_ids = torch.randperm(data_train_size)
        random_ids = random_ids_multi_epochs[j]
    
#         k = 0
        
        learning_rate = lrs[j]
        update_learning_rate(optimizer, learning_rate)
        
#         item0 = data_train_loader.dataset.data[100]
    
#         for i, items in enumerate(data_train_loader):

        i = 0
        for k in range(0, data_train_size, batch_size):

            end_id = k + batch_size
            
            if end_id > data_train_size:
                end_id = data_train_size
            
            
            curr_rand_ids = random_ids[k: end_id]
            
#             if not is_GPU:
#                 images, labels, ids =  items[0], items[1], items[2]
#             else:
#                 images, labels, ids =  items[0].to(device), items[1].to(device), items[2]
            if not is_GPU:
                images, labels = dataset_train.data[curr_rand_ids], dataset_train.labels[curr_rand_ids]
            else:
                images, labels = dataset_train.data[curr_rand_ids].to(device), dataset_train.labels[curr_rand_ids].to(device)
            
#             print(k, end_id)
#             random_ids[k:end_id] = ids
            
            
#             k = k + batch_size
            
            optimizer.zero_grad()
    
#             output = net.forward_with_provenance(images, X_theta_prod_seq, X_theta_prod_softmax_seq)
            
            output = net.forward(images)
    
#             print(torch.unique(labels))
    
            loss = criterion(output, labels)
    
    
#             loss_list.append(loss.detach().cpu().item())
#             batch_list.append(i+1)
    
            if i % 10 == 0:
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (j, i, loss.detach().cpu().item()))
            
#             if i % 10 == 0:
#                 lr_scheduler.step()
                 
            i += 1
            loss.backward()
    
            append_gradient_list(gradient_list_all_epochs, None, para_list_all_epochs, net, None, is_GPU, device)
    
            # Update Visualization
    #         if viz.check_connection():
    #             cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
    #                                      win=cur_batch_win, name='current_batch_loss',
    #                                      update=(None if cur_batch_win is None else 'replace'),
    #                                      opts=cur_batch_win_opts)
#             learning_rate = list(optimizer.param_groups)[0]['lr']
            
#             regularization_rate = list(optimizer.param_groups)[0]['weight_decay']
            
#             exp_model_param = update_model(net, learning_rate, regularization_rate)
            
            
            optimizer.step()
            
#             print('parameter comparison::')
#             
#             compute_model_para_diff(list(net.parameters()), exp_model_param)
            
            
            learning_rate_all_epochs.append(learning_rate)
        
            del images, labels
        
#         item1 = data_train_loader.dataset.data[100]
#         print(torch.norm(item0[0] - item1[0]))
        
#         random_ids_multi_epochs.append(random_ids)
    
#     test(net, dataset_test, criterion, data_test_size, is_GPU, device)
        
    
    t2 = time.time()
    
    print("training_time::", (t2 - t1))
    
    return net, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs
