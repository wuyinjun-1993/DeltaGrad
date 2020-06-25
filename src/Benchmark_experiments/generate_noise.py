'''
Created on Apr 14, 2020

'''
import torch

def select_deletion1(X, Y, model, delta_num, num_class):
    
#     multi_res = softmax_layer(torch.mm(X, res))
    
    model.to('cpu')
    
    multi_res = torch.exp(model(X))
    
    prob, predict_labels = torch.max(multi_res, 1)
    
    _, changed_labels = torch.min(multi_res, 1)
    
    print(prob)
    
#     predict_labels = torch.argmax(multi_res, 1)
    
    
    sorted_prob, indices = torch.sort(prob.view(-1), descending = True)
    
    delta_id_array = []
    
#     delta_data_ids = torch.zeros(delta_num, dtype = torch.long)
    
#     for i in range(delta_num):

    expected_selected_label =0
    
     
#     if torch.sum(Y == 1) > torch.sum(Y == -1):
#         expected_selected_label = 1
#         
#     else:
#         expected_selected_label = -1


    i = 0

    while len(delta_id_array) < delta_num and i < X.shape[0]:
        if Y[indices[i]] == predict_labels[indices[i]]:
#         if True:
#             print(indices[i], changed_labels[indices[i]], Y[indices[i]])
            
            Y[indices[i]] = changed_labels[indices[i]]#(Y[indices[i]] + 1)%num_class
            delta_id_array.append(indices[i])
#             X[indices[i]] *= (-2)
    
        i = i + 1
    
    delta_data_ids = torch.tensor(delta_id_array, dtype = torch.long)
    
#     print(delta_data_ids[:100])
#     delta_data_ids = random_generate_subset_ids(X.shape, delta_num)     
#     torch.save(delta_data_ids, git_ignore_folder+'delta_data_ids')
    return X, Y, delta_data_ids    

    