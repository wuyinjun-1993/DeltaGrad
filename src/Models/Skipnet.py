'''
Created on Jan 8, 2020


'''
import torch
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random


import os,sys



sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Benchmark_experiments')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_IO.Load_data import *
#     from Benchmark_experiments.benchmark_exp import *

except ImportError:
    from Load_data import *
#     from benchmark_exp import *

# os.environ['KMP_DUPLICATE_LIB_OK']= 'True'


# train_loader = torch.utils.data.DataLoader(
#   torchvision.datasets.MNIST('./data/', train=True, download=True,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  (0.1307,), (0.3081,))
#                              ])),
#   batch_size= 64, shuffle=True)


def get_dropouted_ids(input):
    
#     ids_list = []
    
#     other_ids_list = []
    
    res = torch.sum(torch.sum(input, 3), 2)
    
    ids_list = get_dropouted_ids_2d(res)
    
#     for i in range(input.shape[0]):
#         
#         curr_id_list = []
#         
#         for j in range(input[i].shape[0]):
#             
#             if torch.norm(input[i][j]) == 0:
#                 curr_id_list.append(j)
#                 
#         ids_list.append(curr_id_list)
#             else:
#                 other_ids_list.append([i,j])
    
    return ids_list#, torch.cat(other_ids_list, 0)


def get_dropouted_ids_2d(input):
    
#     ids_list = []
    
#     other_ids_list = []
    
    
    all_ids = torch.nonzero(input == 0)
    
#     for i in range(input.shape[0]):
#         
#         ids_list.append(torch.nonzero(input[i] == 0).view(-1).tolist())
        
#         for j in range(input[i].shape[0]):
#             if torch.norm(input[i][j]) == 0:
#                 curr_id_list.append(j)
#                 
#         ids_list.append(curr_id_list)
#             else:
#                 other_ids_list.append([i,j])
    
    return all_ids.cpu()#, torch.cat(other_ids_list, 0)

def set_dropouted_ids(input, ids_list, p):
     
#     ids_list = []
    
#     other_ids_list = []
#     print(input.shape)
#     
#     print(ids_list[-1])
    
    for i in range(len(ids_list)):
        
        curr_id_list = ids_list[i]
        
        for j in range(len(curr_id_list)):
            
            
            
            input[i][curr_id_list[j]] = 0
    
    
    input /= (1-p)
    
#     for i in range(input.shape[0]):
#         for j in range(input[i].shape[0]):
#             if torch.norm(input[i][j]) == 0:
#                 ids_list.append([i,j])
#             else:
#                 other_ids_list.append([i,j])
    
    return input#, torch.cat(other_ids_list, 0)




def set_dropouted_ids_2d(input, ids_list, p):
     
#     ids_list = []
    
#     other_ids_list = []
#     print(input.shape)
#     
#     print(ids_list[-1])

    
    input[ids_list[:,0], ids_list[:,1]] = 0
    
#     for i in range(len(ids_list)):
#         
#         curr_id_list = ids_list[i]
#         
#         input[i][curr_id_list] = 0
        
#         for j in range(len(curr_id_list)):
#             
#             
#             
#             input[i][curr_id_list[j]] = 0
    
    
    input /= (1-p)
    
    ids_list.cpu()
    
#     for i in range(input.shape[0]):
#         for j in range(input[i].shape[0]):
#             if torch.norm(input[i][j]) == 0:
#                 ids_list.append([i,j])
#             else:
#                 other_ids_list.append([i,j])
    
    return input#, torch.cat(other_ids_list, 0)

    
def get_dropouted_ids2(input):
    
    ids_list = []
    
#     other_ids_list = []
    
    for i in range(input.shape[0]):
        
        curr_ids_list = []
        
        for j in range(input[i].shape[0]):
            if torch.norm(input[i][j]) == 0:
                curr_ids_list.append(j)
                
        ids_list.append(curr_ids_list)
                
#             else:
#                 other_ids_list.append([i,j])
    
    
    return ids_list#, torch.cat(other_ids_list, 0)
    
    
class DNNModel_skipnet(nn.Module):
    
    def __init__(self):
        super(DNNModel_skipnet, self).__init__()
        
        self.dropout0 = nn.Dropout(p=0.8).double()
        
        self.fc1 = nn.Linear(28*28, 300).double()
        
        self.dropout1 = nn.Dropout().double()
        
        self.relu1 = nn.ReLU()

#         self.fc2 = nn.Linear(1024, 1024).double()
# 
#         self.dropout2 = nn.Dropout().double()
# 
#         self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(300, 10).double()
        
        self.dropout3 = nn.Dropout().double()
        
        self.fc4 = nn.Softmax()


    def forward_with_known_dropout(self, x, matched_ids, ids_list_all_list, need_load):
#         results = torch.load(git_ignore_folder + 'tmp_res')
#         print(torch.norm(results[0][matched_ids] - x))
#         if need_load:
#             exp_ids_list_all_list = torch.load(git_ignore_folder + 'ids_list0') 
#         else:
#             torch.save(ids_list_all_list, git_ignore_folder + 'ids_list0') 
        out = set_dropouted_ids_2d(x, ids_list_all_list[0], self.dropout0.p)
        
        
        
#         print(torch.norm(results[1][matched_ids] - out))
        
        out = self.fc1(out)

#         print(torch.norm(results[2][matched_ids] - out))

#         out = self.dropout1(out)
        out = set_dropouted_ids_2d(out, ids_list_all_list[1], self.dropout1.p)

#         print(torch.norm(results[3][matched_ids] - out))


        out = self.relu1(out)
         
#         print(torch.norm(results[4][matched_ids] - out)) 
        
#         out = self.fc2(out)
#         
# #         out = self.dropout2(out)
#         out = set_dropouted_ids_2d(out, ids_list_all_list[2], self.dropout2.p)
#         
#         out = self.relu2(out)
        
        out = self.fc3(out)
        
        
#         print(torch.norm(results[5][matched_ids] - out))
#         out = self.dropout3(out)
        
        out = set_dropouted_ids_2d(out, ids_list_all_list[2], self.dropout3.p)
        
#         print(torch.norm(results[6][matched_ids] - out))
        
        out = self.fc4(out)
        
#         print(torch.norm(results[7][matched_ids] - out))
        
        return out
#

    def forward(self, x, need_record):
        
        ids_list_all_list = []
        
#         results_list= []
        
#         results_list.append(x.clone())
        
        out = self.dropout0(x)
#       
#         results_list.append(out.clone())

  
        ids_list0 = get_dropouted_ids_2d(out)
        
        out = self.fc1(out)
        
#         results_list.append(out.clone())
        
        out = self.dropout1(out)
        
#         results_list.append(out.clone())

        ids_list = get_dropouted_ids_2d(out)

        out = self.relu1(out)
         
#         results_list.append(out.clone())
#         out = self.fc2(out)
#         
#         out = self.dropout2(out)
#         
#         ids_list2 = get_dropouted_ids_2d(out)
#         
#         out = self.relu2(out)
        
        out = self.fc3(out)
        
#         results_list.append(out.clone())
        
        out = self.dropout3(out)
#         results_list.append(out.clone())
        ids_list3 = get_dropouted_ids_2d(out)
        
        ids_list_all_list.append(ids_list0)
        
        ids_list_all_list.append(ids_list)
        
#         ids_list_all_list.append(ids_list2)
        
        ids_list_all_list.append(ids_list3)
        
        out = self.fc4(out)
#         results_list.append(out.clone())
        
#         if need_record:
#          
#             torch.save(results_list, git_ignore_folder + 'tmp_res')
#              
#             torch.save(ids_list0, git_ignore_folder + 'ids_list0') 
        
        return out, ids_list_all_list
    
    
    def get_all_parameters(self):
    
        para_list = []
        
        for param in self.parameters():
            para_list.append(param.data.clone())
            
            
        return para_list



    def get_all_gradient(self):
        
        para_list = []
        
        for param in self.parameters():
            para_list.append(param.grad.clone())
            
            
        return para_list    
    
    
    
    def get_output_each_layer(self, x):
        
        
        output_list = [None]*(len(self.linear_layers) + 3)
        
        non_linear_input_list = [None]*(len(self.linear_layers) + 3)
        
        k = 0
        
        output_list[k] = torch.cat((x, torch.ones([x.shape[0], 1], dtype = torch.double)), 1)
        
        non_linear_input_list[k]= x.clone()
        
        k = k + 1
        
        
        out = self.fc1(x)
        # Non-linearity 1
        
        non_linear_input_list[k]= out.clone()
        
        out = self.relu1(out)
        
        
        output_list[k] = torch.cat((out, torch.ones([out.shape[0], 1], dtype = torch.double)), 1)
        
        k = k + 1
        
        for i in range(len(self.linear_layers)):
            out = self.linear_layers[i](out)
            non_linear_input_list[k] = out.clone()
            out = self.activation_layers[i](out)
            output_list[k] = torch.cat((out, torch.ones([out.shape[0], 1], dtype = torch.double)), 1)
            
            k = k + 1
        
        
        # Linear function 2
        out = self.fc2(out)
        
        non_linear_input_list[k] = out.clone()
        
        out2 = self.fc3(out)
        
        output_list[k] = out2
        
        
        return output_list, non_linear_input_list
    

    class MyDataset(Dataset):
        def __init__(self, samples):
            
            self.data = samples
            
        def __getitem__(self, index):
            data, target = self.data[index]
            
            data = data.view(-1)
            
            
            y_onehot = torch.DoubleTensor(1, 10)
    
            target = torch.tensor([target])
            target = target.type(torch.LongTensor)
            
        # In your for loop
            y_onehot.zero_()
            y_onehot.scatter_(1, target.view(-1, 1), 1)
            
            
            # Your transformations here (or set it in CIFAR10)
            
            return data.type(torch.DoubleTensor), y_onehot.view(-1), index
    
        def __len__(self):
            return len(self.data)



class DNNModel_skipnet_full(nn.Module):
    
    def __init__(self):
        super(DNNModel_skipnet_full, self).__init__()
        
#         self.dropout0 = nn.Dropout(p=0.8).double()
        
        self.fc1 = nn.Linear(28*28, 1024).double()
        
        self.dropout1 = nn.Dropout().double()
        
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(1024, 1024).double()
 
        self.dropout2 = nn.Dropout().double()
 
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(1024, 10).double()
        
        self.dropout3 = nn.Dropout().double()
        
        self.fc4 = nn.Softmax()


    def forward_with_known_dropout(self, x, matched_ids, ids_list_all_list):
        
        
#         out = set_dropouted_ids_2d(x, ids_list_all_list[0], self.dropout0.p)
        
        out = self.fc1(x)

#         out = self.dropout1(out)
        out = set_dropouted_ids_2d(out, ids_list_all_list[1], self.dropout1.p)

        out = self.relu1(out)
         
        out = self.fc2(out)
         
#         out = self.dropout2(out)
        out = set_dropouted_ids_2d(out, ids_list_all_list[2], self.dropout2.p)
         
        out = self.relu2(out)
        
        out = self.fc3(out)
        
#         out = self.dropout3(out)
        
        out = set_dropouted_ids_2d(out, ids_list_all_list[2], self.dropout3.p)
        
        out = self.fc4(out)
         
        return out
#

    def forward(self, x):
        
        ids_list_all_list = []
        
#         out = self.dropout0(x)
#         
#         ids_list0 = get_dropouted_ids_2d(out)
        
        out = self.fc1(x)
        
        out = self.dropout1(out)

        ids_list = get_dropouted_ids_2d(out)

        out = self.relu1(out)
         
        out = self.fc2(out)
         
        out = self.dropout2(out)
         
        ids_list2 = get_dropouted_ids_2d(out)
         
        out = self.relu2(out)
        
        out = self.fc3(out)
        
        out = self.dropout3(out)
        
        ids_list3 = get_dropouted_ids_2d(out)
        
#         ids_list_all_list.append(ids_list0)
        
        ids_list_all_list.append(ids_list)
        
        ids_list_all_list.append(ids_list2)
        
        ids_list_all_list.append(ids_list3)
        
        out = self.fc4(out)
         
        return out, ids_list_all_list
    
    
    def get_all_parameters(self):
    
        para_list = []
        
        for param in self.parameters():
            para_list.append(param.data.clone())
            
            
        return para_list



    def get_all_gradient(self):
        
        para_list = []
        
        for param in self.parameters():
            para_list.append(param.grad.clone())
            
            
        return para_list    
    
    
    
    def get_output_each_layer(self, x):
        
        
        output_list = [None]*(len(self.linear_layers) + 3)
        
        non_linear_input_list = [None]*(len(self.linear_layers) + 3)
        
        k = 0
        
        output_list[k] = torch.cat((x, torch.ones([x.shape[0], 1], dtype = torch.double)), 1)
        
        non_linear_input_list[k]= x.clone()
        
        k = k + 1
        
        
        out = self.fc1(x)
        # Non-linearity 1
        
        non_linear_input_list[k]= out.clone()
        
        out = self.relu1(out)
        
        
        output_list[k] = torch.cat((out, torch.ones([out.shape[0], 1], dtype = torch.double)), 1)
        
        k = k + 1
        
        for i in range(len(self.linear_layers)):
            out = self.linear_layers[i](out)
            non_linear_input_list[k] = out.clone()
            out = self.activation_layers[i](out)
            output_list[k] = torch.cat((out, torch.ones([out.shape[0], 1], dtype = torch.double)), 1)
            
            k = k + 1
        
        
        # Linear function 2
        out = self.fc2(out)
        
        non_linear_input_list[k] = out.clone()
        
        out2 = self.fc3(out)
        
        output_list[k] = out2
        
        
        return output_list, non_linear_input_list
    

    class MyDataset(Dataset):
        def __init__(self, samples):
            
            self.data = samples
            
        def __getitem__(self, index):
            data, target = self.data[index]
            
            data = data.view(-1)
            
            
            y_onehot = torch.DoubleTensor(1, 10)
    
            target = torch.tensor([target])
            target = target.type(torch.LongTensor)
            
        # In your for loop
            y_onehot.zero_()
            y_onehot.scatter_(1, target.view(-1, 1), 1)
            
            
            # Your transformations here (or set it in CIFAR10)
            
            return data.type(torch.DoubleTensor), y_onehot.view(-1), index
    
        def __len__(self):
            return len(self.data)




class SkipNet(nn.Module):
    def __init__(self):
#         torch.manual_seed(1)
        super(SkipNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5).double()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5).double()
        self.conv2_drop = nn.Dropout2d().double()
        self.fc1 = nn.Linear(320, 50).double()
        self.fc2 = nn.Linear(50, 10).double()
        self.fc3 = nn.Dropout().double()

    def forward(self, x, need_record):
#         torch.manual_seed(1)
#         results = []
 
#         results.append(x.clone())
        
        all_ids_list = []
        
        out = self.conv1(x)
        
#         results.append(out.clone())

        out = F.max_pool2d(out, 2)

#         results.append(out.clone())

        out = F.relu(out)
        
#         results.append(out.clone())
        
        out = self.conv2(out)
        
#         results.append(out.clone())
        
        out = self.conv2_drop(out)
        
#         results.append(out.clone())
        
        ids_list = get_dropouted_ids(out)
        
#       
        all_ids_list.append(ids_list)
  
        out = F.max_pool2d(out, 2)
        
#         results.append(out.clone())
        
        out = F.relu(out)

#         results.append(out.clone())

#         flat_x= x.view(-1,784 )
        
        
        

        out = out.view(-1, 320)

#         out= torch.cat((flat_x,out),1)
        out = F.relu(self.fc1(out))
        
#         results.append(out.clone())
        
        out = self.fc3(out)
        
#         results.append(out.clone())
#         
        ids_list2 = get_dropouted_ids_2d(out)
        
        all_ids_list.append(ids_list2)
        
        out = self.fc2(out)
        
#         results.append(out.clone())
         
        res = F.log_softmax(out)
        
#         print(ids_list)
#         
#         print(ids_list2)
        
#         results.append(res.clone())
        
#         torch.save(results, git_ignore_folder + "tmp_res")
        
        return res, all_ids_list


    def forward_with_known_dropout(self, x, matched_ids, all_ids_list, needed):
        
#         results = torch.load(git_ignore_folder + "tmp_res")
        
#         print(torch.norm(results[0][matched_ids] - x))
        
        out = self.conv1(x)
         
#         print(torch.norm(results[1][matched_ids] - out))

        out = F.max_pool2d(out, 2)
        
#         print(torch.norm(results[2][matched_ids] - out))

        out = F.relu(out)
        
#         print(torch.norm(results[3][matched_ids] - out))
        
        out = self.conv2(out)
        
#         print(torch.norm(results[4][matched_ids] - out))
        
        out = set_dropouted_ids_2d(out, all_ids_list[0], self.conv2_drop.p)
        
#         print(torch.norm(results[5][matched_ids] - out))
#         out = self.conv2_drop(out)
        
#         ids_list = get_dropouted_ids(out)
         
        out = F.max_pool2d(out, 2)
        
#         print(torch.norm(results[6][matched_ids] - out))
        
        out = F.relu(out)
        
#         print(torch.norm(results[7][matched_ids] - out))

#         flat_x= x.view(-1,784 )

        out = out.view(-1, 320)

#         out= torch.cat((flat_x,out),1)
        out = F.relu(self.fc1(out))
        
#         print(torch.norm(results[8][matched_ids] - out))
        
        out = set_dropouted_ids_2d(out, all_ids_list[1], self.fc3.p)
        
#         print(torch.norm(results[9][matched_ids] - out))
        
#         out = self.fc3(out)
#         
#         ids_list2 = get_dropouted_ids2(out)
        
        out = self.fc2(out)
        
#         print(torch.norm(results[10][matched_ids] - out))
        
        res = F.log_softmax(out)
        
#         print(torch.norm(results[11][matched_ids] - out))
        
        return res


    class MyDataset(Dataset):
        def __init__(self, samples):
            
            self.data = samples
            
        def __getitem__(self, index):
            data, target = self.data[index]
            
    #         data = data.view(-1)
            
            # Your transformations here (or set it in CIFAR10)
            
            return data.type(torch.DoubleTensor), target, index
    
        def __len__(self):
            return len(self.data)
        
    def get_all_parameters(self):
    
        para_list = []
        
        for param in self.parameters():
            para_list.append(param.data.clone())
            print(param.data.shape)
            
            
        return para_list
    def get_all_gradient(self):
        
        para_list = []
        
        for param in self.parameters():
            para_list.append(param.grad.clone())
            
            
        return para_list    

def get_model_para_shape_list(para_list):
    
    shape_list = []
    
    full_shape_list = []
    
    total_shape_size = 0
    
    for para in list(para_list):
        
        all_shape_size = 1
        
        
        for i in range(len(para.shape)):
            all_shape_size *= para.shape[i]
        
        total_shape_size += all_shape_size
        shape_list.append(all_shape_size)
        full_shape_list.append(para.shape)
        
    return full_shape_list, shape_list, total_shape_size





def train (train_loader, optimizer, network, criterion):
    loss_list = []
    iteration_list = []
    accuracy_list = []

    correct = 0
    total = 0
    count=0
    
    

    for i, (inputs, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        outputs = network(inputs.type(torch.DoubleTensor))
        print(outputs[0])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        temp, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

        accuracy = 100 * correct / float(total)

        loss_list.append(loss.item())
        iteration_list.append(count)
        accuracy_list.append(accuracy)

            # Print Loss
        print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.item(), accuracy))


    return accuracy_list, loss_list




if __name__ == '__main__':

    torch.manual_seed(12)
    torch.cuda.manual_seed(12)
    np.random.seed(12)
    random.seed(12)

    network = SkipNet()
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(network.parameters())
    
    # optimizer = optim.Adam(network.parameters(),lr=0.001)
    # criterion = nn.CrossEntropyLoss()
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.2)
    # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    
    
    train_data = torchvision.datasets.MNIST(git_ignore_folder + '/mnist',
           download=True,
           transform=transforms.Compose([
    #            transforms.Resize((32, 32)),
               transforms.ToTensor()]))
    
    test_data = torchvision.datasets.MNIST(git_ignore_folder + '/mnist',
              train=False,
              download=True,
              transform=transforms.Compose([
    #               transforms.Resize((32, 32)),
                  transforms.ToTensor()]))
    
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size= 64, shuffle=True)
    
    losslist=[]
    acclist=[]
    
    for epoch in range (1):
    
    
        acclist,loss_list= train (train_loader, optimizer, network, criterion)
    
    #     plt.plot(loss_list, 'r')
    #     plt.xlabel('batches')
    #     plt.ylabel('loss')
    #     plt.show()
    #     #
    #     plt.plot(acclist, 'r')
    # 
    #     plt.xlabel('batches')
    #     plt.ylabel('accuracy')
    #     plt.show()