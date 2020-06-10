'''
Created on Feb 4, 2019

'''
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))





try:
    from data_IO.Load_data import *
    from utils import *
    from Interpolation.piecewise_linear_interpolation_2D import *

except ImportError:
    from Load_data import *
    from utils import *
    from piecewise_linear_interpolation_2D import *

loss_threshold = 0.1

    
#         if type(m) == nn.Linear:
#             nn.init.constant_(m.weight, 0)
#             
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
            
softmax_func = nn.Softmax(dim = 1)

sigmoid_func = nn.Sigmoid()

cut_off_epoch = 100

# def create_models(input_dim, hidden_dims, output_dim):
#     layers = []
#     layers.append(nn.Linear(input_dim, hidden_dims[0]))
#     layers.append(nn.Sigmoid())
#     
#     for i in range(len(hidden_dims) - 1):
#         layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
#         layers.append(nn.Sigmoid())
#     
#     
#     
#     layers.append(nn.Linear(hidden_dims[-1], output_dim))
#     layers.append(nn.Sigmoid())
#     
#     net = nn.Sequential(*layers)
#     
#     return net
# 
# def get_output_each_layer(model, x):
#         
#         
#         output_list = []
#         
#         output_list.append(torch.cat((x, torch.ones(x.shape[0], 1)), 1))
#         
#         
#         out = self.fc1(x)
#         # Non-linearity 1
#         out = self.relu1(out)
#         
#         
#         output_list.append(torch.cat((out, torch.ones(out.shape[0], 1)), 1))
#         
#         for i in range(len(self.linear_layers)):
#             out = self.linear_layers[i](out)
#             out = self.activation_layers[i](out)
#             output_list.append(torch.cat((out, torch.ones(out.shape[0], 1)), 1))
#         
#         
#         
#         
#         # Linear function 2
#         out = self.fc2(out)
#         
#         out2 = self.fc3(out)
#         
#         output_list.append(out2)
#         
#         
#         return output_list
    
    
    
class DNNModel(nn.Module):
    
    
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(DNNModel, self).__init__()
        # Linear function 1: 784 --> 100
        self.fc1 = nn.Linear(input_dim, hidden_dims[0]).double()
        
#         self.fc1.weight.data.fill_(0)
#         self.fc1.bias.data.fill_(0)

#         nn.init.constant_(self.fc1.weight, float(0.5))
#         
#         nn.init.constant_(self.fc1.bias, float(0.5))
#         
#         self.fc1.weight.requires_grad = True
#         
#         self.fc1.bias.requires_grad = True
        
        

        # Non-linearity 1
        self.relu1 = nn.Sigmoid()
        
        
        self.linear_layers = nn.ModuleList([])
        
        
        self.activation_layers = nn.ModuleList([])
        
        for i in range(len(hidden_dims) - 1):
            self.linear_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.activation_layers.append(nn.Sigmoid())
        
        # Linear function 2: 100 --> 100
#         print(hidden_dims[len(hidden_dims) - 1])
        self.fc2 = nn.Linear(hidden_dims[len(hidden_dims) - 1], output_dim).double()
        
#         self.fc2.weight.fill_(0)
#         self.fc2.bias.fill_(0)
#         nn.init.constant_(self.fc2.weight, float(0.5))
#         
#         
#         nn.init.constant_(self.fc2.bias, float(0.5))
        
#         self.fc3 = nn.Softmax(dim=1)

        self.fc3 = nn.Sigmoid()
        
        
#         # Non-linearity 2
#         self.tanh2 = nn.Tanh()
#         
#         # Linear function 3: 100 --> 100
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim)
#         # Non-linearity 3
#         self.elu3 = nn.ELU()
#         
#         # Linear function 4 (readout): 100 --> 10
#         self.fc4 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function 1
                
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)
        
        
        for i in range(len(self.linear_layers)):
            out1 = self.linear_layers[i](out)
            out = self.activation_layers[i](out1)
        
        
#         print("dim::", out.shape)
#         
#         print(len(list(self.parameters())))
#         
#         print(self.fc2)
        # Linear function 2
        out = self.fc2(out)
        
        out = self.fc3(out)
#         # Non-linearity 2
#         out = self.tanh2(out)
#         
#         # Linear function 2
#         out = self.fc3(out)
#         # Non-linearity 2
#         out = self.elu3(out)
#         
#         # Linear function 4 (readout)
#         out = self.fc4(out)

        return out
    
    
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
        
        
        
    

def get_onehot_y(Y, dim, num_class):
    
#     x_sum_by_class = torch.zeros([num_class, dim[1]], dtype = torch.double)
    
    
    y_onehot = torch.DoubleTensor(dim[0], num_class)

    Y = Y.type(torch.LongTensor)

# In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, Y.view(-1, 1), 1)
    
    
    return y_onehot


def get_onehot_y_time_X(X, Y, dim, num_class):
    
#     x_sum_by_class = torch.zeros([num_class, dim[1]], dtype = torch.double)
    
    
    y_onehot = torch.DoubleTensor(dim[0], num_class)

    Y = Y.type(torch.LongTensor)

# In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, Y.view(-1, 1), 1)
    
    
    return y_onehot

# def model_update_standard_lib(num_epochs, X, Y, test_X, test_Y, learning_rate, error, model):
#     count = 0
# #     for epoch in range(num_epochs):
# 
#     loss = np.infty
# 
#     while loss > loss_threshold and count < num_epochs:
# #         for i, (images, labels) in enumerate(train_loader):
#             
# #         for i in range(X.shape[0]):
#     
#     
# #             train = Variable(images.view(-1, 28*28))
#         train = Variable(X)
#         labels = Variable(Y.view(-1))
#         
#         # Clear gradients
# #         optimizer.zero_grad()
#         
#         # Forward propagation
#         outputs = model(train)
#         
#         # Calculate softmax and ross entropy loss
#         
# #         print(outputs)
# #         
# #         print(labels)
#         
#         labels = labels.type(torch.LongTensor)
#         
#         loss = error(outputs, labels)
#         
#         # Calculating gradients
# #         loss.backward(retain_graph = True, create_graph=True)
#         
#         loss.backward()
#         
#         update_and_zero_model_gradient(model,learning_rate)
#         
#         print("loss:", loss)
#         
#         if count % 10 == 0:
#             # Calculate Accuracy         
#             correct = 0
#             total = 0
#             # Predict test dataset
# #                 for images, labels in test_loader:
# #             for j in range(test_X.shape[0]):
# 
# #                     test = Variable(images.view(-1, 28*28))
#             test = Variable(test_X)
#             
#             labels = test_Y.view(-1).type(torch.LongTensor)
#             
#             # Forward propagation
#             outputs = model(test)
#             
#             # Get predictions from the maximum value
#             predicted = torch.max(outputs.data, 1)[1]
#             
#             # Total number of labels
#             total += len(labels)
# 
#             # Total correct predictions
#             correct += (predicted == labels).sum()
#             
#             accuracy = 100 * correct / float(total)
# #             if count % 500 == 0:
#                 # Print Loss
#                 
#                 
#             print("accuracy:: {} %", format(accuracy))
#         
#         
# #         print("epoch::", epoch)
#         
# #         print_model_para(model)
#         
#         
#         
#         # Update parameters
# #         optimizer.step()
#         
#         count += 1
# #             print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data[0].item(), accuracy.item()))
#     return model



def construct_gradient_list(gradient_list, res_list, model):
    
    gradient_list.clear()
    del gradient_list[:]
    
    for param in model.parameters():
        gradient_list.append(param.grad.clone())
        res_list.append(param.data.clone())
        
        
def append_gradient_list(gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, model, X):
    
    
    gradient_list = []
    
    para_list = []
    
    
    for param in model.parameters():
        gradient_list.append(param.grad.clone())
        para_list.append(param.data.clone())
        
        
    output_list,_ = model.get_output_each_layer(X)   
        
        
    gradient_list_all_epochs.append(gradient_list)
    
    output_list_all_epochs.append(output_list)
    
    para_list_all_epochs.append(para_list)
    
    
        
def get_model_gradient(model):
    
    gradient_list = []
    
    for param in model.parameters():
        gradient_list.append(param.grad.clone())
        
    return gradient_list

def create_piecewise_linea_class(linearized_Function):
#     x_l = torch.tensor(-10, dtype=torch.double)
#     x_u = torch.tensor(10, dtype=torch.double)
    x_l = -20.0
    x_u =20.0
    num = 1000000
    Pi = piecewise_linear_interpolication(x_l, x_u, linearized_Function, num)
    
    return Pi


def loss_function2(output, Y, dim):
    
#     res = 0
    
    
#     sigmoid_res = torch.stack(list(map(bia_function, Y*torch.mm(X, theta))))

#     sigmoid_res = Y*torch.mm(X, theta)
#     data_trans = sigmoid_res.apply(lambda x :  ())

#     sigmoid_res = -log_sigmoid_layer(Y*torch.mm(X, theta))
#     theta = theta.view(dim[1], num_class)




    X_theta_prod = output
    
    
    X_theta_prod_softmax = softmax_func(X_theta_prod)
    
    res = -torch.sum(torch.log(torch.gather(X_theta_prod_softmax, 1, Y.view(-1,1))))/dim[0]
    
    return res
    
#     return res + beta/2*torch.pow(torch.norm(theta, p =2), 2)

def model_update_standard_lib(num_epochs, X, Y, learning_rate, error, model, beta):
    count = 0
#     for epoch in range(num_epochs):
    loss = np.infty

    elapse_time = 0

    t1 = time.time()
    train = Variable(X)
    labels = Variable(Y.view(-1))
    labels = labels.type(torch.LongTensor)
    while count < num_epochs:
    
        
        
#         outputs = model(train)
        
#         loss = error(outputs, labels)
        
        loss = compute_loss(model, error, X, Y, beta)
        
        loss.backward()
        
        update_and_zero_model_gradient(model,learning_rate)
        
#         print("iteration::", count)
#         
#         print("loss::", loss)
        
        count += 1
         
#         print("loss::", loss)
        
    t2 = time.time()
        
    elapse_time += (t2 - t1)  

    print("training time is", elapse_time)
    
    return model, count

def model_update_standard_lib_stochastic(batch_size, num_epochs, X, Y, learning_rate, error, model):
    count = 0
#     for epoch in range(num_epochs):
    loss = np.infty

    elapse_time = 0

    t1 = time.time()

#     train = Variable(X)

    labels = Variable(Y.view(-1))
    
    labels = labels.type(torch.LongTensor)
    
    while count < num_epochs:
        
        
        
        random_ids = torch.randperm(X.shape[0])
        
        curr_X = Variable(X[random_ids])
        
        curr_Y = Variable(labels[random_ids])
        
        
        for i in range(0, X.shape[0], batch_size):
            
            
            end_id = i + batch_size
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
        
                
                
            batch_x = curr_X[i: end_id]
            
            batch_y = curr_Y[i: end_id]    
            
        
            outputs = model(batch_x)
            
            loss = error(outputs, batch_y)
            
            loss.backward()
            
            update_and_zero_model_gradient(model,learning_rate)
            
    #         print("iteration::", count)
    #         
    #         print("loss::", loss)
            
        count += 1
         
#         print("loss::", loss)
        
    t2 = time.time()
        
    elapse_time += (t2 - t1)  

    print("training time is", elapse_time)
    
    return model, count

def model_update_standard_lib_with_recording_parameters(num_epochs, X, Y, learning_rate, error, model):
    count = 0
#     for epoch in range(num_epochs):
    loss = np.infty

    elapse_time = 0

    t1 = time.time()
    
    para_list = []
    
    
    para_list.append(get_all_parameters(model))

    while count < num_epochs:
    
        train = Variable(X)
        labels = Variable(Y.view(-1))
        
        outputs = model(train)
        
        labels = labels.type(torch.LongTensor)
        
        loss = error(outputs, labels)
        
        loss.backward()
        
        update_and_zero_model_gradient(model,learning_rate)
        
        
        para_list.append(get_all_parameters(model))
        
        print("iteration::", count)
        
        print("loss::", loss)
        
        count += 1
         
#         print("loss::", loss)
        
    t2 = time.time()
        
    elapse_time += (t2 - t1)  

    print("training time is", elapse_time)
    
    return model, count, para_list


# def compute_hessian(model, error):


def model_compute_loss(model, X, Y, error):
    train = Variable(X)
    outputs = model(train)
    labels = Variable(Y.view(-1))
    
    loss = error(outputs, labels)
    
    return loss


def compute_test_acc(model, test_X, test_Y):
    correct = 0
    total = 0
    # Predict test dataset
#                 for images, labels in test_loader:
#             for j in range(test_X.shape[0]):

#                     test = Variable(images.view(-1, 28*28))
    test = Variable(test_X)
    
    labels = test_Y.view(-1).type(torch.LongTensor)
    
    # Forward propagation
    outputs = model(test)
    
    # Get predictions from the maximum value
    predicted = torch.max(outputs.data, 1)[1]
    
    # Total number of labels
    total += len(labels)

    # Total correct predictions
    correct += (predicted == labels).sum()
    
    accuracy = 100 * correct / float(total)
    
    # store loss and iteration
#     loss_list.append(loss.data)
#     
#     iteration_list.append(count)
#     accuracy_list.append(accuracy)
#             if count % 500 == 0:
        # Print Loss
        
        
    print("accuracy:: {} %", format(accuracy))




def model_training(num_epochs, X, Y, test_X, test_Y, learning_rate, regularization_coeff, error, model, is_tracking_paras):
    count = 0
#     loss_list = []
#     iteration_list = []
#     accuracy_list = []
#     for epoch in range(num_epochs):
    loss = np.infty

    gradient_list_all_epochs = []
    
    para_list_all_epochs = []
    
    output_list_all_epochs = []
    
#     para_lists_all_epochs = []
    
#     gradient_lists_all_epochs = []
    
#     para_lists_all_epochs.append(get_all_parameters(model))
    
#     gradient_lists_all_epochs.append(get_all_gradient(model))

#     construct_gradient_list(gradient_list, model)


    elapse_time = 0
    
    b_time = 0
    
    f_time = 0

    while loss > loss_threshold and count < num_epochs:
#         for i, (images, labels) in enumerate(train_loader):
            
#         for i in range(X.shape[0]):
        t1 = time.time()
    
#             train = Variable(images.view(-1, 28*28))
        train = Variable(X)
        labels = Variable(Y.view(-1))
        
        # Clear gradients
#         optimizer.zero_grad()
        
        # Forward propagation
        
        t3 = time.time()
        
        outputs = model(train)
        
        t4 = time.time()
        
        f_time += (t4 - t3)
        
        # Calculate softmax and ross entropy loss
        
#         print(outputs)
#         
#         print(labels)
        
        labels = labels.type(torch.LongTensor)
        
        loss = error(outputs, labels) + regularization_coeff*get_regularization_term(model.parameters())
        
#         print("loss0::", loss)
# 
#         loss = loss_function2(outputs, labels, X.shape)
        
        # Calculating gradients
#         loss.backward(retain_graph = True, create_graph=True)
        t3 = time.time()
        
        loss.backward()
        
        t4 = time.time()
        
        b_time += (t4 - t3)
        
#         construct_gradient_list(gradient_list_all_epochs, res_list, model)
        if is_tracking_paras:
            append_gradient_list(gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, model,train)
        
#         gradient_lists_all_epochs.append(get_all_gradient(model))
        
        
        update_and_zero_model_gradient(model,learning_rate)
        
        
        t2 = time.time()
        
        elapse_time += (t2 - t1)
        
        
#         print("training time::", (t2 - t1))
        
        print("iteration::", count)
        
#         print_model_para(model)
        
#         para_lists_all_epochs.append(get_all_parameters(model))
        
#         print("epoch::", epoch)
        
#         print_model_para(model)
        
        
        
        # Update parameters
#         optimizer.step()
        
        count += 1
        
        print("loss::", loss)
        
        if count % 10 == 0:
            compute_test_acc(model, test_X, test_Y)
            # Calculate Accuracy         

#             print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data[0].item(), accuracy.item()))


    print("training time is ", elapse_time)
    
    print("forward time is ", f_time)
    
    print("backaward time is ", b_time)
    
#     para_list_all_epochs.append(get_all_parameters(model))
    
    return model, gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, count


# def compute_hessian(model, error):

def get_all_gradients(selected_rows, output_list_all_epochs, model, para_list_all_epochs, input_dim, hidden_dims, output_dim, num_class, X, Y):
    
    depth = len(hidden_dims) + 1
    

    delta_gradient_all_epochs = []
    
    delta_all_epochs = []
    
    for k in range(len(output_list_all_epochs)):

#     for k in range(1):
        
        output_list = output_list_all_epochs[k]
        pred = output_list[len(output_list) - 1][selected_rows]
            
        para_list = para_list_all_epochs[k]
        
        init_model(model, para_list)
    
    
    
#         A_list = [None]*depth
#     
#     
#         B_list = [None]*depth
#         
#         
#     #     A0_list = [None]*depth
#     #     
#     #     B0_list = [None]*depth
#     #         loss = error(pred, Y)
        
        delta = softmax_func(pred) - get_onehot_y(Y[selected_rows], X[selected_rows].shape, num_class)
        
#         delta = delta[selected_rows]
        
        delta_all_epochs.append(delta)
        delta_gradient = []
        
        for m in range(20):
            
            curr_gradient_list = []
            
            print("derivitive_times::", m)
            
            for l in range(delta.shape[1]):
                delta[m][l].backward(retain_graph=True)
                
                curr_gradient = get_all_gradient(model)
                
                curr_gradient_list.append(curr_gradient)
                
            delta_gradient.append(curr_gradient_list)
            
            
        delta_gradient_all_epochs.append(delta_gradient)
        
    return delta_gradient_all_epochs, delta_all_epochs   
            
            
            
            
            
    
        
        
    
    




def get_all_parameters(model):
    
    para_list = []
    
    for param in model.parameters():
        para_list.append(param.data.clone())
        
        
    return para_list



def get_all_gradient(model):
    
    para_list = []
    
    for param in model.parameters():
        para_list.append(param.grad.clone())
        
        
    return para_list    


def get_all_vectorized_gradients(para_list):
    
    res_list = []
    
    for param in para_list:
        res_list.append(param.grad.clone().view(-1))
        
        
#         para_list.append(param.grad.clone())
        
        
    return torch.cat(res_list, 0)    


def clear_gradients(para_list):
    for param in para_list:
        param.grad.zero_()


def get_all_vectorized_parameters(para_list):
    
    res_list = []
    
    for param in para_list:
        res_list.append(param.data.contiguous().view(-1))
        
        
#         para_list.append(param.grad.clone())
        
        
    return torch.cat(res_list, 0).view(1,-1)


def get_regularization_term(para_list):
    
    regularization_term = 0
    
    for param in para_list:
        regularization_term += param.pow(2).sum()
        
        
    return regularization_term
        
        
        
        
        


        
def get_devectorized_parameters(params, input_dim, hidden_dims, output_dim):
    
    params = params.view(-1)
    
    para_list = []
    
    para_list.append(params[0: input_dim*hidden_dims[0]].view(hidden_dims[0], input_dim))
    
    para_list.append(params[input_dim*hidden_dims[0]: (input_dim+1)*hidden_dims[0]].view(hidden_dims[0]))
    
    pos = (input_dim+1)*hidden_dims[0]
    
    for i in range(len(hidden_dims) - 2):
        para_list.append(params[pos: pos+hidden_dims[i]*hidden_dims[i+1]].view(hidden_dims[i+1], hidden_dims[i]))
        pos = pos+hidden_dims[i]*hidden_dims[i+1]
        para_list.append(params[pos: pos+hidden_dims[i+1]].view(hidden_dims[i+1]))
        pos = pos+hidden_dims[i+1]
        
    
    
    para_list.append(params[pos: pos+ hidden_dims[-1]*output_dim].view(output_dim, hidden_dims[-1]))
    
    pos = pos+hidden_dims[-1]*output_dim
    
    para_list.append(params[pos: pos + output_dim].view(output_dim))
    
    return para_list
    

def update_and_zero_model_gradient(model, alpha):
    
#     all_parameters = list(model.parameters())
    
#     for i in range(len(all_parameters)):

    for param in model.parameters():
        param.data = param.data - alpha*param.grad
        
#         print(param.grad)
        
        param.grad.zero_()
        
#         all_parameters[i].data = all_parameters[i] - alpha*all_parameters[i].grad
        
#         all_parameters[i].grad.zero_()


def zero_model_gradient(model):
    for param in model.parameters():
        param.grad.zero_()

def init_model(model, para_list):
    
    i = 0
    
    for m in model.parameters():
        m.data = para_list[i].clone()
        i += 1
        
    model.zero_grad()


def get_model_para(model):
    
    para_list = []
    
    for param in model.parameters():
        print(param.data.shape)
        para_list.append(param.data.clone())
        
    return para_list

def print_model_para(model):
    
    for param in model.parameters():
        print(param.data.shape)
        print(param.data)



def sigmoid_diff_function(x):
    return np.exp(-x)/(np.power(1+np.exp(-x), 2))


def model_update_provenance_by_dual(alpha, dim, dual_para_list, hessian_matrix, origin_gradient_list, vectorized_orign_params, epoch, input_dim, hidden_dims, output_dim, delta_ids, expected_gradient_list_all_epochs, epxected_para_list_all_epochs_all_epochs):
    
    
    vectorized_origin_gradient = get_all_vectorized_parameters(origin_gradient_list)
    
    res_vectorized_paras = vectorized_orign_params.clone()
    
    n = dim[0]
    
    
    k = delta_ids.shape[0]
    
    
    
    for i in range(epoch):
        
#         expected_para = get_all_vectorized_parameters(epxected_para_list_all_epochs_all_epochs[i])
        
        dual_vectorized_paras = get_all_vectorized_parameters(dual_para_list[i])
        
        dual_vectorized_paras_next_epoch = get_all_vectorized_parameters(dual_para_list[i+1])
        
        res_vectorized_paras = (0 - (n-k)*res_vectorized_paras - k *dual_vectorized_paras - alpha*(torch.mm((vectorized_orign_params - res_vectorized_paras), hessian_matrix)) + k*dual_vectorized_paras_next_epoch)/(k-n)
    
#         expected_grad = get_all_vectorized_parameters(expected_gradient_list_all_epochs[i])
    
    return get_devectorized_parameters(res_vectorized_paras, input_dim, hidden_dims, output_dim)


def update_hessian(hessian_matrix, old_para, new_para, old_gradient, new_gradient):
    
    y = get_all_vectorized_parameters(new_gradient) - get_all_vectorized_parameters(old_gradient)
    
    s = get_all_vectorized_parameters(new_para) - get_all_vectorized_parameters(old_para)
    
    
    
    updated_hessian_matrix = hessian_matrix + torch.mm(torch.t(y), y)/(torch.mm(y, torch.t(s))) - (torch.mm(torch.mm(hessian_matrix, torch.t(s)), torch.mm(s, torch.t(hessian_matrix)))/torch.mm(torch.mm(s, (hessian_matrix)), torch.t(s)))

    
    return updated_hessian_matrix
    
def cal_approx_hessian_vec_prod3(i, m, v_vec, para_list_all_epochs, gradient_list_all_epoch):
    
    curr_S_k = torch.zeros([v_vec.shape[0],m], dtype = torch.double)#S_k_list[:,i-m:i]
        
    curr_Y_k = torch.zeros([v_vec.shape[0],m], dtype = torch.double)
    
    for k in range(m):
        curr_S_k[:,k] = get_all_vectorized_parameters(para_list_all_epochs[i-(k+1)]) - get_all_vectorized_parameters(para_list_all_epochs[i])
    
        curr_Y_k[:,k] = get_all_vectorized_parameters(gradient_list_all_epoch[i-(k+1)]) - get_all_vectorized_parameters(gradient_list_all_epoch[i])
    
    
    res = torch.mm(torch.inverse(torch.mm(torch.t(curr_S_k), curr_S_k)), torch.mm(torch.t(curr_S_k), v_vec))
    
    return torch.mm(curr_Y_k, res.view(-1,1))





def cal_approx_hessian_vec_prod4(truncted_s, extended_Y_k_list, i, m, v_vec):
    
#     curr_S_k = torch.zeros([v_vec.shape[0],m], dtype = torch.double)#S_k_list[:,i-m:i]
#         
#     curr_Y_k = torch.zeros([v_vec.shape[0],m], dtype = torch.double)
#     
#     for k in range(m):
#         curr_S_k[:,k] = get_all_vectorized_parameters(para_list_all_epochs[i-(k+1)]) - get_all_vectorized_parameters(para_list_all_epochs[i])
#     
#         curr_Y_k[:,k] = get_all_vectorized_parameters(gradient_list_all_epoch[i-(k+1)]) - get_all_vectorized_parameters(gradient_list_all_epoch[i])
    
    
    res = torch.mm(torch.inverse(torch.mm(torch.t(truncted_s), truncted_s)), torch.mm(torch.t(truncted_s), v_vec))
    
    print(res)
    
    print(torch.norm(torch.mm(truncted_s, res) - v_vec))
    
    return torch.mm(extended_Y_k_list, res.view(-1,1)), torch.mm(truncted_s, res)


    
#     results = np.linalg.solve(curr_S_k.numpy(), v_vec.numpy())
#     
#     results_tensor = torch.from_numpy(results)
#     
#     
#     return torch.mm(curr_Y_k, results_tensor.view(-1,1))
    
    
    
    
    


def cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec):
    
    curr_S_k = S_k_list[:,i-m:i]
        
    curr_Y_k = Y_k_list[:,i-m:i]
    
    S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
    
    
    S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)
    
    
    R_k = torch.triu(S_k_time_Y_k)
    
    L_k = S_k_time_Y_k - R_k
    
    D_k_diag = torch.diag(S_k_time_Y_k)
    
    
    sigma_k = torch.dot(Y_k_list[:,i-1],S_k_list[:,i-1])/(torch.dot(S_k_list[:,i-1], S_k_list[:,i-1]))
    
    
    interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
    
    J_k = torch.from_numpy(np.linalg.cholesky(interm.detach().numpy())).type(torch.DoubleTensor)
    
    
#     v_vec = S_k_list[:,i-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
    
#         v_vec = torch.rand([para_num, 1], dtype = torch.double)
    
    
    p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
    
    
    D_k_sqr_root = torch.pow(D_k_diag, 0.5)
    
    D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
    
    upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
    
    lower_mat_1 = torch.cat([torch.zeros([m, m], dtype = torch.double), torch.t(J_k)], dim = 1)
    
    
    mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
    
    
    upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([m, m], dtype = torch.double)], dim = 1)
    
    lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
    
    mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
    
    
    p_mat = torch.mm(torch.inverse(mat_1), torch.mm(torch.inverse(mat_2), p_mat))
    
    
    approx_prod = sigma_k*v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    
    return approx_prod



def cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, m, k, v_vec, period):
    
    period_num = int(i/period)
    
    ids = period*torch.tensor(range(period_num))
    
    
    ids = ids[ids > 0]
    
    if ids.shape[0] > m:
        ids = ids[-m:]
    
#     if i-k >= 1:
#         lb = i-k
#         
#         zero_mat_dim = ids.shape[0] + k
#         
#     else:
#         lb = 1
#         
#         zero_mat_dim = ids.shape[0] + i-1
    zero_mat_dim = ids.shape[0]
    
    ids = ids - 1
    
#     curr_S_k = torch.cat([S_k_list[:, ids],S_k_list[:,lb:i]], dim=1) 
#           
#     curr_Y_k = torch.cat([Y_k_list[:, ids],Y_k_list[:,lb:i]], dim=1)

    curr_S_k = torch.cat([S_k_list[:, ids]], dim=1) 
          
    curr_Y_k = torch.cat([Y_k_list[:, ids]], dim=1)

    
#     curr_S_k = S_k_list[:,k:m] 
#          
#     curr_Y_k = Y_k_list[:,k:m] 
    
    S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
    
    
    S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)
    
    
    R_k = torch.triu(S_k_time_Y_k)
    
    L_k = S_k_time_Y_k - R_k
    
    D_k_diag = torch.diag(S_k_time_Y_k)
    
    
    sigma_k = torch.dot(Y_k_list[:,i-1],S_k_list[:,i-1])/(torch.dot(S_k_list[:,i-1], S_k_list[:,i-1]))
    
    
    interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
    
    J_k = torch.from_numpy(np.linalg.cholesky(interm.detach().numpy())).type(torch.DoubleTensor)
    
    
#     v_vec = S_k_list[:,i-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
    
#         v_vec = torch.rand([para_num, 1], dtype = torch.double)
    
    
    p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
    
    
    D_k_sqr_root = torch.pow(D_k_diag, 0.5)
    
    D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
    
    upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
    
    lower_mat_1 = torch.cat([torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double), torch.t(J_k)], dim = 1)
    
    
    mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
    
    
    upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double)], dim = 1)
    
    lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
    
    mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
    
    
    p_mat = torch.mm(torch.inverse(mat_1), torch.mm(torch.inverse(mat_2), p_mat))
    
    
    approx_prod = sigma_k*v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    
    return approx_prod

def cal_approx_hessian_vec_prod2(S_k_list, Y_k_list, i, m, v_vec, delta_para, delta_grad):
    
    curr_S_k = torch.cat([S_k_list[:,i-m:i], delta_para], dim = 1)
                         
        
    curr_Y_k = torch.cat([Y_k_list[:,i-m:i], delta_grad], dim = 1)
    
    
    
    
    S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
    
    
    S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)
    
    
    R_k = torch.triu(S_k_time_Y_k)
    
    L_k = S_k_time_Y_k - R_k
    
    D_k_diag = torch.diag(S_k_time_Y_k)
    
    
    sigma_k = torch.dot(Y_k_list[:,i-1],S_k_list[:,i-1])/(torch.dot(S_k_list[:,i-1], S_k_list[:,i-1]))
    
    
    interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
    
    J_k = torch.from_numpy(np.linalg.cholesky(interm.detach().numpy())).type(torch.DoubleTensor)
    
    
#     v_vec = S_k_list[:,i-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
    
#         v_vec = torch.rand([para_num, 1], dtype = torch.double)
    
    
    p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
    
    
    D_k_sqr_root = torch.pow(D_k_diag, 0.5)
    
    D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
    
    upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
    
    lower_mat_1 = torch.cat([torch.zeros([m+1, m+1], dtype = torch.double), torch.t(J_k)], dim = 1)
    
    
    mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
    
    
    upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([m+1, m+1], dtype = torch.double)], dim = 1)
    
    lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
    
    mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
    
    
    p_mat = torch.mm(torch.inverse(mat_1), torch.mm(torch.inverse(mat_2), p_mat))
    
    
    approx_prod = sigma_k*v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    
    return approx_prod


def model_update_provenance_test(truncted_s, extended_Y_k_list, exp_gradient_list_all_epochs, exp_para_list_all_epochs, X, Y, model, S_k_list, Y_k_list, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, input_dim, hidden_dims, output_dim, m, alpha, beta, selected_rows, error):
    
    
    para_num = S_k_list.shape[0]
    
    
    model_dual = DNNModel(input_dim, hidden_dims, output_dim)
    
    
    para = list(model.parameters())
    
    expected_para = list(model.parameters())
    
    last_gradient_full = None

    last_para = None
    
    for i in range(epoch):
        
        print('epoch::', i)
        
        init_model(model, para)
        
        compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
        
        expect_gradients = get_all_vectorized_parameters(get_all_gradient(model))
        
        
        init_model(model, expected_para)
        
        compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
        
#         compute_model_para_diff(expected_para, exp_para_list_all_epochs[i])

        
        expected_para = get_devectorized_parameters(get_all_vectorized_parameters(expected_para) - alpha*get_all_vectorized_parameters(get_all_gradient(model)), input_dim, hidden_dims, output_dim)
        
        
        
        if i >= 50:
        
            init_model(model_dual, para)
            
            compute_derivative_one_more_step(model_dual, error, X[delta_ids], Y[delta_ids], beta)
            
            
            gradient_dual = get_all_vectorized_parameters(get_all_gradient(model_dual))
            
            
            v_vec = get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])
            
            print('para_diff::', torch.norm(v_vec))
            
            print('para_angle::', torch.dot(get_all_vectorized_parameters(para).view(-1), get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))/(torch.norm(get_all_vectorized_parameters(para).view(-1))*torch.norm(get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))))
            
            
            
            init_model(model, para_list_all_epochs[i])
             
            curr_vec_gradient,_ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, gradient_list_all_epochs[i], beta)
             
            hessian_matrix1 = compute_hessian_matrix(model, curr_vec_gradient, input_dim, hidden_dims, output_dim)
            
            
#             hessian_para_prod = cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, 50, m, v_vec.view(-1,1))
            
            hessian_para_prod = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1))
            
#             hessian_para_prod = cal_approx_hessian_vec_prod2(S_k_list, Y_k_list, i, m-1, v_vec.view(-1,1), last_v_vec.view(-1,1), last_gradient_full.view(-1, 1) - get_all_vectorized_parameters(gradient_list_all_epochs[i-1]).view(-1,1))
            
#             hessian_para_prod = cal_approx_hessian_vec_prod3(i, m, v_vec.view(-1,1), para_list_all_epochs, gradient_list_all_epochs)
            
#             hessian_para_prod, tmp_res = cal_approx_hessian_vec_prod4(truncted_s, extended_Y_k_list[i], i, m, v_vec.view(-1,1))
            
            
            delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
            
#             para_diff = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
            
            gradient_full = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
            
            delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1), error, model, beta)
            gradient_full2 = torch.mm(v_vec.view(1,-1), hessian_matrix1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
#             gradient_full2 = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) - delta_const*v_vec.view(1,-1)
            
            gradients = (gradient_full*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
            
            gradients2 = (gradient_full2*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
            
            print('hessian_vector_prod_diff::', torch.norm(torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1) - hessian_para_prod))
            
            print('gradient_diff::', torch.norm(gradients - expect_gradients))
            
            print('gradient2_diff::', torch.norm(gradients2 - expect_gradients))
            
            compute_model_para_diff(exp_para_list_all_epochs[i], para)
            
            S_k_list[:,i-1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
            
            para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradients2, input_dim, hidden_dims, output_dim)
        
            Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
            
            last_gradient_full = gradient_full
            
            last_v_vec = v_vec.clone()
        
        
        else:
            if i >= 1:
                S_k_list[:,i - 1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
            last_para = para
            
            para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*expect_gradients, input_dim, hidden_dims, output_dim)
        
            if i == m-1:
                init_model(model, para)
                
                compute_derivative_one_more_step(model, error, X, Y, beta)
            
            
                last_gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
                
                last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
                
            if i >= 1:
                
                init_model(model, para)
                
                compute_derivative_one_more_step(model, error, X, Y, beta)
            
            
                gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
                
#                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
                
                Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
        
#         last_gradient = expect_gradients
#             init_model(model, get_devectorized_parameters(updated_para, input_dim, hidden_dims, output_dim))
            
            
            
            
            
            
            
    return para
            




def model_update_provenance_test1(period, length, truncted_s, extended_Y_k_list, exp_gradient_list_all_epochs, exp_para_list_all_epochs, X, Y, model, S_k_list, Y_k_list, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, input_dim, hidden_dims, output_dim, m, alpha, beta, selected_rows, error):
    
    
    para_num = S_k_list.shape[0]
    
    
    model_dual = DNNModel(input_dim, hidden_dims, output_dim)
    
    
    para = list(model.parameters())
    
    expected_para = list(model.parameters())
    
    last_gradient_full = None

    last_para = None
    
    use_standard_way = False
    
    recorded = 0
    
    for i in range(epoch):
        
#         if i == epoch - 2:
#             y = 1
#             y+=1
        
        
#         print('epoch::', i)
        
#         init_model(model, para)
#         
#         compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
#         
#         expect_gradients = get_all_vectorized_parameters(get_all_gradient(model))
        
        
#         init_model(model, expected_para)
#         
#         compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
#         
# #         compute_model_para_diff(expected_para, exp_para_list_all_epochs[i])
# 
#         
#         expected_para = get_devectorized_parameters(get_all_vectorized_parameters(expected_para) - alpha*get_all_vectorized_parameters(get_all_gradient(model)), input_dim, hidden_dims, output_dim)
        
        if i%period == 0:
            
            recorded = 0
            
            use_standard_way = True
            
            
        if i< 20 or use_standard_way == True:
            
            
            init_model(model, para)
            
            compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
             
            expect_gradients = get_all_vectorized_parameters(get_all_gradient(model))
            
            
            if i>0:
                S_k_list[:,i - 1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
            
#             init_model(model, para)
            clear_gradients(model.parameters())
                
            compute_derivative_one_more_step(model, error, X[delta_ids], Y[delta_ids], beta)
        
        
            gradient_remaining = get_all_vectorized_parameters(get_all_gradient(model))
            
            
            gradient_full = (expect_gradients*selected_rows.shape[0] + gradient_remaining*delta_ids.shape[0])/(selected_rows.shape[0] + delta_ids.shape[0])
            
            para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*expect_gradients, input_dim, hidden_dims, output_dim)
#                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
            
            if i>0:
                Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
            
            
            recorded += 1
            
            
            if recorded >= length:
                use_standard_way = False
            
            
        else:
        
            init_model(model_dual, para)
            
            compute_derivative_one_more_step(model_dual, error, X[delta_ids], Y[delta_ids], beta)
            
            
            gradient_dual = get_all_vectorized_parameters(get_all_gradient(model_dual))
            
            
            v_vec = get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])
            
#             print('para_diff::', torch.norm(v_vec))
#             
#             print('para_angle::', torch.dot(get_all_vectorized_parameters(para).view(-1), get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))/(torch.norm(get_all_vectorized_parameters(para).view(-1))*torch.norm(get_all_vectorized_parameters(para_list_all_epochs[i]).view(-1))))
            
            
            
#             init_model(model, para_list_all_epochs[i])
             
#             curr_vec_gradient,_ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, gradient_list_all_epochs[i], beta)
             
#             hessian_matrix1 = compute_hessian_matrix(model, curr_vec_gradient, input_dim, hidden_dims, output_dim)
            
            
            hessian_para_prod = cal_approx_hessian_vec_prod0(S_k_list, Y_k_list, i, 5, m, v_vec.view(-1,1), 10)
            
#             hessian_para_prod = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1))
            
#             hessian_para_prod = cal_approx_hessian_vec_prod2(S_k_list, Y_k_list, i, m-1, v_vec.view(-1,1), last_v_vec.view(-1,1), last_gradient_full.view(-1, 1) - get_all_vectorized_parameters(gradient_list_all_epochs[i-1]).view(-1,1))
            
#             hessian_para_prod = cal_approx_hessian_vec_prod3(i, m, v_vec.view(-1,1), para_list_all_epochs, gradient_list_all_epochs)
            
#             hessian_para_prod, tmp_res = cal_approx_hessian_vec_prod4(truncted_s, extended_Y_k_list[i], i, m, v_vec.view(-1,1))
            
            
            delta_const = 0#compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
            
#             para_diff = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
            
            gradient_full = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
            
#             delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1), error, model, beta)
#             gradient_full2 = torch.mm(v_vec.view(1,-1), hessian_matrix1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
#             gradient_full2 = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) - delta_const*v_vec.view(1,-1)
            
            gradients = (gradient_full*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
            
#             gradients2 = (gradient_full2*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
            
#             print('hessian_vector_prod_diff::', torch.norm(torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1) - hessian_para_prod))
            
#             print('gradient_diff::', torch.norm(gradients - expect_gradients))
            
#             print('gradient2_diff::', torch.norm(gradients2 - expect_gradients))
            
#             compute_model_para_diff(exp_para_list_all_epochs[i], para)
            
            S_k_list[:,i-1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
            
            para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradients, input_dim, hidden_dims, output_dim)
        
            Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
            
#             last_gradient_full = gradient_full
#             
#             last_v_vec = v_vec.clone()
        
        
#         else:
#             if i >= 1:
#                 S_k_list[:,i - 1] = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i])).view(-1)
#             last_para = para
#             
#             para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*expect_gradients, input_dim, hidden_dims, output_dim)
#         
#             if i == m-1:
#                 init_model(model, para)
#                 
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             
#                 last_gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
#                 
#                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
#                 
#             if i >= 1:
#                 
#                 init_model(model, para)
#                 
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             
#                 gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
#                 
# #                 last_v_vec = get_all_vectorized_parameters(last_para) - get_all_vectorized_parameters(para_list_all_epochs[i])
#                 
#                 Y_k_list[:,i-1] = (gradient_full - get_all_vectorized_parameters(gradient_list_all_epochs[i])).view(-1)
        
#         last_gradient = expect_gradients
#             init_model(model, get_devectorized_parameters(updated_para, input_dim, hidden_dims, output_dim))
            
            
            
            
            
            
            
    return para




# def model_update_provenance_test2(exp_gradient_list_all_epochs, exp_para_list_all_epochs, X, Y, model, S_k_list, Y_k_list, gradient_list_all_epochs, para_list_all_epochs, epoch, delta_ids, input_dim, hidden_dims, output_dim, m, alpha, beta, selected_rows, error):
#     
#     
#     para_num = S_k_list.shape[0]
#     
#     
#     model_dual = DNNModel(input_dim, hidden_dims, output_dim)
#     
#     
#     para = list(model.parameters())
#     
#     expected_para = list(model.parameters())
#     
#     
#     gradient_full = None
#     
#     last_para = None
#     
#     for i in range(epoch):
#         
#         init_model(model, para)
#         
#         compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
#         
#         expect_gradients = get_all_vectorized_parameters(get_all_gradient(model))
#         
#         
#         init_model(model, expected_para)
#         
#         compute_derivative_one_more_step(model, error, X[selected_rows], Y[selected_rows], beta)
#         
# #         compute_model_para_diff(expected_para, exp_para_list_all_epochs[i])
# 
#         
#         expected_para = get_devectorized_parameters(get_all_vectorized_parameters(expected_para) - alpha*get_all_vectorized_parameters(get_all_gradient(model)), input_dim, hidden_dims, output_dim)
#         
#         
#         if i >= m + 1:
#         
#             init_model(model_dual, para)
#              
#             compute_derivative_one_more_step(model_dual, error, X[delta_ids], Y[delta_ids], beta)
#              
#             gradient_dual = get_all_vectorized_parameters(get_all_gradient(model_dual))
# #             
# #             
# #             
# #             
# #             print('para_diff::', torch.norm(v_vec))
# #             
# #             
# #             
# #             init_model(model, para_list_all_epochs[i])
# #              
# #             curr_vec_gradient,_ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, gradient_list_all_epochs[i], beta)
# #              
# # #             hessian_matrix1 = compute_hessian_matrix(model, curr_vec_gradient, input_dim, hidden_dims, output_dim)
# #             
# #             
# #             
# #             
# #             hessian_para_prod = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1))
#             
# #             delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), hessian_para_prod, error, model, beta)
# #             
# # #             para_diff = (get_all_vectorized_parameters(para) - get_all_vectorized_parameters(para_list_all_epochs[i]))
# #             
# #             gradient_full = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
# #             
# #             delta_const = compute_delta_constant(X, Y, para_list_all_epochs[i], para, get_all_vectorized_parameters(gradient_list_all_epochs[i]), torch.mm(v_vec.view(1,-1), hessian_matrix1).view(-1,1), error, model, beta)
# #             gradient_full2 = torch.mm(v_vec.view(1,-1), hessian_matrix1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) + delta_const*v_vec.view(1,-1)
# # #             gradient_full2 = hessian_para_prod.view(1,-1) + get_all_vectorized_parameters(gradient_list_all_epochs[i]) - delta_const*v_vec.view(1,-1)
# #             
#             gradients = (gradient_full*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
# #             
# #             gradients2 = (gradient_full2*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
# #             
# #             print('gradient_diff::', torch.norm(gradients - expect_gradients))
# #             
# #             print('gradient2_diff::', torch.norm(gradients2 - expect_gradients))
# #             
# #             compute_model_para_diff(exp_para_list_all_epochs[i], para)
# 
# 
#             last_para = para
# 
# 
# #             gradients = (gradient_full*X.shape[0] - gradient_dual*delta_ids.shape[0])/(X.shape[0] - delta_ids.shape[0])
# 
#             
#             para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*gradients, input_dim, hidden_dims, output_dim)
#         
#             compute_model_para_diff(exp_para_list_all_epochs[i], para)
#         
#         
#             v_vec = get_all_vectorized_parameters(para) - get_all_vectorized_parameters(last_para)
#         
#             
#             gradient_full = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1)) + gradient_full
#             
#             
#             init_model(model, para)
#         
#             compute_derivative_one_more_step(model, error, X, Y, beta)
#             
#             expected_gradient_full = get_all_gradient(model)
#             
#             print('gradient_diff::')
#             
#             compute_model_para_diff(get_devectorized_parameters(gradient_full, input_dim, hidden_dims, output_dim), expected_gradient_full)
#         
#         else:
#             
# #             last_para = para.clone()
#             
#             if i == m:
#                 
# #                 gradients = expect_gradients
#                 last_para = para
#             
#             para = get_devectorized_parameters(get_all_vectorized_parameters(para) - alpha*expect_gradients, input_dim, hidden_dims, output_dim)
#         
#         
#             if i == m:
# #                 v_vec = get_all_vectorized_parameters(para) - get_all_vectorized_parameters(last_para)
#                 
#                 init_model(model, para)
#         
#                 compute_derivative_one_more_step(model, error, X, Y, beta)
#                 
#                 gradient_full = get_all_vectorized_parameters(get_all_gradient(model))
#                 
# #                 gradient_full = cal_approx_hessian_vec_prod(S_k_list, Y_k_list, i, m, v_vec.view(-1,1)) + expect_gradients
#         
#         
#         
# #             init_model(model, get_devectorized_parameters(updated_para, input_dim, hidden_dims, output_dim))
#             
#             
#             
#             
#             
#             
#             
#     return para



def model_update_provenance_cp0(alpha, X, Y, hessian_matrix, origin_gradient_list, vectorized_orign_params, epoch, model, dim, w_list, b_list, input_dim, hidden_dims, output_dim, delta_ids, expected_gradient_list_all_epochs, expected_para_list_all_epochs_all_epochs, selelcted_rows):

    vectorized_gradient = get_all_vectorized_parameters(origin_gradient_list) 
    
    t1  = time.time()
    
    para_list = list(model.parameters())
    
    depth = len(hidden_dims) + 1
    
    delta_id_num = delta_ids.shape[0]
    
    old_vec_gradient_list = None
    
    old_vec_para_list = None
    
    error = nn.CrossEntropyLoss()
    
    model_dual = DNNModel(input_dim, hidden_dims, output_dim)
    
    for i in range(epoch):
    
#         output_list,input_to_non_linear_layer_list = model.get_output_each_layer(X[delta_ids])
#         
#         input_to_non_linear_layer_list = input_to_non_linear_layer_list[1:]
        
        para_list = list(model.parameters())

        
        
        init_model(model_dual, para_list)

        
        compute_derivative_one_more_step(model_dual, error, X[delta_ids], Y[delta_ids])
        
        
        gradient_dual = get_all_gradient(model_dual)
        
        
        
        curr_vectorized_params = get_all_vectorized_parameters(para_list)        
        
        delta_vectorized_gradient_parameters = torch.mm(hessian_matrix, (curr_vectorized_params - vectorized_orign_params).view(-1, 1)).view(1,-1)
        
        gradient_list = get_devectorized_parameters(delta_vectorized_gradient_parameters + vectorized_gradient, input_dim, hidden_dims, output_dim)
        
        
#         old_vec_para_list = get_all_vectorized_parameters(para_list)
        '''delta: n*output_dim'''
        
        
        '''A: output_dim, hidden_dim[depth-2]^2'''    
        
#         print(depth)
        
        
#         pred = output_list[len(output_list) - 1]
       
        '''A: output_dim, hidden_dims[-1]*hiddem_dims[-1]'''
        
#         input_to_non_linear_layer = input_to_non_linear_layer_list[depth -1]



#         delta = Variable(softmax_func(pred) - get_onehot_y(Y[delta_ids], [delta_id_num, output_dim], output_dim))
#         
#         non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[depth-1][delta_ids] + b_list[depth-1][delta_ids]))
#         
#         delta_A = torch.mm(torch.t(non_linear_output), output_list[depth-1])
        
#         delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(non_linear_output))))
        
        gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
        
        gradient_dual_curr_layer = torch.cat((gradient_dual[2*depth - 2].data, gradient_dual[2*depth - 1].data.view(-1,1)), 1) 
        
        para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 

        curr_A = gradient_curr_layer - gradient_dual_curr_layer*delta_ids.shape[0]

        '''B: output_dim, hidden_dim[depth-2]'''
        
        curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
            
        para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
            
        para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
            
        para_list[2*depth - 1].data = para_curr_layer[:, -1]
        
        '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
    
        for i in range(depth - 2):
            
            '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
            
#             input_to_non_linear_layer = input_to_non_linear_layer_list[depth - i - 2]
#             delta = Variable(delta_para_prod)
# 
#             non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[depth- i - 2][delta_ids] + b_list[depth-i-2][delta_ids]))
#             
#             delta_A = torch.mm(torch.t(non_linear_output), output_list[depth-i-2])
            
            gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
             
            gradient_dual_curr_layer = torch.cat((gradient_dual[2*depth - 2*i - 4].data, gradient_dual[2*depth - 2*i - 3].data.view(-1,1)), 1) 
             
            para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
    
            curr_A = gradient_curr_layer - gradient_dual_curr_layer*delta_ids.shape[0]
            
            '''B: output_dim, hidden_dim[depth-2]'''
#             delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(non_linear_output))))
            
            curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
                
            para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
                
            para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                
            para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
        
#         input_to_non_linear_layer = input_to_non_linear_layer_list[0]
# 
#         delta = Variable(delta_para_prod)
# 
#         non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[0][delta_ids] + b_list[0][delta_ids]))
#         
#         delta_A = torch.mm(torch.t(non_linear_output), output_list[0])

        gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)), 1) 
        
        gradient_dual_curr_layer = torch.cat((gradient_dual[0].data, gradient_dual[1].data.view(-1,1)), 1) 
        
        para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)

        curr_A = gradient_curr_layer - gradient_dual_curr_layer*delta_ids.shape[0]
        
        '''B: output_dim, hidden_dim[depth-2]'''
        
        curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
            
        para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
    
        para_list[0].data = para_curr_layer[:, 0:-1]
            
        para_list[1].data = para_curr_layer[:, -1]
        
        init_model(model, para_list)
        
        '''B: output_dim, hidden_dim[depth-2]'''

    t2  = time.time()
    
    print("time0:", t2 - t1)   
    
    return para_list



def model_update_provenance_cp0_stochastic(batch_size, alpha, X, Y, hessian_matrix, origin_gradient_list, vectorized_orign_params, epoch, model, dim, w_list, b_list, input_dim, hidden_dims, output_dim, delta_ids, expected_gradient_list_all_epochs, expected_para_list_all_epochs_all_epochs, selelcted_rows):

    vectorized_gradient = get_all_vectorized_parameters(origin_gradient_list) 
    
    t1  = time.time()
    
    para_list = list(model.parameters())
    
    depth = len(hidden_dims) + 1
    
    delta_id_num = delta_ids.shape[0]
    
    old_vec_gradient_list = None
    
    old_vec_para_list = None
    
    error = nn.CrossEntropyLoss()
    
    model_dual = DNNModel(input_dim, hidden_dims, output_dim)
    
    
    delta_X = X[delta_ids]
    
    delta_Y = Y[delta_ids]
    
    for i in range(epoch):
        
        
        random_ids = torch.randperm(delta_ids.shape[0])
        
        delta_X = delta_X[random_ids]
        
        delta_Y = delta_Y[random_ids]
        
        
        for i in range(0, delta_ids.shape[0], batch_size):
            
            
            end_id = i + batch_size
            
            if end_id > delta_ids.shape[0]:
                end_id = delta_ids.shape[0]
        
        
        
        
        
        
        
            delta_X_curr_batch = delta_X[i: end_id]
            
            delta_Y_curr_batch = delta_Y[i: end_id]
        
        
        
        
        
        
    
#         output_list,input_to_non_linear_layer_list = model.get_output_each_layer(X[delta_ids])
#         
#         input_to_non_linear_layer_list = input_to_non_linear_layer_list[1:]
        
            para_list = list(model.parameters())
    
            
            
            init_model(model_dual, para_list)
    
            
            compute_derivative_one_more_step(model_dual, error, delta_X_curr_batch, delta_Y_curr_batch)
            
            
            gradient_dual = get_all_gradient(model_dual)
            
            
            
            curr_vectorized_params = get_all_vectorized_parameters(para_list)        
            
            delta_vectorized_gradient_parameters = torch.mm(hessian_matrix, (curr_vectorized_params - vectorized_orign_params).view(-1, 1)).view(1,-1)
            
            gradient_list = get_devectorized_parameters(delta_vectorized_gradient_parameters + vectorized_gradient, input_dim, hidden_dims, output_dim)
            
            '''delta: n*output_dim'''
            
            
            '''A: output_dim, hidden_dim[depth-2]^2'''    
           
            '''A: output_dim, hidden_dims[-1]*hiddem_dims[-1]'''
            
           
            gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
            
            gradient_dual_curr_layer = torch.cat((gradient_dual[2*depth - 2].data, gradient_dual[2*depth - 1].data.view(-1,1)), 1) 
            
            para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 
    
            curr_A = gradient_curr_layer - gradient_dual_curr_layer*delta_ids.shape[0]
    
            '''B: output_dim, hidden_dim[depth-2]'''
            
            curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
                
            para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
                
            para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
                
            para_list[2*depth - 1].data = para_curr_layer[:, -1]
            
            '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
        
            for i in range(depth - 2):
                
                '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
                gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
                 
                gradient_dual_curr_layer = torch.cat((gradient_dual[2*depth - 2*i - 4].data, gradient_dual[2*depth - 2*i - 3].data.view(-1,1)), 1) 
                 
                para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
        
                curr_A = gradient_curr_layer - gradient_dual_curr_layer*delta_ids.shape[0]
                
                '''B: output_dim, hidden_dim[depth-2]'''
                
                curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
                    
                para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
                    
                para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                    
                para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
            
            gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)), 1) 
            
            gradient_dual_curr_layer = torch.cat((gradient_dual[0].data, gradient_dual[1].data.view(-1,1)), 1) 
            
            para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)
    
            curr_A = gradient_curr_layer - gradient_dual_curr_layer*delta_ids.shape[0]
            
            '''B: output_dim, hidden_dim[depth-2]'''
            
            curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
                
            para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
        
            para_list[0].data = para_curr_layer[:, 0:-1]
                
            para_list[1].data = para_curr_layer[:, -1]
            
            init_model(model, para_list)
            
            '''B: output_dim, hidden_dim[depth-2]'''

    t2  = time.time()
    
    print("time0:", t2 - t1)   
    
    return para_list


def model_update_provenance_cp(alpha, X, Y, hessian_matrix, origin_gradient_list, vectorized_orign_params, epoch, model, dim, w_list, b_list, input_dim, hidden_dims, output_dim, delta_ids, expected_gradient_list_all_epochs, expected_para_list_all_epochs_all_epochs, selelcted_rows):
    
    
#     loss = np.infty
#     
#     count = 0
#     
#     
#     output_list = model.get_output_each_layer(X)
#     
#     para_list = model.parameters()
# 
#     while loss > loss_threshold and count < num_epochs:
#         
#         delta_list = []
#         
#         outer_gradient = softmax_func(output_list[len(output_list) - 1]) - get_onehot_y(Y, dim, num_class)       
#         
#         para_list_len = len(para_list)
#         
#         
#         for i in range(para_list_len):
#             
#             
#             '''w_res: n * hidden[len-i], b_res: n*1, para_list[2*para_list_len - 2*i]: hidden[len-i]*hidden[len-i-1], output_list[len-i]: n*hidden[len-i-1]'''
#             
#             
#             w_res[para_list_len - i]*torch.sum(para_list[2*para_list_len - 2*i]*output_list[para_list_len - i], 2) + b_res[para_list_len - i]
#             
#             
#             
#             softmax_func()

    vectorized_gradient = get_all_vectorized_parameters(origin_gradient_list) 
    
    t1  = time.time()
    
    para_list = list(model.parameters())
    
    depth = len(hidden_dims) + 1
    
#     curr_gradient_list = get_all_vectorized_parameters(para_list)
    
    delta_id_num = delta_ids.shape[0]
    
    old_vec_gradient_list = None
    
    old_vec_para_list = None
    
    for i in range(epoch):
        
    
    
#         curr_exp_gradient_list = expected_gradient_list_all_epochs[i]
#         
#         
#         curr_exp_para_list = expected_para_list_all_epochs_all_epochs[i]
#         
#         
#         
#         expected_full_gradient_list, _ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, para_list)
#         
#         hessian_matrix2 = compute_hessian_matrix(model, expected_full_gradient_list, input_dim, hidden_dims, output_dim)
#         
#         hessian_matrix = (hessian_matrix2)
#         
#         vectorized_gradient = get_all_vectorized_parameters(expected_full_gradient_list)
        
#         vectorized_expected_full_gradient = get_all_vectorized_parameters(expected_full_gradient_list)/X.shape[0]
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
#     pred = output_list[len(output_list) - 1]
    
        output_list,input_to_non_linear_layer_list = model.get_output_each_layer(X[delta_ids])
        
        input_to_non_linear_layer_list = input_to_non_linear_layer_list[1:]
        
        para_list = list(model.parameters())

        
        curr_vectorized_params = get_all_vectorized_parameters(para_list)        
        
        delta_vectorized_gradient_parameters = torch.mm(hessian_matrix, (curr_vectorized_params - vectorized_orign_params).view(-1, 1)).view(1,-1)
        
        gradient_list = get_devectorized_parameters(delta_vectorized_gradient_parameters + vectorized_gradient, input_dim, hidden_dims, output_dim)
        
        
        old_vec_para_list = get_all_vectorized_parameters(para_list)
#         vectorized_gradient_list = get_all_vectorized_parameters(gradient_list)/X.shape[0]
        
    
    #         loss = error(pred, Y)
        
    #     delta = softmax_func(pred) - get_onehot_y(Y, dim, num_class)
        
        
        '''delta: n*output_dim'''
        
        
        '''A: output_dim, hidden_dim[depth-2]^2'''    
    #     print(w_res[depth - 1])
        
        print(depth)
        
        
        pred = output_list[len(output_list) - 1]
#         delta_A_list = [None]*depth
#         
#     #     delta_A_list0 = [None]*depth
#         
#         
#         delta_B_list = [None]*depth
        
    #     delta_B_list0 = [None]*depth
        
    
#         w_delta_prod = w_delta_prod_list[depth - 1][delta_ids]
#         
#         b_delta_prod = b_delta_prod_list[depth - 1][delta_ids]
        
        '''A: output_dim, hidden_dims[-1]*hiddem_dims[-1]'''
        
        input_to_non_linear_layer = input_to_non_linear_layer_list[depth -1]



        delta = Variable(softmax_func(pred) - get_onehot_y(Y[delta_ids], [delta_id_num, output_dim], output_dim))
        
#         derivative_non_linear_layer = Variable(w_list[delta_ids]*input_to_non_linear_layer + b_list[delta_ids])
        
        
        non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[depth-1][delta_ids] + b_list[depth-1][delta_ids]))
        
        
        
        delta_A = torch.mm(torch.t(non_linear_output), output_list[depth-1])
        
        
#         non_linear_output = Variable(delta*derivative_non_linear_layer[:, torch.sum(hidden_dim_tensor):torch.sum(hidden_dim_tensor)+output_dim])
        
        
        
        delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(non_linear_output))))
        
#         input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
            
#         delta_A = Variable(torch.mm(torch.t((w_delta_prod*input_to_non_linear_layer + b_delta_prod)), output_list[depth-1]))
        
        gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
        
#         gradient_curr_layer_curr_epoch = torch.cat((curr_exp_gradient_list[2*depth - 2].data, curr_exp_gradient_list[2*depth - 1].data.view(-1,1)), 1)
#         
#         
#         exp_para_curr_layer_curr_epoch = torch.cat((curr_exp_para_list[2*depth - 2].data, curr_exp_para_list[2*depth - 1].data.view(-1,1)), 1)
        
        para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 

        curr_A = gradient_curr_layer - delta_A
#         delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
        
    #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
        
#         delta_A_list[depth - 1] = delta_A
        
    #     delta_A_list0.append(delta_A0)
        
        '''B: output_dim, hidden_dim[depth-2]'''
#         delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-1])
        
#         delta_B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-1]))
        
#         curr_B = Variable(0 - delta_B)
        
        curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
            
#         curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
        
        para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
            
    #         para_curr_layer = para_curr_layer - alpha/
    
        para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
            
        para_list[2*depth - 1].data = para_curr_layer[:, -1]
        
    #     delta_B0 = torch.t(b_delta_prod)
        
    #     delta_B_list0.append(delta_B0)
        
#         delta_B_list[depth - 1] = delta_B
        
        
    #     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1])))
    #     
    #     
    #     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
        
        
    #     delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
    #     
    #     deriv = torch.mm(torch.t(delta), output_list[depth - 1])
    
        '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
    
    #     delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(entire_delta_expression)))
        
    #     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
    #     
    #     para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
    #     
    #     para_list[2*depth - 1].data = para_curr_layer[:, -1]
            
        for i in range(depth - 2):
            
            '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
            
    #         para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
    #         delta = delta_para_prod#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
#             w_delta_prod = w_delta_prod_list[depth - i - 2][delta_ids]
#         
#             b_delta_prod = b_delta_prod_list[depth - i - 2][delta_ids]
            
            input_to_non_linear_layer = input_to_non_linear_layer_list[depth - i - 2]

        
            delta = Variable(delta_para_prod)
        
#         input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
            non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[depth- i - 2][delta_ids] + b_list[depth-i-2][delta_ids]))
            
            delta_A = torch.mm(torch.t(non_linear_output), output_list[depth-i-2])
            
#             delta_A = Variable(torch.mm(torch.t((w_delta_prod*input_to_non_linear_layer + b_delta_prod)), output_list[depth- i - 2]))
            
            gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
            
            
#             gradient_curr_layer_curr_epoch = torch.cat((curr_exp_gradient_list[2*depth - 2*i - 4].data, curr_exp_gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
            
             
            para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
    
            curr_A = gradient_curr_layer - delta_A
    #         delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
            
        #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
            
    #         delta_A_list[depth - 1] = delta_A
            
        #     delta_A_list0.append(delta_A0)
            
            '''B: output_dim, hidden_dim[depth-2]'''
    #         delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-1])
            
#             delta_B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth- i - 2]))
#             
#             curr_B = Variable(0 - delta_B)
            
            
            delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(non_linear_output))))
            
            
            curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
                
#             curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
            
            para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
                
        #         para_curr_layer = para_curr_layer - alpha/
        
            para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                
            para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
#             delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - i - 2].view(delta_id_num, hidden_dims[depth - i - 3] + 1, 1), output_list[depth - i - 2].view(delta_id_num, 1, hidden_dims[depth - i - 3] + 1)).view(delta_id_num, (hidden_dims[depth-i-3]+1)*(hidden_dims[depth-i-3]+1)))
#             
#     #         delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - i - 2][delta_ids])
#             
#             delta_A_list[depth - i - 2] = delta_A
#             
#     #         delta_A_list0.append(delta_A0)
#             
#             '''B: output_dim, hidden_dim[depth-2]'''
#             delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-i-2])
#             
#     #         delta_B0 = torch.t(b_delta_prod)
#             
#     #         delta_B_list0.append(delta_B0)
#             
#             
#             delta_B_list[depth - i - 2] = delta_B
            
    #         w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - i - 2])))
    #         
    #         
    #         entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    #         
    #         
    #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(entire_delta_expression)))
    #         
    #         A_list[depth - 2 - i] = A
    #         
    #         B_list[depth - 2 - i] = B
            
    #         deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])
    #         
    #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))                    
    #         
    #         '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
    #         para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
    #     
    #         para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
    #         
    #         para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
                
    #     para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1)
    
    #     delta = delta_para_prod#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
        
#         w_delta_prod = w_delta_prod_list[0][delta_ids]
#     
#         b_delta_prod = b_delta_prod_list[0][delta_ids]
        
        
        input_to_non_linear_layer = input_to_non_linear_layer_list[0]


        delta = Variable(delta_para_prod)
        
#         input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
        non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[0][delta_ids] + b_list[0][delta_ids]))
        
        delta_A = torch.mm(torch.t(non_linear_output), output_list[0])

        
#         input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
            
#         delta_A = Variable(torch.mm(torch.t((w_delta_prod*input_to_non_linear_layer + b_delta_prod)), output_list[0]))
        
        gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)), 1) 
        
#         gradient_curr_layer_curr_epoch = torch.cat((curr_exp_gradient_list[0].data, curr_exp_gradient_list[1].data.view(-1,1)), 1)
        
        para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)

        curr_A = gradient_curr_layer - delta_A
#         delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
        
    #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
        
#         delta_A_list[depth - 1] = delta_A
        
    #     delta_A_list0.append(delta_A0)
        
        '''B: output_dim, hidden_dim[depth-2]'''
#         delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-1])
        
#         delta_B = Variable(torch.mm(torch.t(b_delta_prod), output_list[0]))
#         
#         curr_B = Variable(0 - delta_B)
        
        curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
            
#         curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
        
        para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
            
    #         para_curr_layer = para_curr_layer - alpha/
    
        para_list[0].data = para_curr_layer[:, 0:-1]
            
        para_list[1].data = para_curr_layer[:, -1]
        
        init_model(model, para_list)
        
#         if old_vec_gradient_list is not None:
#             hessian_matrix = update_hessian(hessian_matrix, old_vec_para_list, get_all_vectorized_parameters(para_list), old_vec_gradient_list, get_all_vectorized_parameters(gradient_list))
#         
#         
#             vectorized_orign_params = curr_vectorized_params
#             
#             vectorized_gradient = get_all_vectorized_parameters(gradient_list)
#         
#         old_vec_gradient_list = get_all_vectorized_parameters(gradient_list)
        
        
        
        
        
#         delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[0].view(delta_id_num, input_dim + 1, 1), output_list[0].view(delta_id_num, 1, input_dim + 1)).view(delta_id_num, (input_dim+1)*(input_dim + 1)))
#     
#         delta_A = delta_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)
    
    #     print(w_delta_prod.shape)
    
    #     delta_A = torch.mm(torch.t(output_list[0][delta_ids]), torch.bmm(w_delta_prod.view(delta_id_num, hidden_dims[0], 1), output_list[0][delta_ids].view(delta_id_num, 1, input_dim + 1)).view(delta_id_num, hidden_dims[0]*(input_dim + 1)))
    #     
    #     delta_A = torch.transpose(delta_A.view(input_dim + 1, hidden_dims[0], input_dim + 1), 0, 1)
        
    #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[0][delta_ids])
        
    #     delta_A_list0.append(delta_A0)
        
        '''B: output_dim, hidden_dim[depth-2]'''
#         delta_B = torch.mm(torch.t(b_delta_prod), output_list[0])
#         
#     #     delta_B0 = torch.t(b_delta_prod)
#         
#     #     delta_B_list0.append(delta_B0)
#         
#         
#         delta_A_list[0] = delta_A
#         
#         delta_B_list[0] = delta_B
#     
#     
#     #     weights = para_list[2*depth - 2].data
#     #         
#     #     offsets = para_list[2*depth - 1].data
#         
#         
#         para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 
#         
#         
#         '''A: (output_dim*hidden_dims[depth-1])*hiddem_dims[depth-1]'''
#     
#         '''B: output_dim*hidden_dims[depth-2]'''
#         
#         '''weights: output_dim*hidden_dims[depth-1]'''
#         
#         curr_A = A_list[depth - 1] - delta_A_list[depth - 1] 
#         
#         curr_B = B_list[depth - 1] - delta_B_list[depth - 1]
#         
# #         for j in range(epoch):
#             
#         '''output_dim, 1, hidden_dims[depth-2]'''
#         
# #         curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
#         
#         curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
#         
#         curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
#         
#         
#         gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
#         
#         delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#         
#         para_curr_layer = para_curr_layer - alpha*curr_gradient
#             
#     #         para_curr_layer = para_curr_layer - alpha/
#     
#         para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
#             
#         para_list[2*depth - 1].data = para_curr_layer[:, -1]
#         
#         for i in range(depth-2):
#             
#             para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
#                     
#             '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
#     
#             '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#             
#             '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#             
#             curr_A = A_list[depth - i - 2] - delta_A_list[depth - i - 2] 
#             
#             curr_B = B_list[depth - i - 2] - delta_B_list[depth - i - 2]
#             
#             for j in range(epoch):
#             
#                 gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)),1)
#                 
#                 curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[depth-i-2], 1, hidden_dims[depth-i-3] + 1), curr_A.view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1, hidden_dims[depth-3-i] + 1)).view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1)
#                 
#                 curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
#                 
#                 delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#                 
#                 para_curr_layer = para_curr_layer - alpha*curr_gradient
#             
#             
#             para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
#             
#             para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
#         
#         
#         para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)
#                     
#         '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
#     
#         '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#         
#         '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#         
#         curr_A = A_list[0] - delta_A_list[0] 
#         
#         curr_B = B_list[0] - delta_B_list[0]
#         
#         for j in range(epoch):
#             
#             gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)),1)
#             
#             curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), curr_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
#             
#             curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
#             
#             delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#                     
#             para_curr_layer = para_curr_layer - alpha*curr_gradient
#         
#         
#         para_list[0].data = para_curr_layer[:, 0:-1]
#         
#         para_list[1].data = para_curr_layer[:, -1]

    
    
#     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[0])))
#     
#     
#     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    
#     deriv = torch.mm(torch.t(delta), output_list[0])
#     
#     '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
# 
#     para_list[0].data = para_curr_layer[:, 0:-1]
#     
#     para_list[1].data = para_curr_layer[:, -1]

    t2  = time.time()
    
    print("time0:", t2 - t1)   
    
    return para_list

def model_update_provenance_cp2(alpha, X, Y, epoch, model, dim, w_list_all_epochs, b_list_all_epochs, input_dim, hidden_dims, output_dim, selected_rows, exp_gradient_list_all_epochs, exp_para_list_all_epochs):
    
    
#     loss = np.infty
#     
#     count = 0
#     
#     
#     output_list = model.get_output_each_layer(X)
#     
#     para_list = model.parameters()
# 
#     while loss > loss_threshold and count < num_epochs:
#         
#         delta_list = []
#         
#         outer_gradient = softmax_func(output_list[len(output_list) - 1]) - get_onehot_y(Y, dim, num_class)       
#         
#         para_list_len = len(para_list)
#         
#         
#         for i in range(para_list_len):
#             
#             
#             '''w_res: n * hidden[len-i], b_res: n*1, para_list[2*para_list_len - 2*i]: hidden[len-i]*hidden[len-i-1], output_list[len-i]: n*hidden[len-i-1]'''
#             
#             
#             w_res[para_list_len - i]*torch.sum(para_list[2*para_list_len - 2*i]*output_list[para_list_len - i], 2) + b_res[para_list_len - i]
#             
#             
#             
#             softmax_func()

#     vectorized_gradient = get_all_vectorized_parameters(origin_gradient_list) 
    
    t1  = time.time()
    
    para_list = list(model.parameters())
    
    depth = len(hidden_dims) + 1
    
#     curr_gradient_list = get_all_vectorized_parameters(para_list)
    
#     delta_id_num = delta_ids.shape[0]
    
    selected_row_num = selected_rows.shape[0]
    
    for i in range(epoch):
        
    
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
#     pred = output_list[len(output_list) - 1]

        if i >= cut_off_epoch:
            w_list = w_list_all_epochs[cut_off_epoch-1]
            
            b_list = b_list_all_epochs[cut_off_epoch-1]
        else:
            w_list = w_list_all_epochs[i]
            
            b_list = b_list_all_epochs[i]
    
        output_list,input_to_non_linear_layer_list = model.get_output_each_layer(X[selected_rows])
        
        input_to_non_linear_layer_list = input_to_non_linear_layer_list[1:]
        
        para_list = list(model.parameters())

        
#         curr_vectorized_params = get_all_vectorized_parameters(para_list)
#         
#         
#         
#         delta_vectorized_gradient_parameters = torch.mm(curr_vectorized_params - vectorized_orign_params, hessian_matrix)
#         
#         gradient_list = get_devectorized_parameters(delta_vectorized_gradient_parameters + vectorized_gradient, input_dim, hidden_dims, output_dim)
#         gradient_list = exp_gradient_list_all_epochs[i]
        
    
    #         loss = error(pred, Y)
        
    #     delta = softmax_func(pred) - get_onehot_y(Y, dim, num_class)
        
        
        '''delta: n*output_dim'''
        
        
        '''A: output_dim, hidden_dim[depth-2]^2'''    
    #     print(w_res[depth - 1])
        
#         print(depth)
        
        
        pred = output_list[len(output_list) - 1]
#         delta_A_list = [None]*depth
#         
#     #     delta_A_list0 = [None]*depth
#         
#         
#         delta_B_list = [None]*depth
        
    #     delta_B_list0 = [None]*depth
        
    
#         w_delta_prod = w_delta_prod_list[depth - 1][delta_ids]
#         
#         b_delta_prod = b_delta_prod_list[depth - 1][delta_ids]
        
        '''A: output_dim, hidden_dims[-1]*hiddem_dims[-1]'''
        
        input_to_non_linear_layer = input_to_non_linear_layer_list[depth -1]



        delta = Variable(softmax_func(pred) - get_onehot_y(Y[selected_rows], [selected_row_num, output_dim], output_dim))
        
#         derivative_non_linear_layer = Variable(w_list[delta_ids]*input_to_non_linear_layer + b_list[delta_ids])
        
        
        non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[depth-1][selected_rows] + b_list[depth-1][selected_rows]))
        
        
        
        delta_A = torch.mm(torch.t(non_linear_output), output_list[depth-1])
        
        
#         non_linear_output = Variable(delta*derivative_non_linear_layer[:, torch.sum(hidden_dim_tensor):torch.sum(hidden_dim_tensor)+output_dim])
        
        
        
        delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(non_linear_output))))
        
#         input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
            
#         delta_A = Variable(torch.mm(torch.t((w_delta_prod*input_to_non_linear_layer + b_delta_prod)), output_list[depth-1]))
        
#         gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
        para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 

        curr_A = delta_A
#         delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
        
    #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
        
#         delta_A_list[depth - 1] = delta_A
        
    #     delta_A_list0.append(delta_A0)
        
        '''B: output_dim, hidden_dim[depth-2]'''
#         delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-1])
        
#         delta_B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-1]))
        
#         curr_B = Variable(0 - delta_B)
        
        curr_gradient = Variable((1.0/(selected_row_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
            
#         curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
        
        para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
            
    #         para_curr_layer = para_curr_layer - alpha/
    
        para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
            
        para_list[2*depth - 1].data = para_curr_layer[:, -1]
        
    #     delta_B0 = torch.t(b_delta_prod)
        
    #     delta_B_list0.append(delta_B0)
        
#         delta_B_list[depth - 1] = delta_B
        
        
    #     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1])))
    #     
    #     
    #     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
        
        
    #     delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
    #     
    #     deriv = torch.mm(torch.t(delta), output_list[depth - 1])
    
        '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
    
    #     delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(entire_delta_expression)))
        
    #     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
    #     
    #     para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
    #     
    #     para_list[2*depth - 1].data = para_curr_layer[:, -1]
            
        for i in range(depth - 2):
            
            '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
            
    #         para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
    #         delta = delta_para_prod#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
#             w_delta_prod = w_delta_prod_list[depth - i - 2][delta_ids]
#         
#             b_delta_prod = b_delta_prod_list[depth - i - 2][delta_ids]
            
            input_to_non_linear_layer = input_to_non_linear_layer_list[depth - i - 2]

        
            delta = Variable(delta_para_prod)
        
#         input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
            non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[depth- i - 2][selected_rows] + b_list[depth-i-2][selected_rows]))
            
            delta_A = torch.mm(torch.t(non_linear_output), output_list[depth-i-2])
            
#             delta_A = Variable(torch.mm(torch.t((w_delta_prod*input_to_non_linear_layer + b_delta_prod)), output_list[depth- i - 2]))
            
#             gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1) 
            para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
    
            curr_A = delta_A
    #         delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
            
        #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
            
    #         delta_A_list[depth - 1] = delta_A
            
        #     delta_A_list0.append(delta_A0)
            
            '''B: output_dim, hidden_dim[depth-2]'''
    #         delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-1])
            
#             delta_B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth- i - 2]))
#             
#             curr_B = Variable(0 - delta_B)
            
            
            delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(non_linear_output))))
            
            
            curr_gradient = Variable((1.0/(selected_row_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
                
#             curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
            
            para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
                
        #         para_curr_layer = para_curr_layer - alpha/
        
            para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                
            para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
#             delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - i - 2].view(delta_id_num, hidden_dims[depth - i - 3] + 1, 1), output_list[depth - i - 2].view(delta_id_num, 1, hidden_dims[depth - i - 3] + 1)).view(delta_id_num, (hidden_dims[depth-i-3]+1)*(hidden_dims[depth-i-3]+1)))
#             
#     #         delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - i - 2][delta_ids])
#             
#             delta_A_list[depth - i - 2] = delta_A
#             
#     #         delta_A_list0.append(delta_A0)
#             
#             '''B: output_dim, hidden_dim[depth-2]'''
#             delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-i-2])
#             
#     #         delta_B0 = torch.t(b_delta_prod)
#             
#     #         delta_B_list0.append(delta_B0)
#             
#             
#             delta_B_list[depth - i - 2] = delta_B
            
    #         w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - i - 2])))
    #         
    #         
    #         entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    #         
    #         
    #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(entire_delta_expression)))
    #         
    #         A_list[depth - 2 - i] = A
    #         
    #         B_list[depth - 2 - i] = B
            
    #         deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])
    #         
    #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))                    
    #         
    #         '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
    #         para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
    #     
    #         para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
    #         
    #         para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
                
    #     para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1)
    
    #     delta = delta_para_prod#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
        
#         w_delta_prod = w_delta_prod_list[0][delta_ids]
#     
#         b_delta_prod = b_delta_prod_list[0][delta_ids]
        
        
        input_to_non_linear_layer = input_to_non_linear_layer_list[0]


        delta = Variable(delta_para_prod)
        
#         input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
        non_linear_output = Variable(delta*(input_to_non_linear_layer*w_list[0][selected_rows] + b_list[0][selected_rows]))
        
        delta_A = torch.mm(torch.t(non_linear_output), output_list[0])

        
#         input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
            
#         delta_A = Variable(torch.mm(torch.t((w_delta_prod*input_to_non_linear_layer + b_delta_prod)), output_list[0]))
        
#         gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)), 1) 
        para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)

        curr_A = delta_A
#         delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
        
    #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
        
#         delta_A_list[depth - 1] = delta_A
        
    #     delta_A_list0.append(delta_A0)
        
        '''B: output_dim, hidden_dim[depth-2]'''
#         delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-1])
        
#         delta_B = Variable(torch.mm(torch.t(b_delta_prod), output_list[0]))
#         
#         curr_B = Variable(0 - delta_B)
        
        curr_gradient = Variable((1.0/(selected_row_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
            
#         curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
        
        para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
            
    #         para_curr_layer = para_curr_layer - alpha/
    
        para_list[0].data = para_curr_layer[:, 0:-1]
            
        para_list[1].data = para_curr_layer[:, -1]
        
        init_model(model, para_list)
        
        
        
        
        
        
        
#         delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[0].view(delta_id_num, input_dim + 1, 1), output_list[0].view(delta_id_num, 1, input_dim + 1)).view(delta_id_num, (input_dim+1)*(input_dim + 1)))
#     
#         delta_A = delta_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)
    
    #     print(w_delta_prod.shape)
    
    #     delta_A = torch.mm(torch.t(output_list[0][delta_ids]), torch.bmm(w_delta_prod.view(delta_id_num, hidden_dims[0], 1), output_list[0][delta_ids].view(delta_id_num, 1, input_dim + 1)).view(delta_id_num, hidden_dims[0]*(input_dim + 1)))
    #     
    #     delta_A = torch.transpose(delta_A.view(input_dim + 1, hidden_dims[0], input_dim + 1), 0, 1)
        
    #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[0][delta_ids])
        
    #     delta_A_list0.append(delta_A0)
        
        '''B: output_dim, hidden_dim[depth-2]'''
#         delta_B = torch.mm(torch.t(b_delta_prod), output_list[0])
#         
#     #     delta_B0 = torch.t(b_delta_prod)
#         
#     #     delta_B_list0.append(delta_B0)
#         
#         
#         delta_A_list[0] = delta_A
#         
#         delta_B_list[0] = delta_B
#     
#     
#     #     weights = para_list[2*depth - 2].data
#     #         
#     #     offsets = para_list[2*depth - 1].data
#         
#         
#         para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 
#         
#         
#         '''A: (output_dim*hidden_dims[depth-1])*hiddem_dims[depth-1]'''
#     
#         '''B: output_dim*hidden_dims[depth-2]'''
#         
#         '''weights: output_dim*hidden_dims[depth-1]'''
#         
#         curr_A = A_list[depth - 1] - delta_A_list[depth - 1] 
#         
#         curr_B = B_list[depth - 1] - delta_B_list[depth - 1]
#         
# #         for j in range(epoch):
#             
#         '''output_dim, 1, hidden_dims[depth-2]'''
#         
# #         curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
#         
#         curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
#         
#         curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
#         
#         
#         gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
#         
#         delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#         
#         para_curr_layer = para_curr_layer - alpha*curr_gradient
#             
#     #         para_curr_layer = para_curr_layer - alpha/
#     
#         para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
#             
#         para_list[2*depth - 1].data = para_curr_layer[:, -1]
#         
#         for i in range(depth-2):
#             
#             para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
#                     
#             '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
#     
#             '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#             
#             '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#             
#             curr_A = A_list[depth - i - 2] - delta_A_list[depth - i - 2] 
#             
#             curr_B = B_list[depth - i - 2] - delta_B_list[depth - i - 2]
#             
#             for j in range(epoch):
#             
#                 gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)),1)
#                 
#                 curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[depth-i-2], 1, hidden_dims[depth-i-3] + 1), curr_A.view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1, hidden_dims[depth-3-i] + 1)).view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1)
#                 
#                 curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
#                 
#                 delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#                 
#                 para_curr_layer = para_curr_layer - alpha*curr_gradient
#             
#             
#             para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
#             
#             para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
#         
#         
#         para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)
#                     
#         '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
#     
#         '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#         
#         '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#         
#         curr_A = A_list[0] - delta_A_list[0] 
#         
#         curr_B = B_list[0] - delta_B_list[0]
#         
#         for j in range(epoch):
#             
#             gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)),1)
#             
#             curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), curr_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
#             
#             curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
#             
#             delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#                     
#             para_curr_layer = para_curr_layer - alpha*curr_gradient
#         
#         
#         para_list[0].data = para_curr_layer[:, 0:-1]
#         
#         para_list[1].data = para_curr_layer[:, -1]

    
    
#     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[0])))
#     
#     
#     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    
#     deriv = torch.mm(torch.t(delta), output_list[0])
#     
#     '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
# 
#     para_list[0].data = para_curr_layer[:, 0:-1]
#     
#     para_list[1].data = para_curr_layer[:, -1]

    t2  = time.time()
    
    print("time0:", t2 - t1)   
    
    return para_list



def model_update_provenance(alpha, epoch, model, dim, output_list, A_list, B_list, w_delta_prod_list, b_delta_prod_list, input_dim, hidden_dims, output_dim, delta_ids, gradient_list):
    
    
#     loss = np.infty
#     
#     count = 0
#     
#     
#     output_list = model.get_output_each_layer(X)
#     
#     para_list = model.parameters()
# 
#     while loss > loss_threshold and count < num_epochs:
#         
#         delta_list = []
#         
#         outer_gradient = softmax_func(output_list[len(output_list) - 1]) - get_onehot_y(Y, dim, num_class)       
#         
#         para_list_len = len(para_list)
#         
#         
#         for i in range(para_list_len):
#             
#             
#             '''w_res: n * hidden[len-i], b_res: n*1, para_list[2*para_list_len - 2*i]: hidden[len-i]*hidden[len-i-1], output_list[len-i]: n*hidden[len-i-1]'''
#             
#             
#             w_res[para_list_len - i]*torch.sum(para_list[2*para_list_len - 2*i]*output_list[para_list_len - i], 2) + b_res[para_list_len - i]
#             
#             
#             
#             softmax_func()


    t1  = time.time()
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
#     pred = output_list[len(output_list) - 1]
    
    para_list = list(model.parameters())
    
    
    depth = len(hidden_dims) + 1

#         loss = error(pred, Y)
    
#     delta = softmax_func(pred) - get_onehot_y(Y, dim, num_class)
    
    
    '''delta: n*output_dim'''
    
    
    '''A: output_dim, hidden_dim[depth-2]^2'''    
#     print(w_res[depth - 1])
    
    print(depth)
    
    delta_A_list = [None]*depth
    
#     delta_A_list0 = [None]*depth
    
    
    delta_B_list = [None]*depth
    
#     delta_B_list0 = [None]*depth
    

    delta_id_num = delta_ids.shape[0]
    
    w_delta_prod = w_delta_prod_list[depth - 1][delta_ids]
    
    b_delta_prod = b_delta_prod_list[depth - 1][delta_ids]
    
    '''A: output_dim, hidden_dims[-1]*hiddem_dims[-1]'''
    
    delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1][delta_ids].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1][delta_ids].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
    
#     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
    
    delta_A_list[depth - 1] = delta_A
    
#     delta_A_list0.append(delta_A0)
    
    '''B: output_dim, hidden_dim[depth-2]'''
    delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-1][delta_ids])
    
#     delta_B0 = torch.t(b_delta_prod)
    
#     delta_B_list0.append(delta_B0)
    
    delta_B_list[depth - 1] = delta_B
    
    
#     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1])))
#     
#     
#     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    
    
#     delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
#     
#     deriv = torch.mm(torch.t(delta), output_list[depth - 1])

    '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''

#     delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(entire_delta_expression)))
    
#     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
#     
#     para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
#     
#     para_list[2*depth - 1].data = para_curr_layer[:, -1]
        
    for i in range(depth - 2):
        
        '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
        
#         para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
#         delta = delta_para_prod#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
        w_delta_prod = w_delta_prod_list[depth - i - 2][delta_ids]
    
        b_delta_prod = b_delta_prod_list[depth - i - 2][delta_ids]
        
        delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - i - 2][delta_ids].view(delta_id_num, hidden_dims[depth - i - 3] + 1, 1), output_list[depth - i - 2][delta_ids].view(delta_id_num, 1, hidden_dims[depth - i - 3] + 1)).view(delta_id_num, (hidden_dims[depth-i-3]+1)*(hidden_dims[depth-i-3]+1)))
        
#         delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - i - 2][delta_ids])
        
        delta_A_list[depth - i - 2] = delta_A
        
#         delta_A_list0.append(delta_A0)
        
        '''B: output_dim, hidden_dim[depth-2]'''
        delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-i-2][delta_ids])
        
#         delta_B0 = torch.t(b_delta_prod)
        
#         delta_B_list0.append(delta_B0)
        
        
        delta_B_list[depth - i - 2] = delta_B
        
#         w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - i - 2])))
#         
#         
#         entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
#         
#         
#         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(entire_delta_expression)))
#         
#         A_list[depth - 2 - i] = A
#         
#         B_list[depth - 2 - i] = B
        
#         deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])
#         
#         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))                    
#         
#         '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#         para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
#     
#         para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
#         
#         para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
            
#     para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1)

#     delta = delta_para_prod#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
    
    w_delta_prod = w_delta_prod_list[0][delta_ids]

    b_delta_prod = b_delta_prod_list[0][delta_ids]
    
    delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[0][delta_ids].view(delta_id_num, input_dim + 1, 1), output_list[0][delta_ids].view(delta_id_num, 1, input_dim + 1)).view(delta_id_num, (input_dim+1)*(input_dim + 1)))

    delta_A = delta_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)

#     print(w_delta_prod.shape)

#     delta_A = torch.mm(torch.t(output_list[0][delta_ids]), torch.bmm(w_delta_prod.view(delta_id_num, hidden_dims[0], 1), output_list[0][delta_ids].view(delta_id_num, 1, input_dim + 1)).view(delta_id_num, hidden_dims[0]*(input_dim + 1)))
#     
#     delta_A = torch.transpose(delta_A.view(input_dim + 1, hidden_dims[0], input_dim + 1), 0, 1)
    
#     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[0][delta_ids])
    
#     delta_A_list0.append(delta_A0)
    
    '''B: output_dim, hidden_dim[depth-2]'''
    delta_B = torch.mm(torch.t(b_delta_prod), output_list[0][delta_ids])
    
#     delta_B0 = torch.t(b_delta_prod)
    
#     delta_B_list0.append(delta_B0)
    
    
    delta_A_list[0] = delta_A
    
    delta_B_list[0] = delta_B


#     weights = para_list[2*depth - 2].data
#         
#     offsets = para_list[2*depth - 1].data
    
    
    para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 
    
    
    '''A: (output_dim*hidden_dims[depth-1])*hiddem_dims[depth-1]'''

    '''B: output_dim*hidden_dims[depth-2]'''
    
    '''weights: output_dim*hidden_dims[depth-1]'''
    
    curr_A = A_list[depth - 1] - delta_A_list[depth - 1] 
    
    curr_B = B_list[depth - 1] - delta_B_list[depth - 1]
    
    for j in range(epoch):
        
        '''output_dim, 1, hidden_dims[depth-2]'''
        
        curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
        
        curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
        
        
        gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
        
        delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
        
        para_curr_layer = para_curr_layer - alpha*curr_gradient
        
#         para_curr_layer = para_curr_layer - alpha/

    para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
        
    para_list[2*depth - 1].data = para_curr_layer[:, -1]
    
    for i in range(depth-2):
        
        para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
                
        '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''

        '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
        
        '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
        
        curr_A = A_list[depth - i - 2] - delta_A_list[depth - i - 2] 
        
        curr_B = B_list[depth - i - 2] - delta_B_list[depth - i - 2]
        
        for j in range(epoch):
        
            gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)),1)
            
            curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[depth-i-2], 1, hidden_dims[depth-i-3] + 1), curr_A.view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1, hidden_dims[depth-3-i] + 1)).view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1)
            
            curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
            
            delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
        
        
        para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
        
        para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
    
    
    para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)
                
    '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''

    '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
    
    '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
    
    curr_A = A_list[0] - delta_A_list[0] 
    
    curr_B = B_list[0] - delta_B_list[0]
    
    for j in range(epoch):
        
        gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)),1)
        
        curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), curr_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
        
        curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
        
        delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
                
        para_curr_layer = para_curr_layer - alpha*curr_gradient
    
    
    para_list[0].data = para_curr_layer[:, 0:-1]
    
    para_list[1].data = para_curr_layer[:, -1]

    
    
#     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[0])))
#     
#     
#     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    
#     deriv = torch.mm(torch.t(delta), output_list[0])
#     
#     '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
# 
#     para_list[0].data = para_curr_layer[:, 0:-1]
#     
#     para_list[1].data = para_curr_layer[:, -1]

    t2  = time.time()
    
    print("time0:", t2 - t1)   
    
    return para_list

def model_update_provenance1_2(alpha, init_para, selected_rows, model, X, Y, w_list_all_epochs, b_list_all_epochs, num_class, input_dim, hidden_dims, output_dim, gradient_list_all_epochs):
    
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
#     A_list_all_epochs = []
#     
#     B_list_all_epochs = []
#     
#     w_delta_prod_list_all_epochs = []
#     
#     b_delta_prod_list_all_epochs = []
    
    
    curr_X = X[selected_rows]
    
    
    curr_Y = Y[selected_rows]
    
    
    selected_row_num = selected_rows.shape[0]
    
    init_model(model, init_para)
    
    t1  = time.time()
    
    depth = len(hidden_dims) + 1
    
    overhead = 0.0
    
    with torch.no_grad():
        
        for k in range(len(w_list_all_epochs)):
            w_res = w_list_all_epochs[k]
            
            b_res = b_list_all_epochs[k]
            
#             expected_gradient_list = gradient_list_all_epochs[k]
            
            output_list, input_to_non_linear_layer_list = model.get_output_each_layer(curr_X)
            
            input_to_non_linear_layer_list = input_to_non_linear_layer_list[1:]
            
            pred = output_list[len(output_list) - 1]
            
            para_list = list(model.parameters())
        
        
#             A_list = [None]*depth
#         
#         
#             B_list = [None]*depth
            
            
        #     A0_list = [None]*depth
        #     
        #     B0_list = [None]*depth
        #         loss = error(pred, Y)
            
                   
            
            delta = Variable(softmax_func(pred) - get_onehot_y(curr_Y, [selected_row_num, num_class], num_class))
            
            
            '''delta: n*output_dim'''
            
            para_curr_layer = Variable(torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1)) 
            
            
            '''output_list[depth-1]: n*hidden_dims[depth-2], para_curr_layer: output_dim*hidden_dims[depth-2] -> output_dim*n'''
            
            input_to_non_linear_layer = input_to_non_linear_layer_list[depth -1]#Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth-1]))))
            
            '''n*output_dim'''
            
            t3 = time.time()
            
            non_linear_output = Variable(delta*(input_to_non_linear_layer*w_res[depth-1][selected_rows] + b_res[depth-1][selected_rows]))
            
            t4 = time.time()
            
            curr_overhead = (t4 - t3)
            
            overhead = overhead + curr_overhead
#             w_delta_prod = Variable(w_res[depth - 1][selected_rows]*delta)
#              
#             b_delta_prod = Variable(b_res[depth - 1][selected_rows]*delta)
            
            
#             w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
            
            
#             entire_delta_expression = non_linear_output#w_delta_prod*input_to_non_linear_layer + b_delta_prod
            
            
#             entire_delta_expression2 = w_delta_prod*w_input_prod + b_delta_prod
            
        #     delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
        #     
        #     deriv = torch.mm(torch.t(delta), output_list[depth - 1])
        
            '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
        
            delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(non_linear_output))))
            
#             delta_para_prod2 = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(entire_delta_expression2)))) 
            
            curr_gradient = Variable(torch.mm(torch.t(non_linear_output), output_list[depth-1])/selected_row_num) 
            
#             expected_gradient = Variable(torch.cat((expected_gradient_list[2*depth - 2].data, expected_gradient_list[2*depth - 1].data.view(-1,1)), 1))
            
            
#             gradient_diff = torch.norm(expected_gradient - curr_gradient)
            
#             print("gradient_diff::", gradient_diff)
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
                
                
            para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
            
            para_list[2*depth - 1].data = para_curr_layer[:, -1]
            
            
            
        #     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        #     
        #     para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
        #     
        #     para_list[2*depth - 1].data = para_curr_layer[:, -1         
                
            for i in range(depth - 2):
                
                '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
                
                para_curr_layer = Variable(torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1))
                
                delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
                        
                        
                input_to_non_linear_layer = input_to_non_linear_layer_list[depth-i-2]#Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth- i - 2]))))
                        
                non_linear_output = Variable(delta*(input_to_non_linear_layer*w_res[depth-i-2][selected_rows] + b_res[depth-i-2][selected_rows]))

#                 w_delta_prod = Variable(w_res[depth - i - 2][selected_rows]*delta)
#             
#                 b_delta_prod = Variable(b_res[depth - i - 2][selected_rows]*delta)
                
               
#                 w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - i - 2]))))
                
                
#                 entire_delta_expression = non_linear_output#w_delta_prod*input_to_non_linear_layer + b_delta_prod
                
                
                delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(non_linear_output))))
                                                
                
                curr_gradient = Variable(torch.mm(torch.t(delta*(input_to_non_linear_layer*w_res[depth-i-2][selected_rows] + b_res[depth-i-2][selected_rows])),output_list[depth-i-2])/selected_row_num) 
                        
#                 expected_gradient = Variable(torch.cat((expected_gradient_list[2*depth - 2*i - 4].data, expected_gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1))
            
            
#                 gradient_diff = torch.norm(expected_gradient - curr_gradient)
                
#                 print("gradient_diff::", gradient_diff)
                
                para_curr_layer = para_curr_layer - alpha*curr_gradient
                    
                    
                para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                
                para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
                
#                 A_list[depth - 2 - i] = A
#                 
#                 B_list[depth - 2 - i] = B
                
        #         deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])
        #         
        #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))                    
        #         
        #         '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
        #         para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        #     
        #         para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
        #         
        #         para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
            
            
            
            para_curr_layer = Variable(torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1))
        
            delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
            
            

            input_to_non_linear_layer = input_to_non_linear_layer_list[0]#Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
            
            t3 = time.time()
            
            non_linear_output = Variable(delta*(input_to_non_linear_layer*w_res[0][selected_rows] + b_res[0][selected_rows]))
            t4 = time.time()
            
            curr_overhead = (t4 - t3)
            
            overhead = overhead + curr_overhead
            
            curr_gradient = Variable(torch.mm(torch.t(non_linear_output),output_list[0])/selected_row_num)
                        
                        
            
#             expected_gradient = Variable(torch.cat((expected_gradient_list[0].data, expected_gradient_list[1].data.view(-1,1)), 1))
        
        
#             gradient_diff = torch.norm(expected_gradient - curr_gradient)
            
#             print("gradient_diff::", gradient_diff)
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
                
                
            para_list[0].data = para_curr_layer[:, 0:-1]
            
            para_list[1].data = para_curr_layer[:, -1]

            init_model(model, para_list)
            
#             t4 = time.time()
            
            
            
#             print("curr_overhead::", (t4 - t3))
#             
#             print("overhead::", overhead)
            
#             A_list[0] = A
#             
#             B_list[0] = B
#         
#             A_list_all_epochs.append(A_list)
#             
#             B_list_all_epochs.append(B_list)
    
#     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[0])))
#     
#     
#     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    
#     deriv = torch.mm(torch.t(delta), output_list[0])
#     
#     '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
# 
#     para_list[0].data = para_curr_layer[:, 0:-1]
#     
#     para_list[1].data = para_curr_layer[:, -1]

    t2  = time.time()
    
    print("time0:", t2 - t1)   
    
    print("overhead::", overhead)
    
    
    return list(model.parameters())
    
    
    
    
#     for param in param_list:
#         
#         delta =  


def model_update_provenance1_3(alpha, init_para, selected_rows, model, X, Y, w_list_all_epochs, b_list_all_epochs, num_class, input_dim, hidden_dims, output_dim, gradient_list_all_epochs):
    
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
#     A_list_all_epochs = []
#     
#     B_list_all_epochs = []
#     
#     w_delta_prod_list_all_epochs = []
#     
#     b_delta_prod_list_all_epochs = []
    
    
    curr_X = X[selected_rows]
    
    
    curr_Y = Y[selected_rows]
    
    
    selected_row_num = selected_rows.shape[0]
    
    init_model(model, init_para)
    
    t1  = time.time()
    
    depth = len(hidden_dims) + 1
    
    hidden_dim_tensor = torch.tensor(hidden_dims)
    
    overhead = 0.0
    
    with torch.no_grad():
        
        for k in range(len(w_list_all_epochs)):
            
            
            w_res_cat = torch.cat(w_list_all_epochs[k], 1)
            
            b_res_cat = torch.cat(b_list_all_epochs[k], 1)
            
#             expected_gradient_list = gradient_list_all_epochs[k]
            
            
            
            output_list, input_to_non_linear_layer_list = model.get_output_each_layer(curr_X)
            
            input_to_non_linear_layer_list = input_to_non_linear_layer_list[1:]
            t3 = time.time()

            
#             output_list_cat = torch.cat(output_list, 1)
            
            input_to_non_linear_layer_list_cat = Variable(torch.cat(input_to_non_linear_layer_list, 1))
            
            derivative_non_linear_layer = Variable(w_res_cat[selected_rows]*input_to_non_linear_layer_list_cat + b_res_cat[selected_rows])
            
            
            t4 = time.time()
            
            overhead += (t4 - t3)
            
            pred = output_list[len(output_list) - 1]
            
            para_list = list(model.parameters())
        
        
#             A_list = [None]*depth
#         
#         
#             B_list = [None]*depth
            
            
        #     A0_list = [None]*depth
        #     
        #     B0_list = [None]*depth
        #         loss = error(pred, Y)
            
            delta = Variable(softmax_func(pred) - get_onehot_y(curr_Y, [selected_row_num, num_class], num_class))
            
            
            '''delta: n*output_dim'''
            
            para_curr_layer = Variable(torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1)) 
            
            
            '''output_list[depth-1]: n*hidden_dims[depth-2], para_curr_layer: output_dim*hidden_dims[depth-2] -> output_dim*n'''
            
#             input_to_non_linear_layer = input_to_non_linear_layer_list[depth -1]#Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth-1]))))
            
            '''n*output_dim'''
            
            non_linear_output = Variable(delta*derivative_non_linear_layer[:, torch.sum(hidden_dim_tensor):torch.sum(hidden_dim_tensor)+output_dim])#Variable(delta*(input_to_non_linear_layer*w_res[depth-1][selected_rows] + b_res[depth-1][selected_rows]))
            
            
#             w_delta_prod = Variable(w_res[depth - 1][selected_rows]*delta)
#              
#             b_delta_prod = Variable(b_res[depth - 1][selected_rows]*delta)
            
            
#             w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
            
            
#             entire_delta_expression = non_linear_output#w_delta_prod*input_to_non_linear_layer + b_delta_prod
            
            
#             entire_delta_expression2 = w_delta_prod*w_input_prod + b_delta_prod
            
        #     delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
        #     
        #     deriv = torch.mm(torch.t(delta), output_list[depth - 1])
        
            '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
        
            delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(non_linear_output))))
            
#             delta_para_prod2 = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(entire_delta_expression2)))) 
            
            
            curr_gradient = Variable(torch.mm(torch.t(non_linear_output), output_list[depth-1])/selected_row_num) 
            
            
#             expected_gradient = Variable(torch.cat((expected_gradient_list[2*depth - 2].data, expected_gradient_list[2*depth - 1].data.view(-1,1)), 1))
            
            
#             gradient_diff = torch.norm(expected_gradient - curr_gradient)
            
#             print("gradient_diff::", gradient_diff)
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
                
                
            para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
            
            para_list[2*depth - 1].data = para_curr_layer[:, -1]
            
            
            
        #     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        #     
        #     para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
        #     
        #     para_list[2*depth - 1].data = para_curr_layer[:, -1         
                
            for i in range(depth - 2):
                
                '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
                
                para_curr_layer = Variable(torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1))
                
                delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
                        
                        
#                 input_to_non_linear_layer = input_to_non_linear_layer_list[depth-i-2]#Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth- i - 2]))))
                        
                non_linear_output = Variable(delta*derivative_non_linear_layer[:, torch.sum(hidden_dim_tensor[0:depth - i - 2]):torch.sum(hidden_dim_tensor[0:depth-i-1])])#Variable(delta*(input_to_non_linear_layer*w_res[depth-i-2][selected_rows] + b_res[depth-i-2][selected_rows]))

#                 w_delta_prod = Variable(w_res[depth - i - 2][selected_rows]*delta)
#             
#                 b_delta_prod = Variable(b_res[depth - i - 2][selected_rows]*delta)
                
               
#                 w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - i - 2]))))
                
                
#                 entire_delta_expression = non_linear_output#w_delta_prod*input_to_non_linear_layer + b_delta_prod
                
                
                delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(non_linear_output))))
                                                
                
                curr_gradient = Variable(torch.mm(torch.t(non_linear_output),output_list[depth-i-2])/selected_row_num) 
                        
#                 expected_gradient = Variable(torch.cat((expected_gradient_list[2*depth - 2*i - 4].data, expected_gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1))
            
            
#                 gradient_diff = torch.norm(expected_gradient - curr_gradient)
                
#                 print("gradient_diff::", gradient_diff)
                
                para_curr_layer = para_curr_layer - alpha*curr_gradient
                    
                    
                para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                
                para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
                
#                 A_list[depth - 2 - i] = A
#                 
#                 B_list[depth - 2 - i] = B
                
        #         deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])
        #         
        #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))                    
        #         
        #         '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
        #         para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        #     
        #         para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
        #         
        #         para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
                    
            para_curr_layer = Variable(torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1))
        
            delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
            
            
#             input_to_non_linear_layer = input_to_non_linear_layer_list[0]#Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
            
            non_linear_output = Variable(delta*derivative_non_linear_layer[:, 0:torch.sum(hidden_dim_tensor[0:1])])#Variable(delta*(input_to_non_linear_layer*w_res[0][selected_rows] + b_res[0][selected_rows]))
            
            curr_gradient = Variable(torch.mm(torch.t(non_linear_output),output_list[0])/selected_row_num)
                        
#             expected_gradient = Variable(torch.cat((expected_gradient_list[0].data, expected_gradient_list[1].data.view(-1,1)), 1))
        
        
#             gradient_diff = torch.norm(expected_gradient - curr_gradient)
            
#             print("gradient_diff::", gradient_diff)
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
                
                
            para_list[0].data = para_curr_layer[:, 0:-1]
            
            para_list[1].data = para_curr_layer[:, -1]

            init_model(model, para_list)
            
            
            
#             A_list[0] = A
#             
#             B_list[0] = B
#         
#             A_list_all_epochs.append(A_list)
#             
#             B_list_all_epochs.append(B_list)
    
#     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[0])))
#     
#     
#     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    
#     deriv = torch.mm(torch.t(delta), output_list[0])
#     
#     '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
# 
#     para_list[0].data = para_curr_layer[:, 0:-1]
#     
#     para_list[1].data = para_curr_layer[:, -1]

    t2  = time.time()
    
    print("time0:", t2 - t1)   
    
    print("overhead::", overhead)
    
    
    return list(model.parameters())
    
    
    
    
#     for param in param_list:
#         
#         delta =  


def model_update_provenance1(alpha, init_para, selected_rows, model, X, Y, w_list_all_epochs, b_list_all_epochs, num_class, input_dim, hidden_dims, output_dim, gradient_list_all_epochs):
    
#     t1  = time.time()
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
#     A_list_all_epochs = []
#     
#     B_list_all_epochs = []
#     
#     w_delta_prod_list_all_epochs = []
#     
#     b_delta_prod_list_all_epochs = []
    
    
    curr_X = X[selected_rows]
    
    
    curr_Y = Y[selected_rows]
    
    
    selected_row_num = selected_rows.shape[0]
    
    init_model(model, init_para)
    
    with torch.no_grad():
        
        for k in range(len(w_list_all_epochs)):
            w_res = w_list_all_epochs[k]
            
            b_res = b_list_all_epochs[k]
            
            expected_gradient_list = gradient_list_all_epochs[k]
            
            
            output_list = model.get_output_each_layer(curr_X)
            
            pred = output_list[len(output_list) - 1]
            
            para_list = list(model.parameters())
        
        
            depth = len(hidden_dims) + 1
        
#             A_list = [None]*depth
#         
#         
#             B_list = [None]*depth
            
            
        #     A0_list = [None]*depth
        #     
        #     B0_list = [None]*depth
        #         loss = error(pred, Y)
            
            delta = Variable(softmax_func(pred) - get_onehot_y(curr_Y, [selected_row_num, num_class], num_class))
            
            
            '''delta: n*output_dim'''
            
            para_curr_layer = Variable(torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1)) 
            
            
            '''A: output_dim, hidden_dim[depth-2]^2'''
            
        #     print(len(w_res))
            
        #     print(w_res[depth - 1])
            
        #     print(depth)
            
            
#             w_delta_prod_list = [None]*depth
#             
#             b_delta_prod_list = [None]*depth
            
            
            '''w_delta_prod: n*output_dim'''
            
            w_delta_prod = Variable(w_res[depth - 1][selected_rows]*delta)
            
            b_delta_prod = Variable(b_res[depth - 1][selected_rows]*delta)
            
#             w_delta_prod_list[depth - 1] = w_delta_prod
#             
#             b_delta_prod_list[depth - 1] = b_delta_prod
            
            '''A: output_dim*(hidden_dims[-1]*hiddem_dims[-1])'''
            A = Variable(torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(selected_row_num, hidden_dims[depth-2] + 1, 1), output_list[depth - 1].view(selected_row_num, 1, hidden_dims[depth-2]+ 1)).view(selected_row_num, -1)))
            
        #     A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1])
            
        #     A0_list.append(A0)
            
            '''B: output_dim, hidden_dim[depth-2]'''
            B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-1]))
            
        #     B0 = torch.t(torch.sum(b_delta_prod, 0))
            
        #     B0_list.append(B0)
            
            
            
#             A_list[depth - 1] = A
#             
#             B_list[depth - 1] = B
            
            
            
            w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
            
            
            entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
            
            
        #     delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
        #     
        #     deriv = torch.mm(torch.t(delta), output_list[depth - 1])
        
            '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
        
            delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(entire_delta_expression))))
            
        #     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        #     
        #     para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
        #     
        #     para_list[2*depth - 1].data = para_curr_layer[:, -1]
        
            curr_gradient = (1.0/(selected_row_num))*torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth-2] + 1), A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
                
            curr_gradient += (1.0/(selected_row_num))*B
            
            
            expected_gradient = Variable(torch.cat((expected_gradient_list[2*depth - 2].data, expected_gradient_list[2*depth - 1].data.view(-1,1)), 1))
            
            
            gradient_diff = torch.norm(expected_gradient - curr_gradient)
            
            print("gradient_diff::", gradient_diff)
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
                
                
            para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
            
            para_list[2*depth - 1].data = para_curr_layer[:, -1]
        
                
            for i in range(depth - 2):
                
                '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
                
                para_curr_layer = Variable(torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1))
                
                delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
                        
                w_delta_prod = Variable(w_res[depth - i - 2][selected_rows]*delta)
            
                b_delta_prod = Variable(b_res[depth - i - 2][selected_rows]*delta)
                
#                 w_delta_prod_list[depth - i - 2] = w_delta_prod
#             
#                 b_delta_prod_list[depth - i - 2] = b_delta_prod
                
                A = Variable(torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - i - 2].view(selected_row_num, hidden_dims[depth - i - 3]+ 1, 1), output_list[depth - i - 2].view(selected_row_num, 1, hidden_dims[depth - i - 3]+ 1)).view(selected_row_num, -1)))
                
        #         A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - i - 2])
                
        #         A0_list.append(A0)
                
                '''B: output_dim, hidden_dim[depth-2]'''
                B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-i-2]))
                
        #         B0 = torch.t(torch.sum(b_delta_prod, 0))
                
        #         B0_list.append(B0)
                
                w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - i - 2]))))
                
                
                entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
                
                
                delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(entire_delta_expression))))
                
                
                expected_gradient = Variable(torch.cat((expected_gradient_list[2*depth - 2*i - 4].data, expected_gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1))
                
                
                curr_gradient = (1.0/(selected_row_num))*torch.bmm(para_curr_layer.view(hidden_dims[depth-i-2], 1, hidden_dims[depth-i-3] + 1), A.view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1, hidden_dims[depth-3-i] + 1)).view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1)
                
                curr_gradient += (1.0/(selected_row_num))*B
                
                
                gradient_diff = torch.norm(curr_gradient-expected_gradient)
                
                print("gradient_diff::", gradient_diff)
                
                para_curr_layer = para_curr_layer - alpha*curr_gradient
                    
                    
                para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                
                para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
                
                
#                 A_list[depth - 2 - i] = A
#                 
#                 B_list[depth - 2 - i] = B
                
        #         deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])
        #         
        #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))                    
        #         
        #         '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
        #         para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        #     
        #         para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
        #         
        #         para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
                    
            para_curr_layer = Variable(torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1))
        
            delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
            
            w_delta_prod = Variable(w_res[0][selected_rows]*delta)
        
            b_delta_prod = Variable(b_res[0][selected_rows]*delta)
            
#             w_delta_prod_list[0] = w_delta_prod
#             
#             b_delta_prod_list[0] = b_delta_prod
#             
#             
#             w_delta_prod_list_all_epochs.append(w_delta_prod_list)
#             
#             b_delta_prod_list_all_epochs.append(b_delta_prod_list)
            
        #     A = torch.mm(torch.bmm(w_delta_prod.view(dim[0], )))
            
        #     A = torch.zeros([hidden_dims[0], (input_dim+1)*(input_dim+1)])
            
            
            A = Variable(torch.mm(torch.t(output_list[0]), torch.bmm(w_delta_prod.view(selected_row_num, hidden_dims[0], 1), output_list[0].view(selected_row_num, 1, input_dim + 1)).view(selected_row_num, hidden_dims[0]*(input_dim + 1))))
            
            A = Variable(torch.transpose(A.view(input_dim + 1, hidden_dims[0], input_dim + 1), 0, 1))
            
        #     for k in range(dim[0]):
        #         A += torch.mm(torch.t(w_delta_prod)[:,k].view(-1,1), torch.mm(output_list[0][k].view(input_dim + 1, 1), output_list[0][k].view(1, input_dim + 1)).view(1,-1))
            
        #     A = torch.mm( temp)
            
            '''B: output_dim, hidden_dim[depth-2]'''
            B = Variable(torch.mm(torch.t(b_delta_prod), output_list[0]))
            
            
            curr_gradient = (1.0/(selected_row_num))*torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
            
            curr_gradient += (1.0/(selected_row_num))*B
            
            
            expected_gradient = Variable(torch.cat((expected_gradient_list[0].data, expected_gradient_list[1].data.view(-1,1)), 1))
            
            gradient_diff = torch.norm(expected_gradient - curr_gradient)
            
            print("gradient_diff::", gradient_diff)
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
                
                
            para_list[0].data = para_curr_layer[:, 0:-1]
            
            para_list[1].data = para_curr_layer[:, -1]
            
            
            init_model(model, para_list)
            
#             A_list[0] = A
#             
#             B_list[0] = B
#         
#             A_list_all_epochs.append(A_list)
#             
#             B_list_all_epochs.append(B_list)
    
#     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[0])))
#     
#     
#     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    
#     deriv = torch.mm(torch.t(delta), output_list[0])
#     
#     '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
# 
#     para_list[0].data = para_curr_layer[:, 0:-1]
#     
#     para_list[1].data = para_curr_layer[:, -1]

#     t2  = time.time()
    
#     print("time0:", t2 - t1)   
    
    
    return list(model.parameters())
    
    
    
    
#     for param in param_list:
#         
#         delta =  
        


def model_update_provenance2(alpha, model, dim, output_list_all_epochs, A_list_all_epochs, B_list_all_epochs, w_delta_prod_list_all_epochs, b_delta_prod_list_all_epochs, input_dim, hidden_dims, output_dim, delta_ids, expected_gradient_list_all_epochs):
    
    
#     loss = np.infty
#     
#     count = 0
#     
#     
#     output_list = model.get_output_each_layer(X)
#     
#     para_list = model.parameters()
# 
#     while loss > loss_threshold and count < num_epochs:
#         
#         delta_list = []
#         
#         outer_gradient = softmax_func(output_list[len(output_list) - 1]) - get_onehot_y(Y, dim, num_class)       
#         
#         para_list_len = len(para_list)
#         
#         
#         for i in range(para_list_len):
#             
#             
#             '''w_res: n * hidden[len-i], b_res: n*1, para_list[2*para_list_len - 2*i]: hidden[len-i]*hidden[len-i-1], output_list[len-i]: n*hidden[len-i-1]'''
#             
#             
#             w_res[para_list_len - i]*torch.sum(para_list[2*para_list_len - 2*i]*output_list[para_list_len - i], 2) + b_res[para_list_len - i]
#             
#             
#             
#             softmax_func()


    t1  = time.time()
    
    
    para_list = list(model.parameters())
    
    depth = len(hidden_dims) + 1
    
    '''delta: n*output_dim'''
    
    
    '''A: output_dim, hidden_dim[depth-2]^2'''    
    
    print(depth)
    
    delta_id_num = delta_ids.shape[0]
    
    curr_training_data = output_list_all_epochs[0][0][:, 0:-1]
    
    
    with torch.no_grad():
        
    
        for k in range(len(output_list_all_epochs)):
        
            expected_gradient_list = expected_gradient_list_all_epochs[k]
        
            
            delta_A_list = [None]*depth
            
        #     delta_A_list0 = [None]*depth
            
            
            delta_B_list = [None]*depth
            
        #     delta_B_list0 = [None]*depth
            
            w_delta_prod_list = w_delta_prod_list_all_epochs[k]
            
            b_delta_prod_list = b_delta_prod_list_all_epochs[k]
            
            A_list = A_list_all_epochs[k]
            
            B_list = B_list_all_epochs[k]
            
            
            output_list,_ = model.get_output_each_layer(curr_training_data)#output_list_all_epochs[k]
            
        
            w_delta_prod = w_delta_prod_list[depth - 1][delta_ids]
            
            b_delta_prod = b_delta_prod_list[depth - 1][delta_ids]
            
            '''A: output_dim, hidden_dims[-1]*hiddem_dims[-1]'''
            
            delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1][delta_ids].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1][delta_ids].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
            
        #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
            
            delta_A_list[depth - 1] = delta_A
            
        #     delta_A_list0.append(delta_A0)
            
            '''B: output_dim, hidden_dim[depth-2]'''
            delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-1][delta_ids])
            
        #     delta_B0 = torch.t(b_delta_prod)
            
        #     delta_B_list0.append(delta_B0)
            
            delta_B_list[depth - 1] = delta_B
            
            for i in range(depth - 2):
                
                '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
                
        #         para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
        #         delta = delta_para_prod#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
                w_delta_prod = w_delta_prod_list[depth - i - 2][delta_ids]
            
                b_delta_prod = b_delta_prod_list[depth - i - 2][delta_ids]
                
                delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - i - 2][delta_ids].view(delta_id_num, hidden_dims[depth - i - 3] + 1, 1), output_list[depth - i - 2][delta_ids].view(delta_id_num, 1, hidden_dims[depth - i - 3] + 1)).view(delta_id_num, (hidden_dims[depth-i-3]+1)*(hidden_dims[depth-i-3]+1)))
                
        #         delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - i - 2][delta_ids])
                
                delta_A_list[depth - i - 2] = delta_A
                
        #         delta_A_list0.append(delta_A0)
                
                '''B: output_dim, hidden_dim[depth-2]'''
                delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-i-2][delta_ids])
                
        #         delta_B0 = torch.t(b_delta_prod)
                
        #         delta_B_list0.append(delta_B0)
                
                
                delta_B_list[depth - i - 2] = delta_B
            
            w_delta_prod = w_delta_prod_list[0][delta_ids]
        
            b_delta_prod = b_delta_prod_list[0][delta_ids]
            
            delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[0][delta_ids].view(delta_id_num, input_dim + 1, 1), output_list[0][delta_ids].view(delta_id_num, 1, input_dim + 1)).view(delta_id_num, (input_dim+1)*(input_dim + 1)))
        
            delta_A = delta_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)
        
        #    
            '''B: output_dim, hidden_dim[depth-2]'''
            delta_B = torch.mm(torch.t(b_delta_prod), output_list[0][delta_ids])
            
        #     delta_B0 = torch.t(b_delta_prod)
            
        #     delta_B_list0.append(delta_B0)
            
            
            delta_A_list[0] = delta_A
            
            delta_B_list[0] = delta_B
        
        
        #     weights = para_list[2*depth - 2].data
        #         
        #     offsets = para_list[2*depth - 1].data
            
            
            para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 
            
            '''A: (output_dim*hidden_dims[depth-1])*hiddem_dims[depth-1]'''
        
            '''B: output_dim*hidden_dims[depth-2]'''
            
            '''weights: output_dim*hidden_dims[depth-1]'''
            
            curr_A = A_list[depth - 1] - delta_A_list[depth - 1] 
            
            curr_B = B_list[depth - 1] - delta_B_list[depth - 1]
            
    #         for j in range(epoch):
                
            '''output_dim, 1, hidden_dims[depth-2]'''
                
            curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
            
            curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
            
            gradient_curr_layer = torch.cat((expected_gradient_list[2*depth - 2].data, expected_gradient_list[2*depth - 1].data.view(-1,1)), 1) 

            
    #         gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
    #         
            delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
                
        #         para_curr_layer = para_curr_layer - alpha/
        
            para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
                
            para_list[2*depth - 1].data = para_curr_layer[:, -1]
            
            for i in range(depth-2):
                
                para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
                        
                '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
        
                '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
                
                '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
                
                curr_A = A_list[depth - i - 2] - delta_A_list[depth - i - 2] 
                
                curr_B = B_list[depth - i - 2] - delta_B_list[depth - i - 2]
                
    #             for j in range(epoch):
                
    #             gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)),1)
                
                curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[depth-i-2], 1, hidden_dims[depth-i-3] + 1), curr_A.view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1, hidden_dims[depth-3-i] + 1)).view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1)
                
                curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
                
                gradient_curr_layer = torch.cat((expected_gradient_list[2*depth - 2*i - 4].data, expected_gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1) 

                
                delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
                
                para_curr_layer = para_curr_layer - alpha*curr_gradient
                
                
                para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                
                para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
            
            
            para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)
                        
            '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
        
            '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
            
            '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
            
            curr_A = A_list[0] - delta_A_list[0] 
            
            curr_B = B_list[0] - delta_B_list[0]
            
    #         for j in range(epoch):
                
    #         gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)),1)
            
            curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), curr_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
            
            curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
            
            gradient_curr_layer = torch.cat((expected_gradient_list[0].data, expected_gradient_list[1].data.view(-1,1)), 1) 
            
            delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
                    
            para_curr_layer = para_curr_layer - alpha*curr_gradient
            
            
            para_list[0].data = para_curr_layer[:, 0:-1]
            
            para_list[1].data = para_curr_layer[:, -1] 
            
            init_model(model, para_list)
    
    t2  = time.time()
    
    print("time0:", t2 - t1)   
    
    return para_list


def model_update_provenance3(X, Y, alpha, model, dim, output_list_all_epochs, A_list_all_epochs, B_list_all_epochs, w_delta_prod_list_all_epochs, b_delta_prod_list_all_epochs, input_dim, hidden_dims, output_dim, delta_ids, expected_gradient_list_all_epochs):
    
    
#     loss = np.infty
#     
#     count = 0
#     
#     
#     output_list = model.get_output_each_layer(X)
#     
#     para_list = model.parameters()
# 
#     while loss > loss_threshold and count < num_epochs:
#         
#         delta_list = []
#         
#         outer_gradient = softmax_func(output_list[len(output_list) - 1]) - get_onehot_y(Y, dim, num_class)       
#         
#         para_list_len = len(para_list)
#         
#         
#         for i in range(para_list_len):
#             
#             
#             '''w_res: n * hidden[len-i], b_res: n*1, para_list[2*para_list_len - 2*i]: hidden[len-i]*hidden[len-i-1], output_list[len-i]: n*hidden[len-i-1]'''
#             
#             
#             w_res[para_list_len - i]*torch.sum(para_list[2*para_list_len - 2*i]*output_list[para_list_len - i], 2) + b_res[para_list_len - i]
#             
#             
#             
#             softmax_func()


    t1  = time.time()
    
    
    para_list = list(model.parameters())
    
    depth = len(hidden_dims) + 1
    
    '''delta: n*output_dim'''
    
    
    '''A: output_dim, hidden_dim[depth-2]^2'''    
    
    print(depth)
    
    delta_id_num = delta_ids.shape[0]
    
    curr_training_data = X[delta_ids]#output_list_all_epochs[0][0][:, 0:-1]
    
    
    overhead = 0
    
    with torch.no_grad():
        
    
        for k in range(len(output_list_all_epochs)):
        
            expected_gradient_list = expected_gradient_list_all_epochs[k]
        
            
#             delta_A_list = [None]*depth
            
        #     delta_A_list0 = [None]*depth
            
            
#             delta_B_list = [None]*depth
            
        #     delta_B_list0 = [None]*depth
            
            w_delta_prod_list = w_delta_prod_list_all_epochs[k]
            
            b_delta_prod_list = b_delta_prod_list_all_epochs[k]
            
            A_list = A_list_all_epochs[k]
            
            B_list = B_list_all_epochs[k]
            
            
            output_list,input_to_non_linear_layer_list = model.get_output_each_layer(curr_training_data)#output_list_all_epochs[k]
            
            
        
            input_to_non_linear_layer_list = input_to_non_linear_layer_list[1:]
        
            w_delta_prod = Variable(w_delta_prod_list[depth - 1][delta_ids])
            
            b_delta_prod = Variable(b_delta_prod_list[depth - 1][delta_ids])
            
            '''A: output_dim, hidden_dims[-1]*hiddem_dims[-1]'''
            
            para_curr_layer = Variable(torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1)) 


            input_to_non_linear_layer = input_to_non_linear_layer_list[depth -1]

            t3 = time.time()

            delta_A = Variable(torch.mm(torch.t((w_delta_prod*input_to_non_linear_layer + b_delta_prod)), output_list[depth-1]))


            

            
            curr_A = Variable(torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), A_list[depth - 1].view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1) - delta_A) 
            
            
            
            
            delta_B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-1]))
            
            curr_B = Variable(B_list[depth - 1] - delta_B)
            
    #         for j in range(epoch):
                
            '''output_dim, 1, hidden_dims[depth-2]'''
                
            curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A)#torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
            
            curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
            
            para_curr_layer = Variable(para_curr_layer - alpha*curr_gradient)
                
        #         para_curr_layer = para_curr_layer - alpha/
        
            para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
                
            para_list[2*depth - 1].data = para_curr_layer[:, -1]
            
#             delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1][delta_ids].view(delta_id_num, hidden_dims[depth - 2] + 1, 1), output_list[depth - 1][delta_ids].view(delta_id_num, 1, hidden_dims[depth - 2] + 1)).view(delta_id_num, (hidden_dims[depth - 2] + 1)*(hidden_dims[depth - 2] + 1)))
            
        #     delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1][delta_ids])
                        
        #     delta_A_list0.append(delta_A0)
            
            '''B: output_dim, hidden_dim[depth-2]'''
            
        #     delta_B0 = torch.t(b_delta_prod)
            
        #     delta_B_list0.append(delta_B0)
                        
            for i in range(depth - 2):
                
                '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
                para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
        #         para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
        #         delta = delta_para_prod#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
                w_delta_prod = w_delta_prod_list[depth - i - 2][delta_ids]
            
                b_delta_prod = b_delta_prod_list[depth - i - 2][delta_ids]
                
                delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - i - 2].view(delta_id_num, hidden_dims[depth - i - 3] + 1, 1), output_list[depth - i - 2].view(delta_id_num, 1, hidden_dims[depth - i - 3] + 1)).view(delta_id_num, (hidden_dims[depth-i-3]+1)*(hidden_dims[depth-i-3]+1)))
                
        #         delta_A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - i - 2][delta_ids])
                
#                 delta_A_list[depth - i - 2] = delta_A
                
        #         delta_A_list0.append(delta_A0)
                
                '''B: output_dim, hidden_dim[depth-2]'''
                delta_B = torch.mm(torch.t(b_delta_prod), output_list[depth-i-2][delta_ids])


                curr_A = torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), A_list[depth - i - 2].view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1) - delta_A 
            
                curr_B = B_list[depth - i - 2] - delta_B
                
                
                curr_gradient = (1.0/(dim[0]-delta_id_num))*curr_A #torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), curr_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
            
                curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
                
                para_curr_layer = para_curr_layer - alpha*curr_gradient
                
                para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
                
                para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]

                
        #         delta_B0 = torch.t(b_delta_prod)
                
        #         delta_B_list0.append(delta_B0)
                
                
#                 delta_B_list[depth - i - 2] = delta_B
            
            w_delta_prod = Variable(w_delta_prod_list[0][delta_ids])
        
            b_delta_prod = Variable(b_delta_prod_list[0][delta_ids])
            
#             delta_A = torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[0][delta_ids].view(delta_id_num, input_dim + 1, 1), output_list[0][delta_ids].view(delta_id_num, 1, input_dim + 1)).view(delta_id_num, (input_dim+1)*(input_dim + 1)))
#         
#             delta_A = delta_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)
        
        #
            para_curr_layer = Variable(torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1))

        
            input_to_non_linear_layer = input_to_non_linear_layer_list[0]
        
            delta_A = Variable(torch.mm(torch.t((w_delta_prod*input_to_non_linear_layer + b_delta_prod)), output_list[0]))


            '''B: output_dim, hidden_dim[depth-2]'''
            delta_B = Variable(torch.mm(torch.t(b_delta_prod), output_list[0]))

            t4 = time.time()
            
            overhead += (t4 - t3)
            
            curr_A = Variable(torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), A_list[0].view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1) - delta_A)

            

            
            curr_B = Variable(B_list[0] - delta_B)
            
    #         for j in range(epoch):
                
    #         gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)),1)
            
            curr_gradient = Variable((1.0/(dim[0]-delta_id_num))*curr_A) #torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), curr_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
            
            curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
            
            para_curr_layer = para_curr_layer - alpha*curr_gradient
            
            
            para_list[0].data = para_curr_layer[:, 0:-1]
            
            para_list[1].data = para_curr_layer[:, -1] 
            
            init_model(model, para_list)

        #     delta_B0 = torch.t(b_delta_prod)
            
        #     delta_B_list0.append(delta_B0)
            
            
        #         
        #     offsets = para_list[2*depth - 1].data
            
            
#             para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 
            
            '''A: (output_dim*hidden_dims[depth-1])*hiddem_dims[depth-1]'''
        
            '''B: output_dim*hidden_dims[depth-2]'''
            
            '''weights: output_dim*hidden_dims[depth-1]'''
            
            
            
#             gradient_curr_layer = torch.cat((expected_gradient_list[2*depth - 2].data, expected_gradient_list[2*depth - 1].data.view(-1,1)), 1) 
# 
#             
#     #         gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1) 
#     #         
#             delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#             
#             para_curr_layer = para_curr_layer - alpha*curr_gradient
#                 
#         #         para_curr_layer = para_curr_layer - alpha/
#         
#             para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
#                 
#             para_list[2*depth - 1].data = para_curr_layer[:, -1]
#             
#             for i in range(depth-2):
#                 
#                 para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
#                         
#                 '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
#         
#                 '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#                 
#                 '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#                 
#                 curr_A = A_list[depth - i - 2] - delta_A_list[depth - i - 2] 
#                 
#                 curr_B = B_list[depth - i - 2] - delta_B_list[depth - i - 2]
#                 
#     #             for j in range(epoch):
#                 
#     #             gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)),1)
#                 
#                 curr_gradient = (1.0/(dim[0]-delta_id_num))*torch.bmm(para_curr_layer.view(hidden_dims[depth-i-2], 1, hidden_dims[depth-i-3] + 1), curr_A.view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1, hidden_dims[depth-3-i] + 1)).view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1)
#                 
#                 curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
#                 
#                 gradient_curr_layer = torch.cat((expected_gradient_list[2*depth - 2*i - 4].data, expected_gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1) 
# 
#                 
#                 delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#                 
#                 para_curr_layer = para_curr_layer - alpha*curr_gradient
#                 
#                 
#                 para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
#                 
#                 para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
#             
#             
#             para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)
#                         
#             '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
#         
#             '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#             
#             '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
#             
#             curr_A = torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), A_list[0].view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1) - delta_A_list[0] 
#             
#             curr_B = B_list[0] - delta_B_list[0]
#             
#     #         for j in range(epoch):
#                 
#     #         gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)),1)
#             
#             curr_gradient = (1.0/(dim[0]-delta_id_num))*curr_A #torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), curr_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
#             
#             curr_gradient += (1.0/(dim[0]-delta_id_num))*curr_B
#             
#             gradient_curr_layer = torch.cat((expected_gradient_list[0].data, expected_gradient_list[1].data.view(-1,1)), 1) 
#             
#             delta_gradient = torch.norm(curr_gradient - gradient_curr_layer)
#                     
#             para_curr_layer = para_curr_layer - alpha*curr_gradient
#             
#             
#             para_list[0].data = para_curr_layer[:, 0:-1]
#             
#             para_list[1].data = para_curr_layer[:, -1] 
            
    
    t2  = time.time()
    
    print("time0:", t2 - t1)  
    
    print("overhead:", overhead) 
    
    return para_list


def compute_derivative_with_provenance(dim, hidden_dims, A_list, B_list, para_list):
     
#     para_list = list(model.parameters())
     
    depth = len(hidden_dims) + 1
     
    para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 
     
    curr_A = A_list[depth  -1]
    
    curr_B = B_list[depth - 1]
     
    der_list = [None]*(2*depth)
     
    derivative1 =1.0/(dim[0])*torch.bmm(para_curr_layer.view(output_dim, 1, hidden_dims[depth - 2] + 1), curr_A.view(output_dim, hidden_dims[depth-2] + 1, hidden_dims[depth-2] + 1)).view(output_dim, hidden_dims[depth-2] + 1)
    
    
    derivative1 += 1.0/(dim[0])*curr_B
 
    der_list[2*depth - 2] = derivative1[:,0:-1]
    
    der_list[2*depth - 1] = derivative1[:,-1]
 
    for i in range(depth-2):
         
        para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)),1)
                 
        '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
 
        '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
         
        '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
         
        curr_A = A_list[depth - i - 2]
         
        curr_B = B_list[depth - i - 2]
         
        derivative1 = 1.0/(dim[0])*torch.bmm(para_curr_layer.view(hidden_dims[depth-i-2], 1, hidden_dims[depth-i-3] + 1), curr_A.view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1, hidden_dims[depth-3-i] + 1)).view(hidden_dims[depth-i-2], hidden_dims[depth-3-i] + 1)
         
        derivative1 += 1.0/(dim[0])*curr_B
        
        der_list[2*depth - 2*i - 4] = derivative1[:,0:-1]
    
        der_list[2*depth - 2*i - 3] = derivative1[:,-1]
        
     
    para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)),1)
                 
    '''A: (hidden_dims[depth-i-1]*hidden_dims[depth-i-2])*hiddem_dims[depth-i-2]'''
 
    '''B: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
     
    '''weights: hidden_dims[depth-i-1]*hidden_dims[depth-i-2]'''
     
    curr_A = A_list[0] 
     
    curr_B = B_list[0]
    
    derivative1 = 1./(dim[0])*torch.bmm(para_curr_layer.view(hidden_dims[0], 1, input_dim + 1), curr_A.view(hidden_dims[0], input_dim + 1, input_dim + 1)).view(hidden_dims[0], input_dim + 1)
     
    derivative1 += 1.0/(dim[0])*curr_B
    
    der_list[0] = derivative1[:,0:-1]
    
    der_list[1] = derivative1[:,-1]
    
    return der_list
    

def compute_predictions(X, para_list):
    
    output = X
    
    for para in para_list:
        output = torch.mm(para_list, output)
        
        output = sigmoid_func(output)
        
        
    return output
    


def sigmoid_func_derivitive(x):
    
    res = sigmoid_func(x)
    
    
    
    return (1-res)*res


    
def compute_gradient_diff(curr_gradient_list, gradient_list):
    
    print("gradient diff:")
    
    for i in range(len(curr_gradient_list)):
        
        print(curr_gradient_list[i])
        
        print(gradient_list[i])
        
        
        print("gradient_dff::", i, torch.norm(curr_gradient_list[i]-gradient_list[i]))



def compute_updated_parameters(old_delta, expected_delta, delta_gradient_list, old_para_list, para_list):
    
    
    new_delta = old_delta.clone()
    
    
    for i in range(len(old_para_list)):
        para_updates = para_list[i] - old_para_list[i]
        
        for j in range(20):
            
            for k in range(old_delta.shape[1]):
                new_delta[j][k] += torch.sum(delta_gradient_list[j][k][i]*para_updates)
            
    
    print("delta_diff::", torch.norm(new_delta[0:20]-expected_delta[0:20]))
    
        
            
        
    
    
    
    
    
   
def compute_model_parameter_iteration(num_epochs, model, X, Y, alpha, error, dim, num_class, input_dim, hidden_dims, output_dim, para_list_all_epochs, delta_gradient_all_epochs, delta_all_epochs):
    
    loss = np.infty
    
    count = 0
    
#     for para in init_para_list:
#         para.requires_grad = True
        
    
    
    
    
    '''hidden_dim len: depth - 1'''
        
    '''output_list len: depth'''
    
    depth = len(hidden_dims) + 1

    para_list = list(model.parameters())
    
    print("depth::", depth)
    
    print("para_len::", len(para_list))

    while count < num_epochs:    
        
        
        print("iteration::", count)
        
#         compute_model_para_diff2(para_lists_all_epochs[count], para_list)
        
        t1  = time.time()
        
        
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
        old_para_list = para_list_all_epochs[count]
        
        
        old_delta_list = delta_all_epochs[count]
        
        delta_gradient_list = delta_gradient_all_epochs[count]
        
        
        output_list,_ = model.get_output_each_layer(X)
        
        pred = output_list[len(output_list) - 1]
        
#         loss = error(pred, Y)
        
        delta = softmax_func(pred) - get_onehot_y(Y, dim, num_class)
        
        
        '''delta: n*output_dim'''
        
        para_curr_layer = torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1) 
        
        compute_updated_parameters(old_delta_list, delta, delta_gradient_list, old_para_list, para_list)

#         print(output_list[depth - 1].shape)
        t3 = time.time()
#         delta = delta*sigmoid_func_derivitive(torch.sum(para_curr_layer.view(1, output_dim, hidden_dims[depth - 2] + 1)*output_list[depth - 1].view(dim[0], 1, hidden_dims[depth - 2] + 1), 2))
        delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
        t4 = time.time()

        


#         deriv = torch.bmm(delta.view(dim[0], output_dim, 1), output_list[depth - 1].view(dim[0], 1, hidden_dims[depth - 2] + 1))
        
        deriv = torch.mm(torch.t(delta), output_list[depth - 1])

        '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
        
        
#         delta_para_prod = torch.sum(torch.t(para_list[2*depth - 2].data).view(1, hidden_dims[depth - 2], output_dim)*delta.view(dim[0], 1, output_dim), 2)

        delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(delta)))
        
        para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        
        para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
        
        para_list[2*depth - 1].data = para_curr_layer[:, -1]
        
#         this_gradient_list = [None]*len(para_list)
#         
#         this_gradient_list[2*depth - 2] = (deriv[:,0:-1]/dim[0])
#          
#         this_gradient_list[2*depth - 1] = (deriv[:,-1]/dim[0])
         
        print("iteration::", count)
         
#         print("diff1::", torch.norm(curr_para_list[2*depth - 2].data - para_list[2*depth - 2].data))
#          
#         print("diff2::", torch.norm(curr_para_list[2*depth - 1].data - para_list[2*depth - 1].data))
        
        for i in range(depth - 2):
            
            '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
            
            para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1)
            
            
#             delta = delta_para_prod*sigmoid_func_derivitive(torch.sum(para_curr_layer.view(1, hidden_dims[depth - 2 - i], hidden_dims[depth - 3 - i] + 1)*output_list[depth - 2 - i].view(dim[0], 1, hidden_dims[depth - 3 - i]), 2))
        
            delta = delta_para_prod*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
            
#             deriv = torch.bmm(delta.view(dim[0], hidden_dims[depth - 2 - i], 1), output_list[depth - 2 - i].view(dim[0], 1, hidden_dims[depth- 3 - i] + 1))



            deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])

#             delta_para_prod = torch.sum(torch.t(para_list[2*depth - 2*i - 4].data).view(1, hidden_dims[depth - 3 - i], hidden_dims[depth - 2 - i])*delta.view(dim[0], 1, hidden_dims[depth - 2 - i]), 2)
            
            
            delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))            
            
            
#             para_list[len(para_list) - 2 - i].data = para_list[len(para_list) - 2 - i].data - alpha/dim[0]*torch.sum(deriv, 0)
            
            
            '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
            para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        
            para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
            
            para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
            
            
#             this_gradient_list[2*depth - 2*i - 4] = (deriv[:,0:-1]/dim[0])
#           
#             this_gradient_list[2*depth - 2*i - 3] = (deriv[:,-1]/dim[0])
          
#             print("diff1::", torch.norm(curr_para_list[2*depth - 2*i - 4].data - para_list[2*depth - 2*i - 4].data))
#           
#             print("diff2::", torch.norm(curr_para_list[2*depth - 2*i - 3].data - para_list[2*depth - 2*i - 3].data))
        
        para_curr_layer = torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1)
            
                    
#         delta = delta_para_prod*sigmoid_func_derivitive(torch.sum(para_curr_layer.view(1, hidden_dims[0], input_dim + 1)*output_list[0].view(dim[0], 1, input_dim + 1), 2))
#         print(para_curr_layer.shape)
#         
#         print(output_list[0].shape)

        delta = delta_para_prod*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
        
#         deriv = torch.bmm(delta.view(dim[0], hidden_dims[0], 1), output_list[0].view(dim[0], 1, input_dim + 1))

        deriv = torch.mm(torch.t(delta), output_list[0])

#         delta_para_prod = torch.sum(torch.t(para_list[0].data).view(1, input_dim, hidden_dims[0])*delta.view(dim[0], 1, hidden_dims[0]), 2)            
        
        
#             para_list[len(para_list) - 2 - i].data = para_list[len(para_list) - 2 - i].data - alpha/dim[0]*torch.sum(deriv, 0)
        
        
        '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
        para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
    
        para_list[0].data = para_curr_layer[:, 0:-1]
        
        para_list[1].data = para_curr_layer[:, -1]
        
#         this_gradient_list[0] = (deriv[:,0:-1]/dim[0])
#          
#         this_gradient_list[1] = (deriv[:,-1]/dim[0])
        
        t2  = time.time()
        
        print("time0:", t2 - t1)
        
        print("time1:", t4 - t3)
        
        
#         compute_gradient_diff(curr_gradient_list, this_gradient_list)
#         print("diff1::", torch.norm(curr_para_list[0].data - para_list[0].data))
#          
#         print("diff2::", torch.norm(curr_para_list[1].data - para_list[1].data))
        
        
        
        
#         delta = delta_para_prod*sigmoid_func_derivitive(torch.sum(para_list[0].data.view(1, hidden_dims[0], input_dim)*X.view(dim[0], 1, input_dim), 2))
#     
#         deriv = torch.bmm(delta.view(dim[0], hidden_dims[0], 1), torch.transpose(X, 1, 2).view(dim[0], 1, input_dim))
#         
#         
#         para_list[0].data = para_list[0].data - alpha/dim[0]*torch.sum(deriv, 0)
        
        
        '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
        
#         delta_para_prod = torch.sum(torch.t(para_list[0].data).view(1, hidden_dims[0], input_dim)*delta.view(dim[0], 1, hidden_dims[0]), 2)
        
        
        count += 1
        
        
        
        
#         for i in range(len(para_list)):
#             print(para_list[i].data)
#             
#             print(para_list[i].data.shape)
        
        
    print("iteration::", num_epochs)
        
#     compute_model_para_diff2(expected_model_paras, para_list)    

def compute_x_sum_by_class(X, Y, num_class, dim):
    
#     x_sum_by_class = torch.zeros([num_class, dim[1]], dtype = torch.double)
    
    
    y_onehot = torch.DoubleTensor(dim[0], num_class)

    Y = Y.type(torch.LongTensor)

# In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, Y.view(-1, 1), 1)
    
    
    x_sum_by_class = torch.mm(torch.t(y_onehot), X)
    
#     for i in range(num_class):
#         Y_mask = (Y == i)
#         
#         Y_mask = Y_mask.type(torch.DoubleTensor)
#         
#         x_sum_by_class[i] = torch.mm(torch.t(Y_mask), X) 
        
    return x_sum_by_class.view(-1, 1)

def compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, expected_gradient, beta):     
     
    output_list,_ = model.get_output_each_layer(X)
    
    para_list = list(model.parameters())
    
    pred = output_list[len(output_list) - 1]
    delta = softmax_func(pred) - get_onehot_y(Y, X.shape, output_dim)
#     delta = softmax_func(pred) - compute_x_sum_by_class(X, Y, output_dim, X.shape)
    
    depth = len(hidden_dims) + 1
    
    para_curr_layer = torch.cat((para_list[2*depth - 2], para_list[2*depth - 1].view(-1,1)), 1) 
    
    
    expected_gradient_curr_layer = torch.cat((expected_gradient[2*depth - 2], expected_gradient[2*depth - 1].view(-1,1)), 1) 
    
    
    delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
    
    
    deriv_list = [None]*(len(para_list))
    
    deriv = torch.mm(torch.t(delta), output_list[depth - 1])
    
#     deriv_list[depth-1] = deriv
    
    deriv_list[2*depth - 2] = deriv[:, 0:-1]/X.shape[0] + 2*beta*para_list[2*depth - 2]
            
    deriv_list[2*depth - 1] = deriv[:, -1]/X.shape[0] + 2*beta*para_list[2*depth - 1]
    
    
    
    
    delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2]),torch.t(delta)))

    for i in range(depth - 2):
        
        '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
        
        para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4], para_list[2*depth - 2*i - 3].view(-1,1)), 1)
            
#             delta = delta_para_prod*sigmoid_func_derivitive(torch.sum(para_curr_layer.view(1, hidden_dims[depth - 2 - i], hidden_dims[depth - 3 - i] + 1)*output_list[depth - 2 - i].view(dim[0], 1, hidden_dims[depth - 3 - i]), 2))
    
        delta = delta_para_prod*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
        
#             deriv = torch.bmm(delta.view(dim[0], hidden_dims[depth - 2 - i], 1), output_list[depth - 2 - i].view(dim[0], 1, hidden_dims[depth- 3 - i] + 1))



        deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])

#             delta_para_prod = torch.sum(torch.t(para_list[2*depth - 2*i - 4].data).view(1, hidden_dims[depth - 3 - i], hidden_dims[depth - 2 - i])*delta.view(dim[0], 1, hidden_dims[depth - 2 - i]), 2)
#         deriv_list[depth-2-i] = deriv
        
        deriv_list[2*depth - 2*i - 4] = deriv[:, 0:-1]/X.shape[0] + 2*beta*para_list[2*depth - 2*i - 4]
            
        deriv_list[2*depth - 2*i - 3] = deriv[:, -1]/X.shape[0] + 2*beta*para_list[2*depth - 2*i - 3]
        
        delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4]), torch.t(delta)))            
        
    para_curr_layer = torch.cat((para_list[0], para_list[1].view(-1,1)), 1)
    
    expected_gradient_curr_layer = torch.cat((expected_gradient[0], expected_gradient[1].view(-1,1)), 1) 

#         delta = delta_para_prod*sigmoid_func_derivitive(torch.sum(para_curr_layer.view(1, hidden_dims[0], input_dim + 1)*output_list[0].view(dim[0], 1, input_dim + 1), 2))
#         print(para_curr_layer.shape)
#         
#         print(output_list[0].shape)

    delta = delta_para_prod*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
    
#         deriv = torch.bmm(delta.view(dim[0], hidden_dims[0], 1), output_list[0].view(dim[0], 1, input_dim + 1))

    deriv = torch.mm(torch.t(delta), output_list[0])

#     deriv_list[0] = deriv

    deriv_list[0] = deriv[:, 0:-1]/X.shape[0] + 2*beta*para_list[0]
            
    deriv_list[1] = deriv[:, -1]/X.shape[0] + 2*beta*para_list[1]
    
    return deriv_list, para_list
#             para_list[len(para_list) - 2 - i].data = para_list[len(para_list) - 2 - i].data - alpha/dim[0]*torch.sum(deriv, 0)
    

     
def compute_model_derivitive_iteration(X, Y, alpha, error, dim, num_class, input_dim, hidden_dims, output_dim, expected_model_paras, gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, init_para_list):
    
    loss = np.infty
    
    count = 0


    para_list = []
    
    for para in init_para_list:
        
        para_list.append(para.data.clone())
        
        
#         para.requires_grad = True
        
    
    
    
    
    '''hidden_dim len: depth - 1'''
        
    '''output_list len: depth'''
    
    depth = len(hidden_dims) + 1

    print("depth::", depth)
    
    while count < num_epochs:    
        
        
        print("iteration::", count)
        
#         compute_model_para_diff2(para_lists_all_epochs[count], para_list)
        
        t1  = time.time()
        
        
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
        
        
        output_list = output_list_all_epochs[count]
        
        
#         para_list = para_list_all_epochs[count]
        
        pred = output_list[len(output_list) - 1]
        
        
        gradient_list = gradient_list_all_epochs[count]
        
#         loss = error(pred, Y)
        
        delta = softmax_func(pred) - get_onehot_y(Y, dim, num_class)
        
        
        expected_para_list = para_list_all_epochs[count]
        
        
        '''delta: n*output_dim'''
        
        para_curr_layer = torch.cat((para_list[2*depth - 2], para_list[2*depth - 1].view(-1,1)), 1) 
        
        curr_expected_para = torch.cat((expected_para_list[2*depth - 2], expected_para_list[2*depth - 1].view(-1,1)), 1)
        
        print(torch.norm(curr_expected_para - para_curr_layer))
        
#         print(output_list[depth - 1].shape)
        t3 = time.time()
#         delta = delta*sigmoid_func_derivitive(torch.sum(para_curr_layer.view(1, output_dim, hidden_dims[depth - 2] + 1)*output_list[depth - 1].view(dim[0], 1, hidden_dims[depth - 2] + 1), 2))
        delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
        t4 = time.time()

#         deriv = torch.bmm(delta.view(dim[0], output_dim, 1), output_list[depth - 1].view(dim[0], 1, hidden_dims[depth - 2] + 1))
        
        deriv = torch.mm(torch.t(delta), output_list[depth - 1])
        
        
        
        
        

        '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
        
        
#         delta_para_prod = torch.sum(torch.t(para_list[2*depth - 2].data).view(1, hidden_dims[depth - 2], output_dim)*delta.view(dim[0], 1, output_dim), 2)

        delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2]),torch.t(delta)))
        
        
        gradient_curr_layer = torch.cat((gradient_list[2*depth - 2].data, gradient_list[2*depth - 1].data.view(-1,1)), 1)

      
        gradient_delta = torch.norm(gradient_curr_layer - deriv/dim[0])
        
        print("gradient delta::", gradient_delta)
        
        
        para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        
        para_list[2*depth - 2] = para_curr_layer[:, 0:-1]
        
        para_list[2*depth - 1] = para_curr_layer[:, -1]
        
#         this_gradient_list = [None]*len(para_list)
#         
#         this_gradient_list[2*depth - 2] = (deriv[:,0:-1]/dim[0])
#          
#         this_gradient_list[2*depth - 1] = (deriv[:,-1]/dim[0])
         
        print("iteration::", count)
         
#         print("diff1::", torch.norm(curr_para_list[2*depth - 2].data - para_list[2*depth - 2].data))
#          
#         print("diff2::", torch.norm(curr_para_list[2*depth - 1].data - para_list[2*depth - 1].data))
        
        for i in range(depth - 2):
            
            '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
            
            para_curr_layer = torch.cat((para_list[2*depth - 2*i - 4], para_list[2*depth - 2*i - 3].view(-1,1)), 1)
            
            curr_expected_para = torch.cat((expected_para_list[2*depth - 2*i - 4], expected_para_list[2*depth - 2*i - 3].view(-1,1)), 1)
        
            print(torch.norm(curr_expected_para - para_curr_layer))
#             delta = delta_para_prod*sigmoid_func_derivitive(torch.sum(para_curr_layer.view(1, hidden_dims[depth - 2 - i], hidden_dims[depth - 3 - i] + 1)*output_list[depth - 2 - i].view(dim[0], 1, hidden_dims[depth - 3 - i]), 2))
        
            delta = delta_para_prod*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
            
#             deriv = torch.bmm(delta.view(dim[0], hidden_dims[depth - 2 - i], 1), output_list[depth - 2 - i].view(dim[0], 1, hidden_dims[depth- 3 - i] + 1))



            deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])


            gradient_curr_layer = torch.cat((gradient_list[2*depth - 2*i - 4].data, gradient_list[2*depth - 2*i - 3].data.view(-1,1)), 1)

      
            gradient_delta = torch.norm(gradient_curr_layer - deriv/dim[0])
            
            print("gradient delta::", gradient_delta)


#             delta_para_prod = torch.sum(torch.t(para_list[2*depth - 2*i - 4].data).view(1, hidden_dims[depth - 3 - i], hidden_dims[depth - 2 - i])*delta.view(dim[0], 1, hidden_dims[depth - 2 - i]), 2)
            
            
            delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4]), torch.t(delta)))            
            
            
#             para_list[len(para_list) - 2 - i].data = para_list[len(para_list) - 2 - i].data - alpha/dim[0]*torch.sum(deriv, 0)
            
            
            '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
            para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        
            para_list[2*depth - 2*i - 4] = para_curr_layer[:, 0:-1]
            
            para_list[2*depth - 2*i - 3] = para_curr_layer[:, -1]
            
            
#             this_gradient_list[2*depth - 2*i - 4] = (deriv[:,0:-1]/dim[0])
#           
#             this_gradient_list[2*depth - 2*i - 3] = (deriv[:,-1]/dim[0])
          
#             print("diff1::", torch.norm(curr_para_list[2*depth - 2*i - 4].data - para_list[2*depth - 2*i - 4].data))
#           
#             print("diff2::", torch.norm(curr_para_list[2*depth - 2*i - 3].data - para_list[2*depth - 2*i - 3].data))
        
        para_curr_layer = torch.cat((para_list[0], para_list[1].view(-1,1)), 1)
        
        curr_expected_para = torch.cat((expected_para_list[0], expected_para_list[1].view(-1,1)), 1)
        
        print(torch.norm(curr_expected_para - para_curr_layer))

        
                    
#         delta = delta_para_prod*sigmoid_func_derivitive(torch.sum(para_curr_layer.view(1, hidden_dims[0], input_dim + 1)*output_list[0].view(dim[0], 1, input_dim + 1), 2))
#         print(para_curr_layer.shape)
#         
#         print(output_list[0].shape)

        delta = delta_para_prod*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
        
#         deriv = torch.bmm(delta.view(dim[0], hidden_dims[0], 1), output_list[0].view(dim[0], 1, input_dim + 1))

        deriv = torch.mm(torch.t(delta), output_list[0])


        gradient_curr_layer = torch.cat((gradient_list[0].data, gradient_list[1].data.view(-1,1)), 1)

      
        gradient_delta = torch.norm(gradient_curr_layer - deriv/dim[0])
        
        print("gradient delta::", gradient_delta)


#         delta_para_prod = torch.sum(torch.t(para_list[0].data).view(1, input_dim, hidden_dims[0])*delta.view(dim[0], 1, hidden_dims[0]), 2)            
        
        
#             para_list[len(para_list) - 2 - i].data = para_list[len(para_list) - 2 - i].data - alpha/dim[0]*torch.sum(deriv, 0)
        
        
        '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
        para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
    
        para_list[0] = para_curr_layer[:, 0:-1]
        
        para_list[1] = para_curr_layer[:, -1]
        
#         this_gradient_list[0] = (deriv[:,0:-1]/dim[0])
#          
#         this_gradient_list[1] = (deriv[:,-1]/dim[0])
        
        t2  = time.time()
        
        print("time0:", t2 - t1)
        
        print("time1:", t4 - t3)
        
        
#         compute_gradient_diff(curr_gradient_list, this_gradient_list)
#         print("diff1::", torch.norm(curr_para_list[0].data - para_list[0].data))
#          
#         print("diff2::", torch.norm(curr_para_list[1].data - para_list[1].data))
        
        
        
        
#         delta = delta_para_prod*sigmoid_func_derivitive(torch.sum(para_list[0].data.view(1, hidden_dims[0], input_dim)*X.view(dim[0], 1, input_dim), 2))
#     
#         deriv = torch.bmm(delta.view(dim[0], hidden_dims[0], 1), torch.transpose(X, 1, 2).view(dim[0], 1, input_dim))
#         
#         
#         para_list[0].data = para_list[0].data - alpha/dim[0]*torch.sum(deriv, 0)
        
        
        '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
        
#         delta_para_prod = torch.sum(torch.t(para_list[0].data).view(1, hidden_dims[0], input_dim)*delta.view(dim[0], 1, hidden_dims[0]), 2)
        
        
        count += 1
        
        
        
        
#         for i in range(len(para_list)):
#             print(para_list[i].data)
#             
#             print(para_list[i].data.shape)
        
        
    print("iteration::", num_epochs)
        
    compute_model_para_diff2(expected_model_paras, para_list)    
   


def capture_provenance(X, Y, w_list_all_epochs, b_list_all_epochs, output_list_all_epochs, para_list_all_epochs, dim, num_class, input_dim, hidden_dims, output_dim, gradient_list_all_epochs):
    
#     t1  = time.time()
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
    A_list_all_epochs = []
    
    B_list_all_epochs = []
    
    w_delta_prod_list_all_epochs = []
    
    b_delta_prod_list_all_epochs = []
    
    with torch.no_grad():
        
        for k in range(len(w_list_all_epochs)):
            w_res = w_list_all_epochs[k]
            
            b_res = b_list_all_epochs[k]
            
            
            
            output_list = output_list_all_epochs[k]
            
            pred = output_list[len(output_list) - 1]
            
            para_list = para_list_all_epochs[k]
        
        
            depth = len(hidden_dims) + 1
        
            A_list = [None]*depth
        
        
            B_list = [None]*depth
            
            
        #     A0_list = [None]*depth
        #     
        #     B0_list = [None]*depth
        #         loss = error(pred, Y)
            
            delta = Variable(softmax_func(pred) - get_onehot_y(Y, dim, num_class))
            
            
            '''delta: n*output_dim'''
            
            para_curr_layer = Variable(torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1)) 
            
            
            '''A: output_dim, hidden_dim[depth-2]^2'''
            
        #     print(len(w_res))
            
        #     print(w_res[depth - 1])
            
        #     print(depth)
            
            
            w_delta_prod_list = [None]*depth
            
            b_delta_prod_list = [None]*depth
            
            
            '''w_delta_prod: n*output_dim'''
            
            w_delta_prod = Variable(w_res[depth - 1]*delta)
            
            b_delta_prod = Variable(b_res[depth - 1]*delta)
            
            w_delta_prod_list[depth - 1] = w_delta_prod
            
            b_delta_prod_list[depth - 1] = b_delta_prod
            
            '''A: output_dim*(hidden_dims[-1]*hiddem_dims[-1])'''
            A = Variable(torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(dim[0], hidden_dims[depth-2] + 1, 1), output_list[depth - 1].view(dim[0], 1, hidden_dims[depth-2]+ 1)).view(dim[0], -1)))
            
        #     A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1])
            
        #     A0_list.append(A0)
            
            '''B: output_dim, hidden_dim[depth-2]'''
            B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-1]))
            
        #     B0 = torch.t(torch.sum(b_delta_prod, 0))
            
        #     B0_list.append(B0)
            
            
            
            A_list[depth - 1] = A
            
            B_list[depth - 1] = B
            
            
            
            w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
            
            
            entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
            
            
        #     delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
        #     
        #     deriv = torch.mm(torch.t(delta), output_list[depth - 1])
        
            '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
        
            delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(entire_delta_expression))))
            
        #     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        #     
        #     para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
        #     
        #     para_list[2*depth - 1].data = para_curr_layer[:, -1]
                
            for i in range(depth - 2):
                
                '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
                
                para_curr_layer = Variable(torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1))
                
                delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
                        
                w_delta_prod = Variable(w_res[depth - i - 2]*delta)
            
                b_delta_prod = Variable(b_res[depth - i - 2]*delta)
                
                w_delta_prod_list[depth - i - 2] = w_delta_prod
            
                b_delta_prod_list[depth - i - 2] = b_delta_prod
                
                A = Variable(torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - i - 2].view(dim[0], hidden_dims[depth - i - 3]+ 1, 1), output_list[depth - i - 2].view(dim[0], 1, hidden_dims[depth - i - 3]+ 1)).view(dim[0], -1)))
                
        #         A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - i - 2])
                
        #         A0_list.append(A0)
                
                '''B: output_dim, hidden_dim[depth-2]'''
                B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-i-2]))
                
        #         B0 = torch.t(torch.sum(b_delta_prod, 0))
                
        #         B0_list.append(B0)
                
                w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - i - 2]))))
                
                
                entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
                
                
                delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(entire_delta_expression))))
                
                A_list[depth - 2 - i] = A
                
                B_list[depth - 2 - i] = B
                
        #         deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])
        #         
        #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))                    
        #         
        #         '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
        #         para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
        #     
        #         para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
        #         
        #         para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
                    
            para_curr_layer = Variable(torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1))
        
            delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
            
            w_delta_prod = Variable(w_res[0]*delta)
        
            b_delta_prod = Variable(b_res[0]*delta)
            
            w_delta_prod_list[0] = w_delta_prod
            
            b_delta_prod_list[0] = b_delta_prod
            
            
            w_delta_prod_list_all_epochs.append(w_delta_prod_list)
            
            b_delta_prod_list_all_epochs.append(b_delta_prod_list)
            
        #     A = torch.mm(torch.bmm(w_delta_prod.view(dim[0], )))
            
        #     A = torch.zeros([hidden_dims[0], (input_dim+1)*(input_dim+1)])
            
            
            A = Variable(torch.mm(torch.t(output_list[0]), torch.bmm(w_delta_prod.view(dim[0], hidden_dims[0], 1), output_list[0].view(dim[0], 1, input_dim + 1)).view(dim[0], hidden_dims[0]*(input_dim + 1))))
            
            A = Variable(torch.transpose(A.view(input_dim + 1, hidden_dims[0], input_dim + 1), 0, 1))
            
        #     for k in range(dim[0]):
        #         A += torch.mm(torch.t(w_delta_prod)[:,k].view(-1,1), torch.mm(output_list[0][k].view(input_dim + 1, 1), output_list[0][k].view(1, input_dim + 1)).view(1,-1))
            
        #     A = torch.mm( temp)
            
            '''B: output_dim, hidden_dim[depth-2]'''
            B = Variable(torch.mm(torch.t(b_delta_prod), output_list[0]))
            
            
            A_list[0] = A
            
            B_list[0] = B
        
            A_list_all_epochs.append(A_list)
            
            B_list_all_epochs.append(B_list)
    
#     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[0])))
#     
#     
#     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
    
#     deriv = torch.mm(torch.t(delta), output_list[0])
#     
#     '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
# 
#     para_list[0].data = para_curr_layer[:, 0:-1]
#     
#     para_list[1].data = para_curr_layer[:, -1]

#     t2  = time.time()
    
#     print("time0:", t2 - t1)   
    
    
    return A_list_all_epochs, B_list_all_epochs, w_delta_prod_list_all_epochs, b_delta_prod_list_all_epochs
    
    
    
    
#     for param in param_list:
#         
#         delta =  
        
    
    
def capture_provenance2(X, Y, w_list, b_list, output_list, para_list, dim, num_class, input_dim, hidden_dims, output_dim, gradient_list):
     
#     t1  = time.time()
#         curr_gradient_list = gradient_lists_all_epochs[count]
        
#     hessian_matrix = compute_hessian_matrix(model, para_list, gradient_list, input_dim, hidden_dims, output_dim)    
        
    
#     A_list_all_epochs = []
#      
#     B_list_all_epochs = []
#      
#     w_delta_prod_list_all_epochs = []
#      
#     b_delta_prod_list_all_epochs = []
     
     
    depth = len(hidden_dims) + 1
    
    w_delta_prod_list = [None]*depth
     
    b_delta_prod_list = [None]*depth
     
    with torch.no_grad():
         
#         for k in range(len(w_list_all_epochs)):
        w_res = w_list
         
        b_res = b_list
         
         
         
#         output_list = output_list_all_epochs[k]
         
        pred = output_list[len(output_list) - 1]
         
#         para_list = para_list_all_epochs[k]
     
     
        A_list = [None]*depth
     
     
        B_list = [None]*depth
         
         
    #     A0_list = [None]*depth
    #     
    #     B0_list = [None]*depth
    #         loss = error(pred, Y)
         
        delta = Variable(softmax_func(pred) - get_onehot_y(Y, dim, num_class))
         
         
        '''delta: n*output_dim'''
         
        para_curr_layer = Variable(torch.cat((para_list[2*depth - 2].data, para_list[2*depth - 1].data.view(-1,1)), 1)) 
         
         
        '''A: output_dim, hidden_dim[depth-2]^2'''
         
    #     print(len(w_res))
         
    #     print(w_res[depth - 1])
         
    #     print(depth)
         
         
         
        '''w_delta_prod: n*output_dim'''
         
        w_delta_prod = Variable(w_res[depth - 1]*delta)
         
        b_delta_prod = Variable(b_res[depth - 1]*delta)
         
        w_delta_prod_list[depth - 1] = w_delta_prod
         
        b_delta_prod_list[depth - 1] = b_delta_prod
         
        '''A: output_dim*(hidden_dims[-1]*hiddem_dims[-1])'''
#         A = Variable(torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - 1].view(dim[0], hidden_dims[depth-2] + 1, 1), output_list[depth - 1].view(dim[0], 1, hidden_dims[depth-2]+ 1)).view(dim[0], -1)))
         
    #     A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - 1])
         
    #     A0_list.append(A0)
         
        '''B: output_dim, hidden_dim[depth-2]'''
#         B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-1]))
         
    #     B0 = torch.t(torch.sum(b_delta_prod, 0))
         
    #     B0_list.append(B0)
         
         
         
#         A_list[depth - 1] = A
#          
#         B_list[depth - 1] = B
         
         
         
        w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
         
         
        entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
         
         
    #     delta = delta*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 1]))))
    #     
    #     deriv = torch.mm(torch.t(delta), output_list[depth - 1])
     
        '''delta_para_prod: n*hidden_dims[len(para_list) - 1]'''
     
        delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2].data),torch.t(entire_delta_expression))))
         
    #     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
    #     
    #     para_list[2*depth - 2].data = para_curr_layer[:, 0:-1]
    #     
    #     para_list[2*depth - 1].data = para_curr_layer[:, -1]
             
        for i in range(depth - 2):
             
            '''delta: n*hidden_dims[len(hidden_dims) - 1 - i]'''
             
            para_curr_layer = Variable(torch.cat((para_list[2*depth - 2*i - 4].data, para_list[2*depth - 2*i - 3].data.view(-1,1)), 1))
             
            delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - 2 - i]))))
                     
            w_delta_prod = Variable(w_res[depth - i - 2]*delta)
         
            b_delta_prod = Variable(b_res[depth - i - 2]*delta)
             
            w_delta_prod_list[depth - i - 2] = w_delta_prod
         
            b_delta_prod_list[depth - i - 2] = b_delta_prod
             
            A = Variable(torch.mm(torch.t(w_delta_prod), torch.bmm(output_list[depth - i - 2].view(dim[0], hidden_dims[depth - i - 3]+ 1, 1), output_list[depth - i - 2].view(dim[0], 1, hidden_dims[depth - i - 3]+ 1)).view(dim[0], -1)))
             
    #         A0 = torch.mm(torch.t(w_delta_prod), output_list[depth - i - 2])
             
    #         A0_list.append(A0)
             
            '''B: output_dim, hidden_dim[depth-2]'''
            B = Variable(torch.mm(torch.t(b_delta_prod), output_list[depth-i-2]))
             
    #         B0 = torch.t(torch.sum(b_delta_prod, 0))
             
    #         B0_list.append(B0)
             
            w_input_prod = Variable(torch.t(torch.mm(para_curr_layer, torch.t(output_list[depth - i - 2]))))
             
             
            entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
             
             
            delta_para_prod = Variable(torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data),torch.t(entire_delta_expression))))
             
            A_list[depth - 2 - i] = A
             
            B_list[depth - 2 - i] = B
             
    #         deriv = torch.mm(torch.t(delta), output_list[depth - 2 - i])
    #         
    #         delta_para_prod = torch.t(torch.mm(torch.t(para_list[2*depth - 2*i - 4].data), torch.t(delta)))                    
    #         
    #         '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
    #         para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
    #     
    #         para_list[2*depth - 2*i - 4].data = para_curr_layer[:, 0:-1]
    #         
    #         para_list[2*depth - 2*i - 3].data = para_curr_layer[:, -1]
                 
        para_curr_layer = Variable(torch.cat((para_list[0].data, para_list[1].data.view(-1,1)), 1))
     
        delta = Variable(delta_para_prod)#*sigmoid_func_derivitive(torch.t(torch.mm(para_curr_layer, torch.t(output_list[0]))))
         
        w_delta_prod = Variable(w_res[0]*delta)
     
        b_delta_prod = Variable(b_res[0]*delta)
         
        w_delta_prod_list[0] = w_delta_prod
         
        b_delta_prod_list[0] = b_delta_prod
         
    
    
    return w_delta_prod_list, b_delta_prod_list
       
#         w_delta_prod_list_all_epochs.append(w_delta_prod_list)
         
#         b_delta_prod_list_all_epochs.append(b_delta_prod_list)
         
    #     A = torch.mm(torch.bmm(w_delta_prod.view(dim[0], )))
         
    #     A = torch.zeros([hidden_dims[0], (input_dim+1)*(input_dim+1)])
         
         
#         A = Variable(torch.mm(torch.t(output_list[0]), torch.bmm(w_delta_prod.view(dim[0], hidden_dims[0], 1), output_list[0].view(dim[0], 1, input_dim + 1)).view(dim[0], hidden_dims[0]*(input_dim + 1))))
#          
#         A = Variable(torch.transpose(A.view(input_dim + 1, hidden_dims[0], input_dim + 1), 0, 1))
#          
#     #     for k in range(dim[0]):
#     #         A += torch.mm(torch.t(w_delta_prod)[:,k].view(-1,1), torch.mm(output_list[0][k].view(input_dim + 1, 1), output_list[0][k].view(1, input_dim + 1)).view(1,-1))
#          
#     #     A = torch.mm( temp)
#          
#         '''B: output_dim, hidden_dim[depth-2]'''
#         B = Variable(torch.mm(torch.t(b_delta_prod), output_list[0]))
#          
#          
#         A_list[0] = A
#          
#         B_list[0] = B
#      
#         A_list_all_epochs.append(A_list)
#          
#         B_list_all_epochs.append(B_list)
     
#     w_input_prod = torch.t(torch.mm(para_curr_layer, torch.t(output_list[0])))
#     
#     
#     entire_delta_expression = w_delta_prod*w_input_prod + b_delta_prod
     
#     deriv = torch.mm(torch.t(delta), output_list[0])
#     
#     '''delta_para_prod: n*hidden_dims[len(para_list) - 2 - i]'''
#     para_curr_layer = para_curr_layer - alpha/dim[0]*deriv
# 
#     para_list[0].data = para_curr_layer[:, 0:-1]
#     
#     para_list[1].data = para_curr_layer[:, -1]
 
#     t2  = time.time()
     
#     print("time0:", t2 - t1)   
     
     
     
#     for param in param_list:
#         
#         delta =  


def compute_linearized_coefficieent_each_epoch(depth, para_list, output_list, Pi):
    w_list = [None]*depth
            
    b_list = [None]*depth
            
        #     for m in model.children():
    for i in range(depth):
        
        
#         if type(m) == nn.modules.activation.Sigmoid: 
        
#             curr_weight = para_list[2*i].view(1, para_list[2*i].shape[0], para_list[2*i].shape[1])

            '''para_curr_layer:: hidden_dims[i], hidden_dims[i-1]'''
            
            para_curr_layer = torch.cat((para_list[2*depth - 2*i - 2].data, para_list[2*depth - 2*i - 1].data.view(-1,1)), 1)
            
#             curr_weight = para_list[2*i]
            
#             curr_input = output_list[i].view(output_list[i].shape[0], 1, output_list[i].shape[1])
            '''curr_input:: n, hidden_dims[i-1]'''
            curr_input = Variable(output_list[depth - i - 1])
            
            
#             print(para_curr_layer.shape)
            
#             print(curr_input.shape)
            
#             curr_offset = para_list[2*i+1]
            
#             print(curr_offset.shape)
            
#             print(torch.sum(curr_weight*curr_input, 2).shape)
            '''w_paras: n, hidden_dims[i]'''
            w_paras, b_paras = Pi.piecewise_linear_interpolate_coeff_batch2(torch.t(torch.mm(para_curr_layer, torch.t(curr_input))))
        
        
            w_list[depth - i - 1] = w_paras
            
            b_list[depth - i - 1] = b_paras
            
    return w_list, b_list


def compute_linearized_coeffcient_single_epoch(X, input_dim, hidden_dims, output_dim, output_list, para_list):
    Pi = create_piecewise_linea_class(sigmoid_diff_function)
        
        
    depth = len(hidden_dims) + 1
    
    
    return compute_linearized_coefficieent_each_epoch(depth, para_list, output_list, Pi)
    


def compute_linearized_coefficient(X, input_dim, hidden_dims, output_dim, output_list_all_epochs, para_list_all_epochs):
    
    
    w_list_all_epochs = []
    
    b_list_all_epochs = []


    Pi = create_piecewise_linea_class(sigmoid_diff_function)
        
        
    depth = len(hidden_dims) + 1

    with torch.no_grad():
    
        for k in range(len(output_list_all_epochs)):
            
        
        
            para_list = para_list_all_epochs[k]#list(model.parameters())
            
            
            output_list = output_list_all_epochs[k]#model.get_output_each_layer(X)
            

            w_list, b_list = compute_linearized_coefficieent_each_epoch(depth, para_list, output_list, Pi)
            
            w_list_all_epochs.append(w_list)
            
            b_list_all_epochs.append(b_list)
#                 i += 1
            
            
            
    return w_list_all_epochs, b_list_all_epochs
            
#         print(type(m))
# 
#         if t

#         print(param.grad)
        
        
#         for grd in param.grad:
#             for single_grad in grd:
#                 print(single_grad)
#                 
#                 model.zero_grad()
#                 
#                 single_grad.backward()
#                 
#                 print(single_grad.grad)


def compute_hessian_matrix(model, gradient_list, input_dim, hidden_dims, output_dim):
    
    
    total_attr_num = 0
    
    para_list = list(model.parameters())
    
    for k in range(len(hidden_dims) -1):
        total_attr_num += hidden_dims[k] * hidden_dims[k+1]
    
    total_attr_num += hidden_dims[0]*(input_dim + 1)
    
    total_attr_num += output_dim*(hidden_dims[-1]+1)
    
    print("attribute num::", total_attr_num)
    
    hessian_matrix = torch.zeros([total_attr_num, total_attr_num], dtype = torch.float64)
    
    print("init_tensor_done!!")
    
    clear_gradients(para_list)
    
    
    i = 0
    
    for k in range(len(gradient_list)):
        
        curr_gradient = gradient_list[k]
        
        
        
        for m in range(curr_gradient.shape[0]):
            
            if(len(curr_gradient.shape) > 1):
            
                for l in range(curr_gradient.shape[1]):
                
#                     print(i)
                
                    curr_gradient[m][l].backward(retain_graph = True)
                    
                    vectorized_gradient = get_all_vectorized_gradients(para_list)
                    
                    clear_gradients(para_list)
                    
                    hessian_matrix[:,i] = vectorized_gradient
                
                    i = i + 1
        
    
            else:
#                 print(i)
                
                curr_gradient[m].backward(retain_graph = True)
                
                vectorized_gradient = get_all_vectorized_gradients(para_list)
                
                clear_gradients(para_list)
                
                hessian_matrix[:,i] = vectorized_gradient
            
                i = i + 1
    
    return hessian_matrix
    

# def compute_near_hessian_matrix(model, gradient_list, input_dim, hidden_dims, output_dim):
#     
#     
#     total_attr_num = 0
#     
#     para_list = list(model.parameters())
#     
#     for k in range(len(hidden_dims) -1):
#         total_attr_num += hidden_dims[k] * hidden_dims[k+1]
#     
#     total_attr_num += hidden_dims[0]*(input_dim + 1)
#     
#     total_attr_num += output_dim*(hidden_dims[-1]+1)
#     
#     print("attribute num::", total_attr_num)
#     
#     hessian_matrix = torch.zeros([total_attr_num, total_attr_num], dtype = torch.float32)
#     
#     print("init_tensor_done!!")
#     
#     clear_gradients(para_list)
#     
#     
#     i = 0
#     
#     for k in range(len(gradient_list)):
#         
#         curr_gradient = gradient_list[k]
#         
#         
#         
#         for m in range(curr_gradient.shape[0]):
#             
#             if(len(curr_gradient.shape) > 1):
#             
#                 for l in range(curr_gradient.shape[1]):
#                 
#                     print(i)
#                 
#                     curr_gradient[m][l].backward(retain_graph = True)
#                     
#                     vectorized_gradient = get_all_vectorized_gradients(para_list)
#                     
#                     clear_gradients(para_list)
#                     
#                     hessian_matrix[:,i] = vectorized_gradient
#                 
#                     i = i + 1
#         
#     
#             else:
#                 print(i)
#                 
#                 curr_gradient[m].backward(retain_graph = True)
#                 
#                 vectorized_gradient = get_all_vectorized_gradients(para_list)
#                 
#                 clear_gradients(para_list)
#                 
#                 hessian_matrix[:,i] = vectorized_gradient
#             
#                 i = i + 1
#     
#     return hessian_matrix
def compute_loss(model, error, X, Y, regularization_coeff):
    
    train = Variable(X)
    labels = Variable(Y.view(-1))
    
    # Clear gradients
#         optimizer.zero_grad()
    
    # Forward propagation
    outputs = model(train)
    
    # Calculate softmax and ross entropy loss
    
#         print(outputs)
#         
#         print(labels)
    
    labels = labels.type(torch.LongTensor)
    
    loss = error(outputs, labels) + regularization_coeff*get_regularization_term(model.parameters())
    
    return loss


def compute_delta_constant(X, Y, para1, para2, gradient1, hessian_para_prod, error, model, beta):
    
    init_model(model, para1)
    
    loss1 = compute_loss(model, error, X, Y, beta)
    
    init_model(model, para2)
    
    loss2 = compute_loss(model, error, X, Y, beta)
    
    para_diff = (get_all_vectorized_parameters(para2) - get_all_vectorized_parameters(para1))
    
    delta_loss = loss2 - loss1 - torch.mm(gradient1.view(1,-1), para_diff.view(-1,1)) - 0.5*torch.mm(para_diff.view(1,-1), hessian_para_prod)
    
    epsilon = delta_loss/(torch.mm(para_diff.view(1,-1), para_diff.view(-1,1)))
    
    return epsilon
    
    
    
    
    
    
    

def compute_derivative_one_more_step(model, error, X, Y, beta):
    
    
    loss = compute_loss(model, error, X, Y, beta)
    
#     train = Variable(X)
#     labels = Variable(Y.view(-1))
#     
#     # Clear gradients
# #         optimizer.zero_grad()
#     
#     # Forward propagation
#     outputs = model(train)
#     
#     # Calculate softmax and ross entropy loss
#     
# #         print(outputs)
# #         
# #         print(labels)
#     
#     labels = labels.type(torch.LongTensor)
    
#     loss = error(outputs, labels)
    
    
#         print("loss0::", loss)
# 
#         loss = loss_function2(outputs, labels, X.shape)
    
    # Calculating gradients
#         loss.backward(retain_graph = True, create_graph=True)
    
    loss.backward()
    
    return loss



def add_small_variation(para_list):
    
    res_para_list = []
    
    
    for i in range(len(para_list)):
        res_para_list.append(para_list[i] + 0.00001)

    return res_para_list

def verify_hessian_matrix(hessian_mat, para_list, model, input_dim, hidden_dims, output_dim, X, Y, gradient_list):
    
    
    updated_para_list = add_small_variation(para_list)
    
    
    init_model(model, updated_para_list)
    
    udpated_gradient_list, _ = compute_gradient_iteration(model, input_dim, hidden_dims, output_dim, X, Y, gradient_list)
    
    expected_gradient_delta = torch.mm(hessian_mat, torch.t(get_all_vectorized_parameters(updated_para_list) - get_all_vectorized_parameters(para_list)))/X.shape[0]
    
    real_gradient_delta = get_all_vectorized_parameters(udpated_gradient_list)/X.shape[0] - get_all_vectorized_parameters(gradient_list)


    print(expected_gradient_delta - real_gradient_delta)
    
    
    
    
    

def compute_first_derivative(model, X, Y, error, beta):
    
    outputs = model(X)
    
    labels = Y.view(-1).type(torch.LongTensor)
        
#     loss = error(outputs, labels)
    loss = compute_loss(model, error, X, Y, beta)
    
    loss.backward()
    
    first_derivative = get_all_vectorized_gradients(list(model.parameters()))
    
    
#     zero_model_gradient(model)
    return first_derivative



def add_noise_data(X, Y, num, num_class, model):
    
    
#     X_distance = torch.sqrt(torch.bmm(X.view(dim[0], 1, dim[1]), X.view(dim[0],dim[1], 1))).view(-1,1)
    
    expected_selected_label =0
    
    updated_selected_label = 0
    
    max_count = -1
    
    min_count = Y.shape[0] + 1
    
    
    mean_list = [] 
    
    for i in range(num_class):
        
        curr_mean = torch.mean(X[Y.view(-1) == i], 0)
        
        mean_list.append(curr_mean)
        
        
        label_count = torch.sum((Y == i)) 
        if label_count > max_count:
            max_count = label_count
            expected_selected_label = i
        
        if label_count < min_count:
            min_count = label_count
            updated_selected_label = i
     
     
    '''n*q'''
    multi_res = model(X)
    
    prob, predict_labels = torch.max(multi_res, 1)
    
    print(prob)
    
#     predict_labels = torch.argmax(multi_res, 1)
    
    
    sorted_prob, indices = torch.sort(prob.view(-1), descending = True)
#     sorted, indices = torch.sort(torch.abs(multi_res), dim = 0, descending = True)
    selected_point = None
    
    selected_label = None
    
    selected_id = 0
    
    
    selected_points = []
    
    
    noise_data_X = torch.zeros((num, X.shape[1]), dtype = torch.float64)

    noise_data_Y = torch.zeros((num, 1), dtype = torch.long)
    
    for i in range(num):
        
        curr_class = Y[indices[i], 0]
        
        curr_coeff = mean_list[curr_class]/mean_list[(curr_class + 1)%(num_class)]        
         
        curr_coeff = curr_coeff[curr_coeff != np.inf]
        
        curr_coeff = torch.sum(curr_coeff[curr_coeff == curr_coeff])
         
#         curr_coeff = torch.sum(curr_coeff[curr_coeff != np.inf and np.isnan(curr_coeff.numpy())])
        
#         print(curr_coeff)
        
        selected_point = (X[indices[i]].clone())*curr_coeff*5
        
        
        if predict_labels[indices[i]] == curr_class:
            selected_label = (curr_class + 1)%num_class
        else:
            selected_label = curr_class
        
        noise_data_X[i,:] = selected_point
        
        noise_data_Y[i] = selected_label
        
        
    X = torch.cat([X, noise_data_X], 0)
        
    Y = torch.cat([Y, noise_data_Y], 0)    
        
        
        
#     class_list = []
#     
#     for j in range(num_class):
# #         if j == expected_selected_label:
#             for i in range(indices.shape[0]):
#                 
#                 if Y[indices[i], 0].numpy() == j and predict_labels[indices[i]].numpy() == j:
#                     selected_point = X[indices[i]].clone()
#                     selected_points.append(selected_point)
#                     class_list.append(j)
#                     
#                     selected_id = indices[i]
#                     selected_label = updated_selected_label
#                     break
#     
#         
#         
#     selected_num = int(num/len(selected_points))
#     
#     for i in range(len(selected_points)):
#         selected_point = selected_points[i]
# #     for selected_point in selected_points:
#     
# 
#         print(torch.mm(selected_point.view(1,-1), res))        
#                 
#         curr_coeff = mean_list[class_list[i]]/mean_list[class_list[(i+1)%(len(class_list))]]        
#          
#         curr_coeff = torch.mean(curr_coeff[curr_coeff != np.inf])
#            
#         
# #         selected_point = selected_point - 5*(mean_list[class_list[i]] - mean_list[updated_selected_label])# + torch.rand(selected_point.shape, dtype = torch.double)
#         selected_point = selected_point*curr_coeff
#         
#         print('distance::', torch.mm(selected_point.view(1,-1), res))       
#         
#         dist_range = torch.rand(selected_point.view(-1).shape, dtype = torch.double)
#         
#         
#         dist = torch.distributions.Normal(selected_point.view(-1), dist_range)
#     
#     
# #     noise_X = []
# #     
# #     for i in range(num):
# #         
# #         noise_X.append(dist.sample())
# #     
# #     
# #     noise_X = torch.cat(noise_X, 0)
# 
#         noise_X = dist.sample((selected_num,))
#         
#         noise_Y = torch.zeros([selected_num, 1], dtype = torch.long)
#         
#         
#         noise_Y[:,0] = class_list[(i+1)%(len(class_list))]
#         
#         X = torch.cat([X, noise_X], 0)
#         
#         Y = torch.cat([Y, noise_Y], 0)
    
    
    
    
    
    
    
    
    
    
    
#     uniqe_Y_values = torch.unique(Y)
#     
#     
#     new_X = torch.zeros([num, X.shape[1]], dtype= torch.double)
#     
#     new_Y = torch.zeros([num, Y.shape[1]], dtype= torch.double)
#     
#     for i in range(num):
#         curr_X = torch.rand(X.shape[1], dtype = torch.double)
#         
# #         curr_Y = uniqe_Y_values[torch.randint(low = 0, high = uniqe_Y_values.shape[0], size = 1)]
# 
#         curr_Y = uniqe_Y_values[torch.LongTensor(1).random_(0, uniqe_Y_values.shape[0])]
#         
#         new_X[i] = curr_X
#         
#         new_Y[i] = curr_Y
#         
# #         X = torch.cat((X, curr_X.view(1,-1)), 0)
# #         
# #         Y = torch.cat((Y, curr_Y.view(1,-1)), 0)
#         
#     X = torch.cat((X, new_X), 0)
#     
#     Y = torch.cat((Y, new_Y), 0)    
    
    return X, Y

def compute_taylor_expansion(dim, loss_1, gradient_1, para_1, para_2, hessian_1):
    
    del_para = get_all_vectorized_parameters(para_2) - get_all_vectorized_parameters(para_1)
    
    
    approx_loss_2 = loss_1 + torch.mm(del_para.view(1,-1), get_all_vectorized_parameters(gradient_1).view(-1,1)) + 0.5*torch.mm(torch.mm(del_para.view(1,-1), hessian_1/dim[0]), del_para.view(-1,1)) 
    
    return approx_loss_2, del_para
    
    
def compute_taylor_expansion_gradient(dim, gradient_1, para_1, para_2, hessian_1, epsilon):
    
    del_para = get_all_vectorized_parameters(para_2) - get_all_vectorized_parameters(para_1)
    
    approx_gradient_2 = get_all_vectorized_parameters(gradient_1) + torch.mm(del_para.view(1,-1), hessian_1/dim[0]) + epsilon*del_para.view(1,-1)
    
    return approx_gradient_2
    
    
    

def random_deletion(X, Y, delta_num, num_class, model):
    
    multi_res = model(X)#softmax_layer(torch.mm(X, res))
    
    prob, predict_labels = torch.max(multi_res, 1)
    
    changed_values, changed_label = torch.max(-multi_res, 1)
    
    
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
#             Y[indices[i]] = (Y[indices[i]] + 1)%num_class
            Y[indices[i]] = changed_label[indices[i]]
            delta_id_array.append(indices[i])
            
            X[indices[i]] = X[indices[i]] * (-2)
    
    
        i = i + 1
    
    delta_data_ids = torch.tensor(delta_id_array, dtype = torch.long)
    
#     print(delta_data_ids[:100])
#     delta_data_ids = random_generate_subset_ids(X.shape, delta_num)     
#     torch.save(delta_data_ids, git_ignore_folder+'delta_data_ids')
    return X, Y, delta_data_ids    

def verify_incremental_hessian(X, Y, dim, para_list_all_epochs, gradient_list_all_epochs, model, num, beta):
    
    
    last_para_list_all_epochs = None
    
    last_gradient_list_all_epochs = None
    
    
#     last_real_hessian_matrix = None

    last_loss = None

    
    last_exp_hessian_matrix = None
    
    first_real_hessian_matrix = None
    
    
    for i in range(num):
        id = -(num-i)
    
    
        init_model(model, para_list_all_epochs[id])
            
#         pred_2 = model(X)
        
        loss_2 = compute_loss(model, error, X, Y, beta)#error(pred_2, Y.view(-1).type(torch.LongTensor)) 
        
        curr_vec_gradient_2,_ = compute_gradient_iteration(model, input_dim, hidden_dim, output_dim, X, Y, gradient_list_all_epochs[-2])
        
        exp_vec_gradient_2 = get_all_vectorized_parameters(gradient_list_all_epochs[id]) 
        
        
        print(torch.norm(get_all_vectorized_parameters(curr_vec_gradient_2)/dim[0] - exp_vec_gradient_2))
        
        
        hessian_matrix2 = compute_hessian_matrix(model, curr_vec_gradient_2, input_dim, hidden_dim, output_dim)
    
    
        if last_para_list_all_epochs is not None and last_gradient_list_all_epochs is not None and last_exp_hessian_matrix is not None:
    
    
#             pred_1 = model(X)
#         
#             loss_1 = error(pred_1, Y.view(-1).type(torch.LongTensor))
    
    
            approx_loss_1,del_para = compute_taylor_expansion(dim, last_loss, last_gradient_list_all_epochs, last_para_list_all_epochs, para_list_all_epochs[id], last_exp_hessian_matrix)
            
            
            print(loss_2 - approx_loss_1)
            
            epsilon = (loss_2 - approx_loss_1)/(torch.pow(torch.norm(del_para),2))
            
            print('epsilon::', epsilon)
            
            approx_gradient_1 = compute_taylor_expansion_gradient(dim, last_gradient_list_all_epochs, last_para_list_all_epochs, para_list_all_epochs[id], last_exp_hessian_matrix, epsilon)
    
    
            print('gradient_diff::', torch.norm(approx_gradient_1 - exp_vec_gradient_2))
    
    
    
    
            approx_gradient_1_2 = compute_taylor_expansion_gradient(dim, last_gradient_list_all_epochs, last_para_list_all_epochs, para_list_all_epochs[id], first_real_hessian_matrix, epsilon)
    
    
            print('gradient_diff_2::', torch.norm(approx_gradient_1_2 - exp_vec_gradient_2))
    
    
    
    
            
            last_exp_hessian_matrix = last_exp_hessian_matrix + epsilon*torch.eye(last_exp_hessian_matrix.shape[1], dtype = torch.double)*dim[0]
            
            print('hessian_matrix_diff::', torch.norm(last_exp_hessian_matrix - hessian_matrix2))
        
        
        else:
            last_exp_hessian_matrix = hessian_matrix2   
        
#         last_real_hessian_matrix = hessian_matrix2
        
        
        if first_real_hessian_matrix is None:
            first_real_hessian_matrix = hessian_matrix2
        
        
        
        last_para_list_all_epochs = para_list_all_epochs[id]
        
        last_gradient_list_all_epochs = gradient_list_all_epochs[id]
        
#         last_real_hessian_matrix = hessian_matrix2
        
        last_loss = loss_2
    

    
            
#         print(torch.norm(exp_vec_gradient_2 - last_gradient_list_all_epochs))




def store_bfgs_values(para_list_all_epochs, gradient_list_all_epochs):


    para_num = get_all_vectorized_parameters(para_list_all_epochs[0]).shape[1]


    print('para_dim::', para_num)

    S_k_list = torch.zeros([para_num, len(para_list_all_epochs) - 1], dtype = torch.double)
    
    Y_k_list = torch.zeros([para_num, len(para_list_all_epochs) - 1], dtype = torch.double)
    
    for i in range(len(para_list_all_epochs)-1):
        s_k = get_all_vectorized_parameters(para_list_all_epochs[i + 1]) - get_all_vectorized_parameters(para_list_all_epochs[i])
        
        y_k = get_all_vectorized_parameters(gradient_list_all_epochs[i + 1]) - get_all_vectorized_parameters(gradient_list_all_epochs[i])
        
        
        S_k_list[:,i] = s_k
        
        Y_k_list[:,i] = y_k 
        
        
    torch.save(S_k_list, git_ignore_folder + 'S_k_list')
    
    torch.save(Y_k_list, git_ignore_folder + 'Y_k_list')


   
            
def verify_incremental_BFGS_hessian(X, Y, dim, para_list_all_epochs, gradient_list_all_epochs, model, num, m):


    para_num = get_all_vectorized_parameters(para_list_all_epochs[0]).shape[1]


    print('para_dim::', para_num)

    S_k_list = torch.zeros([para_num, len(para_list_all_epochs) - 1], dtype = torch.double)
    
    Y_k_list = torch.zeros([para_num, len(para_list_all_epochs) - 1], dtype = torch.double)
    
    for i in range(len(para_list_all_epochs)-1):
        s_k = get_all_vectorized_parameters(para_list_all_epochs[i + 1]) - get_all_vectorized_parameters(para_list_all_epochs[i])
        
        y_k = get_all_vectorized_parameters(gradient_list_all_epochs[i + 1]) - get_all_vectorized_parameters(gradient_list_all_epochs[i])
        
        
        S_k_list[:,i] = s_k
        
        Y_k_list[:,i] = y_k 
    
        
    
    
    


    for i in range(num):
        
        id = -(num-i)
        
        curr_S_k = S_k_list[:,id-m:id]
        
        curr_Y_k = Y_k_list[:,id-m:id]
        
        S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
        
        
        S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)
        
        
        R_k = torch.triu(S_k_time_Y_k)
        
        L_k = S_k_time_Y_k - R_k
        
        D_k_diag = torch.diag(S_k_time_Y_k)
        
        
        sigma_k = torch.dot(Y_k_list[:,id-1],S_k_list[:,id-1])/(torch.dot(S_k_list[:,id-1], S_k_list[:,id-1]))
        
        
        interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
        
        J_k = torch.from_numpy(np.linalg.cholesky(interm.numpy())).type(torch.DoubleTensor)
        
        
        v_vec = S_k_list[:,id-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
        
#         v_vec = torch.rand([para_num, 1], dtype = torch.double)
        
        
        p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
        
        
        D_k_sqr_root = torch.pow(D_k_diag, 0.5)
        
        D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
        
        upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
        
        lower_mat_1 = torch.cat([torch.zeros([m, m], dtype = torch.double), torch.t(J_k)], dim = 1)
        
        
        mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
        
        
        upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([m, m], dtype = torch.double)], dim = 1)
        
        lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
        
        mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
        
        
        p_mat = torch.mm(torch.inverse(mat_1), torch.mm(torch.inverse(mat_2), p_mat))
        
        
        approx_prod = sigma_k*v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
        
        
        
        init_model(model, para_list_all_epochs[id])
        
        curr_vec_gradient,_ = compute_gradient_iteration(model, input_dim, hidden_dim, output_dim, X, Y, gradient_list_all_epochs[id])
        
        hessian_matrix = compute_hessian_matrix(model, curr_vec_gradient, input_dim, hidden_dim, output_dim)/dim[0]

        
        exp_prod = torch.mm(hessian_matrix, v_vec)
        
        print(torch.norm(approx_prod - exp_prod))
        
        
        
        
    
    
        
            
#         pred = model(X)
        
#         loss_2 = error(pred, Y.view(-1).type(torch.LongTensor)) 
        
                
        
    

    


if __name__ == '__main__':
    
    configs = load_config_data(config_file)
    
    print(configs)
    
    git_ignore_folder = configs['git_ignore_folder']


#     file_name = '../../../data/heartbeat/mitbih_train.csv'

#     file_name = '../../../data/covtype_binary'
#     
    
#     file_name = '../../../data/Sensorless.scale'
    

    file_name = '../../../data/shuttle.scale.tr'
    
#     file_name = '../../../data/skin_nonskin'

#     file_name = '../../../data/minist.csv'

    sys_argv = sys.argv
    
    start = bool(int(sys_argv[1]))
    
    noise_rate = float(sys_argv[2])
    
    
#     if start:
# 
#         [X, Y, test_X, test_Y] = load_data_multi_classes(True, file_name)
# 
#         X, Y, test_X, test_Y = X.type(torch.FloatTensor), Y.type(torch.FloatTensor), test_X.type(torch.FloatTensor), test_Y.type(torch.FloatTensor)
#     
#     else:
#         
#         test_X = torch.load(git_ignore_folder + 'test_X')
#     
#         test_Y = torch.load(git_ignore_folder + 'test_Y')
#         
#         X = torch.load(git_ignore_folder + 'X')
#         
#         Y = torch.load(git_ignore_folder + 'Y')
        
    
    
    
    
    num_epochs = 500
    # Create ANN
    learning_rate = 0.2
    
    regularization_coeff = 0.05
    
    
    hidden_dim = [5] #hidden layer dim is one of the hyper parameter and it should be chosen and tuned. For now I only say 150 there is no reason.
    
    if start:
        [X, Y, test_X, test_Y] = load_data_multi_classes(True, file_name)

        X, Y, test_X, test_Y = X.type(torch.DoubleTensor), Y.type(torch.DoubleTensor), test_X.type(torch.DoubleTensor), test_Y.type(torch.DoubleTensor)
        
        
        input_dim = X.shape[1]
        
        num_class = torch.unique(Y).shape[0]
        
        output_dim = num_class
        
        
        dim = X.shape
        
        print('X_shape::', dim)
        
        model = DNNModel(input_dim, hidden_dim, output_dim)

        init_para_list = get_model_para(model)


        print("start training::")
    
        error = nn.CrossEntropyLoss()
    
    
    #     compute_linearized_coefficient([], [], model)
    
    
                                      
#         optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
        
        model, gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, count = model_training(num_epochs, X, Y, test_X, test_Y, learning_rate, regularization_coeff, error, model, True)


        torch.save(model, git_ignore_folder + 'origin_model')
        
        torch.save(init_para_list, git_ignore_folder + 'init_para')
        
        torch.save(test_X, git_ignore_folder + 'test_X')
     
        torch.save(test_Y, git_ignore_folder + 'test_Y')
         
        torch.save(X, git_ignore_folder + 'X')
         
        torch.save(Y, git_ignore_folder + 'Y')
        
        
        
#         verify_incremental_hessian(X, Y, dim, para_list_all_epochs, gradient_list_all_epochs, model, cut_off_epoch - 1)
#         verify_incremental_BFGS_hessian(X, Y, dim, para_list_all_epochs, gradient_list_all_epochs, model, 10, 3)
        
        
        
#         init_model(model, para_list_all_epochs[-2])
#         
#         pred_2 = model(X)
#         
#         loss_2 = error(pred_2, Y.view(-1).type(torch.LongTensor)) 
#         
#         curr_vec_gradient_2,_ = compute_gradient_iteration(model, input_dim, hidden_dim, output_dim, X, Y, gradient_list_all_epochs[-2])
#         
#         exp_vec_gradient_2 = get_all_vectorized_parameters(gradient_list_all_epochs[-2]) 
#         
#         
#         print(torch.norm(get_all_vectorized_parameters(curr_vec_gradient_2)/dim[0] - exp_vec_gradient_2))
#         
#         
#         hessian_matrix2 = compute_hessian_matrix(model, curr_vec_gradient_2, input_dim, hidden_dim, output_dim)
#         
#         
#         init_model(model, para_list_all_epochs[-1])
#         
#         pred_1 = model(X)
#         
#         loss_1 = error(pred_1, Y.view(-1).type(torch.LongTensor))
#         
#         curr_vec_gradient_1,_ = compute_gradient_iteration(model, input_dim, hidden_dim, output_dim, X, Y, gradient_list_all_epochs[-1])
#         
#         exp_vec_gradient_1 = get_all_vectorized_parameters(gradient_list_all_epochs[-1])
#         
#         
#         hessian_matrix1 = compute_hessian_matrix(model, curr_vec_gradient_1, input_dim, hidden_dim, output_dim)
#         
#         
#         print(torch.norm(get_all_vectorized_parameters(curr_vec_gradient_1)/dim[0] - exp_vec_gradient_1))
#         
#         approx_loss_1,del_para = compute_taylor_expansion(dim, loss_2, gradient_list_all_epochs[-2], para_list_all_epochs[-2], para_list_all_epochs[-1], hessian_matrix2)
#         
#         
#         print(loss_1 - approx_loss_1)
#         
#         epsilon = (loss_1 - approx_loss_1)/(torch.pow(torch.norm(del_para),2))
#         
#         print('epsilon::', epsilon)
#         
#         approx_gradient_1 = compute_taylor_expansion_gradient(dim, gradient_list_all_epochs[-2], para_list_all_epochs[-2], para_list_all_epochs[-1], hessian_matrix2)
#         
#         
#         print(approx_gradient_1 - get_all_vectorized_parameters(gradient_list_all_epochs[-1]))
#         
#         print(torch.norm(approx_gradient_1 - get_all_vectorized_parameters(gradient_list_all_epochs[-1])))
#         
#         
#         print(torch.norm(approx_gradient_1 + epsilon*del_para - get_all_vectorized_parameters(gradient_list_all_epochs[-1])))
#         
#     
#         print(torch.norm(hessian_matrix1/dim[0]- hessian_matrix2/dim[0]))
#         
#         print(torch.norm(hessian_matrix2/dim[0] + epsilon/2*torch.eye(hessian_matrix2.shape[1], dtype = torch.double) - hessian_matrix1/dim[0]))
#         
#         print(torch.norm(hessian_matrix2/dim[0] + epsilon/2*torch.eye(hessian_matrix2.shape[1], dtype = torch.double) - hessian_matrix1/dim[0]) - torch.norm(hessian_matrix1/dim[0]- hessian_matrix2/dim[0]))
    
    else:
    
    
        origin_model = torch.load(git_ignore_folder + 'origin_model')
        
        
        X = torch.load(git_ignore_folder + 'X')
        
        Y = torch.load(git_ignore_folder + 'Y')
        
        print(X.shape)
        
        print(Y.shape)
        
        test_X = torch.load(git_ignore_folder + 'test_X')
        
        test_Y = torch.load(git_ignore_folder + 'test_Y')
    
        input_dim = X.shape[1]
#         hidden_dim = [10] #hidden layer dim is one of the hyper parameter and it should be chosen and tuned. For now I only say 150 there is no reason.
        
        num_class = torch.unique(Y).shape[0]
        
        output_dim = num_class
        X, Y, noise_data_ids = random_deletion(X, Y.type(torch.LongTensor), int(X.shape[0]*noise_rate), num_class, origin_model)
    
        dim = X.shape
    
#         X, Y = add_noise_data(X, Y.type(torch.LongTensor), int(X.shape[0]*noise_rate), num_class, origin_model)
#         noise_data_ids = torch.tensor(list(set(range(X.shape[0])) - set(range(dim[0]))))
        
        dim = X.shape

        init_para_list = torch.load(git_ignore_folder + 'init_para')


#     transform = transforms.Compose([transforms.ToTensor(),
#                               transforms.Normalize((0.5,), (0.5,)),
#                               ])
# 
# 
#     trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
#     valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
#     valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)



# Layer details for the neural network

# Build a feed-forward network

    
    
#     init_model(model)

        print("initialized model parameters::")
    
    #     if start:
    #         init_para_list = get_model_para(model)
    #     else:
    #         init_para_list = list(torch.load(git_ignore_folder + 'init_para'))
    #         
    #         init_model(model, init_para_list)
    
    #     print_model_para(model)
        
        
        print("start training::")
        
        model = DNNModel(input_dim, hidden_dim, output_dim)
        
        init_model(model, init_para_list)

        error = nn.CrossEntropyLoss()
    
    
    #     compute_linearized_coefficient([], [], model)
    
    
#         optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
        cut_off_epoch = num_epochs
    
        
        model, gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, epoch = model_training(num_epochs, X, Y, test_X, test_Y, learning_rate, regularization_coeff, error, model, True)
        
    #     print_model_para(model)
        
        torch.save(gradient_list_all_epochs, git_ignore_folder + 'gradient_list_all_epochs')
        
        torch.save(para_list_all_epochs, git_ignore_folder + 'para_list_all_epochs')

        store_bfgs_values(para_list_all_epochs, gradient_list_all_epochs)





        
#         loss = compute_derivative_one_more_step(model, error, X, Y)
#     #       
#         gradient_list = get_model_gradient(model)
#         
#         output_list,_ = model.get_output_each_layer(X)
#         
#         parameter_list = get_model_para(model)
#         
#         
#         w_list, b_list = compute_linearized_coeffcient_single_epoch(X, input_dim, hidden_dim, output_dim, output_list, parameter_list)
        
        
        
        
        
        
        
        t1 =time.time()
             
    #     w_list_all_epochs, b_list_all_epochs = compute_linearized_coefficient(X, input_dim, hidden_dim, output_dim, output_list_all_epochs, para_list_all_epochs)
             
             
             
        
    #     t3 = time.time()
    #          
    #     A_list_all_epochs, B_list_all_epochs, w_delta_prod_list_all_epochs, b_delta_prod_list_all_epochs = capture_provenance(X, Y, w_list_all_epochs, b_list_all_epochs, output_list_all_epochs, para_list_all_epochs, dim, output_dim, input_dim, hidden_dim, output_dim, gradient_list_all_epochs)
        
    #     w_delta_prod_list, b_delta_prod_list = capture_provenance2(X, Y, w_list, b_list, output_list_all_epochs[-1], para_list_all_epochs[-1], dim, num_class, input_dim, hidden_dim, output_dim, gradient_list_all_epochs[-1])
        
    #     t2 = time.time()
    #         
    #         
    #     prov_time = t2 - t1
    #         
    #     prov_time1 = t2 - t3
    #         
    #     prov_time2 = t3 - t1
    #         
    #     print("time for capturing provenance::", prov_time)
    #         
    #     print("time for capturing provenance 1::", prov_time1)
    #         
    #     print("time for capturing provenance 2::", prov_time2)
        
        
        
        
        print("verify provenance results::")
        
#         init_model(model, para_list_all_epochs[-1])
#     #          
#         curr_gradient_list, curr_para_list = compute_gradient_iteration(model, input_dim, hidden_dim, output_dim, X, Y, gradient_list_all_epochs[-1], regularization_coeff)
#         
#         init_model(model, para_list_all_epochs[-1])
#         
#         compute_derivative_one_more_step(model, error, X, Y, regularization_coeff)
#         
#         exp_gradient_list = get_all_gradient(model)
#         
#         
#         
# #         for i in range(len(exp_gradient_list)):
# #             exp_gradient_list[i] = exp_gradient_list[i]*X.shape[0]
#         
#         compute_model_para_diff(curr_gradient_list, exp_gradient_list)
#         
#         print('here')
        
    #          
    #     hessian_matrix1 = compute_hessian_matrix(model, curr_gradient_list, input_dim, hidden_dim, output_dim)
    #          
    #     verify_hessian_matrix(hessian_matrix1, para_list_all_epochs[-1], model, input_dim, hidden_dim, output_dim, X, Y, gradient_list_all_epochs[-1])  
    #          
    #        
    #          
    #     init_model(model, para_list_all_epochs[-2])
    #          
    #     curr_gradient_list, curr_para_list = compute_gradient_iteration(model, input_dim, hidden_dim, output_dim, X, Y, gradient_list_all_epochs[-2])
    #          
    #     hessian_matrix2 = compute_hessian_matrix(model, curr_gradient_list, input_dim, hidden_dim, output_dim)
        
        
        
        
        
    #     final_model_paras = list(model.parameters()) 
    #     
    #     
    #     compute_model_derivitive_iteration(X, Y, learning_rate, error, dim, output_dim, input_dim, hidden_dim, output_dim, final_model_paras, gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, init_para_list)
        
        
    
         
    #     gradient_list2 = compute_derivative_with_provenance(dim, hidden_dim, A_list_all_epochs[-1], B_list_all_epochs[-1], para_list_all_epochs[-1])
    #       
    #     compute_gradient_diff(gradient_list_all_epochs[-1], gradient_list2)
        
    #     compute_linearized_coefficient(gradient_list, res_list, model)
         
    #     final_model_paras = list(model.parameters()) 
    # #     
    #     init_model(model, init_para_list)
    # #     
    #     print("retraining start::")
    # #     
    #     compute_model_parameter_iteration(model, X, Y, learning_rate, error, dim, output_dim, input_dim, hidden_dim, output_dim, final_model_paras)
        
        
        
        
        
        
#         torch.save(model, git_ignore_folder + 'model_without_noise')    # Cross Entropy Loss 
#         
#         model = DNNModel(input_dim, hidden_dim, output_dim)
#         
#         init_model(model, parameter_list)
#         
#         compute_derivative_one_more_step(model, error, X, Y)
#         
#         expected_gradient = get_all_gradient(model)#(list(model.parameters()))
#         
#         
#         gradient_list2, _ = compute_gradient_iteration(model, input_dim, hidden_dim, output_dim, X, Y, expected_gradient)
#         
#         para_list2 = get_model_para(model)
#         
#     #     init_model(model, para_list2)
#     #     
#     #     
#     #     gradient_list3, para_list3 = compute_gradient_iteration(model, input_dim, hidden_dim, output_dim, X, Y, expected_gradient)
#         
#         
#         
#         computed_gradient = get_all_vectorized_parameters(gradient_list2)/X.shape[0]
#         
#         print(torch.norm(get_all_vectorized_parameters(expected_gradient) - computed_gradient))
#         
#         
#         
#         
#         print("get_gradient_done!!")
#         
#         hessian_matrix = compute_hessian_matrix(model, gradient_list2, input_dim, hidden_dim, output_dim)
#         
#         update_and_zero_model_gradient(model, alpha)
#         
#         para_list3 = get_all_parameters(model)
#         
#         init_model(model, para_list3)
#         
#         compute_derivative_one_more_step(model, error, X, Y)
#         
#         computed_gradient_gap = (torch.t(torch.mm(torch.t(hessian_matrix), torch.t((get_all_vectorized_parameters(para_list3) - get_all_vectorized_parameters(parameter_list))))))/X.shape[0]
#         
#         expected_gradient_gap = get_all_vectorized_parameters(get_all_gradient(model)) - get_all_vectorized_parameters(gradient_list2)/X.shape[0]
#         
#         print("gradient_diff:::", torch.norm(computed_gradient_gap - expected_gradient_gap))
#         
#         
#         Hessian_inverse = torch.inverse(hessian_matrix)
#         
#         print("hessian prepare done!!")
#         
#         torch.save(Hessian_inverse, git_ignore_folder + 'Hessian_inverse')
#         
#         
#         torch.save(hessian_matrix, git_ignore_folder + 'hessian_matrix')
        
        
        torch.save(X, git_ignore_folder + 'noise_X')
        
        torch.save(Y, git_ignore_folder + 'noise_Y')
        
        torch.save(noise_data_ids, git_ignore_folder + 'noise_data_ids')
        
    #     delta_gradient_all_epochs, delta_all_epochs = get_all_gradients(selected_rows, output_list_all_epochs, model, para_list_all_epochs, input_dim, hidden_dim, output_dim, output_dim, X, Y)
        
        
        torch.save(learning_rate, git_ignore_folder + 'alpha')
        
        torch.save(regularization_coeff, git_ignore_folder + 'beta')
        
    #     torch.save(delta_gradient_all_epochs, git_ignore_folder + 'delta_gradient_all_epochs')
    #     
    #     torch.save(delta_all_epochs, git_ignore_folder + 'delta_all_epochs')
        
        
    #     torch.save(update_X, git_ignore_folder + 'noise_X')
    #     
    #     torch.save(update_Y, git_ignore_folder + 'noise_Y')
         
         
    
         
        torch.save(hidden_dim, git_ignore_folder + 'hidden_dims')
             
        torch.save(epoch, git_ignore_folder+'epoch')    
         
    #     torch.save(A_list_all_epochs, git_ignore_folder + 'A_list')
    #         
    #     torch.save(B_list_all_epochs, git_ignore_folder + 'B_list')
    #         
    #         
    #     torch.save(w_delta_prod_list_all_epochs, git_ignore_folder + 'w_delta_prod_list')
    #         
    #     torch.save(b_delta_prod_list_all_epochs, git_ignore_folder + 'b_delta_prod_list')
           
    #         torch.save(A_list_all_epochs, git_ignore_folder + 'A_list')
    #         
    #     torch.save(B_list_all_epochs, git_ignore_folder + 'B_list')
            
            
    #     torch.save(w_delta_prod_list, git_ignore_folder + 'w_delta_prod_list')
    #         
    #     torch.save(b_delta_prod_list, git_ignore_folder + 'b_delta_prod_list')   
        
#         torch.save(gradient_list, git_ignore_folder + 'gradient_list')
#         
#         torch.save(output_list, git_ignore_folder + 'output_list')
#            
#            
#         torch.save(w_list, git_ignore_folder + 'w_seq')
#               
#         torch.save(b_list, git_ignore_folder + 'b_seq')
#     #       
#     #     torch.save(output_list_all_epochs, git_ignore_folder + 'output_list')
#          
#          
#         torch.save(parameter_list, git_ignore_folder + "old_para_list")
#         
#         
#         torch.save(loss, git_ignore_folder + 'loss')
#         
#         init_model(model, para_list2)
#         
#         compute_derivative_one_more_step(model, error, X, Y)
#         
#         expected_full_gradient_list2 = get_all_gradient(model)
#             
#         expected_full_gradient_list, expected_full_para_list = compute_gradient_iteration(model, input_dim, hidden_dim, output_dim, X, Y, gradient_list2)
#     
#         
#         print(torch.norm(get_all_vectorized_parameters(expected_full_gradient_list)/X.shape[0] - get_all_vectorized_parameters(expected_full_gradient_list2)))
#         
#         
#         print(torch.norm(get_all_vectorized_parameters(gradient_list2) - get_all_vectorized_parameters(expected_full_gradient_list)))
#     
#         print(torch.norm(get_all_vectorized_parameters(expected_gradient) - get_all_vectorized_parameters(expected_full_gradient_list2)))
#         
#         torch.save(para_list2, git_ignore_folder + 'hessian_para_list')
#         
#         torch.save(gradient_list2, git_ignore_folder + 'hessian_gradient_list')
#         
#         print(gradient_list2[-1])
    
#     model_para_list = model_update_provenance_cp2(alpha, X, Y, max_epoch, model, dim, w_list_all_epochs, b_list_all_epochs, input_dim, hidden_dim, output_dim, torch.tensor(range(X.shape[0])), gradient_list_all_epochs, para_list_all_epochs)
    
    
#     print("model parameters::")
#     
#     print(list(model.parameters())[0])
    
    
    
    # SGD Optimizer


#     model = construct_model(input_size, hidden_sizes, output_size)
# 
# 
#     criterion = nn.NLLLoss()
#     images, labels = next(iter(trainloader))
#     images = images.view(images.shape[0], -1)
# 
#     logps = model(images)
#     loss = criterion(logps, labels)










