'''
Created on Feb 4, 2019

'''

# import torch
# from sensitivity_analysis.data_IO import load_data
# 
# torch.set_printoptions(precision=14)
# 
# 
# # [X, Y] = load_data()
# # 
# # index_list = range(10)
# # 
# # indices = torch.tensor(index_list)
# # 
# # X = torch.index_select(X, 0, indices)
# # 
# # 
# # dim = X.shape
# # 
# # expect_X_product = torch.mm(torch.t(X), X)
# # 
# # 
# # 
# # compute_X_product = torch.zeros([dim[1], dim[1]], dtype=torch.float64)
# # 
# # 
# # print(dim)
# # 
# # for i in range(dim[0]):
# # #     print(X[i,:])
# # #     print(X[i,:].resize_(dim[1],1))
# # #     print(X[i,:].resize_(1, dim[1]))
# #     compute_X_product = compute_X_product + torch.mm(X[i,:].resize_(dim[1],1), X[i,:].resize_(1, dim[1]))
# # 
# # 
# # gap = expect_X_product - compute_X_product
# # 
# # print('gap', gap)
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# x = torch.randn(2,3)
# 
# 
# mask = torch.diag(x[0,:])
# 
# print(x)
# 
# print(mask)
# 
# 
# dim = x.shape
# print(x)
# indices = torch.tensor([0, 1])
# 
# 
# b = torch.zeros(dim)
# 
# for i in range(dim[0]):
#     b = b
# 
# 
# print('here',torch.mm(torch.t(x), x) - torch.mm(x[0,:].resize_(dim[1],1), x[0, :].resize_(1,dim[1])) - torch.mm(x[1,:].resize_(dim[1],1), x[1, :].resize_(1,dim[1])))
# 
# 
# print(torch.index_select(x, 0, indices))
# print(torch.index_select(x, 1, indices))
# from torch import nn
import torch
import numpy as np
import time
from torch.autograd import Variable
# import matplotlib.pyplot as plt
# from torch import nn,optim
# from torch.utils.data import Dataset, DataLoader
# class Data(Dataset):
#     def __init__(self):
#         self.x=torch.arange(-3,3,0.1).view(-1, 1)
#         self.y=-3*self.x+1+0.1*torch.randn(self.x.size())
#         self.len=self.x.shape[0]
#     def __getitem__(self,index):    
#             
#         return self.x[index],self.y[index]
#     def __len__(self):
#         return self.len
# class linear_regression(nn.Module):
#     def __init__(self,input_size,output_size):
#         super(linear_regression,self).__init__()
#         self.linear=nn.Linear(input_size,output_size)
#     def forward(self,x):
#         yhat=self.linear(x)
#         return yhat
# 
# # class linear_regression(nn.Module):
# #     def __init__(self,input_size,output_size):
# #         super(linear_regression,self).__init__()
# #         self.linear=nn.Linear(input_size,output_size)
# #     def forward(self,x):
# #         yhat=self.linear(x)
# #         return yhat
# model=linear_regression(1,1)
# optimizer = optim.SGD(model.parameters(), lr = 0.01)
# criterion = nn.MSELoss()
# dataset=Data()
# trainloader=DataLoader(dataset=dataset,batch_size=dataset.len)
# LOSS=[]
# 
# n=1;
# for epoch in range(10):
#     for x,y in trainloader:
#         yhat=model(x)
#         loss=criterion(yhat,y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         LOSS.append(loss)
# 
# w = list(model.parameters())
# 
# print(w)

# m = torch.nn.Sigmoid()
# 
# tensor = torch.tensor([[1,-1],[-2, -3]], dtype = torch.double)
# 
# 
# b = torch.tensor([2,3], dtype = torch.double)
# 
# res = tensor.mul(b)
# 
# print(res)
# 
# res = torch.sum(res, dim = 0)
# 
# # res = m(tensor)
# 
# print(res)

# import time
# 
# X = torch.tensor([1,2,5])
# 
# 
# Y = torch.tensor([0,3, 5])
# 
# 
# t1 = time.time()
# 
# for i in range(10000):
#     X * Y
# 
# t2 = time.time()
# 
# 
# t3 = time.time()
# 
# for i in range(10000):
#     torch.mul(X, Y)
# 
# t4 = time.time()
# 
# 
# t5 = time.time()
# 
# for i in range(10000):
#     X.mul_(Y)
# 
# t6 = time.time()
# 
# time1 = (t2 - t1)
# 
# time2 = (t4 - t3)
# 
# time3 = (t6 - t5)
# 
# print(time1)
# 
# print(time2)
# 
# print(time3)
# 
# print(np.power(2,3))
# 
# 
# 
# 
# 
# def second_derivative_function(X):
#     return (-torch.exp(-X) + torch.exp(-2*X))/torch.pow((1+torch.exp(-X)), 3) 
# 
# def first_derivative_function(X):
#     return torch.exp(-X)/(torch.pow((1 + torch.exp(-X)), 2))
# 
# X = torch.tensor(10, dtype = torch.double)
# 
# print(second_derivative_function(X)*20)
# 
# print(first_derivative_function(torch.tensor(0, dtype = torch.double)))
# 
# 
# t = torch.randn(3, 3).type(torch.double)
# t1 = torch.randn(3, 1).type(torch.double)
# t2 = torch.randn(1, 3).type(torch.double)
# 
# 
# print('t*t1', t*t1)
# 
# 
# a1 = time.time()
# 
# res = torch.addcmul(t, 0.1, t1, t2)
# 
# 
# a2 = time.time()
# 
# res2 = t+0.1*t1*t2
# 
# 
# a3 = time.time()
# 
# print('time1::', a2 - a1)
# 
# print('time2::', a3 - a2)
# 
# 
# print(t)
# print(t1)
# print(t2)
# 
# print(res)
# 
# print(res2)








# x = np.array([1, 2, 3, 4, 5, 6])
# 
# print(x[[0, 1, 2]])






X = torch.tensor([[1,2,3],[1,5,3], [2,1,4]])

dim = X.shape

weights = torch.tensor([[1,2],[3,1], [2,5]])

w_dim = weights.shape

res = torch.bmm(X.view(dim[0], dim[1], 1), weights.view(dim[0], 1, w_dim[1]))


# dim = X.shape
# 
# w_dim = weights.shape
# 
# X_product_batch = torch.bmm(X.view(dim[0], dim[1], 1), X.view(dim[0], 1, dim[1]))
# 
# print(X_product_batch.shape)
# 
# X_product_batch = torch.transpose(X_product_batch, 0, 1).repeat(w_dim[1],1, 1)
# 
# X_product_batch = X_product_batch.view(w_dim[1], dim[0], dim[1], dim[1])
# 
# # print(X_product_batch.shape)
# 
# print(weights.view(w_dim[1], dim[0], 1,1).shape)
# 
# print(X_product_batch)
# 
# res = (X_product_batch*(weights.view(w_dim[1], dim[0], 1, 1)))
# 
# print(res)
# 
# print(res.shape)
# 
# selected_rows = torch.tensor([0,2])
# 
# res = torch.index_select(res, 1, selected_rows)
# 
# print(res)




X = torch.rand(10)


print(X/0.0001)

# X += 1

print(X)






# data = [1,2]#np.random.randint(10, size=2)
# 


X = torch.rand(3,10)


print(X)

print(X.view(-1,1))


Y = torch.LongTensor(3).random_(0, 10)


res = torch.gather(X, 1, Y.view(3, 1))





print(X)

s_layer = torch.nn.Softmax(dim = 1) 

print(s_layer(X))

s_layer_sum = torch.sum(s_layer(X), dim = 1)

softmax_sum = torch.sum(torch.exp(X), dim = 1)

print(s_layer_sum)

print(softmax_sum)

print(torch.exp(X)/(softmax_sum.view(X.shape[0], 1)))

print(Y)

print(res)



X = torch.tensor([[1,2, 3, 3, 5,3,2]])

X = torch.t(X)


Y = torch.rand(10, 1)


print((X == 2))

print((X==2).nonzero()[:, 0])

X[(X==2).nonzero()[:, 0]] = 4

print(X)

# print(torch.unique(X))

print(Y)


X = Variable(torch.rand([2, 3]))

X.requires_grad = True

Y = torch.sum(X*2)

print(X)

Y.backward()

print(X.grad)

# print(X.repeat(5, 1) - Y)

batch_size = 5
nb_digits = 10
# Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
y = torch.LongTensor(batch_size,1).random_() % nb_digits
# One hot encoding buffer that you create out of the loop and just keep reusing
y_onehot = torch.FloatTensor(batch_size, nb_digits)

# In your for loop
y_onehot.zero_()
y_onehot.scatter_(1, y, 1)

print(y)
print(y_onehot)



X = torch.tensor([[[0.8310, 0.8651]],

        [[0.8064, 0.3189]],

        [[0.3677, 0.4951]]], dtype = torch.double)

# print(X)
# 
# 
# softmax_layer = torch.nn.Softmax(dim = 2)
# 
# print(softmax_layer(X))
# 
# ids = torch.argsort(X, 2)
# 
# print(ids)




ordered_ids = torch.argsort(X, dim = 2)


print(X)

print('ordered_ids::', ordered_ids)


ordered_ids2 = X.numpy().argsort(axis  =2)

print('ordered_ids2::', ordered_ids2)



# X = torch.rand(3,2,2)
# 
# print(X)
# 
# curr_diag = torch.diagonal(X, dim1=1, dim2=2)
# 
# print(torch.diag_embed(curr_diag))
# 
# print(torch.diag_embed(curr_diag) - X)
# 
# curr_sum= torch.sum(torch.diag_embed(curr_diag) - X, dim = 1)
# 
# print(curr_sum)
# 
# res = (torch.diag_embed(curr_sum) + torch.diag_embed(curr_diag) - X)
# 
# 
# print(res)
# 
# res = torch.diag_embed(res)
# 
# print(res)
# 
# print(res.shape)
# import os
# 
# import psutil
# 
# X = torch.rand([42000, 70, 70], dtype = torch.double)
# pid = os.getpid() 
# py = psutil.Process(pid) 
# memoryUse = py.memory_info()[0] / 2e9
# 
# print(memoryUse)

# X = torch.rand(3,2,2)
# 
# print('X::',X)
# 
# print(torch.sum(X, dim = 1))
# 
# print(torch.diag_embed(torch.sum(X, dim = 1)))

# from torch.autograd import Variable
# import torch
# x = Variable(torch.FloatTensor([[1, 2, 3, 4], [1, 2, 3, 4]]), requires_grad=True)
# 
# dim = x.shape
# z = torch.bmm(x.view(dim[0], 1, dim[1]), x.view(dim[0], dim[1], 1)).view(dim[0], 1)
# 
# loss = torch.sum(z, dim = 0)
# 
# 
# 
# print(z, loss)
# 
# z.backward(torch.FloatTensor([[1],[0]]), retain_graph=True)
# 
# print(x.grad.data)
# 
# x.grad.data.zero_()
# 
# 
# loss.backward(retain_graph=True)
# 
# print(x.grad.data)
# 
# print(x.grad.grad_fn)
# 
# r = x.grad.data.clone()
# 
# x.grad.data.zero_()
# 
# 
# from torch.autograd import Variable, grad
# import torch
# 
# x = Variable(torch.FloatTensor([[1, 2, 3, 4], [1, 2, 3, 4]]), requires_grad=True)
# z = torch.bmm(x.view(dim[0], 1, dim[1]), x.view(dim[0], dim[1], 1)).view(dim[0], 1)
# 
# y = torch.sum(z, dim = 0)
# 
# g = grad(y, x, create_graph=True)
# print(g) # g = 3
# 
# g2 = grad(g, x)
# print(g2) # g2 = 6



set1 = {1,2}

set2 = {2,3}



selected_rows = np.random.choice(list(range(20)), 10, replace = False)

print(selected_rows)



X = torch.randn(10,50)


X = torch.mm(torch.t(X), X)

u, s, v = torch.svd(X)


X = X/torch.max(s)


epoch = 50

t1 = time.time()

res1 = torch.eye(X.shape[0])

# term2 = torch.zeros(X.shape)

for i in range(epoch):
    
#     term2 += res1
    res1 = torch.mm(res1, X)




t2 = time.time()


print(res1)

print(t2 - t1)


t3 = time.time()

u, s, v = torch.svd(X)

# loadings = torch.mm(u, s)

s = torch.pow(s, epoch)

res2 = u.mul(s.view(1,-1))

# s_diag = torch.diag(s)

# print(s)

# s_diag = torch.pow(s_diag, epoch)

res2 = torch.mm(res2, torch.t(v))
# res2 = torch.t((torch.bmm(s.view(u.shape[0], 1, 1), torch.t(u).view(u.shape[0], 1, u.shape[1]))).view(u.shape[0], u.shape[1]))

# print(s)

# res2 = torch.mm(u, res2)



t4 = time.time()

print(t4 - t3)

print(res1  -res2)



A = torch.rand((4,4,2))

B = torch.tensor([[[1],[0],[1],[0]],[[0],[0],[0],[1]]], dtype = torch.float)


print(A)

print(B)

print(B.shape)

non_zero_B = torch.nonzero(B)

print(torch.nonzero(B))

print(non_zero_B[:, 1:3])

print(A[non_zero_B[:, 1], non_zero_B[:, 2]])



# A = torch.rand((10000,10000), dtype = torch.double)
# 
# print('here')
# 
# A_inverse = torch.tensor(np.linalg.inv(A.numpy()))
# 
# A_inverse2 = torch.inverse(A)
# 
# print(torch.norm(torch.mm(A, A_inverse) - torch.eye(10000, dtype = torch.double)))
# 
# print(torch.norm(torch.mm(A, A_inverse2) - torch.eye(10000, dtype = torch.double)))

# B = torch.eye(2,2)

# t1 = time.time()
# 
# torch.inverse(A)
# 
# t2 = time.time()
# 
# print(t2 - t1)

# print(A - B)





# m = torch.distributions.Normal(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.5, 0.5, 0.2]))
# 
# print(m.sample((int(2),)))





# print(z.backward(x[0], retain_graph = True))



# print(ordered_ids[:,:,0])
# 
# e_i = np.zeros(X.shape)
# 
# 
# 
# 
# 
# 
# X = torch.rand([3,2,4])
# 
# exp_res = torch.exp(X)
# 
# print(X)
# 
# print(exp_res)
# 
# print(torch.sum(exp_res, dim = 2))
# 
# print(exp_res/(torch.sum(exp_res, dim = 2).view(X.shape[0], X.shape[1], 1)))

    
#     
# print(X)
# 
# print(ordered_ids)    
# 
# 
# for i in range(X.shape[2]):
#     e_i = torch.zeros(X.shape, dtype = torch.double)
#     print(ordered_ids[:,:,i])
#     print(e_i)
#     e_i.scatter_(2, ordered_ids[:,:,i].view(X.shape[0],X.shape[1],1), 1)
# 
#     print(e_i)

# data2 = torch.rand(10,3)
# 
# print(data2)
# 
# print(data2[0:5,:])
# 
# 
# curr_set = set()
# 
# curr_set.add(1)
# 
# curr_set.add(1)
# 
# print(len(curr_set))

# 
# print(data2*data)
# 
# 
# def function(x):
# #     if x <2:
# #         return 2
# #     else:
# #         if x > 5:
# #             return 5
# #         else:
#     return data[x]
# 
# 
# 
# ids = np.random.randint(1000, size=[10000, 2])
# 
# 
# 
# 
# myfunc_vec = np.vectorize(function)
# 
# 
# t1 = time.time()
# 
# res = np.array([function(xi) for xi in ids])
# 
# 
# t2 = time.time()
# 
# 
# res2 = np.array(list(map(function, ids)))
# 
# t3 = time.time()
# 
# 
# result = myfunc_vec(ids)
# 
# t4 = time.time()
# 
# 
# # print(ids)
# 
# res3 = data[ids]
# 
# t5 = time.time()
# 
# print(t2 - t1)
# 
# print(t3 - t2)
# 
# print(t4 - t3)
# 
# print(t5 - t4)
# 
# print(res3 - res)

# print(torch.diag(X))

# print(X.view([2, 3, 1]))
# 
# print(X.view([2, 1, 3]))

