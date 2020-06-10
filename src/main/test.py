from torch.autograd import Variable

from scipy.interpolate import interp1d
from torch import nn, optim
import torch

from main.matrix_prov_sample_level import M_prov
import matplotlib.pyplot as plt
import numpy as np


# x = np.linspace(-10, 10, num=100000, endpoint=True)
# y = 1/(1+np.exp(-x))
# f = interp1d(x, y)
# print(f._y_axis)
# 
# 
# # f2 = interp1d(x, y, kind='cubic')
# xnew = np.linspace(-10, 10, num=1000000, endpoint=True)
# ynew = 1/(1+np.exp(-xnew))
# 
# 
# # plt.plot(xnew, ynew, 'o', xnew, f(xnew), '-')
# # plt.legend(['data', 'linear'], loc='best')
# # plt.show()
# print(max(abs(f(xnew) - ynew)))
# theArray = [[['a','s','l'],['b','s'],['c','s']],[['d','s'],['e','s'],['f','s']],[['g','s'],['h','s'],['i','s']]]
# # theArray = [['a','b','c'],['d','e','f'],['g','h','i']]
# print(theArray)
# arr = map(list, zip(*theArray))
# print(len(theArray))
# print(list(arr))
# 
# 
# x_train = np.array([[3.3, 1], [4.4, 2], [5.5, 0]], dtype=np.float32)
# 
# # y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
# #                     [3.366], [2.596], [2.53], [1.221], [2.827],
# #                     [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
# 
# y_train = np.array([[1.7, 0], [2.06, 3], [2.59,5]], dtype=np.float32)
#  
#  
# x_train = torch.from_numpy(x_train)
#  
# y_train = torch.from_numpy(y_train)
#  
# X = Variable(x_train)
#  
# Y = Variable(y_train)
#  
# P1 = M_prov.add_prov_token(X, 'x', X.size())
#  
# P0 = M_prov.add_prov_token(Y, 'y', Y.size())
#  
# P2 = M_prov.prov_matrix_transpose(P1, torch.transpose(X, 0, 1))
#  
# P3 = M_prov.prov_matrix_mul_prov_matrix(P2, P1, torch.mm(torch.transpose(X, 0, 1), X))
#  
#  
# P4 = M_prov.prov_matrix_add_prov_matrix(P1, P0, X+Y)
#  
#  
# print(P1.data_matrix_list)
#  
# print(P1.prov_list)
#  
# print(P2.data_matrix_list)
#  
# print(P2.prov_list)
#  
# print(P3.data_matrix_list)
#  
# print(P3.prov_list)
#  
# print(P4.data_matrix_list)
 
# print(P4.prov_list)


# import torch
# from torch import optim
# from hessian import hessian
# from data_IO.generate_config_files import git_ignore_folder
# 
# 
# def f(z):
#     x = z[0:2]
#     y = z[2:4]
#     
#     x = x.view(-1,1)
#     y = y.view(-1,1)
#     
#     return torch.mm(torch.t((x**2)), y)
# 
# def g(x, y):
#     x = x.view(-1,1)
#     y = y.view(-1,1)
#     return torch.mm(torch.t((x**2)), y)[0][0]
# 
# x = torch.tensor([3., 5.], requires_grad=True)
# x_detached = x.detach().requires_grad_()
# y = torch.tensor([4., 1.], requires_grad=True)
# x_optim = optim.SGD([x], lr=1.)
# y_optim = optim.SGD([y], lr=1.)
# ddx, = torch.autograd.grad(f(x_detached, y).mean(), x_detached, create_graph=True)
# 2(x + y) = 14

# ddx.mean().backward()
# x_detached.grad = d^2/dx^2 f(x, y) = 2
# x.grad unaffected
# y.grad = d/dy d/dx f(x, y) = 2

# x.grad.data.zero_()
# y.grad.data.zero_()




'''
z = torch.cat((x,y))

print(g(x,y))

h = hessian(g(x,y), [x,y], create_graph=True)


print(h)


g(x, y).backward(retain_graph=True, create_graph = True)
# x.grad = d/dx g(x, y) = y = 4
print(x.grad)
print(y.grad)


y.grad.data.zero_()

x.grad.data.zero_()


# x_detached.grad.data.zero_()

ddx, = torch.autograd.grad(g(x, y), x, create_graph=True)

print(g(x, y))

print(ddx)

y.grad.data.zero_()

x.grad.data.zero_()

# x_detached.grad.data.zero_()




print("output second derivatives::")

ddx[0].backward(create_graph=True, retain_graph = True)

print(x.grad)
print(y.grad)

y.grad.data.zero_()

x.grad.data.zero_()

ddx[1].backward()

print(x.grad)
print(y.grad)

y.grad.data.zero_()

x.grad.data.zero_()


ddx, = torch.autograd.grad(g(x, y), y, create_graph=True)



ddx[0].backward(create_graph=True, retain_graph = True)

print(x.grad)
print(y.grad)

y.grad.data.zero_()

x.grad.data.zero_()

ddx[1].backward()

print(x.grad)
print(y.grad)

y.grad.data.zero_()

x.grad.data.zero_()



# y_optim.step()
# # y = 2
# # x = 3
# 
# x_optim.step()

'''



# x = torch.tensor([[1.,2.,3.],[3.,1.,2.]])
# 
# r = torch.tensor([[3.,4.,5.],[1.,3.,2.],[2.,2.,2.]])
# 
# 
# print(x.shape)
# 
# print(r.shape)
# 
# 
# converted_x = x.view(x.shape[0], 1, x.shape[1])
# 
# converted_r = r.view(1, r.shape[0], r.shape[1]) 
# 
# print(converted_x.shape)
# 
# print(converted_r.shape)
# 
# 
# 
# res = converted_x*converted_r
# 
# print(res)
# 
# print(torch.sum(res, 2))




import torch
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.datasets import fetch_rcv1
import time
from scipy import sparse
import os
import psutil

def check_and_convert_to_sparse_tensor(res):
    non_zero_entries = torch.nonzero(res)
    final_res = res
    if non_zero_entries.shape[0] < res.shape[0]*res.shape[1]:
        final_res_values = res[res != 0]
        
        final_res = coo_matrix((final_res_values.detach().numpy(), (non_zero_entries[:,0].detach().numpy(), non_zero_entries[:,1].detach().numpy())), shape=list(res.shape))
        
#         print(len(final_res.row))

        final_res = final_res.tocsr()
        
#         final_res = torch.sparse.DoubleTensor(torch.t(non_zero_entries), final_res_values, torch.Size(res.shape))
        
        
        
    return final_res

# rcv1 = fetch_rcv1()
 
 
 
# X = rcv1.data[0:23149].tocsr()#coo_matrix(([3,4,5], ([0,1,1], [2,0,2])), shape=(2,3))
#  
# Y_coo = rcv1.target[0:23149].tocsr()

import scipy

from scipy.sparse import csc_matrix

from scipy.sparse.linalg import svds, eigs

# len = mat.shape[1]
# # mat = scipy.sparse.random(len,len,density=0.0001, format='csr')
# 
# weight = np.random.rand(len, 1) 
# 
# 
# u, s, vt = svds(mat, 1000)
# 
# 
# print('here')
# # print(mat)
# 
# print(np.linalg.norm((u*s).dot(vt) - mat))
# 
# # print(s)
# 
# ids = (s!=0)
# 
# print(ids.shape)
# 
# # print(ids)
# # 
# # 
# # print(vt.shape)
# # 
# # print(u)
# # 
# # print(u*s)
# 
# sub_u = u[:,ids]
# 
# sub_s = s[ids]
# 
# sub_v = vt[ids]
# 
# print(sub_u.shape)
# 
# print(np.nonzero(sub_u != 0)[0].shape)
# 
# print(sub_s.shape)
# 
# print(sub_v.shape)
# 
# print(np.linalg.norm((u[:,ids]*s[ids]).dot(vt[ids]) - mat))
# 
# 
# 
# 
# t1 = time.time()
# 
# 
# res2 = (sub_u).dot(sub_s*(sub_v.dot(weight)))
# 
# 
# t2 = time.time()
# 
# res1 = mat.dot(weight)
# 
# 
# t3 = time.time()
# 
# 
# print('time1::', t2 - t1)
# 
# print('time2::', t3 - t2)

# total_num = 20
# 
# 
# all_id = np.array(range(total_num))
# 
# random_deleted_ids = np.random.choice(all_id, int(total_num*0.2), replace = False)
# 
# 
# 
# random_ids = np.random.permutation(all_id)
# 
# sort_idx = random_ids.argsort()#random_ids.numpy().argsort()
# #         all_indexes = np.sort(np.searchsorted(random_ids.numpy(), delta_ids.numpy()))
#         
# # found_res = np.searchsorted(random_ids,random_deleted_ids,sorter = sort_idx)
#         
# all_indexes = np.sort(sort_idx[random_deleted_ids])
# 
# 
# print(all_id)
# 
# print(random_ids)
# 
# print(random_deleted_ids)
# 
# print(all_indexes)
# 
# 
# sorted_ids = np.argsort(random_ids)
# 
# 
# found_res = np.searchsorted(all_id,random_deleted_ids,sorter = sort_idx)
# 
# all_indexes2 = np.sort(sorted_ids[random_deleted_ids])
# 
# print(all_indexes2)

import torch


def quantize_vectors(data, epsilon):
    
    theta = torch.rand(data.shape, dtype = torch.double) - 0.5
    
    
    print((data - theta*epsilon)/epsilon)
    
    ids = (data - theta*epsilon)/epsilon
    
    
    discretized_ids = ids.type(torch.IntTensor)
    
    signs = (((discretized_ids > 0).type(torch.DoubleTensor) - 0.5)*2).type(torch.IntTensor)
    
    
    res_id = (torch.abs(ids - discretized_ids.type(torch.DoubleTensor)) + 0.5).type(torch.IntTensor)*signs
    
    res_id += discretized_ids
    
#     res_id = ((data - theta*epsilon)/epsilon + 0.5).type(torch.IntTensor)
    
    res = (res_id.type(torch.DoubleTensor) + theta)*epsilon
    
    print(res_id)
    
    print(torch.max(torch.abs(res - data)))
    
    return res





data = (torch.rand([5], dtype = torch.double)-1)





quantize_vectors(data, torch.tensor(1e-5))

# print(t2 - t1)




# print(t2 - t1)
# 
# print(t3 - t2)




# w = np.ones((10, 1))
# 
# print(mat)
# 
# print(w.shape)
# 
# res = np.multiply(mat.todense(), w)
# 
# print(res) 



# print(u[:ids].dot())



# 
# process = psutil.Process(os.getpid())
#     
# print('memory usage1::', process.memory_info().rss)
# 
# print(w.shape)

# print(X.shape)
# 
# t1 = time.time()
# 
# for i in range(w.shape[1]):
#     print(w[:,i].shape)
# 
#     res0 = X.multiply(np.reshape(w[:,i],[w.shape[0], 1]))
#     
#     res = X.transpose().dot(res0)
#     
#     del res0
#     
#     res2 = sparse.csr_matrix(res)
#     
#     del res
# 
# t2 = time.time()
# 
# print('time1::', t2 - t1)
# 
# process = psutil.Process(os.getpid())
#     
# print('memory usage2::', process.memory_info().rss)
# 
# print(res2[0:10])
# # X = X.todense()
# 
# X_tensor = torch.from_numpy(X.todense())
# 
# t3 = time.time()
# res2 = torch.mm(torch.t(X_tensor), X_tensor)
# t4 = time.time()
# 
# 
# 
# 
# print('time2::', t4 - t3)













# values = X_coo.data
# print(X_coo)
# indices = np.vstack((X_coo.row, X_coo.col))
#  
# i = torch.LongTensor(indices)
# v = torch.FloatTensor(values)
# shape = X_coo.shape
#  
# X = torch.sparse.FloatTensor(i, v, torch.Size(shape))
#  
#  
# i_transpose = torch.stack((i[1], i[0]), 0)
 
 
# X_transpose = torch.sparse.FloatTensor(i_transpose, v, torch.Size((shape[1], shape[0])))
# 
# 
# X_dense = X.to_dense()
# 
# t1 = time.time()
# 
# torch.sparse.mm(X_transpose, X_dense)
# 
# 
# t2 = time.time()
# 
# 
# torch.mm(torch.t(X_dense), X_dense)
# 
# 
# t3 = time.time()
# 
# 
# print(t3 - t2)
# 
# print(t2 - t1)
 
 
 
 
# indices = np.vstack((Y_coo.row, Y_coo.col))
# values = Y_coo.data
#  
# i = torch.LongTensor(indices)
# v = torch.FloatTensor(values)
# shape = Y_coo.shape
#  
# print(Y_coo)
#  
# Y = torch.zeros([8000, 8000])
#  
# M = torch.rand([shape[0], 1])
# res = None
# 
# 
# n = 400
# c = .01
#  
# Y = torch.randn((n, n))
# Y[torch.rand_like(Y) > c] = 0
# 
# 
# ids = np.array([1,3,5])
# 
# print(Y[ids])
# 
# del n
# 
# print('here')

# Y = check_and_convert_to_sparse_tensor(Y)
# 
# X = torch.ones((n, 1))
# # X[torch.rand_like(X) > c] = 0
# 
# # X = check_and_convert_to_sparse_tensor(X)
# 
# res0 = Y.dot(X.numpy())

# print(Y[0:10])
# 
# print(X)
# 
# print(res0)


# Y = Y.to_sparse()
 
 

# print(Y)
# 
# if res is None:
#     res = Y
# else:
#     res = res + Y
# 
# X = torch.randn((n, n))
# X[torch.rand_like(X) > c] = 0
# X = X.to_sparse()
#  
# print(X)
# 
# Z = np.random.rand(n, 1)
# 
# # print(Z.shape)
# # 
# # print(Y.shape)
# 
# t1 = time.time()
# 
# # res = torch.sparse.mm(Y, Z)
# res = Y.dot(Z)
# 
# 
# 
# t2 = time.time()
# 
# 
# print(res)
# 
# 
# print('running time::', t2 - t1)


 
# print(Y.values().shape[0])
#  
# M = torch.rand([n,n])
#  
# res = torch.sparse.mm(Y, M)
#  
# print(Y)
#  
# print(res)

# A = torch.rand([10, 4, 4], dtype = torch.double)
# 
# B = torch.rand([4, 1], dtype = torch.double)


# res = A*B
# print(res.shape)


# res = None
# 
# try:
#     res = Y.to_dense()
# 
# except:
#     res = Y


# print(res)












