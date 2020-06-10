'''
Created on Feb 9, 2019

'''

import numpy as np
import torch
import time
# try:
#     from sensitivity_analysis.logistic_regression.Logistic_regression import *
# except ImportError:
#     from Logistic_regression import *


class piecewise_linear_interpolication:
    
    def __init__(self, x_l, x_u, function, num):
        self.x_l = x_l
        self.x_u = x_u
        self.function = function
        self.num = num
        self.gap = (x_u - x_l)*1.0/num
        
        self.w_seq, self.b_seq = self.compute_piecewise_linear_interpolation_paras()
        
    
    def compute_piecewise_linear_interpolation_paras(self):
        
        w_seq = []
        
        b_seq = []
        
        w_seq.append(0)
        
        b_seq.append(self.function(self.x_l))
        
        
        for i in range(self.num):
        
            interval_x_1 = self.x_l + self.gap * i
        
        
            interval_x_2 = self.x_l + self.gap * (i + 1)
            
            interval_y_1 = self.function(interval_x_1)
        
            interval_y_2 = self.function(interval_x_2)
            
            w = (interval_y_2  -interval_y_1)/(interval_x_2  -interval_x_1)
        
            b = interval_y_1 - interval_x_1*w
            
            w_seq.append(w)
            
            b_seq.append(b)
            
        w_seq.append(0)
        
        b_seq.append(self.function(self.x_u))
        
        return np.array(w_seq), np.array(b_seq)
    
    
    def compute_piecewise_linear_interpolation_paras2(self):
        
        w_seq = []
        
        b_seq = []
        
        w_seq.append(0)
        
        b_seq.append(self.function(self.x_l))
        
        
        for i in range(self.num):
         
            interval_x_1 = self.x_l + self.gap * i
         
         
            interval_x_2 = self.x_l + self.gap * (i + 1)
             
            interval_y_1 = self.function(interval_x_1)
         
            interval_y_2 = self.function(interval_x_2)
             
            w = (interval_y_2  -interval_y_1)/(interval_x_2  -interval_x_1)
         
            b = interval_y_1 - interval_x_1*w
             
            w_seq.append(w)
             
            b_seq.append(b)
            
        w_seq.append(0)
        
        b_seq.append(self.function(self.x_u))
        
        return np.array(w_seq), np.array(b_seq)
    
    
    def get_parameter_w(self, id):
#         if id >= 0 and id < len(self.w_seq):
        return self.w_seq[int(id)]
#         else:
#             return 0
    
    def get_parameter_b(self, id):
#         if id >= 0 and id < len(self.b_seq):        
        return self.b_seq[int(id)]
#         else:
#             return 0

    def piecewise_linear_interpolate_coeff_batch(self, X, total_time):
        
#         pid = os.getpid()
#     
# #     prev_mem=0
#     
#         print('pid::', pid)
#         sig_function = 

        
#         x_l_tensor = torch.tensor(self.x_l, dtype = torch.double)
#         x_u_tensor = torch.tensor(self.x_u, dtype = torch.double)
#         
#         dim = X.shape
        
#         x_l_tensor = x_l_tensor.repeat(dim[0], dim[1])
        
#         x_u_tensor = x_u_tensor.repeat(dim[0], dim[1])
#         t1 = time.time()
#         compare_l_res = torch.lt(X, x_l_tensor).type(torch.DoubleTensor)
#         
#         w_res = self.w_seq[0]*compare_l_res
#         
#         b_res = self.b_seq[0]*compare_l_res
#         
#         compare_u_res = torch.gt(X, x_u_tensor).type(torch.DoubleTensor)
#         
#         w_res = w_res + self.w_seq[-1]*compare_u_res
#         
#         b_res = b_res + self.b_seq[-1]*compare_u_res
#         
#         mask_entries = torch.ones(dim, dtype = torch.int) - compare_l_res.type(torch.IntTensor) - compare_u_res.type(torch.IntTensor)
#         Y_value = self.function(X)
#         
#         Y_value = (1 - self.function(X))
        t1 = time.time()
        
        interval_id =(X - self.x_l)/self.gap
        
#         interval_id = np.floor(interval_id)
        
        interval_id[interval_id < 0] = -1
        
        interval_id[interval_id > len(self.w_seq)] = len(self.w_seq) - 2
        
        interval_id += 1
        
        interval_id = interval_id.astype(int)
        
        
#         w_res = 
#         
#         b_res = 
        
        res = X*self.w_seq[interval_id]+self.b_seq[interval_id]
        
        t2 = time.time()
        
        total_time += (t2 - t1)
        
#         print(interval_id)
# 
#         print(self.w_seq)
#         
#         print(self.b_seq)
        
        return res, total_time
        
#         interval_id = mask_entries.type(torch.DoubleTensor)*interval_id
#           
#         interval_w_res = interval_id.clone()
#         
#         interval_b_res = interval_id.clone()
#         
#         interval_w_res = interval_w_res.detach().apply_(self.get_parameter_w)
#         
#         interval_b_res = interval_b_res.detach().apply_(self.get_parameter_b)
#         t1 = time.time()
        
#         interval_x_1 = self.x_l + self.gap * interval_id
#            
# #         interval_x_2 = self.x_l + self.gap * (interval_id + 1)
#           
# #         interval_x_2 = self.x_u if interval_x_2 > self.x_u else interval_x_2
#           
# #         interval_y_1_1 = interval_x_1.clone()
# #           
# #         interval_y_2_1 = interval_x_2.clone()  
# #             
# #         t1 = time.time()  
# #           
# #         interval_y_1_1.detach().apply_(self.function)
# #           
# #         interval_y_2_1.detach().apply_(self.function)  
# #           
# #         t2 = time.time()
#         
# #         element_wise_time = (t2 - t1)
# #         print('element_wise_time::', element_wise_time)
# 
#         interval_y_1 = 1 - self.function(interval_x_1)
#           
# #         interval_y_2 = 1 - self.function(interval_x_2)
#         
#         
#         curr_gap = interval_x_1 - X
#         
# #         curr_gap.sub_(X)
#         
#         
#         interval_w_res = (interval_y_1 - Y_value)#(interval_x_1 - X)
#         
#         interval_w_res /= curr_gap
#         
#         interval_b_res = interval_y_1
# 
#         interval_x_1 *= (interval_w_res)
# 
#         interval_b_res -= interval_x_1
# 
# #         t2 = time.time()
# #         
# #         total_time += (t2 - t1)
# 
# #         sizes = [get_tensor_size(Y_value), get_tensor_size(interval_id), get_tensor_size(interval_x_1),
# #                  get_tensor_size(interval_y_1), get_tensor_size(curr_gap), get_tensor_size(interval_w_res),  
# #                  get_tensor_size(interval_b_res)]
# #         
# #         print(sizes)
#         
#         
#         del interval_id, interval_y_1, interval_x_1, curr_gap, Y_value
# 
# 
# 
# #         print(sizes)
# 
#         return interval_w_res, interval_b_res, total_time
#         print('x_1::', interval_x_1)
#         
#         print('x_2::', interval_x_2)
#         
#         
#         print('layer_x_1::', sig_function(interval_x_1))
#         
#         print('layer_x_2::', sig_function(interval_x_2))
#         
# 
#         print('y_1::', interval_y_1)
#         
#         print('y_2::', interval_y_2)
# 
#         print('y_1_1::', interval_y_1_1)
#         
#         print('y_2_2::', interval_y_2_1)
        
#         w_res = interval_w_res*mask_entries.type(torch.DoubleTensor) + w_res
#         
#         b_res = interval_b_res*mask_entries.type(torch.DoubleTensor) + b_res
#         
#         
#         
#         
#         
#         return w_res, b_res, total_time

    def piecewise_linear_interpolate_coeff_batch2(self, X):
        
#         pid = os.getpid()
#     
# #     prev_mem=0
#     
#         print('pid::', pid)
#         sig_function = 

        
#         x_l_tensor = torch.tensor(self.x_l, dtype = torch.double)
#         x_u_tensor = torch.tensor(self.x_u, dtype = torch.double)
#         
#         dim = X.shape
        
#         x_l_tensor = x_l_tensor.repeat(dim[0], dim[1])
        
#         x_u_tensor = x_u_tensor.repeat(dim[0], dim[1])
#         t1 = time.time()
#         compare_l_res = torch.lt(X, x_l_tensor).type(torch.DoubleTensor)
#         
#         w_res = self.w_seq[0]*compare_l_res
#         
#         b_res = self.b_seq[0]*compare_l_res
#         
#         compare_u_res = torch.gt(X, x_u_tensor).type(torch.DoubleTensor)
#         
#         w_res = w_res + self.w_seq[-1]*compare_u_res
#         
#         b_res = b_res + self.b_seq[-1]*compare_u_res
#         
#         mask_entries = torch.ones(dim, dtype = torch.int) - compare_l_res.type(torch.IntTensor) - compare_u_res.type(torch.IntTensor)
#         Y_value = self.function(X)
#         
#         Y_value = (1 - self.function(X))
        interval_id =(X - self.x_l)/self.gap
#         print(interval_id)
        
#         interval_id = np.floor(interval_id)
        
        interval_id[interval_id < 0] = -1
        
        interval_id[interval_id >= len(self.w_seq) - 1] = len(self.w_seq) - 2
        
        interval_id += 1
        
        interval_id = interval_id.type(torch.IntTensor)
        
        
#         w_res = 
#         
#         b_res = 
#         print(interval_id)
        
        return torch.as_tensor(self.w_seq[interval_id.numpy()], dtype = X.dtype), torch.as_tensor(self.b_seq[interval_id.numpy()], dtype = X.dtype)
        
#         interval_id = mask_entries.type(torch.DoubleTensor)*interval_id
#           
#         interval_w_res = interval_id.clone()
#         
#         interval_b_res = interval_id.clone()
#         
#         interval_w_res = interval_w_res.detach().apply_(self.get_parameter_w)
#         
#         interval_b_res = interval_b_res.detach().apply_(self.get_parameter_b)
#         t1 = time.time()
        
#         interval_x_1 = self.x_l + self.gap * interval_id
#            
# #         interval_x_2 = self.x_l + self.gap * (interval_id + 1)
#           
# #         interval_x_2 = self.x_u if interval_x_2 > self.x_u else interval_x_2
#           
# #         interval_y_1_1 = interval_x_1.clone()
# #           
# #         interval_y_2_1 = interval_x_2.clone()  
# #             
# #         t1 = time.time()  
# #           
# #         interval_y_1_1.detach().apply_(self.function)
# #           
# #         interval_y_2_1.detach().apply_(self.function)  
# #           
# #         t2 = time.time()
#         
# #         element_wise_time = (t2 - t1)
# #         print('element_wise_time::', element_wise_time)
# 
#         interval_y_1 = 1 - self.function(interval_x_1)
#           
# #         interval_y_2 = 1 - self.function(interval_x_2)
#         
#         
#         curr_gap = interval_x_1 - X
#         
# #         curr_gap.sub_(X)
#         
#         
#         interval_w_res = (interval_y_1 - Y_value)#(interval_x_1 - X)
#         
#         interval_w_res /= curr_gap
#         
#         interval_b_res = interval_y_1
# 
#         interval_x_1 *= (interval_w_res)
# 
#         interval_b_res -= interval_x_1
# 
# #         t2 = time.time()
# #         
# #         total_time += (t2 - t1)
# 
# #         sizes = [get_tensor_size(Y_value), get_tensor_size(interval_id), get_tensor_size(interval_x_1),
# #                  get_tensor_size(interval_y_1), get_tensor_size(curr_gap), get_tensor_size(interval_w_res),  
# #                  get_tensor_size(interval_b_res)]
# #         
# #         print(sizes)
#         
#         
#         del interval_id, interval_y_1, interval_x_1, curr_gap, Y_value
# 
# 
# 
# #         print(sizes)
# 
#         return interval_w_res, interval_b_res, total_time
#         print('x_1::', interval_x_1)
#         
#         print('x_2::', interval_x_2)
#         
#         
#         print('layer_x_1::', sig_function(interval_x_1))
#         
#         print('layer_x_2::', sig_function(interval_x_2))
#         
# 
#         print('y_1::', interval_y_1)
#         
#         print('y_2::', interval_y_2)
# 
#         print('y_1_1::', interval_y_1_1)
#         
#         print('y_2_2::', interval_y_2_1)
        
#         w_res = interval_w_res*mask_entries.type(torch.DoubleTensor) + w_res
#         
#         b_res = interval_b_res*mask_entries.type(torch.DoubleTensor) + b_res
#         
#         
#         
#         
#         
#         return w_res, b_res, total_time

    def piecewise_linear_interpolate_coeff_batch2_sparse(self, X):
         
        interval_id =(X - self.x_l)/self.gap
         
        interval_id[interval_id < 0] = -1
         
        interval_id[interval_id >= len(self.w_seq) - 1] = len(self.w_seq) - 2
         
        interval_id += 1
         
        interval_id = interval_id.astype(int)
         
        return torch.as_tensor(self.w_seq[interval_id]), torch.as_tensor(self.b_seq[interval_id])

    def piecewise_linear_interpolate_coeff(self, x):
#         print(x)
        if x < self.x_l:
#             return 0, self.function(self.x_l)
            return self.w_seq[0], self.b_seq[0]
        if x > self.x_u:
#             return 0, self.function(self.x_u)
            return self.w_seq[-1], self.b_seq[-1]
        
        interval_id = int((x - self.x_l)/self.gap)
         
        return [self.w_seq[interval_id + 1], self.b_seq[interval_id + 1]]
        
#         interval_id = int((x - self.x_l)/self.gap)
#          
#         interval_x_1 = self.x_l + self.gap * interval_id
#          
#         interval_x_2 = self.x_l + self.gap * (interval_id + 1)
#          
#         interval_x_2 = self.x_u if interval_x_2 > self.x_u else interval_x_2
#          
#         interval_y_1 = self.function(interval_x_1)
#          
#         interval_y_2 = self.function(interval_x_2)
#          
#         w = (interval_y_2  -interval_y_1)/(interval_x_2  -interval_x_1)
#          
#         b = interval_y_1 - interval_x_1*w
#          
#         return w, b
    


# if __name__ == '__main__':
#     x_l = torch.tensor(-10, dtype=torch.double)
#     x_u = torch.tensor(10, dtype=torch.double)
#     num = 10000
#     Pi = piecewise_linear_interpolication(x_l, x_u, sigmoid_function, num)
#     
#     x = torch.tensor(20, dtype=torch.double)
#     
#     w,b = Pi.piecewise_linear_interpolate_coeff(x)
#     
#     y_est = w*x + b
#     
#     y_real = sigmoid_function(x)
#     
#     print(w,b)
#     
#     print(y_est, y_real)

    
    
        