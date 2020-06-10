'''
Created on Feb 9, 2019

'''
# from sensitivity_analysis.logistic_regression.Logistic_regression import *
import torch

import numpy as np

from torch.autograd import Variable


softmax_layer = torch.nn.Softmax(dim = 0)

softmax_layer2 = torch.nn.Softmax(dim = 2)


class piecewise_linear_interpolation_multi_dimension:
    
    
    
    
    def __init__(self, x_l, x_u, function, gap):
        
        
        '''x_l, column vector'''
        
        self.x_l = x_l
        self.x_u = x_u
        self.function = function
        self.gap = gap
        
        vec_dim = x_l.shape
        
        self.gap_cubic = torch.zeros(vec_dim[0])
        
        self.gap_cubic += gap
        
        self.dim = vec_dim[0]
        
        
#         self.w_seq, self.b_seq = self.compute_piecewise_linear_interpolation_paras()

    def map_to_original_space(self, x, cube_lower):
        return x*self.gap + cube_lower
    
    def map_from_original_space(self, x, cube_lower):
        return (x - cube_lower)/self.gap

    def get_cube_lower_vec(self, x):
        ids_in_all_dims = (x - self.x_l)/self.gap
        
        ids_in_all_dims.floor_()
        
        
        cube_lower = ids_in_all_dims*self.gap + self.x_l
        
        return cube_lower
    
    
    
    def compute_approximate_value_batch(self, x):
        '''x:: n*t*q = dim[0]*max_epoch*num_class'''
        
#         cube_higher = self.gap_cubic + cube_lower

        cube_lower = self.get_cube_lower_vec(x)
        
        position_in_unit_cubic = (x - cube_lower)/self.gap
        
        
#         position_in_unit_cubic_list = list(position_in_unit_cubic)
        
#         ordered_ids = sorted(range(len(position_in_unit_cubic_list)), key = lambda k: position_in_unit_cubic_list[k])
        
        ordered_ids = torch.argsort(position_in_unit_cubic, dim = 2)
        
        
#         print('x::', position_in_unit_cubic)
#         
#         print('ordered_ids::', ordered_ids)
        
#         ordered_ids = position_in_unit_cubic.numpy().argsort()
        
        
        s_i = torch.ones(x.shape, dtype = torch.double)
        
        
        
        offset = self.function(self.map_to_original_space(s_i, cube_lower))
        
        weights = torch.zeros([x.shape[0], x.shape[1], x.shape[2], x.shape[2]],dtype = torch.double)
        
        for id in range(x.shape[2]):
            
            e_i = torch.zeros(x.shape, dtype = torch.double)
        
#             print(ordered_ids)
         
#             print(ordered_ids[:,:,id].shape)
        
            e_i.scatter_(2, ordered_ids[:,:,id].view(ordered_ids.shape[0],ordered_ids.shape[1],1), 1)
        
        
#             print(e_i)
#             e_i[id] = 1
            
            s_i_previous = s_i
            
            s_i = s_i - e_i
            
#             print(s_i_previous)
#             
#             print(s_i)
#             
#             print(cube_lower)
            
            curr_weights = (self.function(self.map_to_original_space(s_i_previous, cube_lower)) - self.function(self.map_to_original_space(s_i, cube_lower)))/self.gap
            
#             print(curr_weights)
#             
#             print(curr_weights.shape)
#             
#             print(ordered_ids[:,:,id].view(ordered_ids.shape[0],ordered_ids.shape[1],1))
#             
#             print(ordered_ids[:,:,id].view(ordered_ids.shape[0],ordered_ids.shape[1],1).shape)
            
            for j in range(x.shape[2]):
                weights[:,:,:,j].scatter_(2, ordered_ids[:,:,id].view(ordered_ids.shape[0],ordered_ids.shape[1],1), curr_weights[:,:,j].view(x.shape[0], x.shape[1], 1))
            
#             weights[:,:,id] = 
            
            curr_cube_lower = torch.gather(cube_lower, 2, ordered_ids[:,:,id].view(ordered_ids.shape[0],ordered_ids.shape[1],1))
            
            offset += (1 + curr_cube_lower/self.gap).view(x.shape[0], x.shape[1], 1)*(self.function(self.map_to_original_space(s_i, cube_lower)) - self.function(self.map_to_original_space(s_i_previous, cube_lower)))
            
        return weights, offset
            
#             Y = 
     
     
     
    def compute_approximate_value_batch2(self, x):
        '''x:: n*t*q = dim[0]*max_epoch*num_class'''
        
#         cube_higher = self.gap_cubic + cube_lower

        cube_lower = self.get_cube_lower_vec(x)
        
        position_in_unit_cubic = (x - cube_lower)/self.gap
        
        
#         position_in_unit_cubic_list = list(position_in_unit_cubic)
        
#         ordered_ids = sorted(range(len(position_in_unit_cubic_list)), key = lambda k: position_in_unit_cubic_list[k])
        
        ordered_ids = torch.argsort(position_in_unit_cubic, dim = 2)
        
        
#         print('x::', position_in_unit_cubic)
#         
#         print('ordered_ids::', ordered_ids)
        
#         ordered_ids = position_in_unit_cubic.numpy().argsort()
        
        
        s_i = torch.ones(x.shape, dtype = torch.double)
        
        
        
        offset = Variable(self.function(self.map_to_original_space(s_i, cube_lower)))
        
#         weights = torch.zeros([x.shape[0], x.shape[1], x.shape[2], x.shape[2]],dtype = torch.double)
        
        for id in range(x.shape[2]):
            
            e_i = torch.zeros(x.shape, dtype = torch.double)
        
#             print(ordered_ids)
         
#             print(ordered_ids[:,:,id].shape)
        
            e_i.scatter_(2, ordered_ids[:,:,id].view(ordered_ids.shape[0],ordered_ids.shape[1],1), 1)
        
        
#             print(e_i)
#             e_i[id] = 1
            
            s_i_previous = s_i
            
            s_i = s_i - e_i
            
#             print(s_i_previous)
#             
#             print(s_i)
#             
#             print(cube_lower)
            
#             curr_weights = (self.function(self.map_to_original_space(s_i_previous, cube_lower)) - self.function(self.map_to_original_space(s_i, cube_lower)))/self.gap
            
#             print(curr_weights)
#             
#             print(curr_weights.shape)
#             
#             print(ordered_ids[:,:,id].view(ordered_ids.shape[0],ordered_ids.shape[1],1))
#             
#             print(ordered_ids[:,:,id].view(ordered_ids.shape[0],ordered_ids.shape[1],1).shape)
            
#             for j in range(x.shape[2]):
#                 weights[:,:,:,j].scatter_(2, ordered_ids[:,:,id].view(ordered_ids.shape[0],ordered_ids.shape[1],1), curr_weights[:,:,j].view(x.shape[0], x.shape[1], 1))
            
#             weights[:,:,id] = 
            
            curr_cube_lower = torch.gather(cube_lower, 2, ordered_ids[:,:,id].view(ordered_ids.shape[0],ordered_ids.shape[1],1))
            
            offset += (1 + curr_cube_lower/self.gap).view(x.shape[0], x.shape[1], 1)*(self.function(self.map_to_original_space(s_i, cube_lower)) - self.function(self.map_to_original_space(s_i_previous, cube_lower)))
        
        
        
        return offset
            
#             Y = 
        

    def compute_approximate_value(self, x):
        
        
#         cube_higher = self.gap_cubic + cube_lower

        cube_lower = self.get_cube_lower_vec(x)
        
        position_in_unit_cubic = (x - cube_lower)/self.gap
        
        
        position_in_unit_cubic_list = list(position_in_unit_cubic)
        
        ordered_ids = sorted(range(len(position_in_unit_cubic_list)), key = lambda k: position_in_unit_cubic_list[k])
        
        
        s_i = torch.ones(self.dim, dtype = torch.double)
        
        
        
        offset = self.function(self.map_to_original_space(s_i, cube_lower))
        
        weights = torch.zeros([self.dim, offset.shape[0]],dtype = torch.double)
        
        for id in ordered_ids:
            
            e_i = torch.zeros(self.dim, dtype = torch.double)
            
            e_i[id] = 1
            
            s_i_previous = s_i
            
            s_i = s_i - e_i
            
#             print(s_i_previous)
#             
#             print(s_i)
#             
#             print(cube_lower)


            weights[id] = (self.function(self.map_to_original_space(s_i_previous, cube_lower)) - self.function(self.map_to_original_space(s_i, cube_lower)))/self.gap
            
            offset += (1 + cube_lower[id]/self.gap)*(self.function(self.map_to_original_space(s_i, cube_lower)) - self.function(self.map_to_original_space(s_i_previous, cube_lower)))


            
#             weights[id] = self.function(self.map_to_original_space(s_i_previous, cube_lower)) - self.function(self.map_to_original_space(s_i, cube_lower))
#             
#             offset += self.function(self.map_to_original_space(s_i, cube_lower)) - self.function(self.map_to_original_space(s_i_previous, cube_lower))
            
        return weights, offset
            
#             Y = 
        
        
        
        
        
        
    
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

#     def piecewise_linear_interpolate_coeff_batch(self, X, total_time):
#         
# #         pid = os.getpid()
# #     
# # #     prev_mem=0
# #     
# #         print('pid::', pid)
# #         sig_function = 
# 
#         
# #         x_l_tensor = torch.tensor(self.x_l, dtype = torch.double)
# #         x_u_tensor = torch.tensor(self.x_u, dtype = torch.double)
# #         
# #         dim = X.shape
#         
# #         x_l_tensor = x_l_tensor.repeat(dim[0], dim[1])
#         
# #         x_u_tensor = x_u_tensor.repeat(dim[0], dim[1])
# #         t1 = time.time()
# #         compare_l_res = torch.lt(X, x_l_tensor).type(torch.DoubleTensor)
# #         
# #         w_res = self.w_seq[0]*compare_l_res
# #         
# #         b_res = self.b_seq[0]*compare_l_res
# #         
# #         compare_u_res = torch.gt(X, x_u_tensor).type(torch.DoubleTensor)
# #         
# #         w_res = w_res + self.w_seq[-1]*compare_u_res
# #         
# #         b_res = b_res + self.b_seq[-1]*compare_u_res
# #         
# #         mask_entries = torch.ones(dim, dtype = torch.int) - compare_l_res.type(torch.IntTensor) - compare_u_res.type(torch.IntTensor)
# #         Y_value = self.function(X)
# #         
# #         Y_value = (1 - self.function(X))
#         t1 = time.time()
#         
#         interval_id =(X - self.x_l)/self.gap
#         
# #         interval_id = np.floor(interval_id)
#         
#         interval_id[interval_id < 0] = -1
#         
#         interval_id[interval_id > len(self.w_seq)] = len(self.w_seq) - 2
#         
#         interval_id += 1
#         
#         interval_id = interval_id.astype(int)
#         
#         
# #         w_res = 
# #         
# #         b_res = 
#         
#         res = X*self.w_seq[interval_id]+self.b_seq[interval_id]
#         
#         t2 = time.time()
#         
#         total_time += (t2 - t1)
#         
# #         print(interval_id)
# # 
# #         print(self.w_seq)
# #         
# #         print(self.b_seq)
#         
#         return res, total_time
#         
# #         interval_id = mask_entries.type(torch.DoubleTensor)*interval_id
# #           
# #         interval_w_res = interval_id.clone()
# #         
# #         interval_b_res = interval_id.clone()
# #         
# #         interval_w_res = interval_w_res.detach().apply_(self.get_parameter_w)
# #         
# #         interval_b_res = interval_b_res.detach().apply_(self.get_parameter_b)
# #         t1 = time.time()
#         
# #         interval_x_1 = self.x_l + self.gap * interval_id
# #            
# # #         interval_x_2 = self.x_l + self.gap * (interval_id + 1)
# #           
# # #         interval_x_2 = self.x_u if interval_x_2 > self.x_u else interval_x_2
# #           
# # #         interval_y_1_1 = interval_x_1.clone()
# # #           
# # #         interval_y_2_1 = interval_x_2.clone()  
# # #             
# # #         t1 = time.time()  
# # #           
# # #         interval_y_1_1.detach().apply_(self.function)
# # #           
# # #         interval_y_2_1.detach().apply_(self.function)  
# # #           
# # #         t2 = time.time()
# #         
# # #         element_wise_time = (t2 - t1)
# # #         print('element_wise_time::', element_wise_time)
# # 
# #         interval_y_1 = 1 - self.function(interval_x_1)
# #           
# # #         interval_y_2 = 1 - self.function(interval_x_2)
# #         
# #         
# #         curr_gap = interval_x_1 - X
# #         
# # #         curr_gap.sub_(X)
# #         
# #         
# #         interval_w_res = (interval_y_1 - Y_value)#(interval_x_1 - X)
# #         
# #         interval_w_res /= curr_gap
# #         
# #         interval_b_res = interval_y_1
# # 
# #         interval_x_1 *= (interval_w_res)
# # 
# #         interval_b_res -= interval_x_1
# # 
# # #         t2 = time.time()
# # #         
# # #         total_time += (t2 - t1)
# # 
# # #         sizes = [get_tensor_size(Y_value), get_tensor_size(interval_id), get_tensor_size(interval_x_1),
# # #                  get_tensor_size(interval_y_1), get_tensor_size(curr_gap), get_tensor_size(interval_w_res),  
# # #                  get_tensor_size(interval_b_res)]
# # #         
# # #         print(sizes)
# #         
# #         
# #         del interval_id, interval_y_1, interval_x_1, curr_gap, Y_value
# # 
# # 
# # 
# # #         print(sizes)
# # 
# #         return interval_w_res, interval_b_res, total_time
# #         print('x_1::', interval_x_1)
# #         
# #         print('x_2::', interval_x_2)
# #         
# #         
# #         print('layer_x_1::', sig_function(interval_x_1))
# #         
# #         print('layer_x_2::', sig_function(interval_x_2))
# #         
# # 
# #         print('y_1::', interval_y_1)
# #         
# #         print('y_2::', interval_y_2)
# # 
# #         print('y_1_1::', interval_y_1_1)
# #         
# #         print('y_2_2::', interval_y_2_1)
#         
# #         w_res = interval_w_res*mask_entries.type(torch.DoubleTensor) + w_res
# #         
# #         b_res = interval_b_res*mask_entries.type(torch.DoubleTensor) + b_res
# #         
# #         
# #         
# #         
# #         
# #         return w_res, b_res, total_time
# 
#     def piecewise_linear_interpolate_coeff_batch2(self, X):
#         
# #         pid = os.getpid()
# #     
# # #     prev_mem=0
# #     
# #         print('pid::', pid)
# #         sig_function = 
# 
#         
# #         x_l_tensor = torch.tensor(self.x_l, dtype = torch.double)
# #         x_u_tensor = torch.tensor(self.x_u, dtype = torch.double)
# #         
# #         dim = X.shape
#         
# #         x_l_tensor = x_l_tensor.repeat(dim[0], dim[1])
#         
# #         x_u_tensor = x_u_tensor.repeat(dim[0], dim[1])
# #         t1 = time.time()
# #         compare_l_res = torch.lt(X, x_l_tensor).type(torch.DoubleTensor)
# #         
# #         w_res = self.w_seq[0]*compare_l_res
# #         
# #         b_res = self.b_seq[0]*compare_l_res
# #         
# #         compare_u_res = torch.gt(X, x_u_tensor).type(torch.DoubleTensor)
# #         
# #         w_res = w_res + self.w_seq[-1]*compare_u_res
# #         
# #         b_res = b_res + self.b_seq[-1]*compare_u_res
# #         
# #         mask_entries = torch.ones(dim, dtype = torch.int) - compare_l_res.type(torch.IntTensor) - compare_u_res.type(torch.IntTensor)
# #         Y_value = self.function(X)
# #         
# #         Y_value = (1 - self.function(X))
#         interval_id =(X - self.x_l)/self.gap
# #         print(interval_id)
#         
# #         interval_id = np.floor(interval_id)
#         
#         interval_id[interval_id < 0] = -1
#         
#         interval_id[interval_id >= len(self.w_seq) - 1] = len(self.w_seq) - 2
#         
#         interval_id += 1
#         
#         interval_id = interval_id.type(torch.IntTensor)
#         
#         
# #         w_res = 
# #         
# #         b_res = 
# #         print(interval_id)
#         
#         return torch.as_tensor(self.w_seq[interval_id.numpy()], dtype = torch.double), torch.as_tensor(self.b_seq[interval_id.numpy()], dtype = torch.double)
#         
# #         interval_id = mask_entries.type(torch.DoubleTensor)*interval_id
# #           
# #         interval_w_res = interval_id.clone()
# #         
# #         interval_b_res = interval_id.clone()
# #         
# #         interval_w_res = interval_w_res.detach().apply_(self.get_parameter_w)
# #         
# #         interval_b_res = interval_b_res.detach().apply_(self.get_parameter_b)
# #         t1 = time.time()
#         
# #         interval_x_1 = self.x_l + self.gap * interval_id
# #            
# # #         interval_x_2 = self.x_l + self.gap * (interval_id + 1)
# #           
# # #         interval_x_2 = self.x_u if interval_x_2 > self.x_u else interval_x_2
# #           
# # #         interval_y_1_1 = interval_x_1.clone()
# # #           
# # #         interval_y_2_1 = interval_x_2.clone()  
# # #             
# # #         t1 = time.time()  
# # #           
# # #         interval_y_1_1.detach().apply_(self.function)
# # #           
# # #         interval_y_2_1.detach().apply_(self.function)  
# # #           
# # #         t2 = time.time()
# #         
# # #         element_wise_time = (t2 - t1)
# # #         print('element_wise_time::', element_wise_time)
# # 
# #         interval_y_1 = 1 - self.function(interval_x_1)
# #           
# # #         interval_y_2 = 1 - self.function(interval_x_2)
# #         
# #         
# #         curr_gap = interval_x_1 - X
# #         
# # #         curr_gap.sub_(X)
# #         
# #         
# #         interval_w_res = (interval_y_1 - Y_value)#(interval_x_1 - X)
# #         
# #         interval_w_res /= curr_gap
# #         
# #         interval_b_res = interval_y_1
# # 
# #         interval_x_1 *= (interval_w_res)
# # 
# #         interval_b_res -= interval_x_1
# # 
# # #         t2 = time.time()
# # #         
# # #         total_time += (t2 - t1)
# # 
# # #         sizes = [get_tensor_size(Y_value), get_tensor_size(interval_id), get_tensor_size(interval_x_1),
# # #                  get_tensor_size(interval_y_1), get_tensor_size(curr_gap), get_tensor_size(interval_w_res),  
# # #                  get_tensor_size(interval_b_res)]
# # #         
# # #         print(sizes)
# #         
# #         
# #         del interval_id, interval_y_1, interval_x_1, curr_gap, Y_value
# # 
# # 
# # 
# # #         print(sizes)
# # 
# #         return interval_w_res, interval_b_res, total_time
# #         print('x_1::', interval_x_1)
# #         
# #         print('x_2::', interval_x_2)
# #         
# #         
# #         print('layer_x_1::', sig_function(interval_x_1))
# #         
# #         print('layer_x_2::', sig_function(interval_x_2))
# #         
# # 
# #         print('y_1::', interval_y_1)
# #         
# #         print('y_2::', interval_y_2)
# # 
# #         print('y_1_1::', interval_y_1_1)
# #         
# #         print('y_2_2::', interval_y_2_1)
#         
# #         w_res = interval_w_res*mask_entries.type(torch.DoubleTensor) + w_res
# #         
# #         b_res = interval_b_res*mask_entries.type(torch.DoubleTensor) + b_res
# #         
# #         
# #         
# #         
# #         
# #         return w_res, b_res, total_time
# 
# 
# 
#     def piecewise_linear_interpolate_coeff(self, x):
# #         print(x)
#         if x < self.x_l:
# #             return 0, self.function(self.x_l)
#             return self.w_seq[0], self.b_seq[0]
#         if x > self.x_u:
# #             return 0, self.function(self.x_u)
#             return self.w_seq[-1], self.b_seq[-1]
#         
#         interval_id = int((x - self.x_l)/self.gap)
#          
#         return [self.w_seq[interval_id + 1], self.b_seq[interval_id + 1]]
#         
# #         interval_id = int((x - self.x_l)/self.gap)
# #          
# #         interval_x_1 = self.x_l + self.gap * interval_id
# #          
# #         interval_x_2 = self.x_l + self.gap * (interval_id + 1)
# #          
# #         interval_x_2 = self.x_u if interval_x_2 > self.x_u else interval_x_2
# #          
# #         interval_y_1 = self.function(interval_x_1)
# #          
# #         interval_y_2 = self.function(interval_x_2)
# #          
# #         w = (interval_y_2  -interval_y_1)/(interval_x_2  -interval_x_1)
# #          
# #         b = interval_y_1 - interval_x_1*w
# #          
# #         return w, b
#     

def function(x):
    return softmax_layer2(x)


if __name__ == '__main__':
    Pi = piecewise_linear_interpolation_multi_dimension(torch.tensor([-10,-10], dtype = torch.double), torch.tensor([10,10], dtype = torch.double), function, 1e-4)
    
    Pi2 = piecewise_linear_interpolation_multi_dimension(torch.tensor([-10,-10], dtype = torch.double), torch.tensor([10,10], dtype = torch.double), softmax_layer, 1e-4)

    
    x = torch.rand([3,1,2], dtype = torch.double)
    
    
#     x = torch.tensor([[[1,1]]], dtype = torch.double)
    
    print(x)
    
    expect_values = softmax_layer2(x)
    
    '''weights:: x.shape[0], x.shape[1], x.shape[2], offset.shape[0]'''
    
    '''offsets:: x.shape[0], x.shape[1], offset.shape[0]'''
    
    
    
    
    
    
    
    weights, offsets = Pi.compute_approximate_value_batch(x)
    
    
    
    
    
    
    
    approx_weights = torch.mm(softmax_layer(x[0][0]).view(x.shape[2], 1), softmax_layer(x[0][0]).view(1, x.shape[2]))
    
    print('approx_weight1::', approx_weights)
    
    approx_weights = torch.diag(torch.diag(approx_weights)) - approx_weights
    
    print('approx_weight1::', approx_weights)
    
    approx_weights = torch.diag(-torch.sum(approx_weights, dim = 0)) + approx_weights 
    
    
    print('approx_weight1::', approx_weights)
    
    print('expect_weight2::', weights[0][0])
    
    print(approx_weights - weights[0][0])
    
    
    
    
    
    
    
    
    
    print(weights.shape)
     
#     x_unit_space = Pi.map_from_original_space(x, Pi.get_cube_lower_vec(x))
     
    approx_value = torch.bmm(weights.view(3*1, 2, 2), x.view(3*1, 2, 1)) 
     
    
     
#     print(x_unit_space)
#      
#      
#     approx_value = torch.bmm(x_unit_space.view(x.shape[0]*x.shape[1], 1, x.shape[2]), weights.view(x.shape[0]*x.shape[1], x.shape[2], x.shape[2]))
     
    approx_value = approx_value.view(offsets.shape) + offsets
     
#     approx_value = torch.t(torch.mm(weights, x_unit_space.view(-1, 1))) + offsets
    
    
    approx_value2 = torch.zeros(x.shape, dtype = torch.double)
      
    weights2 = torch.zeros([x.shape[0], x.shape[1], x.shape[2], x.shape[2]], dtype = torch.double)
      
    offsets2 = torch.zeros([x.shape[0], x.shape[1], x.shape[2]], dtype = torch.double)
      
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
              
            print('x::', x[i][j])
              
            print('expected::', softmax_layer(x[i][j]))
              
            curr_weight, curr_offset = Pi2.compute_approximate_value(x[i][j])
              
            weights2[i][j] = curr_weight
              
            print(curr_weight)
              
            offsets2[i][j] = curr_offset
              
            print(curr_offset)
              
#             x_unit_space = Pi.map_from_original_space(x[i][j], Pi.get_cube_lower_vec(x[i][j]))
#              
#             print(x_unit_space)
#              
#             print(torch.mm(weights[i][j], x_unit_space.view(-1,1)))
              
            approx_value2[i][j] = (torch.mm(weights2[i][j], x[i][j].view(-1,1)) + offsets2[i][j].view(-1,1)).view(approx_value2[i][j].shape)
      
            print('approx::', approx_value2[i][j])
    
    print('weight_gap::', weights - weights2)
    
    print('offsets_gap::', offsets - offsets2)
    
    print('offsets2::', offsets2)
    
    print('weights2::', weights2)
    
    print('approx2::', approx_value2)
    
    print('approx::', approx_value)
    
    print('exact::', expect_values)
    
    print('weights::', weights)
    
    print('offsets::', offsets)
    
    print(approx_value - expect_values)
    
    print(approx_value2 - expect_values)
    
    
    
    
    
    
        