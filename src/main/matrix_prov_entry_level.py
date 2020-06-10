'''
Created on Oct 20, 2018

'''

from bokeh.models import ranges
import torch

import numpy as np;


# from blaze.expr.strings import str
prov_one = "1"
prov_zero = "0"

class M_prov(object):
    '''
    classdocs
    '''

    def __init__(self, M, data_matrix_list, prov_list):
        self.M = M
        self.data_matrix_list = data_matrix_list
        self.prov_list = prov_list
#     def __init__(self, M, name, partition):
#         '''
#         Constructor
#         '''
#         
#         '''matrix'''
#         self.M = M
#         
#         '''shape'''
#         self.shape = list(M.size())
#         
#         '''name'''
#         
#         self.name = name
#         
#         prov_list, data_matrix_list = add_prov_token_per_row(M, name, self.shape, partition)
#         
#         '''list of list'''
#         self.prov_list = prov_list
#         
#         '''list of matrix'''
#         self.data_matrix_list = data_matrix_list
#         
#         
#         '''list of matrix'''
#         self.supple_matrix_list1 = supple_matrix_list1
#         
#         '''list of matrix'''
#         self.supple_matrix_list2 = supple_matrix_list2
    
    def create_supple_matrix(n, m):
        min_len = min(n, m)
        identity_matrix = torch.eye(min_len)
        res = torch.zeros([n,m])
        res[0:min_len, 0:min_len] = identity_matrix
        return res
    
    def add_prov_token_constants(M, shape):
        '''lists of lists of sets'''
        prov_list = []
    
        '''lists of lists of matrix'''
        data_matrix_list = []
    
        for i in range(shape[0]):
            
            this_data_matrix_list = []
            
            this_prov_list = []
            
            for j in range(shape[1]):
            
                curr_prov = set()
                unique_id = prov_one
                curr_prov.add(unique_id)
                curr_prov_list = []
                curr_prov_list.append(curr_prov)
                this_prov_list.append(curr_prov_list)
                
                curr_data_matrix_list = []
                curr_data_matrix_list.append(M[i,j])
                this_data_matrix_list.append(curr_data_matrix_list)
            
            prov_list.append(this_prov_list)
            data_matrix_list.append(this_data_matrix_list)


        return M_prov(M, data_matrix_list, prov_list)
    
    
    def add_prov_token(M, name, shape):
        
        
        '''lists of lists of sets'''
        prov_list = []
    
        '''lists of lists of matrix'''
        data_matrix_list = []
    
#         supple_matrix_list1 = []
#         
#         supple_matrix_list2 = []
        
        start = 0
        
#         shape = list(M.size())
        
        for i in range(shape[0]):
            
            this_prov_list = []
            
            this_matrix_entry_list = []
            
            for j in range(shape[1]):
            
                curr_prov = set()
                unique_id_str = name + "'" + str(i) + "," + str(j)
                unique_id = unique_id_str#hash(unique_id_str)
                curr_prov.add(unique_id)
                curr_prov_list = []
                curr_prov_list.append(curr_prov)
                this_prov_list.append(curr_prov_list)
                
                end = -1
                
    #             if i >= len(partition):
    #                 end = shape[0]
    #             else:
    #                 end = partition[i]
                
    #             data_matrix = torch.zeros(shape[0]. shape[1])
    #             
    #             data_matrix[start:end,:] = M[start:end,:]
                curr_data_matrix_list = []
                curr_data_matrix_list.append(M[i,j])
                this_matrix_entry_list.append(curr_data_matrix_list)
                
#             data_matrix_list.append(M[start:end,:])
#
            prov_list.append(this_prov_list)
            data_matrix_list.append(this_matrix_entry_list)
#             supple_matrix_list1.append(create_supple_matrix(shape[0], end - start))
#             
#             supple_matrix_list2.append(torch.eye(shape[1]))
            
#             if(i < len(partition)):
#                 start = partition[i]
        
        
        
        
        return M_prov(M, data_matrix_list, prov_list)
    
    def prov_matrix_transpose(P, res):
        
        data_matrix_list = list(map(list, zip(*(P.data_matrix_list))))
        
        prov_matrix_list = list(map(list, zip(*(P.prov_list))))
        
#         for i in range(len(P.data_matrix_list)):
#             curr_data_matrix_list = []
#             for j in range(len(P.data_matrix_list[i])):
#                 curr_data_matrix_list.append(torch.transpose(P.data_matrix_list[i][j], 0, 1))
#             data_matrix_list.append(curr_data_matrix_list)
        
        return M_prov(res, data_matrix_list, prov_matrix_list)
    
#     def prov_matrix_mul_matrix(P, M, res):
#         
#         shape1 = list(P.M.size())
#         shape2 = list(M.size())
#         
#         assert shape1[1] == shape2[0]
#         
#         
#         data_matrix_list = []
#         for i in range(len(P.data_matrix_list)):
#             data_matrix_list.append(torch.mm(P.data_matrix_list[i], M))
#         
#         return M_prov(res, data_matrix_list, P.prov_list)
    
    
#     def matrix_mul_prov_matrix(M, P, res):
#         
#         shape1 = list(M.size())
#         shape2 = list(P.M.size())
#         
#         assert shape1[1] == shape2[0]
#         
#         
#         data_matrix_list = []
#         for i in range(len(P.data_matrix_list)):
#             data_matrix_list.append(torch.mm(M, P.data_matrix_list[i]))
#         
#         return M_prov(res, data_matrix_list, P.prov_list)
#     
#     def prov_matrix_add_matrix(P, M, res):
#         shape1 = list(P.M.size())
#         shape2 = list(M.size())
#         
#         assert shape1[0] == shape2[0] and shape1[1] == shape2[1]
#         
#         prov_list = []
#         prov_list.extend(P.prov_list) 
#         
#         data_matrix_list = []
#         data_matrix_list.extend(P.data_matrix_list)
#         
#         
#         for i in range(len(P.data_matrix_list)):
#             
# #             curr_prov_list = []
# #             curr_data_matrix_list = []
# #             curr_prov_list.extend(P.prov_list[i])
# #             curr_data_matrix_list.extend(P.data_matrix_list[i])
#             curr_prov = {}
#             curr_prov.add(hash(prov_one))
#             prov_list[i].append(curr_prov)
#             data_matrix_list[i].append(M[i:(i+1),:])
# #             prov_list.append(curr_prov_list)
# #             data_matrix_list.append(curr_data_matrix_list)
#         
#         return M_prov(res, data_matrix_list, prov_list)
#     
#     def matrix_add_prov_matrix(M, P, res):
#         shape1 = list(P.M.size())
#         shape2 = list(M.size())
#         
#         assert shape1[0] == shape2[0] and shape1[1] == shape2[1]
#         
#         prov_list = []
#         
#         
#         data_matrix_list = []
#         
#         for i in range(len(P.data_matrix_list)):
# 
#             curr_prov_list=[]
#             curr_data_matrix_list = []
#             curr_prov = {}
#             curr_prov.add(hash(prov_one))
#             curr_prov_list.append(curr_prov)
#             curr_data_matrix_list.append(M[i:(i+1),:])
#             curr_prov_list.extend(P.prov_list[i])
#             curr_data_matrix_list.extend(P.data_matrix_list[i])
# 
#             
#             
#             prov_list.append(curr_prov_list)
#             data_matrix_list.append(curr_data_matrix_list)
#              
#         return M_prov(res, data_matrix_list, prov_list)
#     
        
    def prov_matrix_add_prov_matrix(P1, P2, res):
        
        
        
        
        shape11 = len(P1.data_matrix_list)
        shape12 = len(P1.data_matrix_list[0]) 
        shape21 = len(P2.data_matrix_list)
        shape22 = len(P2.data_matrix_list[0])
        assert shape11 == shape21 and shape12 == shape22
        
        
         
        prov_list = []
        data_matrix_list = []
        
        for i in range(shape11):
            this_prov_list = []
            this_data_matrix_list = []
            
            for j in range(shape12):
                curr_prov_list1 = P1.prov_list[i][j]
                curr_prov_list2 = P2.prov_list[i][j]
                curr_data_matrix_list1 = P1.data_matrix_list[i][j]
                curr_data_matrix_list2 = P2.data_matrix_list[i][j]
                
                curr_prov_list = []
                curr_data_matrix_list = []
                curr_prov_list.extend(curr_prov_list1)
                curr_data_matrix_list.extend(curr_data_matrix_list1)
                
                for k in range(len(curr_prov_list2)):
                    try:
                        index = curr_prov_list.index(curr_prov_list2[k])
                        curr_data_matrix_list[index] = curr_data_matrix_list[index] + curr_data_matrix_list2[k]
                    except ValueError:
                        curr_prov_list.append(curr_prov_list2[k])
                        curr_data_matrix_list.append(curr_data_matrix_list2[k])
                
                this_prov_list.append(curr_prov_list)
                this_data_matrix_list.append(curr_data_matrix_list)
            prov_list.append(this_prov_list)
            data_matrix_list.append(this_data_matrix_list)
        
#         for i in range(len(P1.data_matrix_list)):
#             curr_prov_list = []
#             curr_prov_list.extend(P1.prov_list[i])
#             curr_data_matrix_list = []
#             curr_data_matrix_list.extend(P1.data_matrix_list[i])
# #             print(P1.data_matrix_list[i])
# #             print('curr_data_matrix',curr_data_matrix_list)
#             for j in range(len(P2.prov_list[i])):
#                 try:
#                     index = curr_prov_list.index(P2.prov_list[i][j])
#                     curr_data_matrix_list[index] = curr_data_matrix_list[index] + P2.data_matrix_list[i][j]
#                 except ValueError:
#                     curr_prov_list.append(P2.prov_list[i][j])
#                     curr_data_matrix_list.append(P2.data_matrix_list[i][j])
#                      
#             
#             prov_list.append(curr_prov_list)
#             data_matrix_list.append(curr_data_matrix_list)            
         
        return M_prov(res, data_matrix_list, prov_list)
        
    
        
    def prov_matrix_mul_matrix(P, M, res):
        shape1 = list(P.M.size())
        shape2 = list(M.size())
        
        assert shape1[1] == shape2[0]
        sub_shape1 = list(P.data_matrix_list[0][0].size())
        assert sub_shape1[1] == 1
        
        
        data_matrix_list = []
        prov_list = []
        
        for i in range(shape2[0]):
            curr_data_matrix_list = P.data_matrix_list[i]
            curr_prov_list = P.prov_list[i]
#             prov_list.extend(curr_prov_list)
            sub_M = M[i:i+1,:]
            for j in range(len(curr_data_matrix_list)):
                try:
                    index = prov_list.index(curr_prov_list[j])
                    data_matrix_list[index] = torch.add(data_matrix_list[index], torch.mm(curr_data_matrix_list[j],sub_M))
                except ValueError:
                    prov_list.append(curr_prov_list)
                    data_matrix_list.append(torch.mm(curr_data_matrix_list[j],sub_M))
        return M_prov(res, data_matrix_list, prov_list)
    
    def sub_matrix_mul_with_prov(sub_data_matrix_list1, sub_data_matrix_list2, sub_prov_list1, sub_prov_list2, prov_list, data_matrix_list):
        for i in range(len(sub_data_matrix_list1)):
#             print('here!!!!', sub_data_matrix_list1[i])
#             print('here!!!!', sub_data_matrix_list2)
            for j in range(len(sub_data_matrix_list2)):
#                 print('here', sub_data_matrix_list1[i])
#                 print('here', sub_data_matrix_list2[j])
#                 curr_sub_data_matrix = torch.mm(sub_data_matrix_list1[i], sub_data_matrix_list2[j])
                curr_sub_data_matrix = sub_data_matrix_list1[i]*sub_data_matrix_list2[j]
#                 print(sub_prov_list1)
#                 print(sub_prov_list1[i])
                curr_prov = set()
                curr_prov.update(sub_prov_list1[i])
                curr_prov.update(sub_prov_list2[j])
                if len(curr_prov) > 1:
                    curr_prov.discard(prov_one)
                try:
                    index = prov_list.index(curr_prov)
                    data_matrix_list[index] = torch.add(data_matrix_list[index], curr_sub_data_matrix)
                except ValueError:
                    prov_list.append(curr_prov)
                    data_matrix_list.append(curr_sub_data_matrix)
        return
    
    def sub_matrix_mul_with_prov2(sub_data_matrix_list1, sub_data_matrix_list2, sub_prov_list1, sub_prov_list2, prov_list, data_matrix_list):
        curr_data_matrix_list = []
        curr_prov_list = []
        for i in range(len(sub_data_matrix_list1)):
            for j in range(len(sub_data_matrix_list2)):
                curr_sub_data_matrix = torch.mm(sub_data_matrix_list1[i], sub_data_matrix_list2[j])
#                 print(sub_prov_list1)
#                 print(sub_prov_list1[i])
                curr_prov = set()
                curr_prov.update(sub_prov_list1[i])
                curr_prov.update(sub_prov_list2[j])
                if len(curr_prov) > 1:
                    curr_prov.discard(prov_one)
#                 try:
#                     index = prov_list.index(list(curr_prov))
#                     data_matrix_list[index] = torch.add(data_matrix_list[index], curr_sub_data_matrix)
#                 except ValueError:
                
                curr_prov_list.append(curr_prov)
                curr_data_matrix_list.append(curr_sub_data_matrix)
        prov_list.append(curr_prov_list)
        data_matrix_list.append(curr_data_matrix_list)
#         print(data_matrix_list)
        return
    
    def prov_matrix_mul_prov_matrix(P1, P2, res):
#         shape1 = list(P1.M.size())
#         shape2 = list(P2.M.size())
#         
#         assert shape1[1] == shape2[0]
#         sub_shape1 = len(P1.data_matrix_list[0][0])
#         sub_shape2 = list(P2.data_matrix_list[0][0])
        
        size1 = len(P1.data_matrix_list)
        size2 = len(P1.data_matrix_list[0])
        size3 = len(P2.data_matrix_list)
        size4 = len(P2.data_matrix_list[0])
        
#         assert sub_shape1[1] == sub_shape2[0] and sub_shape1[0] == 1
        assert size2 == size3
        
        data_matrix_list = []
        prov_list = []
        
        
        for i in range(size1):
            this_data_matrix_list = []
            this_prov_list = []
            
            for j in range(size4):
                
                curr_data_matrix_list = []
                curr_prov_list = []
                
                for k in range(size3):
                    curr_data_matrix_list1 = P1.data_matrix_list[i][k]
                    curr_data_matrix_list2 = P2.data_matrix_list[k][j]
                    curr_prov_list1 = P1.prov_list[i][k]
                    curr_prov_list2 = P2.prov_list[k][j]
                    
#                     print(i,j,k)
#                     print(curr_data_matrix_list1[0])
#                     print(curr_data_matrix_list2)
                    
                    M_prov.sub_matrix_mul_with_prov(curr_data_matrix_list1, curr_data_matrix_list2, curr_prov_list1, curr_prov_list2, curr_prov_list, curr_data_matrix_list)
                    
                    
                    
                    
                    
                this_data_matrix_list.append(curr_data_matrix_list)
                this_prov_list.append(curr_prov_list)
            
            data_matrix_list.append(this_data_matrix_list)
            prov_list.append(this_prov_list)
        
#         if len(P2.data_matrix_list) == 1:
#             for i in range(len(P1.data_matrix_list)):
#                 for j in range(len(P2.data_matrix_list)):
#                     curr_data_matrix_list1 = P1.data_matrix_list[i]
#                     curr_data_matrix_list2 = P2.data_matrix_list[j]
#                     curr_prov_list1 = P1.prov_list[i]
#                     curr_prov_list2 = P2.prov_list[j]
#                     M_prov.sub_matrix_mul_with_prov2(curr_data_matrix_list1, curr_data_matrix_list2, curr_prov_list1, curr_prov_list2, prov_list, data_matrix_list)
#             
#             return M_prov(res, data_matrix_list, prov_list)
#         
#         else:
#             for i in range(len(P1.data_matrix_list)):
#     #             for j in range(len(P2.data_matrix_list)):
#                     curr_data_matrix_list1 = P1.data_matrix_list[i]
#                     curr_data_matrix_list2 = P2.data_matrix_list[i]
#                     curr_prov_list1 = P1.prov_list[i]
#                     curr_prov_list2 = P2.prov_list[i]
#                     M_prov.sub_matrix_mul_with_prov(curr_data_matrix_list1, curr_data_matrix_list2, curr_prov_list1, curr_prov_list2, prov_list, data_matrix_list)
# 
#             final_prov_list = []
#             final_prov_list.append(prov_list)
#             final_data_matrix = []
#             final_data_matrix.append(data_matrix_list)    
        return M_prov(res, data_matrix_list, prov_list)
    
    def constant_mul_prov_matrix(P1, P2, res):
        if len(P2.data_matrix_list) == 1 and len(P2.data_matrix_list[0]) == 1:
            P = P1
            P1 = P2
            P2 = P
        else:
            assert len(P1.data_matrix_list) == 1 and len(P1.data_matrix_list[0]) == 1
            
        data_matrix_list = []
        prov_list = []
        
            
        
        
        for i in range(len(P2.data_matrix_list)):
            this_prov_list = []
            
            this_data_matrix_list = []
            
            for j in range(len(P2.data_matrix_list[0])):
                curr_data_matrix_list = P2.data_matrix_list[i][j]
                curr_prov_list = P2.prov_list[i][j]
                
                res_data_matrix_list = []
                res_prov_list = []
                M_prov.sub_matrix_mul_with_prov(curr_data_matrix_list, P1.data_matrix_list[0][0], curr_prov_list, P1.prov_list[0][0], res_prov_list, res_data_matrix_list)
            
                this_prov_list.append(res_prov_list)
                this_data_matrix_list.append(res_data_matrix_list)
            prov_list.append(this_prov_list)
            data_matrix_list.append(this_data_matrix_list)
        
        return M_prov(res, data_matrix_list, prov_list)
        
                    
    def add_prov_matrix_add(P1, P2, res):
        len1 = len(P1.prov_list)
        len2 = len(P2.prov_list)
        
        prov_list = []
        data_matrix_list = []
#         supple_matrix_list1 = []
#         supple_matrix_list2 = []
        
        dic_mapping = {}
        
        for i in range(len1):
            prov_list.append(P1.prov_list[i])
            data_matrix_list.append(P1.data_matrix_list[i])
            dic_mapping[P1.prov_list[i]] = i
#             supple_matrix_list1.append(P1.supple_matrix_list1[i])
#             supple_matrix_list2.append(torch.eye(P1.shape[1]))
        
        for i in range(len2):
            if P2.prov_list[i] in dic_mapping:
                index = dic_mapping[P2.prov_list[i]]
                data_matrix_list[index] = data_matrix_list[index] + P2.data_matrix_list[i]
            
            else:
                prov_list.append(P2.prov_list[i])
                data_matrix_list.append(P2.data_matrix_list[i])
            
#             supple_matrix_list1.append(torch.eye(P2.shape[0]))
#             supple_matrix_list2.append(P2.supple_matrix_list2[i]) 
        
        
        
        return M_prov(res, data_matrix_list, prov_list)
    
    def add_prov_matrix_mul(P1, P2, res):
        len1 = len(P1.prov_list)
        len2 = len(P2.prov_list)
        
        prov_list = []
        data_matrix_list = []
#         supple_matrix_list1 = []
#         supple_matrix_list2 = []
        
        dic_mapping = {}
        
        for i in range(len1):
            prov_list.append(P1.prov_list[i])
            data_matrix_list.append(torch.mm(P1.data_matrix_list[i], P2.M))
            dic_mapping[P1.prov_list[i]] = i
#             supple_matrix_list1.append(P1.supple_matrix_list1[i])
#             supple_matrix_list2.append(torch.eye(P1.shape[1]))
        
        for i in range(len2):
            if P2.prov_list[i] in dic_mapping:
                index = dic_mapping[P2.prov_list[i]]
                data_matrix_list[index] = data_matrix_list[index] + torch.mm(P1.M, P2.data_matrix_list[i])
            
            else:
                prov_list.append(P2.prov_list[i])
                data_matrix_list.append(torch.mm(P1.M, P2.data_matrix_list[i]))
            
#             supple_matrix_list1.append(torch.eye(P2.shape[0]))
#             supple_matrix_list2.append(P2.supple_matrix_list2[i]) 
        
        
        
        return M_prov(res, data_matrix_list, prov_list)
    
    
    def negate_prov_matrix(P, res):
        
        data_matrix_list = []
        prov_list = []
        
        shape1 = len(P.data_matrix_list)
        
        shape2 = len(P.data_matrix_list[0])
        
        for i in range(shape1):
            
            this_prov_list = []
            this_data_matrix_list = []
            
            
            
            for j in range(shape2):
                curr_data_matrix_list = P.data_matrix_list[i][j]
                curr_prov_list = P.prov_list[i][j]
                this_prov_list.append(curr_prov_list)
                update_data_matrix = []
                
                for k in range(len(curr_data_matrix_list)):
                    update_data_matrix.append(-curr_data_matrix_list[k])
                
                
                
                this_data_matrix_list.append(update_data_matrix)
        
            data_matrix_list.append(this_data_matrix_list)
            prov_list.append(this_prov_list)
        
#         for i in range(len(P.data_matrix_list)):
#             
#             curr_data_matrix_list = []
#             curr_prov_list = []
#             for j in range(len(P.data_matrix_list[i])):
#                 curr_data_matrix_list.append(-(P.data_matrix_list[i][j]))
#                 curr_prov_list.append(P.prov_list[i][j])
#             data_matrix_list.append(curr_data_matrix_list);
#             prov_list.append(curr_prov_list)
        return M_prov(res, data_matrix_list, prov_list)
#         for i in range(len1):
#             for j in range(len2):
#                 prov1 = P1.prov_list[i]
#                 prov2 = P2.prov_list[j]
#                 d1 = P1.data_matrix_list[i]
#                 d2 = P2.data_matrix_list[j]
#                 s1 = P1.supple_matrix_list1[i]
#                 s2 = P2.supple_matrix_list2[j]
#                 
#                 prov = prov1.copy()
#                 prov.update(prov2)
#                 
#                 if prov in dic_mapping:
#                     
#                 else:
#                     supple_matrix_list1.append(s1)
#                     supple_matrix_list2.append(s2)
        
        
        
        