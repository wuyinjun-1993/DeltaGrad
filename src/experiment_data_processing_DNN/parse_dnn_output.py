'''
Created on Jan 14, 2020

'''

import numpy as np
import csv
import pandas as pd


# file_name = '../../scripts_general/mnist_lr_varied_batch_30_5'
# 
# output_file_suffix = 'lr_30_5'

file_name = '../../scripts_general/mnist_dnn0'
 
output_file_suffix = 'DNN'

headers = ["varied deletion rate::"]

varied_headers = ["deletion rate::"]


# random_set_header = "random set"

random_set_header = "batch size::"

repetition_header = "repetition"

deletion_rates_num = 5

random_set_num = 5


methods = ["baseline::", "incremental updates::"]

time_labels = ["time_baseline::", "time_provenance::"]

distance_prefix = "tensor("


running_time_label = "time"

distance_label = "distance"

accuracy_label = "accuracy"

test_acc_prefix = "Test Avg. Loss:"

test_acc_keyword = "Accuracy:"


def get_dstance_value(distance_prefix, line):
    
    comma_id = line.find(",")
    
    dist = float(line[len(distance_prefix):comma_id])
    
    return dist
    
    
def get_accuracy_value(test_acc_keyword, line):
    
    keyword_id = line.find(test_acc_keyword)
    
    acc = float(line[keyword_id + len(test_acc_keyword):].strip())
    
    return acc
    
    
    
    
def convert2df(arrays, noise_rates, approach_lists):
    d = {}
    
    d['noise_rate'] = np.array(list(noise_rates))
    
    for i in range(arrays.shape[1]):
        d[approach_lists[i]] = arrays[:,i]
        
    df = pd.DataFrame(data=d)
    
    return df
        


def write_to_csv_file(file_name, results, output_file_suffix):
    
    head_line = []
    
    head_line.append('')
    
    deletion_rate_list = list(results.keys())
    
    
    
#     approach_list = list(results.keys())
    
    batch_size_list = list(results[deletion_rate_list[0]].keys())
    
    repetition_list = results[deletion_rate_list[0]][batch_size_list[0]].keys()
    
    
    deletion_rate_num = len(deletion_rate_list)
    
    batch_size_num = len(batch_size_list)
    
    deletion_rates_num = len(repetition_list)
    
    
    training_time_arr = np.zeros((deletion_rate_num, batch_size_num, deletion_rates_num, len(methods)))
    
    distance_arr = np.zeros((deletion_rate_num, batch_size_num, deletion_rates_num, len(methods)))
    
    test_accuracy_arr = np.zeros((deletion_rate_num, batch_size_num, deletion_rates_num, len(methods)), dtype = np.double)
    
    final_training_time_arr = np.zeros((deletion_rate_num, batch_size_num, len(methods)))
    
    final_distance_arr = np.zeros((deletion_rate_num, batch_size_num, len(methods)))
    
    final_test_accuracy_arr = np.zeros((deletion_rate_num, batch_size_num, deletion_rates_num, len(methods)), dtype = np.double)
    
    final_test_accuracy_arr_diff = np.zeros((deletion_rate_num, batch_size_num, len(methods) - 1), dtype = np.double)
    
    final_test_accuracy_arr_var = np.zeros((deletion_rate_num, batch_size_num, len(methods)), dtype = np.double)
    
#     abs_arr = np.zeros((noise_rate_num, 3))
#     
#     relative_arr = np.zeros((noise_rate_num, 3))
# 
#     mem_arr = np.zeros((noise_rate_num, 4))
    
#     print(approach_list)
    for i in range(len(deletion_rate_list)):
        
        curr_results = results[deletion_rate_list[i]]
        
        print("curr result::", deletion_rate_list[i])
        
        print(curr_results)
        
        curr_deletion_rate = deletion_rate_list[i]
        
        for j in range(len(curr_results.keys())):
            curr_batch_size = list(curr_results.keys())[j]
            
            this_res = curr_results[curr_batch_size]
            
#             print(deletion_rate_list[i], curr_batch_size, this_res)
        
        
            for k in range(deletion_rates_num):
                
                curr_repetition = list(this_res.keys())[k]
                
#                 print(curr_deletion_rate, curr_batch_size, curr_repetition)
                
#                 if curr_deletion_rate == 0.001 and curr_batch_size == 3 and curr_repetition == 2:
#                     print("here")
                
#                 print(this_res[curr_repetition])
                
                for p in range(len(methods)):
        
#                     print(this_res[curr_repetition][methods[p]])
        
                    training_time_arr[i][j][k][p] = this_res[curr_repetition][methods[p]][running_time_label]
                    
                    distance_arr[i][j][k][p] = this_res[curr_repetition][methods[p]][distance_label]
                    
                    test_accuracy_arr[i][j][k][p] = this_res[curr_repetition][methods[p]][accuracy_label]
        
        
    final_training_time_arr = np.mean(training_time_arr, 2)
    
    final_distance_arr = np.mean(distance_arr, 2)
    
    final_test_accuracy_arr = np.mean(test_accuracy_arr, 2)
    
    
    final_test_accuracy_arr_var = np.mean((test_accuracy_arr - final_test_accuracy_arr.reshape((deletion_rate_num, batch_size_num, 1, len(methods))))**2, 2)
    
    for i in range(len(methods) - 1):
 
        final_test_accuracy_arr_diff[:,:,i] = (final_test_accuracy_arr[:,:,i+1] - final_test_accuracy_arr[:,:,0])
        
        


        
#         final_test_accuracy_arr_var[:,:,i] = np.mean(curr_test_accuracy_diff, 2) 
            
            
        
#             training_time_arr[j][i] = this_res[training_time_label]
            
#             training_accuracy_arr[j][i] = this_res[training_accuracy_label]
#             
#             test_accuracy_arr[j][i] = this_res[test_accuracy_label]
#             
#             if curr_approach == provenance_label:
#                 abs_arr[j][0] = this_res[absolute_error_label]
#                 relative_arr[j][0] = this_res[relative_error_label]
#                 mem_arr[j][0] = this_res[mem_usage_label]
#                 
#             if curr_approach == provenance_opt_label:
#                 abs_arr[j][1] = this_res[absolute_error_label]
#                 relative_arr[j][1] = this_res[relative_error_label]
#                 mem_arr[j][1] = this_res[mem_usage_label]
#                 
#             if curr_approach == influence_label:
#                 abs_arr[j][2] = this_res[absolute_error_label]
#                 relative_arr[j][2] = this_res[relative_error_label]
#             
#             
#             if curr_approach == iteration_label:
#                 mem_arr[j][2] = this_res[mem_usage_label]
#                 
#             if curr_approach == standard_lib_label:
#                 mem_arr[j][3] = this_res[mem_usage_label]
#     approach_list.insert(0, '')
    
    
    writer = pd.ExcelWriter(file_name + '_' + output_file_suffix + '.xlsx', engine='xlsxwriter')
    
    for i in range(len(deletion_rate_list)):
        convert2df(final_training_time_arr[i], batch_size_list, methods).to_excel(writer, sheet_name='training time dr (' + str(deletion_rate_list[i]) + ")", index=False)
        convert2df(final_distance_arr[i], batch_size_list, methods).to_excel(writer, sheet_name='distance dr (' + str(deletion_rate_list[i]) + ")", index=False)
        convert2df(final_test_accuracy_arr[i], batch_size_list, methods).to_excel(writer, sheet_name='test accuracy dr (' + str(deletion_rate_list[i]) + ")", index=False)
        convert2df(final_test_accuracy_arr_var[i], batch_size_list, methods).to_excel(writer, sheet_name='test accuracy var dr (' + str(deletion_rate_list[i]) + ")", index=False)
        
        convert2df(final_test_accuracy_arr_diff[i], batch_size_list, methods[1:]).to_excel(writer, sheet_name='test accuracy diff dr (' + str(deletion_rate_list[i]) + ")", index=False)
        
        
    for i in range(len(range(batch_size_num))):
        convert2df(final_training_time_arr[:,i], deletion_rate_list, methods).to_excel(writer, sheet_name='training time bz (' + str(batch_size_list[i]) + ")", index=False)
        convert2df(final_distance_arr[:,i], deletion_rate_list, methods).to_excel(writer, sheet_name='distance bz (' + str(batch_size_list[i]) + ")", index=False)
        convert2df(final_test_accuracy_arr[:,i], deletion_rate_list, methods).to_excel(writer, sheet_name='test accuracy bz (' + str(batch_size_list[i]) + ")", index=False)
        convert2df(final_test_accuracy_arr_var[:,i], deletion_rate_list, methods).to_excel(writer, sheet_name='test accuracy var bz (' + str(batch_size_list[i]) + ")", index=False)
        convert2df(final_test_accuracy_arr_diff[:,i], deletion_rate_list, methods[1:]).to_excel(writer, sheet_name='test accuracy diff dr (' + str(batch_size_list[i]) + ")", index=False)
        
#     convert2df(training_accuracy_arr, noise_rate_list, approach_list).to_excel(writer, sheet_name='training_accuracy', index=False)
#     
#     convert2df(test_accuracy_arr, noise_rate_list, approach_list).to_excel(writer, sheet_name='test_accuracy', index=False)
#     
#     convert2df(abs_arr, noise_rate_list, [provenance_label, provenance_opt_label, influence_label]).to_excel(writer, sheet_name='abs_err', index=False)
#     
#     convert2df(relative_arr, noise_rate_list, [provenance_label, provenance_opt_label, influence_label]).to_excel(writer, sheet_name='relative_err', index=False)
# 
#     convert2df(mem_arr, noise_rate_list, [provenance_label, provenance_opt_label, iteration_label, standard_lib_label]).to_excel(writer, sheet_name='memory_usage', index=False)

    writer.save()    



    


if __name__ == '__main__':
    
    
    
    with open(file_name + '.txt') as fp:
    
        line = "123"
    
        curr_header_id = -1
        
        new_start = False
    
        deletion_rate = 0
    
        state = -1
        
        random_set_ids = []
        
        results = {}
        results_curr_deletion = {}
        
        results_all_repetition = {}
        
        results_curr_repe = {}
        
        curr_method_id = -1
        
        curr_random_set_id = -1
        
        curr_repetition_id = -1
    
        while line:
        
            line = fp.readline()
    
#             if line.startswith(varied_headers[0]):
#                 print(line, state)
    
    
            
            new_start = False
            
            
            for i in range(len(headers)):
                
                if line.startswith(headers[i]):
#                     curr_header = headers[i]
                    new_start = True
                    
                    curr_header_id = i
                  
                    state = 0
                    
                    break
            
            if new_start:
                continue
        
            
            if state == 0 and line.startswith(varied_headers[curr_header_id]):
                
#                 if len(results_curr_deletion) > 0:
#                     results[deletion_rate] = results_curr_deletion.copy()
                
                
                deletion_rate = float(line[len(varied_headers[curr_header_id]):].strip())
                
                
#                 print("deletion rate::", deletion_rate)
                
                results_curr_deletion.clear()
                
                state = 1
                
                continue
                
            if state == 1 and line.startswith(random_set_header):
                
#                 if len(results_all_repetition) > 0:
#                     results_curr_deletion[curr_random_set_id] = results_all_repetition.copy()
                
                    
                results_all_repetition.clear()
                
                curr_random_set_id = int(line[len(random_set_header):].strip())
                
                state = 2
                
                
                
                
                continue
            
            if state == 2 and line.startswith(repetition_header):
                
#                 if len(results_curr_repe) > 0:
                    
                
                
                curr_repetition_id = int(line[len(repetition_header):].strip())
                
                state = 3
                
                results_curr_repe.clear()
                
                continue

            if state == 3:
                for i in range(len(methods)):
                    if line.startswith(methods[i]):
                        curr_method_id = i
                        state = 4
                        
                        break
            
            if state == 4 and line.startswith(time_labels[curr_method_id]):
                running_time = float(line[len(time_labels[curr_method_id]):].strip())
                
                results_curr_repe[methods[curr_method_id]] = {}
                
                results_curr_repe[methods[curr_method_id]][running_time_label]= running_time
                
                state = 5
                
                continue
            
            
            if state == 5 and line.startswith(distance_prefix):
                distance = get_dstance_value(distance_prefix, line)
                
                results_curr_repe[methods[curr_method_id]][distance_label] = distance
                
                state = 6
                
                continue
            
            
            if state == 6 and line.startswith(test_acc_prefix):
                acc = get_accuracy_value(test_acc_keyword, line)
                
                results_curr_repe[methods[curr_method_id]][accuracy_label] = acc
                
                '''next deletion rate::'''
#                 if deletion_rate == 0.001 and curr_random_set_id == 2:
#                     print("here")
#                 
#                 
#                 if curr_method_id == 1 and curr_random_set_id == 5 and curr_repetition_id == 5:
#                     print("here")
                    
                    
                if len(results_curr_repe) == len(methods):
                    results_all_repetition[curr_repetition_id] = results_curr_repe.copy()
                    results_curr_repe.clear()
                    
                    if len(results_all_repetition) == deletion_rates_num:
                        results_curr_deletion[curr_random_set_id] = results_all_repetition.copy()
                        
                        results_all_repetition.clear()
                        
                        if len(results_curr_deletion) == random_set_num:
                            
#                             print("deletion_rate::", deletion_rate)
                            
                            results[deletion_rate] = results_curr_deletion.copy()
                            results_curr_deletion.clear()
                            
                            state = 0
                        
                        else:
                            state = 1
                        
                    else:
                        state = 2
                else:
                    state = 3
                    
                    
                    
                
                
#                 if len(results_curr_repe) == len(methods) and len(results_all_repetition) == deletion_rates_num and len(results_curr_deletion) == random_set_num:
#                 
#                     state = 0
#                     
#                 else:
#                     
#                     '''next random set'''
#                     
#                     if len(results_curr_repe) == len(methods) and len(results_all_repetition) == deletion_rates_num:
#                         state = 1
#                 
#                     
#                     else:
#                         
#                         '''next repetition'''
#                         if len(results_curr_repe) == len(methods):
#                             
#                             print(curr_method_id, curr_random_set_id, curr_repetition_id)
#                             
#                             state = 2
#                             
#                         else:
#                             '''next method'''
#                             
#                             print(curr_method_id, curr_random_set_id, curr_repetition_id)
#                             
#                             state = 3
                        
                        
                    
                
                
                
                continue
            
        
#         print(curr_repetition_id, curr_random_set_id, deletion_rate)
#         
# #         results_all_repetition[curr_repetition_id] = results_curr_repe.copy()
# #         
# #         results_curr_deletion[curr_random_set_id] = results_all_repetition.copy()
# #     
# #         results[deletion_rate] = results_curr_deletion.copy()
#         
#         print("deletion_rate::", deletion_rate)
#         
#         
#         
#         
#         
#         print(results)
        print(results)
        
        
        write_to_csv_file("../../scripts_general/results", results, output_file_suffix)
        