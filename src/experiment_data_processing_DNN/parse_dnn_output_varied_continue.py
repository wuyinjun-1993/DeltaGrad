'''
Created on Jan 14, 2020

'''

import numpy as np
import csv
import pandas as pd


# file_name = '../../scripts_general/mnist_lr_varied_batch_30_5'
# 
# output_file_suffix = 'lr_30_5'

file_name = '../../scripts_general/lr_continuous_add'
 
output_file_suffix = 'lr_continue_add'

headers = ["varied deletion rate::"]

varied_headers = ["deletion rate::"]

dataset_name_prefix = "dataset name::"


##''rcv1''

# test_sample_sizes = [[677399]]


##'''higgs''''
# test_sample_size = 500000

## mnist
# test_sample_size = 10000

# covtype
# test_sample_size = 58101
test_sample_sizes = [[10000], [58101], [50000]]

# random_set_header = "random set"

random_set_header = "batch size::"

repetition_header = "repetition"

deletion_rates_num = 2

random_set_num = 1



period_label="period::"

init_iter_label="init_iters::"



methods = ["origin", "baseline", "incremental updates"]

method_headers = ["python3 benchmark_exp_lr.py", "baseline::", "period::"]

time_labels = ["training time full::","time_baseline::", "time_provenance::"]

distance_prefix = "tensor("


running_time_label = "time"

distance_label = "distance"

accuracy_label = "accuracy"

test_acc_prefix = "Test Avg. Loss:"

test_acc_keyword = "Accuracy:"

# test_acc_prefix = "RCV1 Test Avg. Accuracy:: "
# 
# test_acc_keyword = test_acc_prefix


total_experiment_number = 3 


def get_dstance_value(distance_prefix, line):
    
    comma_id = line.find(",")
    
    dist = float(line[len(distance_prefix):comma_id])
    
    return dist
    
    
def get_accuracy_value(test_acc_keyword, line):
    
    keyword_id = line.find(test_acc_keyword)
    
    acc = float(line[keyword_id + len(test_acc_keyword):].strip())
    
    return acc
    
    
    
    
def convert2df(label, arrays, noise_rates, approach_lists):
    d = {}
    
    d[label] = np.array(list(noise_rates))
    
    for i in range(arrays.shape[1]):
        d[approach_lists[i]] = arrays[:,i]
        
    df = pd.DataFrame(data=d)
    
    return df



def convert2df_test_acc(label, arrays, noise_rates, approach_lists):
    d = {}
    
    d[label] = np.array(list(noise_rates))
    
    for i in range(arrays.shape[1]):
        d[approach_lists[i]] = arrays[:,i]
#         d[approach_lists[i]] = d[approach_lists[i]].map(lambda x: '%11.8f' % x) 
        
    df = pd.DataFrame(data=d)
    
    for i in range(len(approach_lists)):
        
        df[approach_lists[i]] = df[approach_lists[i]].map(lambda x: '%11.8f' % x)
    
    return df
        
def compute_confidence_interval(mean_values, variance_values, sample_size):
    
    upper_bound = 2.58*np.sqrt(variance_values/sample_size)
    
    lower_bound = 2.58*np.sqrt(variance_values/sample_size)
    
    return lower_bound, upper_bound

def write_to_csv_file(file_name, results, output_file_suffix):
    
    head_line = []
    
    head_line.append('')
    
#     deletion_rate_list = list(results.keys())
    
    
    
#     approach_list = list(results.keys())
    
    dataset_list = list(results.keys())
    
    repetition_list = results[dataset_list[0]].keys()
    
    
#     deletion_rate_num = len(deletion_rate_list)
    
    dataset_num = len(dataset_list)
    
    deletion_rates_num = len(repetition_list)
    
    
    training_time_arr = np.zeros((dataset_num, deletion_rates_num, total_experiment_number))
    
    distance_arr = np.zeros((dataset_num, deletion_rates_num, total_experiment_number))
    
    test_accuracy_arr = np.zeros((dataset_num, deletion_rates_num, total_experiment_number), dtype = np.double)
    
    final_training_time_arr = np.zeros((dataset_num, total_experiment_number))
    
    final_distance_arr = np.zeros((dataset_num, total_experiment_number))
    
    final_test_accuracy_arr = np.zeros((dataset_num, deletion_rates_num, total_experiment_number), dtype = np.double)
    
#     final_test_accuracy_arr_diff = np.zeros((deletion_rate_num, batch_size_num, total_experiment_number - 2), dtype = np.double)
    
    final_test_accuracy_arr_var = np.zeros((dataset_num, total_experiment_number), dtype = np.double)
    
#     abs_arr = np.zeros((noise_rate_num, 3))
#     
#     relative_arr = np.zeros((noise_rate_num, 3))
# 
#     mem_arr = np.zeros((noise_rate_num, 4))
    
#     print(approach_list)

    final_label_list = []

    for i in range(len(dataset_list)):
        
        curr_results = results[dataset_list[i]]
        
#         print("curr result::", deletion_rate_list[i])
#         
#         print(curr_results)
        
#         curr_dataset_list = dataset_list[i]
#         
#         for j in range(len(curr_results.keys())):
#             curr_batch_size = list(curr_results.keys())[j]
#             
#             this_res = curr_results[curr_batch_size]
#             
# #             print(deletion_rate_list[i], curr_batch_size, this_res)
#         
#         
        curr_repetition_set = list(curr_results.keys())

        for k in range(deletion_rates_num):
            
            curr_repetition = curr_repetition_set[k]
            
#                 print(curr_deletion_rate, curr_batch_size, curr_repetition)
            
#                 if curr_deletion_rate == 0.001 and curr_batch_size == 3 and curr_repetition == 2:
#                     print("here")
            
#                 print(this_res[curr_repetition])
            
            for p in range(total_experiment_number):
    
#                     print(this_res[curr_repetition][methods[p]])
    
                label = None
                
                id = 0
    
                if p <= 1:
                    label = methods[p]
                    
                else:
                    label = methods[2]
                    
                    id = p - 2
                
                if len(final_label_list) < total_experiment_number:
                    curr_final_label = label
                    
                    if p >= 2:
#                             print("here")
#                             
#                             print(this_res[curr_repetition][label])
#                             
#                             print(id)
#                             
#                             print(this_res[curr_repetition][label][id])
                        
                        curr_final_label += "_" + str(curr_results[curr_repetition][label][id][period_label]) + "_" + str(curr_results[curr_repetition][label][id][init_iter_label]) 
                    
                    final_label_list.append(label)
                
                print(curr_results[curr_repetition][label][id])
    
                training_time_arr[i][k][p] = curr_results[curr_repetition][label][id][running_time_label]
                
                distance_arr[i][k][p] = curr_results[curr_repetition][label][id][distance_label]
                
                test_accuracy_arr[i][k][p] = curr_results[curr_repetition][label][id][accuracy_label]
        
        
    final_training_time_arr = np.mean(training_time_arr, 1)
    
    final_distance_arr = np.mean(distance_arr, 1)
    
    test_err_arr = 1 - test_accuracy_arr
    
    final_test_accuracy_arr = np.mean(test_accuracy_arr, 1)
    
    final_test_err_arr = np.mean(test_err_arr, 1)
    
    
    
    final_test_accuracy_arr_var = np.mean((test_accuracy_arr - final_test_accuracy_arr.reshape((dataset_num, 1, total_experiment_number)))**2, 1)
    
    
    final_test_err_arr_var = np.mean((test_err_arr - final_test_err_arr.reshape((dataset_num, 1, total_experiment_number)))**2, 1)
    
    
    lower_bound, upper_bound = compute_confidence_interval(final_test_err_arr, final_test_err_arr_var, np.array(test_sample_sizes))
    
#     for i in range(len(methods) - 1):
#  
#         final_test_accuracy_arr_diff[:,:,i] = (final_test_accuracy_arr[:,:,i+1] - final_test_accuracy_arr[:,:,0])
        
        


        
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
    
    convert2df("dataset", final_training_time_arr, dataset_list, final_label_list).to_excel(writer, sheet_name='training time', index=False)
    
    convert2df("dataset", final_distance_arr, dataset_list, final_label_list).to_excel(writer, sheet_name='distance', index=False)
    
    convert2df_test_acc("dataset", final_test_accuracy_arr, dataset_list, final_label_list).to_excel(writer, sheet_name='test accuracy', index=False, float_format='%11.8f')
    
    convert2df("dataset", lower_bound, dataset_list, final_label_list).to_excel(writer, sheet_name='test accuracy CI lower', index=False)
#     for i in range(len(deletion_rate_list)):
#         
#         convert2df("batch_size", final_distance_arr[i], batch_size_list, final_label_list).to_excel(writer, sheet_name='distance dr (' + str(deletion_rate_list[i]) + ")", index=False)
#         convert2df("batch_size", final_test_accuracy_arr[i], batch_size_list, final_label_list).to_excel(writer, sheet_name='test accuracy dr (' + str(deletion_rate_list[i]) + ")", index=False)
#         convert2df("batch_size", final_test_accuracy_arr_var[i], batch_size_list, final_label_list).to_excel(writer, sheet_name='test accuracy var dr (' + str(deletion_rate_list[i]) + ")", index=False)
#         convert2df("batch_size", lower_bound[i], batch_size_list, final_label_list).to_excel(writer, sheet_name='test accuracy CI lower (' + str(deletion_rate_list[i]) + ")", index=False)
#         convert2df("batch_size", upper_bound[i], batch_size_list, final_label_list).to_excel(writer, sheet_name='test accuracy CI upper (' + str(deletion_rate_list[i]) + ")", index=False)
#         
# #         convert2df(final_test_accuracy_arr_diff[i], batch_size_list, methods[1:]).to_excel(writer, sheet_name='test accuracy diff dr (' + str(deletion_rate_list[i]) + ")", index=False)
#         
#         
#     for i in range(len(range(batch_size_num))):
#         convert2df("deletion rate", final_training_time_arr[:,i], deletion_rate_list, final_label_list).to_excel(writer, sheet_name='training time bz (' + str(batch_size_list[i]) + ")", index=False)
#         convert2df("deletion rate", final_distance_arr[:,i], deletion_rate_list, final_label_list).to_excel(writer, sheet_name='distance bz (' + str(batch_size_list[i]) + ")", index=False)
#         convert2df("deletion rate", final_test_accuracy_arr[:,i], deletion_rate_list, final_label_list).to_excel(writer, sheet_name='test accuracy bz (' + str(batch_size_list[i]) + ")", index=False)
#         convert2df("deletion rate", final_test_accuracy_arr_var[:,i], deletion_rate_list, final_label_list).to_excel(writer, sheet_name='test accuracy var bz (' + str(batch_size_list[i]) + ")", index=False)
#         convert2df("deletion rate", lower_bound[:,i], deletion_rate_list, final_label_list).to_excel(writer, sheet_name='test accuracy CI lower (' + str(batch_size_list[i]) + ")", index=False)
#         convert2df("deletion rate", upper_bound[:,i], deletion_rate_list, final_label_list).to_excel(writer, sheet_name='test accuracy CI upper (' + str(batch_size_list[i]) + ")", index=False)
#         convert2df(final_test_accuracy_arr_diff[:,i], deletion_rate_list, methods[1:]).to_excel(writer, sheet_name='test accuracy diff dr (' + str(batch_size_list[i]) + ")", index=False)
        
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
    
        state = 0
        
        random_set_ids = []
        
        results = {}
        results_curr_deletion = {}
        
        results_all_repetition = {}
        
        results_curr_repe = {}
        
        curr_method_id = -1
        
        curr_random_set_id = -1
        
        curr_repetition_id = -1
        
        curr_period=-1
        
        curr_init_iters=-1
    
        recorded_count = 0
    
        while line:
        
            line = fp.readline()
    
#             if line.startswith(varied_headers[0]):
#                 print(line, state)
    
    
            
            new_start = False
            
            
#             for i in range(len(headers)):
#                 
#                 if line.startswith(headers[i]):
# #                     curr_header = headers[i]
#                     new_start = True
#                     
#                     curr_header_id = i
#                   
#                     state = 0
#                     
#                     break
#             
#             if new_start:
#                 continue
        
            
            if state == 0 and line.startswith(dataset_name_prefix):
                
#                 if len(results_curr_deletion) > 0:
#                     results[deletion_rate] = results_curr_deletion.copy()
                
                
                dataset_name = line[len(dataset_name_prefix):].strip()
                
                
                print("dataset name::", dataset_name)
                
                results_curr_deletion.clear()
                
                state = 2
                
                continue
                
#             if state == 1 and line.startswith(random_set_header):
#                 
# #                 if len(results_all_repetition) > 0:
# #                     results_curr_deletion[curr_random_set_id] = results_all_repetition.copy()
#                     
#                 results_all_repetition.clear()
#                 
#                 curr_random_set_id = int(line[len(random_set_header):].strip())
#                 
#                 state = 2
#                 
#                 continue
            
            if state == 2 and line.startswith(repetition_header):
                
#                 if len(results_curr_repe) > 0:
                
                curr_repetition_id = int(line[len(repetition_header):].strip())
                
                state = 3
                
                results_curr_repe.clear()
                
                print("repetition::", curr_repetition_id)
                
                continue

            if state == 3:
                for i in range(len(method_headers)):
                    if line.startswith(method_headers[i]):
                        curr_method_id = i
                        
                        print("methods::", methods[curr_method_id])
                        
                        
                        if curr_method_id == len(method_headers) - 1:
                            curr_period = int(line[len(method_headers[i]):].strip())
                            
                            line = fp.readline()
                            
                            curr_init_iters = int(line[len(init_iter_label):].strip())
                            
                        
                        state = 4
                        
                        break
            
            if state == 4 and line.startswith(time_labels[curr_method_id]):
                running_time = float(line[len(time_labels[curr_method_id]):].strip())
                
                
                if not methods[curr_method_id] in results_curr_repe:
#                 else:
                    results_curr_repe[methods[curr_method_id]] = []
                
                results_curr_repe[methods[curr_method_id]].append({running_time_label:running_time})
                
                if curr_method_id == len(method_headers) - 1:
                    results_curr_repe[methods[curr_method_id]][len(results_curr_repe[methods[curr_method_id]]) - 1][period_label] = curr_period
                    results_curr_repe[methods[curr_method_id]][len(results_curr_repe[methods[curr_method_id]]) - 1][init_iter_label] = curr_init_iters
                
                
                
                
                if curr_method_id == 0:
                    state = 3
                    results_curr_repe[methods[curr_method_id]][len(results_curr_repe[methods[curr_method_id]]) - 1][distance_label] = 0
                    results_curr_repe[methods[curr_method_id]][len(results_curr_repe[methods[curr_method_id]]) - 1][accuracy_label] = 0
                    recorded_count += 1
                    
                else:
                    state = 5
                
                continue
            
            
            if state == 5 and line.startswith(distance_prefix):
                distance = get_dstance_value(distance_prefix, line)
                
                results_curr_repe[methods[curr_method_id]][len(results_curr_repe[methods[curr_method_id]]) - 1][distance_label] = distance
                
                state = 6
                
#                 if curr_method_id == 0:
#                     recorded_count += 1
                
                continue
            
            
            if state == 6 and line.startswith(test_acc_prefix):
                acc = get_accuracy_value(test_acc_keyword, line)
                
                results_curr_repe[methods[curr_method_id]][len(results_curr_repe[methods[curr_method_id]]) - 1][accuracy_label] = acc
                
                recorded_count += 1
                
                '''next deletion rate::'''
#                 if deletion_rate == 0.001 and curr_random_set_id == 2:
#                     print("here")
#                 
#                 
#                 if curr_method_id == 1 and curr_random_set_id == 5 and curr_repetition_id == 5:
#                     print("here")
                    
                    
#                 if len(results_curr_repe) == len(methods):
                if recorded_count >= total_experiment_number:
                    results_curr_deletion[curr_repetition_id] = results_curr_repe.copy()
                    results_curr_repe.clear()
                    
                    recorded_count = 0
                    
                    if len(results_curr_deletion) == deletion_rates_num:
#                         results_curr_deletion[curr_random_set_id] = results_all_repetition.copy()
#                         
#                         results_all_repetition.clear()
#                         
#                         if len(results_curr_deletion) == random_set_num:
                            
#                             print("deletion_rate::", deletion_rate)
                            
                            results[dataset_name] = results_curr_deletion.copy()
                            results_curr_deletion.clear()
                            
                            state = 0
                        
#                         else:
#                             state = 1
                        
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
        