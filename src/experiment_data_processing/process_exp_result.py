'''
Created on Apr 4, 2019

'''
from statistics import *
import numpy as np
import csv
import pandas as pd



# file_name = '../../scripts/logistic_regression_sensitivity_analysis.txt'

file_name = '../../scripts/output_linear_regression_more_features'


noise_rate_prefix = 'noise_rate::'

# noise_rate_prefix = 'new_noise::'

start_standard_lib_label = 'start_standard_lib'

start_iteration_label = 'start_iteration'

start_provenance_label = 'start_provenance'

start_provenance_opt_label = 'start_provenance_opt'

start_influence_function = 'start_influence_function'

# training_accuracy_label = 'training_accuracy::'

training_accuracy_label = 'train_accuracy::'

training_accuracy_label2 = 'training_accuracy::'

test_accuracy_label = 'test_accuracy::'

training_time_label = 'training_time::'

training_time_standard_lib_label = 'training_time_standard_lib::'

training_time_iteration_label = 'training_time_iteration::'

training_time_provenance_label = 'training_time_provenance::'

training_time_influe_label = 'training_time_influence_function::'

absolute_error_label = 'absolute_error::'

# relative_error_label = 'relative_error::'

relative_error_label = 'angle::'

absolute_error_label2 = 'absolute_errors::'

relative_error_label2 = 'relative_errors::'

# relative_error_label2 = 'angle::'

change_data_label = 'change_data_values'

random_err_label = 'random_deletion'

epoch_num_test_label = 'epoch_num_test1'

cut_off_threshold_test_label = '0'

angle_label = 'angle::'


origin_label = 'origin'

standard_lib_label = 'standard_lib'

iteration_label = 'iteration'

provenance_label = 'provenance'

provenance_opt_label = 'provenance_opt'

influence_label = 'influence'


mem_usage_label = 'memory usage::'


new_err_label = 'adding_noisy_data'


def clear_temp_array_list(array_list):
    
    for i in range(len(array_list)):
        array_list[i].clear()


def clear_temp_arrays(training_accuracy, test_accuracy, training_time, absolute_errors, relative_errors):
    training_accuracy.clear()
    test_accuracy.clear()
    training_time.clear()
    absolute_errors.clear()
    relative_errors.clear()


def to_csv_file(file_name, arr, noise_rates, approaches):
    with open(file_name, 'w') as writeFile:
        writer = csv.writer(writeFile)
        
        writer.writerow(approaches)
        
        contents = []
        
        i = 0
        
        for noise_rate in noise_rates:
            contents.append(noise_rate)
            contents.extend(arr[i])
            
            i = i + 1
            
            writer.writerow(contents)
            
            contents.clear()
            
        writeFile.close()
            
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
    
    approach_list = list(results.keys())
    
    noise_rate_list = results[approach_list[0]].keys()
    
    
    approach_num = len(approach_list)
    
    noise_rate_num = len(noise_rate_list)
    
    
    training_time_arr = np.zeros((noise_rate_num, approach_num))
    
    training_accuracy_arr = np.zeros((noise_rate_num, approach_num))
    
    test_accuracy_arr = np.zeros((noise_rate_num, approach_num))
    
    abs_arr = np.zeros((noise_rate_num, 3))
    
    relative_arr = np.zeros((noise_rate_num, 3))

    mem_arr = np.zeros((noise_rate_num, 4))
    
    print(approach_list)
    for i in range(len(approach_list)):
        
        curr_results = results[approach_list[i]]
        
        curr_approach = approach_list[i]
        
        for j in range(len(curr_results.keys())):
            noise_rate = list(curr_results.keys())[j]
            
            this_res = curr_results[noise_rate]
            
            print(approach_list[i], noise_rate, this_res)
        
            training_time_arr[j][i] = this_res[training_time_label]
            
            training_accuracy_arr[j][i] = this_res[training_accuracy_label]
            
            test_accuracy_arr[j][i] = this_res[test_accuracy_label]
            
            if curr_approach == provenance_label:
                abs_arr[j][0] = this_res[absolute_error_label]
                relative_arr[j][0] = this_res[relative_error_label]
                mem_arr[j][0] = this_res[mem_usage_label]
                
            if curr_approach == provenance_opt_label:
                abs_arr[j][1] = this_res[absolute_error_label]
                relative_arr[j][1] = this_res[relative_error_label]
                mem_arr[j][1] = this_res[mem_usage_label]
                
            if curr_approach == influence_label:
                abs_arr[j][2] = this_res[absolute_error_label]
                relative_arr[j][2] = this_res[relative_error_label]
            
            
            if curr_approach == iteration_label:
                mem_arr[j][2] = this_res[mem_usage_label]
                
            if curr_approach == standard_lib_label:
                mem_arr[j][3] = this_res[mem_usage_label]
#     approach_list.insert(0, '')
    
    
    writer = pd.ExcelWriter(file_name + '_' + output_file_suffix + '.xlsx', engine='xlsxwriter')
    
    
    convert2df(training_time_arr, noise_rate_list, approach_list).to_excel(writer, sheet_name='training_time', index=False)
    
    convert2df(training_accuracy_arr, noise_rate_list, approach_list).to_excel(writer, sheet_name='training_accuracy', index=False)
    
    convert2df(test_accuracy_arr, noise_rate_list, approach_list).to_excel(writer, sheet_name='test_accuracy', index=False)
    
    convert2df(abs_arr, noise_rate_list, [provenance_label, provenance_opt_label, influence_label]).to_excel(writer, sheet_name='abs_err', index=False)
    
    convert2df(relative_arr, noise_rate_list, [provenance_label, provenance_opt_label, influence_label]).to_excel(writer, sheet_name='relative_err', index=False)

    convert2df(mem_arr, noise_rate_list, [provenance_label, provenance_opt_label, iteration_label, standard_lib_label]).to_excel(writer, sheet_name='memory_usage', index=False)

    writer.save()
    
#     if is_add_noise:
#         to_csv_file(file_name + '_add_noise_training_time.csv', training_time_arr, noise_rate_list, approach_list)
#         to_csv_file(file_name + '_add_noise_training_accuracy.csv', training_accuracy_arr, noise_rate_list, approach_list)
#         to_csv_file(file_name + '_add_noise_test_accuracy.csv', test_accuracy_arr, noise_rate_list, approach_list)
#         to_csv_file(file_name + '_add_noise_absolute_errs.csv', abs_arr, noise_rate_list, ['', provenance_label, influence_label])
#         to_csv_file(file_name + '_add_noise_relative_errs.csv', relative_arr, noise_rate_list, ['', provenance_label, influence_label])
#                 
#     else:
#                     
#         to_csv_file(file_name + '_change_value_training_time.csv', training_time_arr, noise_rate_list, approach_list)
#         to_csv_file(file_name + '_change_value_training_accuracy.csv', training_accuracy_arr, noise_rate_list, approach_list)
#         to_csv_file(file_name + '_change_value_test_accuracy.csv', test_accuracy_arr, noise_rate_list, approach_list)
#         to_csv_file(file_name + '_change_value_absolute_errs.csv', abs_arr, noise_rate_list, ['', provenance_label, influence_label])
#         to_csv_file(file_name + '_change_value_relative_errs.csv', relative_arr, noise_rate_list, ['', provenance_label, influence_label])



# file_name = '../../scripts/multi_logistic_regression_sensitivity_analysis_heart'

with open(file_name + '.txt1') as fp:  
    
    cnt = 1
    
    state = 0
    
    noise_rate = 0
    
    training_accuracy = []
    
    test_accuracy = []
    
    mem_usage_list = []
    
    training_time = []
    
    absolute_errors = []
    
    relative_errors = []
    
    
    add_noise_results = {}
    
    change_data_value_results = {}
    
    random_results = {}
    
    epoch_num_results = {}
    
    
    all_results = {}
    
    
    curr_error = None
    
    cut_off_thres_results = {}
    
    
    is_change_data_value = False
    
    is_random_flipping = False
    
    is_epoch_num_test = False
    
    is_cut_off_thres = False
    
    state = 0
    
    line = '123'
    
    started = False
    
    while line:
        
        line = fp.readline()
        
#         print(line)
        
#         if line.startswith(change_data_label):
#             y = 0
#             y = y + 1
        
        
        if not started and (line.startswith(new_err_label) or line.startswith(change_data_label) or line.startswith(random_err_label)):
            state = 1
            started = True
            
            if line.startswith(new_err_label):
                curr_error = new_err_label
            
            if line.startswith(change_data_label):
                curr_error = change_data_label
                
            if line.startswith(random_err_label):
                curr_error = random_err_label
            
            
            all_results[curr_error] = {}
            
#             print(state)
#             
#             print(line)
            
            continue
        
        if state == 1 and line.startswith(noise_rate_prefix):
            state = 2
            noise_rate = float(line[len(noise_rate_prefix):])
            
            clear_temp_arrays(training_accuracy, test_accuracy, training_time, absolute_errors, relative_errors)
            
#             print(state)
#             
#             print(line)
            
            continue
        
        
        if state == 2:
            
#             print(line)
            if line.startswith(training_time_label):
                
                training_time.append(float(line[len(training_time_label):].strip()))
                
                continue
            
            if line.startswith(training_accuracy_label) or line.startswith(training_accuracy_label2):
                
                try:
                    training_accuracy.append(float(line[len(training_accuracy_label):].strip()))
                    
                except ValueError:
                    
                    try: 
                        training_accuracy.append(float(line[len(training_accuracy_label2):].strip()))
                    except:
                        
                        try:
                            training_accuracy.append(float(line[len(training_accuracy_label) + 8:line.index(',')].strip()))
                        except:
                            training_accuracy.append(float(line[len(training_accuracy_label2) + 8:line.index(',')].strip()))                
                continue
        
            if line.startswith(test_accuracy_label):
                try:
                    test_accuracy.append(float(line[len(test_accuracy_label):].strip()))
                    
                except ValueError:
                    test_accuracy.append(float(line[len(test_accuracy_label) + 8:line.index(',')].strip()))
                
                continue
        
        
        if state == 2 and line.startswith(start_standard_lib_label):
            
            avg_training_time = mean(training_time)
            
            avg_training_accuracy = mean(training_accuracy)
            
            avg_test_accuracy = mean(test_accuracy)
            
            
            curr_results = {}
            
            curr_results[str(noise_rate)] = {training_time_label: avg_training_time, training_accuracy_label: avg_training_accuracy, test_accuracy_label: avg_test_accuracy}
            
            if origin_label not in all_results[curr_error]:
                all_results[curr_error][origin_label] = curr_results
            else:
                all_results[curr_error][origin_label][str(noise_rate)] = curr_results[str(noise_rate)]
            
            
#             if not is_change_data_value and not is_random_flipping and not is_epoch_num_test and not is_cut_off_thres:
#             
#                 if origin_label not in add_noise_results:
#                     add_noise_results[origin_label] = curr_results
#                     
#                 else:
#                     add_noise_results[origin_label][str(noise_rate)] = curr_results[str(noise_rate)]
#             
#             else:
#                 
#                 if is_change_data_value:
#                 
#                     if origin_label not in change_data_value_results:
#                         change_data_value_results[origin_label] = curr_results
#                         
#                     else:
#                         change_data_value_results[origin_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                         
#                 else:
#                     
#                     if is_random_flipping:
#                     
#                         if origin_label not in random_results:
#                             random_results[origin_label] = curr_results
#                             
#                         else:
#                             random_results[origin_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                             
#                             
#                     else:
#                         
#                         if is_epoch_num_test:
#                             if origin_label not in epoch_num_results:
#                                 epoch_num_results[origin_label] = curr_results
#                             
#                             else:
#                                 epoch_num_results[origin_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                                 
#                         else:
#                             
#                             if is_cut_off_thres:
#                                 if origin_label not in cut_off_thres_results:
#                                     cut_off_thres_results[origin_label] = curr_results
#                                 
#                                 else:
#                                     cut_off_thres_results[origin_label][str(noise_rate)] = curr_results[str(noise_rate)]
                                
                        
                        
                        
                        
                        
                    
            
            state = 3
            
            clear_temp_arrays(training_accuracy, test_accuracy, training_time, absolute_errors, relative_errors)
            continue
        
        if state == 3:
            if line.startswith(training_time_standard_lib_label):
                
                training_time.append(float(line[len(training_time_standard_lib_label):].strip()))
                
                continue
            
            if line.startswith(training_accuracy_label) or line.startswith(training_accuracy_label2):
                
                
                
                try:
                    training_accuracy.append(float(line[len(training_accuracy_label):].strip()))
                    
                except ValueError:
                    
                    try: 
                        training_accuracy.append(float(line[len(training_accuracy_label2):].strip()))
                    except:
                        
                        try:
                            training_accuracy.append(float(line[len(training_accuracy_label) + 8:line.index(',')].strip()))
                        except:
                            training_accuracy.append(float(line[len(training_accuracy_label2) + 8:line.index(',')].strip()))
                
                continue
        
            if line.startswith(test_accuracy_label):
                
                try:
                    test_accuracy.append(float(line[len(test_accuracy_label):].strip()))
                    
                except ValueError:
                    test_accuracy.append(float(line[len(test_accuracy_label) + 8:line.index(',')].strip()))
                
                continue
            
            
            
            if line.startswith(mem_usage_label):
                mem_usage_list.append(float(line[len(mem_usage_label):].strip()))
                
                continue
            
            
        if state == 3 and line.startswith(start_iteration_label):
            avg_training_time = mean(training_time)
            
            avg_training_accuracy = mean(training_accuracy)
            
            avg_test_accuracy = mean(test_accuracy)
            
            avg_mem_usage = mean(mem_usage_list)
            
            curr_results = {}
            
            curr_results[str(noise_rate)] = {mem_usage_label: avg_mem_usage, training_time_label: avg_training_time, training_accuracy_label: avg_training_accuracy, test_accuracy_label: avg_test_accuracy}
            
            if standard_lib_label not in all_results[curr_error]:
                all_results[curr_error][standard_lib_label] = curr_results
            else:
                all_results[curr_error][standard_lib_label][str(noise_rate)] = curr_results[str(noise_rate)]
#             if not is_change_data_value and not is_random_flipping and not is_epoch_num_test and not is_cut_off_thres:
#                 if standard_lib_label not in add_noise_results:
#                     add_noise_results[standard_lib_label] = curr_results
#                     
#                 else:
#                     add_noise_results[standard_lib_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                     
#             else:
#                 
#                 if is_change_data_value:
#                 
#                     if standard_lib_label not in change_data_value_results:
#                         change_data_value_results[standard_lib_label] = curr_results
#                         
#                     else:
#                         change_data_value_results[standard_lib_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                         
#                 else:
#                     
#                     if is_random_flipping:
#                         if standard_lib_label not in random_results:
#                             random_results[standard_lib_label] = curr_results
#                             
#                         else:
#                             random_results[standard_lib_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                             
#                     else:
#                         
#                         if is_epoch_num_test:
#                             if standard_lib_label not in epoch_num_results:
#                                 epoch_num_results[standard_lib_label] = curr_results
#                                 
#                             else:
#                                 epoch_num_results[standard_lib_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                         
#                         else:
#                             if is_cut_off_thres:
#                                 if standard_lib_label not in cut_off_thres_results:
#                                     cut_off_thres_results[standard_lib_label] = curr_results
#                                     
#                                 else:
#                                     cut_off_thres_results[standard_lib_label][str(noise_rate)] = curr_results[str(noise_rate)]
                                
                            
            
            state = 4
            
            clear_temp_array_list([training_accuracy, test_accuracy, training_time, absolute_errors, relative_errors, mem_usage_list])
            continue
            
        
        if state == 4:
            if line.startswith(training_time_iteration_label):
                
                training_time.append(float(line[len(training_time_iteration_label):].strip()))
                
                continue
            
            if line.startswith(training_accuracy_label) or line.startswith(training_accuracy_label2):
                
                
                try:
                    training_accuracy.append(float(line[len(training_accuracy_label):].strip()))
                    
                except ValueError:
                    
                    try: 
                        training_accuracy.append(float(line[len(training_accuracy_label2):].strip()))
                    except:
                        
                        try:
                            training_accuracy.append(float(line[len(training_accuracy_label) + 8:line.index(',')].strip()))
                        except:
                            training_accuracy.append(float(line[len(training_accuracy_label2) + 8:line.index(',')].strip()))                
                continue
        
            if line.startswith(test_accuracy_label):
                try:
                    test_accuracy.append(float(line[len(test_accuracy_label):].strip()))
                except ValueError:
                    test_accuracy.append(float(line[len(test_accuracy_label) + 8:line.index(',')].strip()))
                
                continue
            
            
            if line.startswith(mem_usage_label):
                mem_usage_list.append(float(line[len(mem_usage_label):].strip()))
                
                continue
            
        if state == 4 and line.startswith(start_provenance_label):
               
            avg_training_time = mean(training_time)
            
            avg_training_accuracy = mean(training_accuracy)
            
            avg_test_accuracy = mean(test_accuracy)
            
            avg_mem_usage = mean(mem_usage_list)
            
            curr_results = {}
            
            curr_results[str(noise_rate)] = {mem_usage_label: avg_mem_usage, training_time_label: avg_training_time, training_accuracy_label: avg_training_accuracy, test_accuracy_label: avg_test_accuracy}
            
            if iteration_label not in all_results[curr_error]:
                all_results[curr_error][iteration_label] = curr_results
            else:
                all_results[curr_error][iteration_label][str(noise_rate)] = curr_results[str(noise_rate)]
            
#             if not is_change_data_value and not is_random_flipping and not is_epoch_num_test and not is_cut_off_thres:
#                 if iteration_label not in add_noise_results:
#                     add_noise_results[iteration_label] = curr_results
#                     
#                 else:
#                     add_noise_results[iteration_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                     
#             else:
#                 
#                 if is_change_data_value:
#                 
#                     if iteration_label not in change_data_value_results:
#                         change_data_value_results[iteration_label] = curr_results
#                         
#                     else:
#                         change_data_value_results[iteration_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                 else:
#                     
#                     if is_random_flipping:
#                         if iteration_label not in random_results:
#                             random_results[iteration_label] = curr_results
#                             
#                         else:
#                             random_results[iteration_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                             
#                     else:
#                         
#                         if is_epoch_num_test:
#                             if iteration_label not in epoch_num_results:
#                                 epoch_num_results[iteration_label] = curr_results
#                                 
#                             else:
#                                 epoch_num_results[iteration_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                                 
#                         else:
#                             
#                             if is_cut_off_thres:
#                                 if iteration_label not in cut_off_thres_results:
#                                     cut_off_thres_results[iteration_label] = curr_results
#                                     
#                                 else:
#                                     cut_off_thres_results[iteration_label][str(noise_rate)] = curr_results[str(noise_rate)]
                                
                        
            
            state = 5
            
            clear_temp_array_list([training_accuracy, test_accuracy, training_time, absolute_errors, relative_errors, mem_usage_list])
            continue
        
        if state == 5:
            if line.startswith(training_time_provenance_label):
                
                training_time.append(float(line[len(training_time_provenance_label):].strip()))
                
                continue
            
            if line.startswith(training_accuracy_label) or line.startswith(training_accuracy_label2):
                
                try:
                    training_accuracy.append(float(line[len(training_accuracy_label):].strip()))
                    
                except ValueError:
                    
                    try: 
                        training_accuracy.append(float(line[len(training_accuracy_label2):].strip()))
                    except:
                        
                        try:
                            training_accuracy.append(float(line[len(training_accuracy_label) + 8:line.index(',')].strip()))
                        except:
                            training_accuracy.append(float(line[len(training_accuracy_label2) + 8:line.index(',')].strip()))                
                continue
        
            if line.startswith(test_accuracy_label):
                
                
                try:
                    test_accuracy.append(float(line[len(test_accuracy_label):].strip()))
                    
                except ValueError:
                    test_accuracy.append(float(line[len(test_accuracy_label) + 8:line.index(',')].strip()))
                
                continue
        
        
            if line.startswith(mem_usage_label):
                mem_usage_list.append(float(line[len(mem_usage_label):].strip()))
                
                continue
        
            if line.startswith(absolute_error_label):
                absolute_errors.append(float(line[len(absolute_error_label)+8:line.index(',')]))
                continue
                
            if line.startswith(relative_error_label):
                relative_errors.append(float(line[len(relative_error_label)+8:line.index(',')]))
                continue
            
        if state == 5 and line.startswith(start_provenance_opt_label):
            
#             print(is_cut_off_thres)
#              
#             print(is_epoch_num_test)
#             
#             print(is_change_data_value)
#             
#             print(is_random_flipping)
#             
#             print(noise_rate)
            
#             if is_random_flipping and noise_rate == 1e-5:
#                 y = 1
#                 y += 1
            
            avg_training_time = mean(training_time)
            
            avg_training_accuracy = mean(training_accuracy)
            
            avg_test_accuracy = mean(test_accuracy)
            
            avg_absolute_error = mean(absolute_errors)
            
            avg_relative_error = mean(relative_errors)
            
            avg_mem_usage = mean(mem_usage_list)
            
            curr_results = {}
            
            curr_results[str(noise_rate)] = {mem_usage_label: avg_mem_usage, training_time_label: avg_training_time, training_accuracy_label: avg_training_accuracy, test_accuracy_label: avg_test_accuracy, absolute_error_label: avg_absolute_error, relative_error_label: avg_relative_error}
            
            if provenance_label not in all_results[curr_error]:
                all_results[curr_error][provenance_label] = curr_results
            else:
                all_results[curr_error][provenance_label][str(noise_rate)] = curr_results[str(noise_rate)]
            
#             if not is_change_data_value and not is_random_flipping and not is_cut_off_thres and not is_epoch_num_test:
#                 if provenance_label not in add_noise_results:
#                     add_noise_results[provenance_label] = curr_results
#                     
#                 else:
#                     add_noise_results[provenance_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                     
#             else:
#                 
#                 if is_change_data_value:
#                 
#                     if provenance_label not in change_data_value_results:
#                         change_data_value_results[provenance_label] = curr_results
#                         
#                     else:
#                         change_data_value_results[provenance_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                         
#                 else:
#                     if is_random_flipping:
#                     
#                         if provenance_label not in random_results:
#                             random_results[provenance_label] = curr_results
#                             
#                         else:
#                             random_results[provenance_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                             
#                     else:
#                         
#                         if is_epoch_num_test:
#                             if provenance_label not in epoch_num_results:
#                                 epoch_num_results[provenance_label] = curr_results
#                                 
#                             else:
#                                 epoch_num_results[provenance_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                         
#                         else:
#                             if is_cut_off_thres:
#                                 if provenance_label not in cut_off_thres_results:
#                                     cut_off_thres_results[provenance_label] = curr_results
#                                     
#                                 else:
#                                     cut_off_thres_results[provenance_label][str(noise_rate)] = curr_results[str(noise_rate)]
            
            state = 6
            
            clear_temp_array_list([training_accuracy, test_accuracy, training_time, absolute_errors, relative_errors, mem_usage_list])
            continue
        
        if state == 6:
            if line.startswith(training_time_provenance_label):
                
                training_time.append(float(line[len(training_time_provenance_label):].strip()))
                
                continue
            
            if line.startswith(training_accuracy_label) or line.startswith(training_accuracy_label2):
                
                try:
                    training_accuracy.append(float(line[len(training_accuracy_label):].strip()))
                    
                except ValueError:
                    
                    try: 
                        training_accuracy.append(float(line[len(training_accuracy_label2):].strip()))
                    except:
                        
                        try:
                            training_accuracy.append(float(line[len(training_accuracy_label) + 8:line.index(',')].strip()))
                        except:
                            training_accuracy.append(float(line[len(training_accuracy_label2) + 8:line.index(',')].strip()))                
                continue
        
            if line.startswith(test_accuracy_label):
                
                try:
                    test_accuracy.append(float(line[len(test_accuracy_label):].strip()))
                except ValueError:
                    test_accuracy.append(float(line[len(test_accuracy_label) + 8:line.index(',')].strip()))
                
                continue
        
            if line.startswith(mem_usage_label):
                mem_usage_list.append(float(line[len(mem_usage_label):].strip()))
                
                continue
        
            if line.startswith(absolute_error_label):
                absolute_errors.append(float(line[len(absolute_error_label)+8:+line.index(',')]))
                continue
                
            if line.startswith(relative_error_label):
                relative_errors.append(float(line[len(relative_error_label)+8:line.index(',')]))
                continue
        
        
        
        if state == 6 and line.startswith(start_influence_function):
            
            
            
            
            avg_training_time = mean(training_time)
            
            avg_training_accuracy = mean(training_accuracy)
            
            avg_test_accuracy = mean(test_accuracy)
            
            avg_absolute_error = mean(absolute_errors)
            
            avg_relative_error = mean(relative_errors)
            
            avg_mem_usage = mean(mem_usage_list)
            
            
            curr_results = {}
            
            curr_results[str(noise_rate)] = {mem_usage_label: avg_mem_usage, training_time_label: avg_training_time, training_accuracy_label: avg_training_accuracy, test_accuracy_label: avg_test_accuracy, absolute_error_label: avg_absolute_error, relative_error_label: avg_relative_error}
            
            if provenance_opt_label not in all_results[curr_error]:
                all_results[curr_error][provenance_opt_label] = curr_results
            else:
                all_results[curr_error][provenance_opt_label][str(noise_rate)] = curr_results[str(noise_rate)]
            
#             if not is_change_data_value and not is_random_flipping and not is_cut_off_thres and not is_epoch_num_test:
#                 if provenance_opt_label not in add_noise_results:
#                     add_noise_results[provenance_opt_label] = curr_results
#                     
#                 else:
#                     add_noise_results[provenance_opt_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                     
#             else:
#                 if is_change_data_value:
#                 
#                     if provenance_opt_label not in change_data_value_results:
#                         change_data_value_results[provenance_opt_label] = curr_results
#                         
#                     else:
#                         change_data_value_results[provenance_opt_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                         
#                 else:
#                     
#                     if is_random_flipping:
#                     
#                         if provenance_opt_label not in random_results:
#                             random_results[provenance_opt_label] = curr_results
#                             
#                         else:
#                             random_results[provenance_opt_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                             
#                     else:
#                         if is_cut_off_thres:
#                             if provenance_opt_label not in cut_off_thres_results:
#                                 cut_off_thres_results[provenance_opt_label] = curr_results
#                                 
#                             else:
#                                 cut_off_thres_results[provenance_opt_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                                 
#                         else:
#                             
#                             if is_epoch_num_test:
#                                 if provenance_opt_label not in epoch_num_results:
#                                     epoch_num_results[provenance_opt_label] = curr_results
#                                     
#                                 else:
#                                     epoch_num_results[provenance_opt_label][str(noise_rate)] = curr_results[str(noise_rate)]
                                
                                
                                
                                
            
            state = 7
            
            clear_temp_array_list([training_accuracy, test_accuracy, training_time, absolute_errors, relative_errors, mem_usage_list])
            continue
            
        if state == 7:
            
            
#             print(line)
            
            if line.startswith(training_accuracy_label) or line.startswith(training_accuracy_label2):
                
                
                try:
                    training_accuracy.append(float(line[len(training_accuracy_label):].strip()))
                    
                except ValueError:
                    
                    try: 
                        training_accuracy.append(float(line[len(training_accuracy_label2):].strip()))
                    except:
                        
                        try:
                            training_accuracy.append(float(line[len(training_accuracy_label) + 8:line.index(',')].strip()))
                        except:
                            training_accuracy.append(float(line[len(training_accuracy_label2) + 8:line.index(',')].strip()))                    
                continue
        
            if line.startswith(test_accuracy_label):
                
                try:
                    test_accuracy.append(float(line[len(test_accuracy_label):].strip()))
                except ValueError:
                    test_accuracy.append(float(line[len(test_accuracy_label) + 8:line.index(',')].strip()))
                state = 8
                continue
        
            if line.startswith(absolute_error_label2):
                absolute_errors.append(float(line[len(absolute_error_label2)+8:line.index(',')]))
                continue
                
            if line.startswith(relative_error_label2):
                relative_errors.append(float(line[len(relative_error_label2)+8:line.index(',')]))
                
                
                continue
            
#             try:
            if line.startswith(training_time_influe_label):
                
#                 print(line[len(training_time_influe_label)].strip())
                
                training_time.append(float(line[len(training_time_influe_label):].strip()))
                
                continue
                
            try:
                training_time.append(float(line.strip()))
            except:
                continue
#                 state = 7
#             
#                 continue
#                 
#             except ValueError:
#                 continue
        
        if state == 8:
            print(line)
            
        
        if state == 8 and (line.startswith(noise_rate_prefix) or line.startswith(change_data_label) or line.startswith(random_err_label) or line.startswith(epoch_num_test_label)):
            
            avg_training_time = mean(training_time)
            
            avg_training_accuracy = mean(training_accuracy)
            
            avg_test_accuracy = mean(test_accuracy)
            
            avg_absolute_error = mean(absolute_errors)
            
            avg_relative_error = mean(relative_errors)
            
            curr_results = {}
            
            curr_results[str(noise_rate)] = {training_time_label: avg_training_time, training_accuracy_label: avg_training_accuracy, test_accuracy_label: avg_test_accuracy, absolute_error_label: avg_absolute_error, relative_error_label: avg_relative_error}
            
            if influence_label not in all_results[curr_error]:
                all_results[curr_error][influence_label] = curr_results
            else:
                all_results[curr_error][influence_label][str(noise_rate)] = curr_results[str(noise_rate)]
                
                
            
#             if not is_change_data_value and not is_random_flipping and not is_epoch_num_test and not is_cut_off_thres:
#             
#                 if influence_label not in add_noise_results:
#                     add_noise_results[influence_label] = curr_results
#                     
#                 else:
#                     add_noise_results[influence_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                     
#             else:
#                 
#                 if is_change_data_value:
#                 
#                     if influence_label not in change_data_value_results:
#                         change_data_value_results[influence_label] = curr_results
#                         
#                     else:
#                         change_data_value_results[influence_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                         
#                 else:
#                     
#                     if is_random_flipping:
#                         if influence_label not in random_results:
#                             random_results[influence_label] = curr_results
#                             
#                         else:
#                             random_results[influence_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                             
#                             if noise_rate == 0.25:
#                                     y = 1
#                             
#                     else:
#                         
#                         if is_epoch_num_test:
#                             if influence_label not in epoch_num_results:
#                                 epoch_num_results[influence_label] = curr_results
#                                 
#                             else:
#                                 epoch_num_results[influence_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                                 
#                         else:
#                             
#                             if is_cut_off_thres:
#                                 if influence_label not in cut_off_thres_results:
#                                     cut_off_thres_results[influence_label] = curr_results
#                                     
#                                 else:
#                                     cut_off_thres_results[influence_label][str(noise_rate)] = curr_results[str(noise_rate)]
                                
                                
            state = 2

            
            
            if line.startswith(change_data_label):
                
                curr_error = change_data_label
                all_results[curr_error] = {}
                is_change_data_value = True
                is_random_flipping = False
                is_cut_off_thres = False
                is_epoch_num_test = False
                state = 1
                continue
            
            if line.startswith(random_err_label):
                
                
                curr_error = random_err_label
                all_results[curr_error] = {}
                is_change_data_value = False
                is_random_flipping = True
                is_cut_off_thres = False
                is_epoch_num_test = False
                state = 1
                continue
                
            if line.startswith(cut_off_threshold_test_label):
                
                curr_error = cut_off_threshold_test_label
                all_results[curr_error] = {}
                is_change_data_value = False
                is_random_flipping = False
                is_cut_off_thres = True
                is_epoch_num_test = False
                state = 1
                continue
            
            if line.startswith(epoch_num_test_label):
                curr_error = epoch_num_test_label
                all_results[curr_error] = {}
                is_change_data_value = False
                is_random_flipping = False
                is_cut_off_thres = False
                is_epoch_num_test = True
                state = 1
                continue
            
            if line.startswith(noise_rate_prefix):
                noise_rate = float(line[len(noise_rate_prefix):])
                print('noise_rate::', noise_rate)
                
                if noise_rate == .2:
                    y = 0
                    y += 1
            
            clear_temp_arrays(training_accuracy, test_accuracy, training_time, absolute_errors, relative_errors)
            continue
            
    
    avg_training_time = mean(training_time)
            
    avg_training_accuracy = mean(training_accuracy)
    
    avg_test_accuracy = mean(test_accuracy)
    
    avg_absolute_error = mean(absolute_errors)
    
    avg_relative_error = mean(relative_errors)
    
    curr_results = {}
    
    curr_results[str(noise_rate)] = {training_time_label: avg_training_time, training_accuracy_label: avg_training_accuracy, test_accuracy_label: avg_test_accuracy, absolute_error_label: avg_absolute_error, relative_error_label: avg_relative_error}
    
    
    
    if influence_label not in all_results[curr_error]:
        all_results[curr_error][influence_label] = curr_results
    else:
        all_results[curr_error][influence_label][str(noise_rate)] = curr_results[str(noise_rate)]
    
    
        if noise_rate == .2:
            y = 0
            y += 1
    
#     if not is_change_data_value and not is_random_flipping and not is_epoch_num_test and not is_cut_off_thres:
#     
#         if influence_label not in add_noise_results:
#             add_noise_results[influence_label] = curr_results
#             
#         else:
#             add_noise_results[influence_label][str(noise_rate)] = curr_results[str(noise_rate)]
#             
#     else:
#         
#         if is_change_data_value:
#         
#             if influence_label not in change_data_value_results:
#                 change_data_value_results[influence_label] = curr_results
#                 
#             else:
#                 change_data_value_results[influence_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                 
#         else:
#             
#             if is_random_flipping:
#                 if influence_label not in random_results:
#                     random_results[influence_label] = curr_results
#                     
#                 else:
#                     random_results[influence_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                     
#                     if noise_rate == 0.25:
#                             y = 1
#                     
#             else:
#                 
#                 if is_epoch_num_test:
#                     if influence_label not in epoch_num_results:
#                         epoch_num_results[influence_label] = curr_results
#                         
#                     else:
#                         epoch_num_results[influence_label][str(noise_rate)] = curr_results[str(noise_rate)]
#                         
#                 else:
#                     
#                     if is_cut_off_thres:
#                         if influence_label not in cut_off_thres_results:
#                             cut_off_thres_results[influence_label] = curr_results
#                             
#                         else:
#                             cut_off_thres_results[influence_label][str(noise_rate)] = curr_results[str(noise_rate)]


    
    
    
    print(all_results[random_err_label])
    write_to_csv_file(file_name, all_results[random_err_label], 'random_flipping')
    write_to_csv_file(file_name, all_results[change_data_label], 'change_data_value')
    write_to_csv_file(file_name, all_results[new_err_label], 'adding_noisy_data')
#     write_to_csv_file(file_name, add_noise_results, 'add_noise')
#     
#     write_to_csv_file(file_name, change_data_value_results, 'change_data_value')
#     
#     write_to_csv_file(file_name, random_results, 'random_flipping')
     
#     write_to_csv_file(file_name, cut_off_thres_results, 'cut_off')
     
#     write_to_csv_file(file_name, epoch_num_results, 'epoch_num')
        
            
            
        