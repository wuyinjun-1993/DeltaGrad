'''
Created on Jan 19, 2020

'''
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure
import numpy as np


max_deletion_rates = 0.01


min_deletion_rates = 0.00002

deletion_rate_label = "deletion rate"

batch_size_label = "batch_size"


methods = ["origin", "baseline", "incremental updates", "incremental updates 0"]

baseline_label = "BaseL"

increm_label = "DeltaGrad"

method_labels = ["origin", baseline_label, increm_label, "PrIU"]

# file_name = '../../scripts_general/results_mnist_lr_compare'

# file_name = '../../scripts_general/results_DNN'

# file_names = ['../../scripts_general/results_lr_add_mnist', '../../scripts_general/results_lr_add_cov', '../../scripts_general/results_lr_add_higgs']

file_name = '../../scripts_general/results_lr_mnist'
# file_names = ['../../scripts_general/results_mnist_lr_compare','../../scripts_general/results_covtype_lr_compare']


file_names = [["../../scripts_general/results_lr_continue_deletion","../../scripts_general/results_lr_continue_deletion_rcv1"], ["../../scripts_general/results_lr_continue_add", "../../scripts_general/results_lr_continue_add_rcv1"]]

# del_file_names = ['../../scripts_general/results_lr_mnist', '../../scripts_general/results_lr_cov', '../../scripts_general/results_lr_higgs','../../scripts_general/results_DNN_deletion']
# del_file_names = ['../../scripts_general/results_lr_mnist', '../../scripts_general/results_lr_cov', '../../scripts_general/results_lr_higgs','../../scripts_general/results_lr_rcv1']
del_file_names = ['../../scripts_general/results_lr_add_mnist', '../../scripts_general/results_lr_add_cov', '../../scripts_general/results_lr_add_higgs','../../scripts_general/results_lr_add_rcv1']
# del_file_names = ['../../scripts_general/results_lr_rcv1']
# del_file_names = ['../../scripts_general/results_lr_add_mnist', '../../scripts_general/results_lr_add_cov', '../../scripts_general/results_lr_add_higgs','../../scripts_general/results_DNN_add']

file_names = ['../../scripts_general/results_DNN_deletion', '../../scripts_general/results_DNN_add']
# 
# file_names = ['../../scripts_general/results_lr_rcv1', '../../scripts_general/results_lr_add_rcv1']

# titles = ['MNIST', 'covtype', 'HIGGS', 'MNIST$^n$']
titles = ['MNIST', 'covtype', 'HIGGS', 'RCV1']

batch_sizes = [16384, 4096, 1024]


time_color_map_time={methods[1]: '#fea3aa', methods[2]: '#f8b88b'}#, influence_label: 'b*', standard_lib_label: 'k+', iteration_label: 'ro', origin_label: 'rs', closed_form_label: 'rs', linview_label: 'g*'}

time_color_map_distance={methods[1]: 'b', methods[2]: 'r'}


markers_map_time={methods[1]: '^', methods[2]: '+'}

markers_map_distance={methods[1]: 'o', methods[2]: 'x'}


all_inits = [10,20,30,40,50]

all_periods = [20,10,5,2,1]

all_periods_color_time = ["#FE6B64", "#B29DD9", "#77DD77", "#779ECB"]

all_periods_marker_time = ["o", "^", "D", "*"]

all_periods_color_distance = ["#341A59", "#D31F20", "#2EEA7A", "#BDF054"]

all_periods_marker_distance = ["o", "^", "D", "*"]



def draw_time_figure_by_batch_size_varied_deletion_rate(x_label, excel_name, output_file_name, batch_size, init_iteration, period):
    df = pd.read_excel(excel_name + '.xlsx', sheet_name='training time bz (' + str(batch_size) + ")")
    
    cln_names = list(df.columns)
    
    
    running_time_base_line = df[methods[1]] 
    
    running_time_incremental = df[methods[2] + "_" + str(period) + "_" + str(init_iteration)]
    
    
    deletion_rates = df[deletion_rate_label]
    
    dr_ids = np.nonzero(deletion_rates <= max_deletion_rates)[0]
    
    deletion_rates = deletion_rates[dr_ids]
    
    
    df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='distance bz (' + str(batch_size) + ")")
    
    
    distance_base_line = df2[methods[1]] 
    
    distance_incremental = df2[methods[2] + "_" + str(period) + "_" + str(init_iteration)]

    distance_base_line = distance_base_line[dr_ids]
    
    distance_incremental = distance_incremental[dr_ids]

    
    running_time_base_line = running_time_base_line[dr_ids]
    
    running_time_incremental = running_time_incremental[dr_ids]
    
    lw = 4
    
    mw = 15
    
    
    print("here")
    
    
#     test_acc_std = df[standard_lib_label]
    
#     test_acc_iter = df[iteration_label]
#     
#     test_acc_prov = df[provenance_label]
#     
#     test_acc_prov_opt = df[provenance_opt_label]
#     
# #     test_acc_closed_form = df[closed_form_label]
#     
#     test_acc_inf = df[influence_label]
#     
#     test_acc_closed_form_label = df[closed_form_label]
#     
#     test_acc_linview_label = df[linview_label]
#     
#     lw = 4
#     
#     mw = 15
#     
    fig, ax1 = plt.subplots(figsize=(15,12))
     
#     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
 
     
#     line1, = ax1.plot(noise_rate[subset_ids], test_acc_std[subset_ids], time_color_map[standard_lib_label]+'-', linewidth=lw, markersize=mw)
     
#     line2, = ax1.plot(deletion_rates, running_time_base_line, time_color_map[methods[1]] + '-', linewidth=lw, markersize=mw)
#      
#     line3, = ax1.plot(deletion_rates, running_time_incremental, time_color_map[methods[2]] + '--', linewidth=lw, markersize=mw)
    
    
    line2, = ax1.plot(deletion_rates, running_time_base_line)
      
    line3, = ax1.plot(deletion_rates, running_time_incremental)
    
    plt.setp(line2, color=time_color_map_time[methods[1]], linewidth=lw, marker=markers_map_time[methods[1]], markersize=mw)
    
    plt.setp(line3, color=time_color_map_time[methods[2]], linewidth=lw, marker=markers_map_time[methods[2]], markersize=mw)
    
    
#     line4, = ax1.plot(noise_rate[subset_ids], test_acc_prov_opt[subset_ids], time_color_map[provenance_opt_label] + '-', linewidth=lw, markersize=mw)
#      
#     line5, = ax1.plot(noise_rate[subset_ids], test_acc_inf[subset_ids], time_color_map[influence_label] + '-', linewidth=lw, markersize=mw)
#     line6, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[closed_form_label] + '-', linewidth=lw, markersize=mw)
#     
#     line7, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[linview_label] + '-', linewidth=lw, markersize=mw)
     
#     ax1.legend([line1, line2, line3, line4], [standard_lib_approach + ":" + test_acc, iteration_approach + ":" + test_acc, provenance_approach + ":" + test_acc, influence_approach + ":" + test_acc], loc="best", prop={'size': 15, 'weight':'bold'})
 
#     ax1.legend([line2, line4, line6, line5, line3], [iteration_approach, provenance_opt_approach, closed_form_approach, influence_approach, provenance_approach], bbox_to_anchor=(0.5, 0.6), bbox_transform=plt.gcf().transFigure, ncol=2, prop={'size': 18, 'weight':'bold'})
    
     
    ax1.set_ylabel('running time (second)', fontsize=20, weight = 'bold')
     
    ax1.set_xscale("log")
     
#     ax1.set_yscale("log")
     
    ax1.set_xlabel(x_label, fontsize=20, weight = 'bold')
     
    ax1.tick_params(axis='both', labelsize = 20)
     
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
     
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Euclidean distance', fontsize=20, weight = 'bold')  # we already handled the x-label with ax1
#     line4, = ax2.plot(noise_rate[subset_ids], prov_rel_errs[subset_ids], time_color_map[provenance_label] + '-', linewidth=lw, markersize=mw)
    line5, = ax2.plot(deletion_rates, distance_base_line)
    
    line6, = ax2.plot(deletion_rates, distance_incremental)
    
    
    plt.setp(line5, color=time_color_map_distance[methods[1]], linewidth=lw, marker=markers_map_distance[methods[1]], markersize=mw)
    
    plt.setp(line6, color=time_color_map_distance[methods[2]], linewidth=lw, marker=markers_map_distance[methods[2]], markersize=mw)
    
    
    ax2.tick_params(axis='y', labelsize = 20)
    
    plt.setp(ax2.get_yticklabels(), fontweight="bold")
    ax2.set_yscale("log")
    
#     ax1.legend([line2, line3, line5, line6], ["Running time " + method_labels[1], "Running time " + method_labels[2], "Distance " + method_labels[1], "Distance " + method_labels[2]], bbox_to_anchor=(0.2, 0.8), bbox_transform=plt.gcf().transFigure, ncol=1, prop={'size': 18, 'weight':'bold'})
    
    
     
#     plt.savefig(output_file_name, quality = 50, dpi = 300)
     
     
#     if show_or_not:
    plt.show()


def draw_time_figure_by_batch_size_varied_deletion_rate_DNN(x_label, excel_name, output_file_name, batch_size, init_iteration, period):
    df = pd.read_excel(excel_name + '.xlsx', sheet_name='training time')
    
    cln_names = list(df.columns)
    
    
    running_time_base_line = df[methods[1]] 
    
    running_time_incremental = df[methods[2] + "_" + str(period) + "_" + str(init_iteration)]
    
    
    deletion_rates = df[deletion_rate_label]
    
    dr_ids = np.nonzero(deletion_rates <= max_deletion_rates)[0]
    
    deletion_rates = deletion_rates[dr_ids]
    
    
    df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='distance')
    
    
    distance_base_line = df2[methods[1]] 
    
    distance_incremental = df2[methods[2] + "_" + str(period) + "_" + str(init_iteration)]

    distance_base_line = distance_base_line[dr_ids]
    
    distance_incremental = distance_incremental[dr_ids]

    
    running_time_base_line = running_time_base_line[dr_ids]
    
    running_time_incremental = running_time_incremental[dr_ids]
    
    lw = 4
    
    mw = 15
    
    
    print("here")
    
    
#     test_acc_std = df[standard_lib_label]
    
#     test_acc_iter = df[iteration_label]
#     
#     test_acc_prov = df[provenance_label]
#     
#     test_acc_prov_opt = df[provenance_opt_label]
#     
# #     test_acc_closed_form = df[closed_form_label]
#     
#     test_acc_inf = df[influence_label]
#     
#     test_acc_closed_form_label = df[closed_form_label]
#     
#     test_acc_linview_label = df[linview_label]
#     
#     lw = 4
#     
#     mw = 15
#     
    fig, ax1 = plt.subplots(figsize=(15,8))
     
#     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
 
     
#     line1, = ax1.plot(noise_rate[subset_ids], test_acc_std[subset_ids], time_color_map[standard_lib_label]+'-', linewidth=lw, markersize=mw)
     
#     line2, = ax1.plot(deletion_rates, running_time_base_line, time_color_map[methods[1]] + '-', linewidth=lw, markersize=mw)
#      
#     line3, = ax1.plot(deletion_rates, running_time_incremental, time_color_map[methods[2]] + '--', linewidth=lw, markersize=mw)
    
    
    line2, = ax1.plot(deletion_rates, running_time_base_line, linestyle='dotted', marker='D', linewidth = lw, markersize = mw)
      
    line3, = ax1.plot(deletion_rates, running_time_incremental, linestyle='dotted', marker='*', linewidth = lw, markersize = mw)
    
#     plt.setp(line2, color=time_color_map_time[methods[1]], linewidth=lw, marker=markers_map_time[methods[1]], markersize=mw)
#     
#     plt.setp(line3, color=time_color_map_time[methods[2]], linewidth=lw, marker=markers_map_time[methods[2]], markersize=mw)
    
    
#     line4, = ax1.plot(noise_rate[subset_ids], test_acc_prov_opt[subset_ids], time_color_map[provenance_opt_label] + '-', linewidth=lw, markersize=mw)
#      
#     line5, = ax1.plot(noise_rate[subset_ids], test_acc_inf[subset_ids], time_color_map[influence_label] + '-', linewidth=lw, markersize=mw)
#     line6, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[closed_form_label] + '-', linewidth=lw, markersize=mw)
#     
#     line7, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[linview_label] + '-', linewidth=lw, markersize=mw)
     
#     ax1.legend([line1, line2, line3, line4], [standard_lib_approach + ":" + test_acc, iteration_approach + ":" + test_acc, provenance_approach + ":" + test_acc, influence_approach + ":" + test_acc], loc="best", prop={'size': 15, 'weight':'bold'})
 
#     ax1.legend([line2, line4, line6, line5, line3], [iteration_approach, provenance_opt_approach, closed_form_approach, influence_approach, provenance_approach], bbox_to_anchor=(0.5, 0.6), bbox_transform=plt.gcf().transFigure, ncol=2, prop={'size': 18, 'weight':'bold'})
    
     
    ax1.set_ylabel('running time (second)', fontsize=20, weight = 'bold')
     
    ax1.set_xscale("log")
     
#     ax1.set_yscale("log")
     
    ax1.set_xlabel(x_label, fontsize=20, weight = 'bold')
     
    ax1.tick_params(axis='both', labelsize = 20)
     
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
     
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Euclidean distance', fontsize=20, weight = 'bold')  # we already handled the x-label with ax1
#     line4, = ax2.plot(noise_rate[subset_ids], prov_rel_errs[subset_ids], time_color_map[provenance_label] + '-', linewidth=lw, markersize=mw)
    line5, = ax2.plot(deletion_rates, distance_base_line, marker='D', linewidth = lw, markersize = mw)
    
    line6, = ax2.plot(deletion_rates, distance_incremental, marker='*', linewidth = lw, markersize = mw)
    
    
#     plt.setp(line5, color=time_color_map_distance[methods[1]], linewidth=lw, marker=markers_map_distance[methods[1]], markersize=mw)
#     
#     plt.setp(line6, color=time_color_map_distance[methods[2]], linewidth=lw, marker=markers_map_distance[methods[2]], markersize=mw)
    
    
    ax2.tick_params(axis='y', labelsize = 20)
    
    plt.setp(ax2.get_yticklabels(), fontweight="bold")
    ax2.set_yscale("log")
    
    ax1.legend([line2, line3, line5, line6], ["Running time " + method_labels[1], "Running time " + method_labels[2], "Distance " + method_labels[1], "Distance " + method_labels[2]], bbox_to_anchor=(0.5, 0.5), bbox_transform=plt.gcf().transFigure, ncol=1, prop={'size': 18, 'weight':'bold'})
    
    
     
#     plt.savefig(output_file_name, quality = 50, dpi = 300)
     
     
#     if show_or_not:
    plt.show()



def draw_time_figure_by_batch_size_varied_deletion_rate_for_comparison(x_label, excel_names, output_file_name, batch_size, init_iteration, period, titles):
    
    lw = 4
    
    mw = 15
    
    
    print("here")
    
    
#     fig, ax1 = plt.subplots(figsize=(10,8))
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    
    axes = [ax1, ax2]
    
    twin_axes = []
    
    lw = 3
    
    mw = 8
    
    lines = []
    
    labels = ["Time (" + method_labels[-1] + ")", "Time (" + increm_label + ")", "Dist (" + method_labels[-1] + ")", "Dist (" + increm_label + ")"]
    
    for r in range(len(excel_names)):
    
        title = titles[r]
    
        excel_name = excel_names[r]
        df = pd.read_excel(excel_name + '.xlsx', sheet_name='training time')
    
        cln_names = list(df.columns)
        
        
        running_time_base_line = df[methods[1]] 
        
        running_time_incremental = df[methods[2]]
        
        running_time_priu = df[methods[3]]
        
        
        deletion_rates = df[deletion_rate_label]
        
        
        df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='distance')
        
        
        distance_base_line = df2[methods[1]] 
        
        distance_incremental = df2[methods[2]]
    
        distance_priu = df2[methods[3]]
    
    
    
    
    
     
        line2, = axes[r].plot(deletion_rates, running_time_priu, linestyle='dotted', marker='D',linewidth = lw, markersize = mw)
          
        line3, = axes[r].plot(deletion_rates, running_time_incremental, linestyle='dotted', marker='*',linewidth = lw, markersize = mw)
        
    #     plt.setp(line2, color=time_color_map_time[methods[1]], linewidth=lw, marker=markers_map_time[methods[1]], markersize=mw)
    #     
    #     plt.setp(line3, color=time_color_map_time[methods[2]], linewidth=lw, marker=markers_map_time[methods[2]], markersize=mw)
        
#         axes[r].set_ylabel('Running time (second)', fontsize=20, weight = 'bold')
         
        axes[r].set_xscale("log")
         
#         axes[r].set_xlabel(x_label, fontsize=20, weight = 'bold')
         
        axes[r].tick_params(axis='both', labelsize = 15)
         
        plt.setp(axes[r].get_xticklabels(), fontweight="bold")
         
        plt.setp(axes[r].get_yticklabels(), fontweight="bold")
        
        
        
        ax2 = axes[r].twinx()  # instantiate a second axes that shares the same x-axis
    
#         ax2.set_ylabel('Euclidean distance', fontsize=20, weight = 'bold')  # we already handled the x-label with ax1
    #     line4, = ax2.plot(noise_rate[subset_ids], prov_rel_errs[subset_ids], time_color_map[provenance_label] + '-', linewidth=lw, markersize=mw)
        line5, = ax2.plot(deletion_rates, distance_priu, marker='D',linewidth = lw, markersize = mw)
        
        line6, = ax2.plot(deletion_rates, distance_incremental, marker='*',linewidth = lw, markersize = mw)
        
        if r == 1:
            lines = [line2,line3,line5,line6]
    #     plt.setp(line5, color=time_color_map_distance[methods[1]], linewidth=lw, marker=markers_map_distance[methods[1]], markersize=mw)
    #     
    #     plt.setp(line6, color=time_color_map_distance[methods[2]], linewidth=lw, marker=markers_map_distance[methods[2]], markersize=mw)
        
        
        ax2.tick_params(axis='y', labelsize = 15)
        
        plt.setp(ax2.get_yticklabels(), fontweight="bold")
        ax2.set_yscale("log")
        
        
        
        
#         ax.set_xscale('log')
        axes[r].set_title(title, fontsize = 15,fontweight="bold" )
#         ax.tick_params(axis='both', labelsize = 15)
#         ax_t.tick_params(axis='both', labelsize = 15)
#         plt.setp(ax.get_xticklabels(), fontweight="bold")
#      
#         plt.setp(ax_t.get_yticklabels(), fontweight="bold")
#         
#         plt.setp(ax.get_yticklabels(), fontweight="bold")
        
        
#         ax1.legend([line2, line3, line5, line6], ["Time " + method_labels[3], "Time " + method_labels[2], "Dist " + method_labels[3], "Dist " + method_labels[2]], bbox_to_anchor=(0.4, 0.5), bbox_transform=plt.gcf().transFigure, ncol=1, prop={'size': 18, 'weight':'bold'})


        twin_axes.append(ax2)
    
    
    for sax in twin_axes[1:]:
        twin_axes[0].get_shared_y_axes().join(twin_axes[0], sax)
    twin_axes[0].autoscale()
    for sax in twin_axes[0:-1]:
        sax.yaxis.set_tick_params(labelright=False)
    
#     ax1.set_ylabel('Running time (seconds)', fontsize=20)
    
    fig.text(0.5, 0.005, x_label, ha='center',fontsize=15, fontweight='bold')
    fig.text(0.08, 0.5, 'Running time (seconds)', va='center', rotation='vertical', fontsize=15, fontweight='bold')
    fig.text(0.95, 0.5, 'Euclidean distance', va='center', rotation='vertical', fontsize=15, fontweight='bold')
    
    
#     handles, labels = ax1.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center')
                
    plt.legend(lines, labels, bbox_to_anchor=(0.7, 0.9), bbox_transform=plt.gcf().transFigure, ncol=1, prop={'size': 15, 'weight':'bold'})
    
    
    
    
    
    twin_axes[-1].set_yscale('log')
    
    
    
    plt.show()


def draw_time_figure_by_batch_size(excel_name, output_file_name, deletion_rate, init_iteration, period):
    df = pd.read_excel(excel_name + '.xlsx', sheet_name='training time dr (' + str(deletion_rate) + ")")
    
    cln_names = list(df.columns)
    
    barWidth = 0.1

    
    running_time_base_line = df[methods[1]] 
    
    running_time_incremental = df[methods[2] + "_" + str(period) + "_" + str(init_iteration)]
    
    
    batch_sizes = df[batch_size_label]
    
    
    df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='distance dr (' + str(deletion_rate) + ")")
    
    
    distance_base_line = df2[methods[1]] 
    
    distance_incremental = df2[methods[2] + "_" + str(period) + "_" + str(init_iteration)]

    
    lw = 4
    
    mw = 15
    
    
    print("here")
    
    
#     test_acc_std = df[standard_lib_label]
    
#     test_acc_iter = df[iteration_label]
#     
#     test_acc_prov = df[provenance_label]
#     
#     test_acc_prov_opt = df[provenance_opt_label]
#     
# #     test_acc_closed_form = df[closed_form_label]
#     
#     test_acc_inf = df[influence_label]
#     
#     test_acc_closed_form_label = df[closed_form_label]
#     
#     test_acc_linview_label = df[linview_label]
#     
#     lw = 4
#     
#     mw = 15
#     

    font = {'family' : 'normal',
        'weight' : 'bold'}
#         'size'   : all_font_size}

    plt.rc('font', **font)
    fig, ax1 = plt.subplots(figsize=(15,12))
     
#     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
 
     
#     line1, = ax1.plot(noise_rate[subset_ids], test_acc_std[subset_ids], time_color_map[standard_lib_label]+'-', linewidth=lw, markersize=mw)
     
#     line2, = ax1.plot(deletion_rates, running_time_base_line, time_color_map[methods[1]] + '-', linewidth=lw, markersize=mw)
#      
#     line3, = ax1.plot(deletion_rates, running_time_incremental, time_color_map[methods[2]] + '--', linewidth=lw, markersize=mw)
    
    
    
    
    r1 = np.arange(len(batch_sizes))
    r2 = [x + barWidth for x in r1]
    
#     r1 = [x + barWidth*0.5 for x in r1]
#     r1 = r1 - barWidth*1.0/2
#     r3 = [x + barWidth for x in r2]

    plt.bar(r1, running_time_base_line, color='#7f6d5f', width=barWidth, edgecolor = 'black', label='Running time ' + str(method_labels[1]))
    plt.bar(r2, running_time_incremental, color='#557f2d', width=barWidth, edgecolor = 'black', label='Running time ' + str(method_labels[2]))
#     plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')
 
    plt.xticks([r + barWidth for r in range(len(batch_sizes))], batch_sizes)

    
    
    
#     line2, = ax1.plot(batch_sizes, running_time_base_line)
#       
#     line3, = ax1.plot(batch_sizes, running_time_incremental)
    
#     plt.setp(line2, color=time_color_map_time[methods[1]], linewidth=lw, marker=markers_map_time[methods[1]], markersize=mw)
#     
#     plt.setp(line3, color=time_color_map_time[methods[2]], linewidth=lw, marker=markers_map_time[methods[2]], markersize=mw)
    
    
#     line4, = ax1.plot(noise_rate[subset_ids], test_acc_prov_opt[subset_ids], time_color_map[provenance_opt_label] + '-', linewidth=lw, markersize=mw)
#      
#     line5, = ax1.plot(noise_rate[subset_ids], test_acc_inf[subset_ids], time_color_map[influence_label] + '-', linewidth=lw, markersize=mw)
#     line6, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[closed_form_label] + '-', linewidth=lw, markersize=mw)
#     
#     line7, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[linview_label] + '-', linewidth=lw, markersize=mw)
     
#     ax1.legend([line1, line2, line3, line4], [standard_lib_approach + ":" + test_acc, iteration_approach + ":" + test_acc, provenance_approach + ":" + test_acc, influence_approach + ":" + test_acc], loc="best", prop={'size': 15, 'weight':'bold'})
 
#     ax1.legend([line2, line4, line6, line5, line3], [iteration_approach, provenance_opt_approach, closed_form_approach, influence_approach, provenance_approach], bbox_to_anchor=(0.5, 0.6), bbox_transform=plt.gcf().transFigure, ncol=2, prop={'size': 18, 'weight':'bold'})
    
     
    ax1.set_ylabel('Running time (second)', fontsize=20, weight = 'bold')
     
#     ax1.set_xscale("log")
     
#     ax1.set_yscale("log")
     
    ax1.set_xlabel('Batch size', fontsize=20, weight = 'bold')
     
    ax1.tick_params(axis='both', labelsize = 20)
     
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
     
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# 
    ax2.set_ylabel('Euclidean distance', fontsize=20, weight = 'bold')  # we already handled the x-label with ax1
# #     line4, = ax2.plot(noise_rate[subset_ids], prov_rel_errs[subset_ids], time_color_map[provenance_label] + '-', linewidth=lw, markersize=mw)
    line5, = ax2.plot(distance_base_line, label = "Distance " + str(method_labels[1]))
     
    line6, = ax2.plot(distance_incremental, label = "Distance " + str(method_labels[2]))
#     
#     
    
    
    plt.setp(line5, color=time_color_map_distance[methods[1]], linewidth=lw, marker=markers_map_time[methods[1]], markersize=mw)
     
    plt.setp(line6, color=time_color_map_distance[methods[2]], linewidth=lw, marker=markers_map_time[methods[2]], markersize=mw)
#     
#     
    ax2.tick_params(axis='y', labelsize = 20)
     
    plt.setp(ax2.get_yticklabels(), fontweight="bold")
    ax2.set_yscale("log")
    
    ax2.set_ylim([1e-6,2e-3])
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax1.legend(lines + lines2, labels + labels2, bbox_to_anchor=(0.5, 0.25), bbox_transform=plt.gcf().transFigure, prop={'size': 20}, ncol=1)
    
#     
#     ax1.legend([line5, line6], ["Distance " + methods[1], "Distance " + methods[2]], bbox_to_anchor=(0.8, 0.6), bbox_transform=plt.gcf().transFigure, ncol=1, prop={'size': 18, 'weight':'bold'})
    
    
     
#     plt.savefig(output_file_name, quality = 50, dpi = 300)
     
     
#     if show_or_not:
    plt.show()
    
    
def draw_time_sub_figures_vary_by_batch_size_diff_periods(excel_name, output_file_name, deletion_rate, init_iteration, periods):

    df_list = []
    
    df2_list = []
    
    titles = []
    
    for r in range(len(periods)):
        titles.append("$T_0=$" + str(periods[r]))
    
#     for del_rate in deletion_rates:
    
    df = pd.read_excel(excel_name + '.xlsx', sheet_name='training time dr (' + str(deletion_rate) + ")")
    
    df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='distance dr (' + str(deletion_rate) + ")")

        
#         df2_list.append(df2)
#         
#         df_list.append(df)
    
    
    fig, axes_tup = plt.subplots(1, len(periods), sharex=True, sharey=True)
    
    axes = list(axes_tup)#[ax1, ax2, ax3]
    
    twin_axes = []
    
    lw = 3
    
    mw = 8
    
    lines = []
    
    labels = ["Time (" + baseline_label + ")", "Time (" + increm_label + ")", "Dist (" + baseline_label + ")", "Dist (" + increm_label + ")"]

    
    
#     cln_names = list(df_list[0].columns)
    
    barWidth = 0.2

    
    font = {'family' : 'normal',
        'weight' : 'bold'}
#         'size'   : all_font_size}

    plt.rc('font', **font)
    
    for r in range(len(periods)):
    
    
    
        running_time_base_line = df[methods[1]] 
        
        running_time_incremental = df[methods[2] + "_" + str(periods[r]) + "_" + str(init_iteration)]
        
        
        batch_sizes = df[batch_size_label]
        
        
        distance_base_line = df2[methods[1]] 
        
        distance_incremental = df2[methods[2] + "_" + str(periods[r]) + "_" + str(init_iteration)]
    
        
        lw = 4
        
        mw = 10
        
        
        print("here")
        
        
    #     test_acc_std = df[standard_lib_label]
        
    #     test_acc_iter = df[iteration_label]
    #     
    #     test_acc_prov = df[provenance_label]
    #     
    #     test_acc_prov_opt = df[provenance_opt_label]
    #     
    # #     test_acc_closed_form = df[closed_form_label]
    #     
    #     test_acc_inf = df[influence_label]
    #     
    #     test_acc_closed_form_label = df[closed_form_label]
    #     
    #     test_acc_linview_label = df[linview_label]
    #     
    #     lw = 4
    #     
    #     mw = 15
    #     
    
    
#         fig, ax1 = plt.subplots(figsize=(15,12))
         
    #     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
     
         
    #     line1, = ax1.plot(noise_rate[subset_ids], test_acc_std[subset_ids], time_color_map[standard_lib_label]+'-', linewidth=lw, markersize=mw)
         
    #     line2, = ax1.plot(deletion_rates, running_time_base_line, time_color_map[methods[1]] + '-', linewidth=lw, markersize=mw)
    #      
    #     line3, = ax1.plot(deletion_rates, running_time_incremental, time_color_map[methods[2]] + '--', linewidth=lw, markersize=mw)
        
        
        
        
        r1 = np.arange(len(batch_sizes))
        r2 = [x + barWidth for x in r1]
        
    #     r1 = [x + barWidth*0.5 for x in r1]
    #     r1 = r1 - barWidth*1.0/2
    #     r3 = [x + barWidth for x in r2]
        ax1 = axes[r] 
        ax1.bar(r1, running_time_base_line, color='#799FCB', width=barWidth, edgecolor = 'black', label='Running time ' + str(method_labels[1]))
        ax1.bar(r2, running_time_incremental, color='#F9665E', width=barWidth, edgecolor = 'black', label='Running time ' + str(method_labels[2]))
    #     plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')
     
#         ax1.set_xticks([t + barWidth for t in range(len(batch_sizes))], batch_sizes)
        ax1.set_xticks([p + barWidth for p in range(len(batch_sizes))])
        
        ax1.set_xticklabels(batch_sizes)
        
    #     line2, = ax1.plot(batch_sizes, running_time_base_line)
    #       
    #     line3, = ax1.plot(batch_sizes, running_time_incremental)
        
    #     plt.setp(line2, color=time_color_map_time[methods[1]], linewidth=lw, marker=markers_map_time[methods[1]], markersize=mw)
    #     
    #     plt.setp(line3, color=time_color_map_time[methods[2]], linewidth=lw, marker=markers_map_time[methods[2]], markersize=mw)
        
        
    #     line4, = ax1.plot(noise_rate[subset_ids], test_acc_prov_opt[subset_ids], time_color_map[provenance_opt_label] + '-', linewidth=lw, markersize=mw)
    #      
    #     line5, = ax1.plot(noise_rate[subset_ids], test_acc_inf[subset_ids], time_color_map[influence_label] + '-', linewidth=lw, markersize=mw)
    #     line6, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[closed_form_label] + '-', linewidth=lw, markersize=mw)
    #     
    #     line7, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[linview_label] + '-', linewidth=lw, markersize=mw)
         
    #     ax1.legend([line1, line2, line3, line4], [standard_lib_approach + ":" + test_acc, iteration_approach + ":" + test_acc, provenance_approach + ":" + test_acc, influence_approach + ":" + test_acc], loc="best", prop={'size': 15, 'weight':'bold'})
     
    #     ax1.legend([line2, line4, line6, line5, line3], [iteration_approach, provenance_opt_approach, closed_form_approach, influence_approach, provenance_approach], bbox_to_anchor=(0.5, 0.6), bbox_transform=plt.gcf().transFigure, ncol=2, prop={'size': 18, 'weight':'bold'})
        
         
#         ax1.set_ylabel('Running time (second)', fontsize=20, weight = 'bold')
         
    #     ax1.set_xscale("log")
         
    #     ax1.set_yscale("log")
         
#         ax1.set_xlabel('Batch size', fontsize=20, weight = 'bold')
         
        
        
        ax1.tick_params(axis='both', labelsize = 15)
         
#         plt.setp(ax1.get_xticklabels(), fontweight="bold")
#          
#         plt.setp(ax1.get_yticklabels(), fontweight="bold")
        
        
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # 
    
        twin_axes.append(ax2)
#         ax2.set_ylabel('Euclidean distance', fontsize=20, weight = 'bold')  # we already handled the x-label with ax1
    # #     line4, = ax2.plot(noise_rate[subset_ids], prov_rel_errs[subset_ids], time_color_map[provenance_label] + '-', linewidth=lw, markersize=mw)
        line5, = ax2.plot(distance_base_line, label = "Distance " + str(method_labels[1]), linestyle='dotted', marker='D', linewidth=lw, markersize=mw)
         
        line6, = ax2.plot(distance_incremental, label = "Distance " + str(method_labels[2]), linestyle='solid', marker = '*', linewidth=lw, markersize=mw)
    #     
    #     
        
        
#         plt.setp(line5, color=time_color_map_distance[methods[1]], linewidth=lw, marker=markers_map_time[methods[1]], markersize=mw)
#          
#         plt.setp(line6, color=time_color_map_distance[methods[2]], linewidth=lw, marker=markers_map_time[methods[2]], markersize=mw)
    #     
    #     
#         ax2.tick_params(axis='y', labelsize = 20)
#          
#         plt.setp(ax2.get_yticklabels(), fontweight="bold")
#         ax2.set_yscale("log")
#         
#         ax2.set_ylim([1e-6,2e-3])
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        if r == len(periods) - 1:
            ax1.legend(lines + lines2, labels + labels2, bbox_to_anchor=(0.9, 0.8), bbox_transform=plt.gcf().transFigure, prop={'size': 12}, ncol=1)
        ax1.set_title(titles[r], fontsize = 15,fontweight="bold" )
        ax1.set_ylim([0,20])
        ax1.tick_params(axis='both', labelsize = 15)
        ax2.tick_params(axis='both', labelsize = 15)
        plt.setp(ax1.get_xticklabels(), fontweight="bold")
     
        plt.setp(ax2.get_yticklabels(), fontweight="bold")
        
        plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    for sax in twin_axes[1:]:
        twin_axes[0].get_shared_y_axes().join(twin_axes[0], sax)
#     twin_axes[0].autoscale()
    for sax in twin_axes[0:-1]:
        sax.yaxis.set_tick_params(labelright=False)
    
#     ax1.set_ylabel('Running time (seconds)', fontsize=20)
    
    fig.text(0.5, 0.005, "Batch size", ha='center',fontsize=15, fontweight='bold')
    fig.text(0.065, 0.5, 'Running time (seconds)', va='center', rotation='vertical', fontsize=15, fontweight='bold')
    fig.text(0.94, 0.5, 'Euclidean distance', va='center', rotation='vertical', fontsize=15, fontweight='bold')
    
    
#     handles, labels = ax1.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center')
                
#     plt.legend(lines, labels, bbox_to_anchor=(0.68, 0.9), bbox_transform=plt.gcf().transFigure, ncol=1, prop={'size': 12, 'weight':'bold'})
    
#     twin_axes[2].set_ylabel('Euclidean distance')
    
    twin_axes[-1].set_yscale('log')
#     
#     ax1.legend([line5, line6], ["Distance " + methods[1], "Distance " + methods[2]], bbox_to_anchor=(0.8, 0.6), bbox_transform=plt.gcf().transFigure, ncol=1, prop={'size': 18, 'weight':'bold'})
    
    
     
#     plt.savefig(output_file_name, quality = 50, dpi = 300)
     
     
#     if show_or_not:
    plt.show()


def draw_time_sub_figures_vary_by_batch_size_diff_inits(excel_name, output_file_name, deletion_rate, init_iterations, period):

    df_list = []
    
    df2_list = []
    
    titles = []
    
    for r in range(len(init_iterations)):
        titles.append("$j_0=$" + str(init_iterations[r]))
    
#     for del_rate in deletion_rates:
    
    df = pd.read_excel(excel_name + '.xlsx', sheet_name='training time dr (' + str(deletion_rate) + ")")
    
    df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='distance dr (' + str(deletion_rate) + ")")

        
#         df2_list.append(df2)
#         
#         df_list.append(df)
    
    
    fig, axes_tup = plt.subplots(1, len(init_iterations), sharex=True, sharey=True)
    
    axes = list(axes_tup)#[ax1, ax2, ax3]
    
    twin_axes = []
    
    lw = 3
    
    mw = 8
    
    lines = []
    
    labels = ["Time (" + baseline_label + ")", "Time (" + increm_label + ")", "Dist (" + baseline_label + ")", "Dist (" + increm_label + ")"]

    
    
#     cln_names = list(df_list[0].columns)
    
    barWidth = 0.2

    
    font = {'family' : 'normal',
        'weight' : 'bold'}
#         'size'   : all_font_size}

    plt.rc('font', **font)
    
    for r in range(len(init_iterations)):
    
    
    
        running_time_base_line = df[methods[1]] 
        
        running_time_incremental = df[methods[2] + "_" + str(period) + "_" + str(init_iterations[r])]
        
        
        batch_sizes = df[batch_size_label]
        
        
        distance_base_line = df2[methods[1]] 
        
        distance_incremental = df2[methods[2] + "_" + str(period) + "_" + str(init_iterations[r])]
    
        
        lw = 4
        
        mw = 10
        
        
        print("here")
        
        
    #     test_acc_std = df[standard_lib_label]
        
    #     test_acc_iter = df[iteration_label]
    #     
    #     test_acc_prov = df[provenance_label]
    #     
    #     test_acc_prov_opt = df[provenance_opt_label]
    #     
    # #     test_acc_closed_form = df[closed_form_label]
    #     
    #     test_acc_inf = df[influence_label]
    #     
    #     test_acc_closed_form_label = df[closed_form_label]
    #     
    #     test_acc_linview_label = df[linview_label]
    #     
    #     lw = 4
    #     
    #     mw = 15
    #     
    
    
#         fig, ax1 = plt.subplots(figsize=(15,12))
         
    #     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
     
         
    #     line1, = ax1.plot(noise_rate[subset_ids], test_acc_std[subset_ids], time_color_map[standard_lib_label]+'-', linewidth=lw, markersize=mw)
         
    #     line2, = ax1.plot(deletion_rates, running_time_base_line, time_color_map[methods[1]] + '-', linewidth=lw, markersize=mw)
    #      
    #     line3, = ax1.plot(deletion_rates, running_time_incremental, time_color_map[methods[2]] + '--', linewidth=lw, markersize=mw)
        
        
        
        
        r1 = np.arange(len(batch_sizes))
        r2 = [x + barWidth for x in r1]
        
    #     r1 = [x + barWidth*0.5 for x in r1]
    #     r1 = r1 - barWidth*1.0/2
    #     r3 = [x + barWidth for x in r2]
        ax1 = axes[r] 
        ax1.bar(r1, running_time_base_line, color='#799FCB', width=barWidth, edgecolor = 'black', label='Running time ' + str(method_labels[1]))
        ax1.bar(r2, running_time_incremental, color='#F9665E', width=barWidth, edgecolor = 'black', label='Running time ' + str(method_labels[2]))
    #     plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')
     
#         ax1.set_xticks([t + barWidth for t in range(len(batch_sizes))], batch_sizes)
        ax1.set_xticks([p + barWidth for p in range(len(batch_sizes))])
        
        ax1.set_xticklabels(batch_sizes)
        
    #     line2, = ax1.plot(batch_sizes, running_time_base_line)
    #       
    #     line3, = ax1.plot(batch_sizes, running_time_incremental)
        
    #     plt.setp(line2, color=time_color_map_time[methods[1]], linewidth=lw, marker=markers_map_time[methods[1]], markersize=mw)
    #     
    #     plt.setp(line3, color=time_color_map_time[methods[2]], linewidth=lw, marker=markers_map_time[methods[2]], markersize=mw)
        
        
    #     line4, = ax1.plot(noise_rate[subset_ids], test_acc_prov_opt[subset_ids], time_color_map[provenance_opt_label] + '-', linewidth=lw, markersize=mw)
    #      
    #     line5, = ax1.plot(noise_rate[subset_ids], test_acc_inf[subset_ids], time_color_map[influence_label] + '-', linewidth=lw, markersize=mw)
    #     line6, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[closed_form_label] + '-', linewidth=lw, markersize=mw)
    #     
    #     line7, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[linview_label] + '-', linewidth=lw, markersize=mw)
         
    #     ax1.legend([line1, line2, line3, line4], [standard_lib_approach + ":" + test_acc, iteration_approach + ":" + test_acc, provenance_approach + ":" + test_acc, influence_approach + ":" + test_acc], loc="best", prop={'size': 15, 'weight':'bold'})
     
    #     ax1.legend([line2, line4, line6, line5, line3], [iteration_approach, provenance_opt_approach, closed_form_approach, influence_approach, provenance_approach], bbox_to_anchor=(0.5, 0.6), bbox_transform=plt.gcf().transFigure, ncol=2, prop={'size': 18, 'weight':'bold'})
        
         
#         ax1.set_ylabel('Running time (second)', fontsize=20, weight = 'bold')
         
    #     ax1.set_xscale("log")
         
    #     ax1.set_yscale("log")
         
#         ax1.set_xlabel('Batch size', fontsize=20, weight = 'bold')
         
        
        
        ax1.tick_params(axis='both', labelsize = 15)
         
#         plt.setp(ax1.get_xticklabels(), fontweight="bold")
#          
#         plt.setp(ax1.get_yticklabels(), fontweight="bold")
        
        
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # 
    
        twin_axes.append(ax2)
#         ax2.set_ylabel('Euclidean distance', fontsize=20, weight = 'bold')  # we already handled the x-label with ax1
    # #     line4, = ax2.plot(noise_rate[subset_ids], prov_rel_errs[subset_ids], time_color_map[provenance_label] + '-', linewidth=lw, markersize=mw)
        line5, = ax2.plot(distance_base_line, label = "Distance " + str(method_labels[1]), linestyle='dotted', marker='D', linewidth=lw, markersize=mw)
         
        line6, = ax2.plot(distance_incremental, label = "Distance " + str(method_labels[2]), linestyle='solid', marker = '*', linewidth=lw, markersize=mw)
    #     
    #     
        
        
#         plt.setp(line5, color=time_color_map_distance[methods[1]], linewidth=lw, marker=markers_map_time[methods[1]], markersize=mw)
#          
#         plt.setp(line6, color=time_color_map_distance[methods[2]], linewidth=lw, marker=markers_map_time[methods[2]], markersize=mw)
    #     
    #     
#         ax2.tick_params(axis='y', labelsize = 20)
#          
#         plt.setp(ax2.get_yticklabels(), fontweight="bold")
#         ax2.set_yscale("log")
#         
#         ax2.set_ylim([1e-6,2e-3])
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        if r == len(init_iterations) - 1:
            ax1.legend(lines + lines2, labels + labels2, bbox_to_anchor=(0.9, 0.8), bbox_transform=plt.gcf().transFigure, prop={'size': 12}, ncol=1)
        ax1.set_title(titles[r], fontsize = 15,fontweight="bold" )
        ax1.set_ylim([0,20])
        ax1.tick_params(axis='both', labelsize = 15)
        ax2.tick_params(axis='both', labelsize = 15)
        plt.setp(ax1.get_xticklabels(), fontweight="bold")
     
        plt.setp(ax2.get_yticklabels(), fontweight="bold")
        
        plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    for sax in twin_axes[1:]:
        twin_axes[0].get_shared_y_axes().join(twin_axes[0], sax)
#     twin_axes[0].autoscale()
    for sax in twin_axes[0:-1]:
        sax.yaxis.set_tick_params(labelright=False)
    
#     ax1.set_ylabel('Running time (seconds)', fontsize=20)
    
    fig.text(0.5, 0.005, "Batch size", ha='center',fontsize=15, fontweight='bold')
    fig.text(0.07, 0.5, 'Running time (seconds)', va='center', rotation='vertical', fontsize=15, fontweight='bold')
    fig.text(0.94, 0.5, 'Euclidean distance', va='center', rotation='vertical', fontsize=15, fontweight='bold')
    
    
#     handles, labels = ax1.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center')
                
#     plt.legend(lines, labels, bbox_to_anchor=(0.68, 0.9), bbox_transform=plt.gcf().transFigure, ncol=1, prop={'size': 12, 'weight':'bold'})
    
#     twin_axes[2].set_ylabel('Euclidean distance')
    
    twin_axes[-1].set_yscale('log')
#     
#     ax1.legend([line5, line6], ["Distance " + methods[1], "Distance " + methods[2]], bbox_to_anchor=(0.8, 0.6), bbox_transform=plt.gcf().transFigure, ncol=1, prop={'size': 18, 'weight':'bold'})
    
    
     
#     plt.savefig(output_file_name, quality = 50, dpi = 300)
     
     
#     if show_or_not:
    plt.show()


def draw_time_figure_by_deletion_rate(excel_name, output_file_name, deletion_rate, init_iteration, period):
    df = pd.read_excel(excel_name + '.xlsx', sheet_name='training time dr (' + str(deletion_rate) + ")")
    
    cln_names = list(df.columns)
    
    barWidth = 0.1
    running_time_base_line = df[methods[1]] 
    
    running_time_incremental = df[methods[2] + "_" + str(period) + "_" + str(init_iteration)]
    
    
    batch_sizes = df[batch_size_label]
    
    
    df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='distance dr (' + str(deletion_rate) + ")")
    
    
    distance_base_line = df2[methods[1]] 
    
    distance_incremental = df2[methods[2] + "_" + str(period) + "_" + str(init_iteration)]

    
    lw = 4
    
    mw = 15
    
    
    print("here")
    
    
#     test_acc_std = df[standard_lib_label]
    
#     test_acc_iter = df[iteration_label]
#     
#     test_acc_prov = df[provenance_label]
#     
#     test_acc_prov_opt = df[provenance_opt_label]
#     
# #     test_acc_closed_form = df[closed_form_label]
#     
#     test_acc_inf = df[influence_label]
#     
#     test_acc_closed_form_label = df[closed_form_label]
#     
#     test_acc_linview_label = df[linview_label]
#     
#     lw = 4
#     
#     mw = 15
#     
    fig, ax1 = plt.subplots(figsize=(15,12))
    
    
    r1 = np.arange(len(batch_sizes))
    r2 = [x + barWidth for x in r1]
    
#     r1 = [x + barWidth*0.5 for x in r1]
#     r1 = r1 - barWidth*1.0/2
#     r3 = [x + barWidth for x in r2]

    plt.bar(r1, running_time_base_line, color='#7f6d5f', width=barWidth, edgecolor = 'black', label='Running time ' + str(method_labels[1]))
    plt.bar(r2, running_time_incremental, color='#557f2d', width=barWidth, edgecolor = 'black', label='Running time ' + str(method_labels[2]))
#     plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')
 
    plt.xticks([r + barWidth for r in range(len(batch_sizes))], batch_sizes)

    
#     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
 
     
#     line1, = ax1.plot(noise_rate[subset_ids], test_acc_std[subset_ids], time_color_map[standard_lib_label]+'-', linewidth=lw, markersize=mw)
     
#     line2, = ax1.plot(deletion_rates, running_time_base_line, time_color_map[methods[1]] + '-', linewidth=lw, markersize=mw)
#      
#     line3, = ax1.plot(deletion_rates, running_time_incremental, time_color_map[methods[2]] + '--', linewidth=lw, markersize=mw)
    
    
#     line2, = ax1.plot(batch_sizes, running_time_base_line)
#       
#     line3, = ax1.plot(batch_sizes, running_time_incremental)
#     
#     plt.setp(line2, color=time_color_map_time[methods[1]], linewidth=lw, marker=markers_map_time[methods[1]], markersize=mw)
#     
#     plt.setp(line3, color=time_color_map_time[methods[2]], linewidth=lw, marker=markers_map_time[methods[2]], markersize=mw)
    
    
#     line4, = ax1.plot(noise_rate[subset_ids], test_acc_prov_opt[subset_ids], time_color_map[provenance_opt_label] + '-', linewidth=lw, markersize=mw)
#      
#     line5, = ax1.plot(noise_rate[subset_ids], test_acc_inf[subset_ids], time_color_map[influence_label] + '-', linewidth=lw, markersize=mw)
#     line6, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[closed_form_label] + '-', linewidth=lw, markersize=mw)
#     
#     line7, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[linview_label] + '-', linewidth=lw, markersize=mw)
     
#     ax1.legend([line1, line2, line3, line4], [standard_lib_approach + ":" + test_acc, iteration_approach + ":" + test_acc, provenance_approach + ":" + test_acc, influence_approach + ":" + test_acc], loc="best", prop={'size': 15, 'weight':'bold'})
 
#     ax1.legend([line2, line4, line6, line5, line3], [iteration_approach, provenance_opt_approach, closed_form_approach, influence_approach, provenance_approach], bbox_to_anchor=(0.5, 0.6), bbox_transform=plt.gcf().transFigure, ncol=2, prop={'size': 18, 'weight':'bold'})
    
     
    ax1.set_ylabel('Running time (second)', fontsize=20, weight = 'bold')
     
#     ax1.set_xscale("log")
     
#     ax1.set_yscale("log")
     
    ax1.set_xlabel('Batch size', fontsize=20, weight = 'bold')
     
    ax1.tick_params(axis='both', labelsize = 20)
     
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
     
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Euclidean distance', fontsize=25, weight = 'bold')  # we already handled the x-label with ax1
#     line4, = ax2.plot(noise_rate[subset_ids], prov_rel_errs[subset_ids], time_color_map[provenance_label] + '-', linewidth=lw, markersize=mw)
    line5, = ax2.plot(batch_sizes, distance_base_line)
    
    line6, = ax2.plot(batch_sizes, distance_incremental)
    
    
    plt.setp(line5, color=time_color_map_distance[methods[1]], linewidth=lw, marker=markers_map_time[methods[1]], markersize=mw)
    
    plt.setp(line6, color=time_color_map_distance[methods[2]], linewidth=lw, marker=markers_map_time[methods[2]], markersize=mw)
    
    
    ax2.tick_params(axis='y', labelsize = 20)
    
    plt.setp(ax2.get_yticklabels(), fontweight="bold")
    ax2.set_yscale("log")
    
    ax1.legend([line2, line3, line5, line6], ["Running time " + method_labels[1], "Running time " + method_labels[2], "Distance " + method_labels[1], "Distance " + method_labels[2]], bbox_to_anchor=(0.8, 0.6), bbox_transform=plt.gcf().transFigure, ncol=1, prop={'size': 18, 'weight':'bold'})
    
    
     
#     plt.savefig(output_file_name, quality = 50, dpi = 300)
     
     
#     if show_or_not:
    plt.show()



def draw_time_figure_by_periods(excel_name, output_file_name, batch_size, init_iteration, deletion_rate):
    df = pd.read_excel(excel_name + '.xlsx', sheet_name='training time bz (' + str(batch_size) + ")")
    
    cln_names = list(df.columns)
    
    
    running_time_base_line = df[methods[1]]
    
        
    lw = 4
    
    mw = 15
    
    
    fig, ax1 = plt.subplots(figsize=(15,12))
    
    ax2 = ax1.twinx()
    
    all_speed_ups = []
    
#     all_speed_up_lines = []
    
    deletion_rates = df[deletion_rate_label]
    
    rid = 0
    
    for k in range(len(deletion_rates)):
        if deletion_rates[k] == deletion_rate:
            rid = k
            
            break
    
    
    all_speed_up_legends = []
    
    for i in range(len(all_periods)):
        
        if i < len(all_periods) - 1:
            curr_running_time = df[methods[2] + "_" + str(all_periods[i]) + "_" + str(init_iteration)]
        else:
            curr_running_time = df[methods[2] + "_" + str(all_periods[i]) + "_" + str(10)]
        
        curr_speed_up = running_time_base_line/curr_running_time
        
        all_speed_ups.append(curr_speed_up[rid])
    
        
        
#         all_speed_up_lines.append(line)
    
#         plt.setp(line, color=all_periods_color_time[i], linewidth=lw, marker=all_periods_marker_time[i], markersize=mw)
#      
#         all_speed_up_legends.append("speed-ups, periods = ", all_periods[i])
#     running_time_incremental = df[methods[2] + "_" + str(period) + "_" + str(init_iteration)]
    
    line1, = ax1.plot(all_periods, all_speed_ups)
    
    
    plt.setp(line1, color=all_periods_color_time[0], linewidth=lw, marker=all_periods_marker_time[1], markersize=mw)
    
    all_distances = []
    
    all_distance_lines = []
    
    all_ditance_legends = []
    
    
    df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='distance bz (' + str(batch_size) + ")")
    
    for i in range(len(all_periods)):
        
        if i < len(all_periods) - 1:
            curr_distance = df2[methods[2] + "_" + str(all_periods[i]) + "_" + str(init_iteration)]
        else:
            curr_distance = df2[methods[2] + "_" + str(all_periods[i]) + "_" + str(10)]
        
        all_distances.append(curr_distance[rid])
    
#         line, = ax2.plot(deletion_rates, curr_distance)
#         
#         all_distance_lines.append(line)
#         
#         plt.setp(line, color=all_periods_color_distance[i], linewidth=lw, marker=all_periods_marker_distance[i], markersize=mw)
#     
#         all_ditance_legends.append("Euclidean distance, periods = ", all_periods[i])
    
#     distance_base_line = df2[methods[1]] 
#     
#     distance_incremental = df2[methods[2] + "_" + str(period) + "_" + str(init_iteration)]


    line2, = ax2.plot(all_periods, all_distances)
    
    plt.setp(line2, color=all_periods_color_distance[0], linewidth=lw, marker=all_periods_marker_distance[0], markersize=mw)
    
    print("here")
    
    
#     line2, = ax1.plot(deletion_rates, running_time_base_line)
#       
#     line3, = ax1.plot(deletion_rates, running_time_incremental)
    
    
    
#     plt.setp(line3, color=time_color_map_time[methods[2]], linewidth=lw, marker=markers_map_time[methods[2]], markersize=mw)
     
    ax1.set_ylabel('Speed ups', fontsize=20, weight = 'bold')
     
#     ax1.set_xscale("log")
     
     
    ax1.set_xlabel('periods', fontsize=20, weight = 'bold')
     
    ax1.tick_params(axis='both', labelsize = 20)
     
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
     
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    
    
    # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Euclidean distance', fontsize=20, weight = 'bold')  # we already handled the x-label with ax1
#     line4, = ax2.plot(noise_rate[subset_ids], prov_rel_errs[subset_ids], time_color_map[provenance_label] + '-', linewidth=lw, markersize=mw)
#     line5, = ax2.plot(deletion_rates, distance_base_line)
#     
#     line6, = ax2.plot(deletion_rates, distance_incremental)
#     
#     
#     plt.setp(line5, color=time_color_map_distance[methods[1]], linewidth=lw, marker=markers_map_time[methods[1]], markersize=mw)
#     
#     plt.setp(line6, color=time_color_map_distance[methods[2]], linewidth=lw, marker=markers_map_time[methods[2]], markersize=mw)
    
    
    ax2.tick_params(axis='y', labelsize = 20)
    
    plt.setp(ax2.get_yticklabels(), fontweight="bold")
    ax2.set_yscale("log")
    
    ax1.legend([line1, line2], ["Speed ups", "Euclidean distance"], bbox_to_anchor=(0.8, 0.6), bbox_transform=plt.gcf().transFigure, ncol=1, prop={'size': 18, 'weight':'bold'})
    
    
     
#     plt.savefig(output_file_name, quality = 50, dpi = 300)
     
     
#     if show_or_not:
    plt.show()

def draw_time_figure_by_inits(excel_name, output_file_name, batch_size, period, deletion_rate):
    df = pd.read_excel(excel_name + '.xlsx', sheet_name='training time bz (' + str(batch_size) + ")")
    
    cln_names = list(df.columns)
    
    
    running_time_base_line = df[methods[1]]
    
        
    lw = 4
    
    mw = 15
    
    
    fig, ax1 = plt.subplots(figsize=(10,15))
    
    ax2 = ax1.twinx()
    
    all_speed_ups = []
    
#     all_speed_up_lines = []
    
    deletion_rates = df[deletion_rate_label]
    
    rid = 0
    
    for k in range(len(deletion_rates)):
        if deletion_rates[k] == deletion_rate:
            rid = k
            
            break
    
    
    all_speed_up_legends = []
    
    for i in range(len(all_inits)):
        
#         if i < len(all_periods) - 1:
        curr_running_time = df[methods[2] + "_" + str(period) + "_" + str(all_inits[i])]
#         else:
#             curr_running_time = df[methods[2] + "_" + str(period) + "_" + str(10)]
        
        curr_speed_up = running_time_base_line/curr_running_time
        
        all_speed_ups.append(curr_speed_up[rid])
    
        
        
#         all_speed_up_lines.append(line)
    
#         plt.setp(line, color=all_periods_color_time[i], linewidth=lw, marker=all_periods_marker_time[i], markersize=mw)
#      
#         all_speed_up_legends.append("speed-ups, periods = ", all_periods[i])
#     running_time_incremental = df[methods[2] + "_" + str(period) + "_" + str(init_iteration)]
    
    line1, = ax1.plot(all_inits, all_speed_ups)
    
    
    plt.setp(line1, color=all_periods_color_time[0], linewidth=lw, marker=all_periods_marker_time[1], markersize=mw)
    
    all_distances = []
    
    all_distance_lines = []
    
    all_ditance_legends = []
    
    
    df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='distance bz (' + str(batch_size) + ")")
    
    for i in range(len(all_periods)):
        
#         if i < len(all_periods) - 1:
        curr_distance = df2[methods[2] + "_" + str(period) + "_" + str(all_inits[i])]
#         else:
#             curr_distance = df2[methods[2] + "_" + str(all_periods[i]) + "_" + str(10)]
        
        all_distances.append(curr_distance[rid])
    
#         line, = ax2.plot(deletion_rates, curr_distance)
#         
#         all_distance_lines.append(line)
#         
#         plt.setp(line, color=all_periods_color_distance[i], linewidth=lw, marker=all_periods_marker_distance[i], markersize=mw)
#     
#         all_ditance_legends.append("Euclidean distance, periods = ", all_periods[i])
    
#     distance_base_line = df2[methods[1]] 
#     
#     distance_incremental = df2[methods[2] + "_" + str(period) + "_" + str(init_iteration)]

    all_distances = np.array(all_distances) 
    
    all_distances = all_distances/(1e-5)
    
    
    line2, = ax2.plot(all_inits, all_distances)
    
    plt.setp(line2, color=all_periods_color_distance[0], linewidth=lw, marker=all_periods_marker_distance[0], markersize=mw)
    
    print("here")
    
    
#     line2, = ax1.plot(deletion_rates, running_time_base_line)
#       
#     line3, = ax1.plot(deletion_rates, running_time_incremental)
    
    
    
#     plt.setp(line3, color=time_color_map_time[methods[2]], linewidth=lw, marker=markers_map_time[methods[2]], markersize=mw)
     
    ax1.set_ylabel('Speed ups', fontsize=20, weight = 'bold')
     
#     ax1.set_xscale("log")
     
     
    ax1.set_xlabel('Initial periods', fontsize=20, weight = 'bold')
     
    ax1.tick_params(axis='both', labelsize = 20)
     
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
     
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    
    
    # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Euclidean distance (x$10^{-5}$)', fontsize=20, weight = 'bold')  # we already handled the x-label with ax1
#     line4, = ax2.plot(noise_rate[subset_ids], prov_rel_errs[subset_ids], time_color_map[provenance_label] + '-', linewidth=lw, markersize=mw)
#     line5, = ax2.plot(deletion_rates, distance_base_line)
#     
#     line6, = ax2.plot(deletion_rates, distance_incremental)
#     
#     
#     plt.setp(line5, color=time_color_map_distance[methods[1]], linewidth=lw, marker=markers_map_time[methods[1]], markersize=mw)
#     
#     plt.setp(line6, color=time_color_map_distance[methods[2]], linewidth=lw, marker=markers_map_time[methods[2]], markersize=mw)
    
    
    ax2.tick_params(axis='y', labelsize = 20)
    
    plt.setp(ax2.get_yticklabels(), fontweight="bold")
#     ax2.set_yscale("log")
    
    ax1.legend([line1, line2], ["Speed ups", "Euclidean distance"], bbox_to_anchor=(0.9, 0.9), bbox_transform=plt.gcf().transFigure, ncol=1, prop={'size': 18, 'weight':'bold'})
    
    
     
#     plt.savefig(output_file_name, quality = 50, dpi = 300)
     
     
#     if show_or_not:
    plt.show()


def draw_test_acc_error_bars_by_batch_size(x_label, excel_name, output_file_name, batch_size, inits, period):
    
    df = pd.read_excel(excel_name + '.xlsx', sheet_name='test accuracy bz (' + str(batch_size) + ")")
    
    cln_names = list(df.columns)
    
    
    
    
    acc_avg_base_line = df[methods[1]]
    
    deletion_rates = df[deletion_rate_label]
    
    selected_ids = np.nonzero(deletion_rates <= max_deletion_rates)[0]
    
#     print(selected_ids)
    
    deletion_rates = deletion_rates[selected_ids]
    
    ind = np.arange(len(deletion_rates))
    
    acc_avg_incre = df[methods[2] + "_" + str(period) + "_" + str(inits)]
    
    
    df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='test accuracy CI lower (' + str(batch_size) + ")")
    
    acc_err_base_line = df2[methods[1]]
    
    acc_err_incre = df2[methods[2] + "_" + str(period) + "_" + str(inits)]
    
    
    width = 0.35
    
    
    fig, ax = plt.subplots(figsize=(10,6))
    rects1 = ax.bar(ind - width/2, acc_avg_base_line[selected_ids], width, yerr=acc_err_base_line[selected_ids],
                    label=method_labels[1])
    rects2 = ax.bar(ind + width/2, acc_avg_incre[selected_ids], width, yerr=acc_err_incre[selected_ids],
                    label=method_labels[2])
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Test accuracy', fontsize=20, weight = 'bold')
    ax.set_xlabel(x_label, fontsize=20, weight = 'bold')
#     ax.set_title('Scores by group and gender')
    ax.set_xticks(ind)
    ax.set_xticklabels(deletion_rates)
    plt.setp(ax.get_xticklabels(), fontweight="bold", fontsize = 20)
    
    plt.setp(ax.get_yticklabels(), fontweight="bold", fontsize = 20)
    ax.legend()
    ax.set_ylim(0.528,0.531)
    
    plt.legend(prop={'size': 20, 'weight': 'bold'})

    
    plt.show()
    
    
def draw_test_acc_error_bars_by_batch_size_DNN(x_label, excel_name, output_file_name, batch_size, inits, period):
    
    df = pd.read_excel(excel_name + '.xlsx', sheet_name='test accuracy')
    
    cln_names = list(df.columns)
    
    
    
    
    acc_avg_base_line = df[methods[1]]
    
    deletion_rates = df[deletion_rate_label]
    
    selected_ids = np.nonzero(deletion_rates <= max_deletion_rates)[0]
    
#     print(selected_ids)
    
    deletion_rates = deletion_rates[selected_ids]
    
    ind = np.arange(len(deletion_rates))
    
    acc_avg_incre = df[methods[2]]
    
    
    df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='test accuracy CI lower')
    
    acc_err_base_line = df2[methods[1]]
    
    acc_err_incre = df2[methods[2]]
    
    
    width = 0.35
    
    
    fig, ax = plt.subplots(figsize=(10,6))
    rects1 = ax.bar(ind - width/2, acc_avg_base_line[selected_ids], width, yerr=acc_err_base_line[selected_ids],
                    label=method_labels[1])
    rects2 = ax.bar(ind + width/2, acc_avg_incre[selected_ids], width, yerr=acc_err_incre[selected_ids],
                    label=method_labels[2])
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Test accuracy', fontsize=20, weight = 'bold')
    ax.set_xlabel(x_label, fontsize=20, weight = 'bold')
#     ax.set_title('Scores by group and gender')
    ax.set_xticks(ind)
    ax.set_xticklabels(deletion_rates)
    plt.setp(ax.get_xticklabels(), fontweight="bold", fontsize = 20)
    
    plt.setp(ax.get_yticklabels(), fontweight="bold", fontsize = 20)
    ax.legend()
    ax.set_ylim(0.924,0.927)
    
    plt.legend(prop={'size': 20, 'weight': 'bold'})

    
    plt.show()




# def plot_sub_figures_error_bar_by_deletion_rate(excel_names, batch_size, init_iterations, periods, titles):
#     
#     for excel_name in excel_names:
#         
#     

def plot_sub_figures_time_distance_by_deletion_rate(x_label, excel_names, batch_sizes, init_iterations, periods, titles):
    
    fig, axes_tup = plt.subplots(1, len(excel_names), sharex=True, sharey=True)
    
    axes = list(axes_tup)#[ax1, ax2, ax3]
    
    twin_axes = []
    
    lw = 3
    
    mw = 8
    
    lines = []
    
    labels = ["Time (" + baseline_label + ")", "Time (" + increm_label + ")", r"||$\bf{W^{U*}-W^*}$||", r"||$\bf{W^{U*}-W^{I*}}$||"]
    
    for r in range(len(excel_names)):
    
        period = periods[r]
        
        init_iteration = init_iterations[r]
        
        excel_name = excel_names[r]
        
        ax = axes[r]
        
#         if r < len(excel_names) - 1:
        df = pd.read_excel(excel_name + '.xlsx', sheet_name='training time bz (' + str(batch_sizes[r]) + ")")
            
#         else:
#             df = pd.read_excel(excel_name + '.xlsx', sheet_name='training time')
        
        cln_names = list(df.columns)
        
        
        running_time_base_line = df[methods[1]] 
        
        running_time_incremental = df[methods[2] + "_" + str(period) + "_" + str(init_iteration)]
        
        
        deletion_rates = df[deletion_rate_label]
        
        dr_ids = np.nonzero(deletion_rates <= max_deletion_rates)[0]
        
        deletion_rates = deletion_rates[dr_ids]
        
#         if r < len(excel_names) - 1:
        df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='distance bz (' + str(batch_sizes[r]) + ")")
#         else:
#             df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='distance')
        
        
        distance_base_line = df2[methods[1]] 
        
        distance_incremental = df2[methods[2] + "_" + str(period) + "_" + str(init_iteration)]
    
        distance_base_line = distance_base_line[dr_ids]
        
        distance_incremental = distance_incremental[dr_ids]
    
        
        running_time_base_line = running_time_base_line[dr_ids]
        
        running_time_incremental = running_time_incremental[dr_ids]
    
    
    
        l1, = ax.plot(deletion_rates, running_time_base_line, linestyle='dotted', marker='D', linewidth = lw, markersize = mw)
        
        l2, = ax.plot(deletion_rates, running_time_incremental, linestyle='dotted', marker = '*', linewidth = lw, markersize = mw)
        
        ax_t = ax.twinx()
        
        twin_axes.append(ax_t)
        
        l3, = ax_t.plot(deletion_rates, distance_base_line, marker = 'D', linewidth = lw, markersize = mw)
        
        l4, = ax_t.plot(deletion_rates, distance_incremental, marker = '*', linewidth = lw, markersize = mw)
        
        title = titles[r]
        
        if r == 1:
            lines = [l1,l2,l3,l4]
            
            
    
#         ax.set_xlabel(x_label)

        ax.set_xscale('log')
        
        ax.set_yscale('log')
        
        if r == len(excel_names) - 1:
            
            ax.set_xlim([5e-5,1e-2])
        else:
            ax.set_xlim([2e-5,1e-2])
        
        
        
        
        ax.set_title(title, fontsize = 15,fontweight="bold" )
        ax.tick_params(axis='both', labelsize = 15)
        ax_t.tick_params(axis='both', labelsize = 15)
        plt.setp(ax.get_xticklabels(), fontweight="bold")
     
        plt.setp(ax_t.get_yticklabels(), fontweight="bold")
        
        plt.setp(ax.get_yticklabels(), fontweight="bold")
    
    for sax in twin_axes[1:]:
        twin_axes[0].get_shared_y_axes().join(twin_axes[0], sax)
#     twin_axes[0].autoscale()
    for sax in twin_axes[0:-1]:
        sax.yaxis.set_tick_params(labelright=False)
    
#     ax1.set_ylabel('Running time (seconds)', fontsize=20)
    
    fig.text(0.47, 0.001, x_label, ha='center',fontsize=15, fontweight='bold')
    fig.text(0.07, 0.5, 'Running time (seconds)', va='center', rotation='vertical', fontsize=15, fontweight='bold')
    fig.text(0.94, 0.5, 'Distance', va='center', rotation='vertical', fontsize=15, fontweight='bold')
    
    
#     handles, labels = ax1.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center')
                
    plt.legend(lines, labels, bbox_to_anchor=(0.68, 0.9), bbox_transform=plt.gcf().transFigure, ncol=1, prop={'size': 15, 'weight':'bold'})
    
#     twin_axes[2].set_ylabel('Euclidean distance')
    
    twin_axes[2].set_yscale('log')
    
    
#     ax1.set_xlabel(x_label, fontsize=20, weight = 'bold')
     
#     fig.legend(lines,     # The line objects
#            labels=labels,   # The labels for each line
#            loc="best",   # Position of legend
#            borderaxespad=0.1,    # Small spacing around legend box
#            title="Legend Title"  # Title for the legend
#            )

     
    
    
    
    plt.show()
    
def plot_sub_figures_time_distance_by_deletion_rate_DNN(x_labels, excel_names, batch_sizes, init_iterations, periods, titles):
    
    fig, axes_tup = plt.subplots(1, len(excel_names), sharex=True, sharey=True)
    
    axes = list(axes_tup)#[ax1, ax2, ax3]
    
    twin_axes = []
    
    lw = 3
    
    mw = 8
    
    lines = []
    
#     labels = ["Time (" + baseline_label + ")", "Time (" + increm_label + ")", "Dist (" + baseline_label + ")", "Dist (" + increm_label + ")"]
    labels = ["Time (" + baseline_label + ")", "Time (" + increm_label + ")", r"||$\bf{W^{U*}-W^*}$||", r"||$\bf{W^{U*}-W^{I*}}$||"]

    
    for r in range(len(excel_names)):
    
        period = periods[r]
        
        init_iteration = init_iterations[r]
        
        excel_name = excel_names[r]
        
        ax = axes[r]
        
#         df = pd.read_excel(excel_name + '.xlsx', sheet_name='training time bz (' + str(batch_sizes[r]) + ")")
        df = pd.read_excel(excel_name + '.xlsx', sheet_name='training time')
        
        cln_names = list(df.columns)
        
        
        running_time_base_line = df[methods[1]] 
        
        running_time_incremental = df[methods[2] + "_" + str(period) + "_" + str(init_iteration)]
        
        
        deletion_rates = df[deletion_rate_label]
        
        dr_ids = np.nonzero(deletion_rates > min_deletion_rates)[0]
        
        deletion_rates = deletion_rates[dr_ids]
        
        
#         df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='distance bz (' + str(batch_sizes[r]) + ")")
        df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='distance')
        
        
        distance_base_line = df2[methods[1]] 
        
        distance_incremental = df2[methods[2] + "_" + str(period) + "_" + str(init_iteration)]
    
        distance_base_line = distance_base_line[dr_ids]
        
        distance_incremental = distance_incremental[dr_ids]
    
        
        running_time_base_line = running_time_base_line[dr_ids]
        
        running_time_incremental = running_time_incremental[dr_ids]
    
    
    
        l1, = ax.plot(deletion_rates, running_time_base_line, linestyle='dotted', marker='D', linewidth = lw, markersize = mw)
        
        l2, = ax.plot(deletion_rates, running_time_incremental, linestyle='dotted', marker = '*', linewidth = lw, markersize = mw)
        
        ax_t = ax.twinx()
        
        twin_axes.append(ax_t)
        
        l3, = ax_t.plot(deletion_rates, distance_base_line, marker = 'D', linewidth = lw, markersize = mw)
        
        l4, = ax_t.plot(deletion_rates, distance_incremental, marker = '*', linewidth = lw, markersize = mw)
        
        title = titles[r]
        
        if r == 1:
            lines = [l1,l2,l3,l4]
    
#         ax.set_xlabel(x_label)
        ax.set_xlabel(x_labels[r], fontsize=18,fontweight="bold")
        ax.xaxis.set_label_coords(0.5, -0.08)
        ax.set_xscale('log')
        ax.set_title(title, fontsize = 18,fontweight="bold")
        ax.tick_params(axis='both', labelsize = 18)
        ax_t.tick_params(axis='both', labelsize = 18)
        plt.setp(ax.get_xticklabels(), fontweight="bold")
     
        plt.setp(ax_t.get_yticklabels(), fontweight="bold")
        
        plt.setp(ax.get_yticklabels(), fontweight="bold")
    
    for sax in twin_axes[1:]:
        twin_axes[0].get_shared_y_axes().join(twin_axes[0], sax)
    twin_axes[0].autoscale()
    for sax in twin_axes[0:-1]:
        sax.yaxis.set_tick_params(labelright=False)
    
#     ax1.set_ylabel('Running time (seconds)', fontsize=20)
    
#     fig.text(0.5, 0.005, x_label, ha='center',fontsize=15, fontweight='bold')
    fig.text(0.04, 0.5, 'Running time (seconds)', va='center', rotation='vertical', fontsize=18, fontweight='bold')
    fig.text(0.98, 0.5, 'Distance', va='center', rotation='vertical', fontsize=18, fontweight='bold')
    
    
#     handles, labels = ax1.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center')
                
    plt.legend(lines, labels, bbox_to_anchor=(0.55, 0.6), bbox_transform=plt.gcf().transFigure, ncol=1, prop={'size': 16, 'weight':'bold'})
    
#     twin_axes[2].set_ylabel('Euclidean distance')
    
    twin_axes[-1].set_yscale('log')
    
    
#     ax1.set_xlabel(x_label, fontsize=20, weight = 'bold')
     
#     fig.legend(lines,     # The line objects
#            labels=labels,   # The labels for each line
#            loc="best",   # Position of legend
#            borderaxespad=0.1,    # Small spacing around legend box
#            title="Legend Title"  # Title for the legend
#            )

     
    
    
    
    plt.show()


def draw_continuous_deletion_addition(excel_names, titles, dataset_labels):
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}

    all_dataset_list = []

#     for i in range(2):

    df_list = []

    for i in range(len(excel_names)):
        
        curr_df = None
        
        for j in range(len(excel_names[i])):
        
            excel_name = excel_names[i][j]
        
            print(excel_name)
            
            df = pd.read_excel(excel_name + '.xlsx', sheet_name='training time')
            
            if curr_df is None:
                
                curr_df = df
            else:
                curr_df = pd.concat([curr_df, df])
            
        df_list.append(curr_df)
#             dataset_list = list(df['dataset'])
#             all_dataset_list.extend(dataset_list)

    plt.rc('font', **font)
    
    
    fig, axes_tup = plt.subplots(1, len(excel_names), sharex=True, sharey=True)
    
    axes = list(axes_tup)
    
    twin_axes = []
    
    lw = 3
    
    mw = 8
    
    lines = []
    
    labels = ["Time (" + baseline_label + ")", "Time (" + increm_label + ")"]
    
    barWidth = 0.3
    
    for r in range(len(df_list)):
#         excel_name = excel_names[r]
        
#         df = pd.read_excel(excel_name + '.xlsx', sheet_name='training time')
        df = df_list[r]
        
        cln_names = list(df.columns)
        
        dataset_list = list(df['dataset'])
        print(dataset_list)
        
        running_time_base_line = df[methods[1]] 
        
        running_time_incremental = df[methods[2]]
        
        r1 = np.arange(len(dataset_list))
        r2 = [x + barWidth for x in r1]
        
    #     r1 = [x + barWidth*0.5 for x in r1]
    #     r1 = r1 - barWidth*1.0/2
    #     r3 = [x + barWidth for x in r2]
    
        ax = axes[r]
    
        ax.set_title(titles[r], fontsize = 15,fontweight="bold" )
    
        br1 = ax.bar(r1, running_time_base_line, color='#799FCB', width=barWidth, edgecolor = 'black', label='Running time ' + str(method_labels[1]), hatch = '//')
        br2 = ax.bar(r2, running_time_incremental, color='#F9665E', width=barWidth, edgecolor = 'black', label='Running time ' + str(method_labels[2]), hatch = '*')
        
        if r == 0:
            lines.append(br1)
            
            lines.append(br2)
        
    #     plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')
     
        ax.set_xticks([p + barWidth for p in range(len(dataset_list))])
        ax.set_xticklabels(dataset_labels)
        ax.set_yscale('log')
#     for sax in twin_axes[1:]:
#         twin_axes[0].get_shared_y_axes().join(twin_axes[0], sax)
#     twin_axes[0].autoscale()
#     for sax in twin_axes:
#         sax.yaxis.set_tick_params(labelright=False)
    
#     ax1.set_ylabel('Running time (seconds)', fontsize=20)
    
#     fig.text(0.5, 0.005, x_label, ha='center',fontsize=15, fontweight='bold')
    fig.text(0.04, 0.5, 'Running time (seconds)', va='center', rotation='vertical', fontsize=15, fontweight='bold')
#     fig.text(0.95, 0.5, 'Euclidean distance', va='center', rotation='vertical', fontsize=15, fontweight='bold')
    
    
#     handles, labels = ax1.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center')
                
    plt.legend(lines, labels, bbox_to_anchor=(0.8, 0.9), bbox_transform=plt.gcf().transFigure, ncol=1, prop={'size': 15, 'weight':'bold'})
    
    plt.show()


if __name__ == '__main__':
    
    
    period = 5
    
    inits = 300
    
#     init_iters = [10, 10, 300, 50]
#     
#     periods = [5, 5, 3, 2]
    
    init_iters = [10, 10, 300, 10]
    
    periods = [5, 5, 3, 10]
    
    batch_sizes = [16384, 16384, 16384, 10200]
    
#     draw_time_sub_figures_vary_by_batch_size_diff_periods(file_name, None, 0.00002, 10, [20, 10,5])
    
#     draw_time_sub_figures_vary_by_batch_size_diff_inits(file_name, None, 0.00002, [5,10,50], 5)
    
#     draw_time_sub_figures_vary_by_batch_size(file_name, None, [0.00002, 0.01], 10, 5)
     
    plot_sub_figures_time_distance_by_deletion_rate_DNN(['Delete rate', 'Add rate'], file_names, [60000, 60000], [50, 50], [2,2], ['Batch Deletion', 'Batch Addition'])
     
#     plot_sub_figures_time_distance_by_deletion_rate('Add rate', del_file_names, batch_sizes, init_iters, periods, titles)
    
    
#     draw_test_acc_error_bars_by_batch_size_DNN('Deletion rate', file_name, None, 4096, inits, period)
    
#     draw_test_acc_error_bars_by_batch_size('Deletion rate', file_name, None, 4096, inits, period)
    
#     draw_time_figure_by_batch_size_varied_deletion_rate_for_comparison("Deletion rate", file_names, None, 16384, 10, 5, titles)
    
#     draw_continuous_deletion_addition(file_names, ["Online Deletion", "Online Addition"],["MNIST", "covtype", "HIGGS", "RCV1"])
    
    
#     draw_time_figure_by_batch_size_varied_deletion_rate('Deletion rate', file_name, None, 4096, inits, period)
#     draw_time_figure_by_batch_size_varied_deletion_rate_DNN('Deletion rate', file_name, None, 4096, 100, 2)
    
#     draw_time_figure_by_batch_size(file_name, None, 0.00002, 10, 5)
    
#     draw_time_figure_by_periods(file_name, None, 4096, inits, 0.00002)
#     draw_time_figure_by_inits(file_name, None, 4096, period, 0.00002)
#     draw_time_figure_by_deletion_rate(file_name, None, 0.00002, 20, period)
#     draw_time_figure_by_batch_size(file_name, None, 0.00002, 20, period)
