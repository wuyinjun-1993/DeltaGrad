'''
Created on Apr 18, 2019

'''
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure

# from experiment_data_processing.process_exp_result import provenance_opt_label, influence_label, iteration_label
# from experiment_data_processing.process_exp_result2 import standard_lib_label


provenance_approach = 'PrIU'

provenance_opt_approach = 'PrIU-opt'

influence_approach = 'INFL'

closed_form_approach = 'Closed-form'

origin = 'origin'

standard_lib_approach = 'STD'

iteration_approach = 'Baseline'

abs_dist = 'absolute_distance'

angle = 'cosine_similarity'

test_acc = 'test_accuray'

origin_label = 'origin'

standard_lib_label = 'standard_lib'

iteration_label = 'iteration'

provenance_label = 'provenance'

provenance_opt_label = 'provenance_opt'

influence_label = 'influence'

iteration_batch_label_1 = 'iteration_batch_1'

iteration_batch_label_2 = 'iteration_batch_2'

iteration_batch_label_3 = 'iteration_batch_3'


closed_form_label = 'closed_form'

linview_label = 'linview'

time_color_map={provenance_label: 'go', provenance_opt_label: 'c^', influence_label: 'b*', standard_lib_label: 'k+', iteration_label: 'ro', origin_label: 'rs', closed_form_label: 'rs', linview_label: 'g*'}

# shape_map = {provenance_label: ':', provenance_opt_label: '-.', influence_label: '--', standard_lib_label: '', iteration_label: 'm'}

# error_color_map=



def draw_accuracy_figure_sparse(excel_name, output_file_name, show_or_not, subset_ids):
    df = pd.read_excel(excel_name + '.xlsx', sheet_name='test_accuracy')
    noise_rate = df['noise_rate']
    
#     test_acc_std = df[standard_lib_label]
    
    test_acc_iter = df[iteration_label]
    
    test_acc_prov = df[provenance_label]
    
#     test_acc_prov_opt = df[provenance_opt_label]
#     
#     test_acc_inf = df[influence_label]
    
    
#     test_acc_origin = df[origin_label]
    
    lw = 4
    
    mw = 15
    
    fig, ax1 = plt.subplots(figsize=(8,6))
    
#     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    
#     line1, = ax1.plot(noise_rate[subset_ids], test_acc_std[subset_ids], time_color_map[standard_lib_label] + '-', linewidth=lw, markersize=mw)
    
    line2, = ax1.plot(noise_rate[subset_ids], test_acc_iter[subset_ids], time_color_map[iteration_label] + '-', linewidth=lw, markersize=mw)
    
    line3, = ax1.plot(noise_rate[subset_ids], test_acc_prov[subset_ids], 'c>-', linewidth=lw, markersize=mw)
    
#     line4, = ax1.plot(noise_rate[subset_ids], test_acc_prov_opt[subset_ids], time_color_map[provenance_opt_label] + '-', linewidth=lw, markersize=mw)
    
#     line5, = ax1.plot(noise_rate[subset_ids], test_acc_inf[subset_ids], time_color_map[influence_label] + '-', linewidth=lw, markersize=mw)
    
#     line6, = ax1.plot(noise_rate[subset_ids], test_acc_origin[subset_ids], time_color_map[origin_label] + '-', linewidth=lw, markersize=mw)
    
#     ax1.legend([line1, line2, line3, line4], [standard_lib_approach + ":" + test_acc, iteration_approach + ":" + test_acc, provenance_approach + ":" + test_acc, influence_approach + ":" + test_acc], loc="best", prop={'size': 15, 'weight':'bold'})

    print('here::', noise_rate[subset_ids])

#     print('here::', test_acc_origin[subset_ids])
    
#     print('here::', test_acc_inf[subset_ids])

    ax1.legend([line2, line3], [iteration_approach, provenance_approach], loc="best", prop={'size': 15, 'weight':'bold'})
    
    ax1.set_ylabel('validation accuracy', fontsize=15, weight = 'bold')
    
#     ax1.set_yscale("log")
    
    ax1.set_xlabel('deletion rate', fontsize=15, weight = 'bold')
    
    ax1.tick_params(axis='both', labelsize = 15)
    
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
    
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    plt.savefig(output_file_name, quality = 50, dpi = 300)
    
    
    if show_or_not:
        plt.show()


def draw_accuracy_figure(excel_name, output_file_name, show_or_not, subset_ids):
    df = pd.read_excel(excel_name + '.xlsx', sheet_name='test_accuracy')
    noise_rate = df['noise_rate']
    
#     test_acc_std = df[standard_lib_label]
    
    test_acc_iter = df[iteration_label]
    
    test_acc_prov = df[provenance_label]
    
    test_acc_prov_opt = df[provenance_opt_label]
    
    test_acc_inf = df[influence_label]
    
    
#     test_acc_origin = df[origin_label]
    
    lw = 4
    
    mw = 15
    
    fig, ax1 = plt.subplots(figsize=(8,6))
    
#     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    
#     line1, = ax1.plot(noise_rate[subset_ids], test_acc_std[subset_ids], time_color_map[standard_lib_label] + '-', linewidth=lw, markersize=mw)
    
    line2, = ax1.plot(noise_rate[subset_ids], test_acc_iter[subset_ids], time_color_map[iteration_label] + '-', linewidth=lw, markersize=mw)
    
#     line3, = ax1.plot(noise_rate[subset_ids], test_acc_prov[subset_ids], 'c>-', linewidth=lw, markersize=mw)
    
    line4, = ax1.plot(noise_rate[subset_ids], test_acc_prov_opt[subset_ids], time_color_map[provenance_opt_label] + '-', linewidth=lw, markersize=mw)
    
    line5, = ax1.plot(noise_rate[subset_ids], test_acc_inf[subset_ids], time_color_map[influence_label] + '-', linewidth=lw, markersize=mw)
    
#     line6, = ax1.plot(noise_rate[subset_ids], test_acc_origin[subset_ids], time_color_map[origin_label] + '-', linewidth=lw, markersize=mw)
    
#     ax1.legend([line1, line2, line3, line4], [standard_lib_approach + ":" + test_acc, iteration_approach + ":" + test_acc, provenance_approach + ":" + test_acc, influence_approach + ":" + test_acc], loc="best", prop={'size': 15, 'weight':'bold'})

    print('here::', noise_rate[subset_ids])

#     print('here::', test_acc_origin[subset_ids])
    
    print('here::', test_acc_inf[subset_ids])

    ax1.legend([line2, line4, line5], [iteration_approach, provenance_opt_approach, influence_approach], loc="best", prop={'size': 15, 'weight':'bold'})
    
    ax1.set_ylabel('validation accuracy', fontsize=15, weight = 'bold')
    
#     ax1.set_yscale("log")
    
    ax1.set_xlabel('deletion rate', fontsize=15, weight = 'bold')
    
    ax1.tick_params(axis='both', labelsize = 15)
    
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
    
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    plt.savefig(output_file_name, quality = 50, dpi = 300)
    
    
    if show_or_not:
        plt.show()

    plt.close()


def draw_time_figure_sparse(excel_name, output_file_name, show_or_not, subset_ids):
    df = pd.read_excel(excel_name + '.xlsx', sheet_name='training_time')
    noise_rate = df['noise_rate']
    
#     test_acc_std = df[standard_lib_label]
    
    test_acc_iter = df[iteration_label]
    
    test_acc_prov = df[provenance_label]
    
#     test_acc_prov_opt = df[provenance_opt_label]
#     
#     test_acc_inf = df[influence_label]
    
#     test_acc_closed_form_label = df[closed_form_label]
    
#     test_acc_linview_label = df[linview_label]
    
    lw = 4
    
    mw = 15
    
    fig, ax1 = plt.subplots(figsize=(8,6))
    
#     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    
#     line1, = ax1.plot(noise_rate[subset_ids], test_acc_std[subset_ids], time_color_map[standard_lib_label]+'-', linewidth=lw, markersize=mw)
    
    line2, = ax1.plot(noise_rate[subset_ids], test_acc_iter[subset_ids], time_color_map[standard_lib_label] + '-', linewidth=lw, markersize=mw)
    
    line3, = ax1.plot(noise_rate[subset_ids], test_acc_prov[subset_ids], time_color_map[provenance_label] + '--', linewidth=lw, markersize=mw)
    
#     line4, = ax1.plot(noise_rate[subset_ids], test_acc_prov_opt[subset_ids], time_color_map[provenance_opt_label] + '-', linewidth=lw, markersize=mw)
#     
#     line5, = ax1.plot(noise_rate[subset_ids], test_acc_inf[subset_ids], time_color_map[influence_label] + '-', linewidth=lw, markersize=mw)
#     line6, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[closed_form_label] + '-', linewidth=lw, markersize=mw)
#     
#     line7, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[linview_label] + '-', linewidth=lw, markersize=mw)
    
#     ax1.legend([line1, line2, line3, line4], [standard_lib_approach + ":" + test_acc, iteration_approach + ":" + test_acc, provenance_approach + ":" + test_acc, influence_approach + ":" + test_acc], loc="best", prop={'size': 15, 'weight':'bold'})

    ax1.legend([line2, line3], [iteration_approach, provenance_approach], loc='best', prop={'size': 15, 'weight':'bold'})
    
    ax1.set_ylabel('update time (second)', fontsize=15, weight = 'bold')
    
    ax1.set_xscale("log")
    
    ax1.set_xlabel('deletion rate', fontsize=15, weight = 'bold')
    
    ax1.tick_params(axis='both', labelsize = 15)
    
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
    
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    plt.savefig(output_file_name, quality = 50, dpi = 300)
    
    
    if show_or_not:
        plt.show()

    plt.close()

def draw_time_figure(excel_name, output_file_name, show_or_not, subset_ids):
    df = pd.read_excel(excel_name + '.xlsx', sheet_name='training_time')
    noise_rate = df['noise_rate']
    
#     test_acc_std = df[standard_lib_label]
    
    test_acc_iter = df[iteration_label]
    
#     test_acc_prov = df[provenance_label]
    
    test_acc_prov_opt = df[provenance_opt_label]
    
#     test_acc_closed_form = df[closed_form_label]
    
    test_acc_inf = df[influence_label]
    
#     test_acc_closed_form_label = df[closed_form_label]
    
#     test_acc_linview_label = df[linview_label]
    
    lw = 4
    
    mw = 15
    
    fig, ax1 = plt.subplots(figsize=(8,6))
    
#     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    
#     line1, = ax1.plot(noise_rate[subset_ids], test_acc_std[subset_ids], time_color_map[standard_lib_label]+'-', linewidth=lw, markersize=mw)
    
    line2, = ax1.plot(noise_rate[subset_ids], test_acc_iter[subset_ids], time_color_map[standard_lib_label] + '-', linewidth=lw, markersize=mw)
    
#     line3, = ax1.plot(noise_rate[subset_ids], test_acc_prov[subset_ids], time_color_map[provenance_label] + '--', linewidth=lw, markersize=mw)
    
    line4, = ax1.plot(noise_rate[subset_ids], test_acc_prov_opt[subset_ids], time_color_map[provenance_opt_label] + '-', linewidth=lw, markersize=mw)
    
    line5, = ax1.plot(noise_rate[subset_ids], test_acc_inf[subset_ids], time_color_map[influence_label] + '-', linewidth=lw, markersize=mw)
#     line6, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[closed_form_label] + '-', linewidth=lw, markersize=mw)
#     
#     line7, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[linview_label] + '-', linewidth=lw, markersize=mw)
    
#     ax1.legend([line1, line2, line3, line4], [standard_lib_approach + ":" + test_acc, iteration_approach + ":" + test_acc, provenance_approach + ":" + test_acc, influence_approach + ":" + test_acc], loc="best", prop={'size': 15, 'weight':'bold'})

#     ax1.legend([line2, line4, line5], [iteration_approach, provenance_opt_approach, influence_approach], bbox_to_anchor=(0.8, 0.8), bbox_transform=plt.gcf().transFigure, prop={'size': 15, 'weight':'bold'})
    ax1.legend([line2, line4, line5], [iteration_approach, provenance_opt_approach, influence_approach], loc = 'best', prop={'size': 15, 'weight':'bold'})    
    ax1.set_ylabel('update time (second)', fontsize=15, weight = 'bold')
    
    ax1.set_xscale("log")
    
#     ax1.set_yscale("log")
    
    ax1.set_xlabel('deletion rate', fontsize=15, weight = 'bold')
    
    ax1.tick_params(axis='both', labelsize = 15)
    
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
    
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    plt.savefig(output_file_name, quality = 50, dpi = 300, format='eps')
    
    
#     if show_or_not:
    plt.show()

  
  
def draw_time_figure_linear_regression(excel_name, output_file_name, show_or_not, subset_ids):
    df = pd.read_excel(excel_name + '.xlsx', sheet_name='training_time')
    noise_rate = df['noise_rate']
    
#     test_acc_std = df[standard_lib_label]
    
    test_acc_iter = df[iteration_label]
    
    test_acc_prov = df[provenance_label]
    
    test_acc_prov_opt = df[provenance_opt_label]
    
#     test_acc_closed_form = df[closed_form_label]
    
    test_acc_inf = df[influence_label]
    
    test_acc_closed_form_label = df[closed_form_label]
    
    test_acc_linview_label = df[linview_label]
    
    lw = 4
    
    mw = 15
    
    fig, ax1 = plt.subplots(figsize=(10,8))
    
#     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    
#     line1, = ax1.plot(noise_rate[subset_ids], test_acc_std[subset_ids], time_color_map[standard_lib_label]+'-', linewidth=lw, markersize=mw)
    
    line2, = ax1.plot(noise_rate[subset_ids], test_acc_iter[subset_ids], time_color_map[standard_lib_label] + '-', linewidth=lw, markersize=mw)
    
    line3, = ax1.plot(noise_rate[subset_ids], test_acc_prov[subset_ids], time_color_map[provenance_label] + '--', linewidth=lw, markersize=mw)
    
    line4, = ax1.plot(noise_rate[subset_ids], test_acc_prov_opt[subset_ids], time_color_map[provenance_opt_label] + '-', linewidth=lw, markersize=mw)
    
    line5, = ax1.plot(noise_rate[subset_ids], test_acc_inf[subset_ids], time_color_map[influence_label] + '-', linewidth=lw, markersize=mw)
    line6, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[closed_form_label] + '-', linewidth=lw, markersize=mw)
#     
#     line7, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[linview_label] + '-', linewidth=lw, markersize=mw)
    
#     ax1.legend([line1, line2, line3, line4], [standard_lib_approach + ":" + test_acc, iteration_approach + ":" + test_acc, provenance_approach + ":" + test_acc, influence_approach + ":" + test_acc], loc="best", prop={'size': 15, 'weight':'bold'})

    ax1.legend([line2, line4, line6, line5, line3], [iteration_approach, provenance_opt_approach, closed_form_approach, influence_approach, provenance_approach], bbox_to_anchor=(0.7, 0.6), bbox_transform=plt.gcf().transFigure, ncol=2, prop={'size': 24, 'weight':'bold'})
#     ax1.legend([line2, line6, line5, line4], [iteration_approach, closed_form_approach, influence_approach, provenance_opt_approach], bbox_to_anchor=(0.8, 0.7), bbox_transform=plt.gcf().transFigure, ncol=2, prop={'size': 28, 'weight':'bold'})
    
    ax1.set_ylabel('update time (second)', fontsize=26, weight = 'bold')
    
    ax1.set_xscale("log")
    
    ax1.set_yscale("log")
    
    ax1.set_xlabel('deletion rate', fontsize=24, weight = 'bold')
    
    ax1.tick_params(axis='both', labelsize = 24)
    
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
    
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    plt.savefig(output_file_name, quality = 50, dpi = 300)
    
    
#     if show_or_not:
    plt.show()

    
# def draw_time_figure_linear_regression(excel_name, output_file_name, show_or_not, subset_ids):
#     df = pd.read_excel(excel_name + '.xlsx', sheet_name='training_time')
#     noise_rate = df['noise_rate']
#     
# #     test_acc_std = df[standard_lib_label]
#     
#     test_acc_iter = df[iteration_label]
#     
#     test_acc_prov = df[provenance_label]
#     
#     test_acc_prov_opt = df[provenance_opt_label]
#     
#     test_acc_closed_form = df[closed_form_label]
#     
# #     test_acc_inf = df[influence_label]
#     
# #     test_acc_closed_form_label = df[closed_form_label]
#     
# #     test_acc_linview_label = df[linview_label]
#     
#     lw = 4
#     
#     mw = 15
#     
#     fig, ax1 = plt.subplots(figsize=(8,6))
#     
# #     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
# 
#     
# #     line1, = ax1.plot(noise_rate[subset_ids], test_acc_std[subset_ids], time_color_map[standard_lib_label]+'-', linewidth=lw, markersize=mw)
#     
#     line2, = ax1.plot(noise_rate[subset_ids], test_acc_iter[subset_ids], time_color_map[standard_lib_label] + '-', linewidth=lw, markersize=mw)
#     
#     line3, = ax1.plot(noise_rate[subset_ids], test_acc_prov[subset_ids], time_color_map[provenance_label] + '--', linewidth=lw, markersize=mw)
#     
#     line4, = ax1.plot(noise_rate[subset_ids], test_acc_prov_opt[subset_ids], time_color_map[provenance_opt_label] + '-', linewidth=lw, markersize=mw)
#     
#     line5, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form[subset_ids], time_color_map[influence_label] + '-', linewidth=lw, markersize=mw)
# #     line6, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[closed_form_label] + '-', linewidth=lw, markersize=mw)
# #     
# #     line7, = ax1.plot(noise_rate[subset_ids], test_acc_closed_form_label[subset_ids], time_color_map[linview_label] + '-', linewidth=lw, markersize=mw)
#     
# #     ax1.legend([line1, line2, line3, line4], [standard_lib_approach + ":" + test_acc, iteration_approach + ":" + test_acc, provenance_approach + ":" + test_acc, influence_approach + ":" + test_acc], loc="best", prop={'size': 15, 'weight':'bold'})
# 
#     ax1.legend([line2, line3, line4, line5], [iteration_approach, provenance_approach, provenance_opt_approach, closed_form_approach], loc='best', prop={'size': 15, 'weight':'bold'})
#     
#     ax1.set_ylabel('update time (second)', fontsize=15, weight = 'bold')
#     
#     ax1.set_xscale("log")
#     
#     ax1.set_yscale("log")
#     
#     ax1.set_xlabel('deletion rate', fontsize=15, weight = 'bold')
#     
#     ax1.tick_params(axis='both', labelsize = 15)
#     
#     plt.setp(ax1.get_xticklabels(), fontweight="bold")
#     
#     plt.setp(ax1.get_yticklabels(), fontweight="bold")
#     
#     plt.savefig(output_file_name, quality = 50, dpi = 300)
#     
#     
#     if show_or_not:
#         plt.show()

def draw_time_figure_with_varied_epochs(excel_name, output_file_name, show_or_not, subset_ids):
    df = pd.read_excel(excel_name + '.xlsx', sheet_name='training_time')
    noise_rate = df['noise_rate']
    
    test_acc_std = df[standard_lib_label]
    
    test_acc_iter = df[iteration_label]
    
    test_acc_prov = df[provenance_label]
    
    test_acc_prov_opt = df[provenance_opt_label]
    
    test_acc_inf = df[influence_label]
    
    lw = 4
    
    mw = 15
    
    fig, ax1 = plt.subplots(figsize=(8,6))
    
#     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    
    line1, = ax1.plot(noise_rate[subset_ids], test_acc_std[subset_ids], 'ms:', linewidth=lw, markersize=mw)
    
    line2, = ax1.plot(noise_rate[subset_ids], test_acc_iter[subset_ids], 'r*--', linewidth=lw, markersize=mw)
    
    line3, = ax1.plot(noise_rate[subset_ids], test_acc_prov[subset_ids], 'kx-', linewidth=lw, markersize=mw)
    
    line4, = ax1.plot(noise_rate[subset_ids], test_acc_prov_opt[subset_ids], 'c^-', linewidth=lw, markersize=mw)
    
    line5, = ax1.plot(noise_rate[subset_ids], test_acc_inf[subset_ids], 'gp-.', linewidth=lw, markersize=mw)
    
#     ax1.legend([line1, line2, line3, line4], [standard_lib_approach + ":" + test_acc, iteration_approach + ":" + test_acc, provenance_approach + ":" + test_acc, influence_approach + ":" + test_acc], loc="best", prop={'size': 15, 'weight':'bold'})

    ax1.legend([line1, line2, line3, line4, line5], [standard_lib_approach, iteration_approach, provenance_approach, provenance_opt_approach, influence_approach], bbox_to_anchor=(0, 1.15), loc=2, prop={'size': 15, 'weight':'bold'})
    
    ax1.set_ylabel('update time (second)', fontsize=15, weight = 'bold')
    
    ax1.set_yscale("log")
    
    ax1.set_xlabel('iteration num', fontsize=15, weight = 'bold')
    
    ax1.tick_params(axis='both', labelsize = 15)
    
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
    
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    plt.savefig(output_file_name, quality = 100, dpi = 1200)
    
    
    if show_or_not:
        plt.show()


# def draw_time_figure_with_varied_epoch(excel_name, output_file_name, show_or_not, subset_ids):
#     
#     df = pd.read_excel(excel_name + '.xlsx', sheet_name='training_time')
#     
#     
#     
    

def draw_distance_figure(excel_name, output_file_name, show_or_not, subset_ids):

    df1 = pd.read_excel(excel_name + '.xlsx', sheet_name='abs_err')
    
    df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='relative_err')
        
    noise_rate = df1['noise_rate']
    
#     prov_abs_errs = df1[provenance_label]
    
    prov_opt_abs_errs = df1[provenance_opt_label]
    
    influence_abs_errs = df1[influence_label]
    
    
    
        
#     prov_rel_errs = df2[provenance_label]
    
    prov_opt_rel_errs = df2[provenance_opt_label]
    
    influence_rel_errs = df2[influence_label]
    
    
    
    
    
    lw = 8
    
    mw = 25
    
    print(noise_rate[1:])
    
#     print(prov_abs_errs[1:])
    
    print(influence_abs_errs[1:])
    
    fig, ax1 = plt.subplots(figsize=(14,10))
    
#     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    
#     line1, = ax1.plot(noise_rate[subset_ids], prov_abs_errs[subset_ids], time_color_map[provenance_label] + ':', linewidth=lw, markersize=mw)
    
    line2, = ax1.plot(noise_rate[subset_ids], prov_opt_abs_errs[subset_ids], time_color_map[provenance_opt_label]+':', linewidth=lw, markersize=mw)
    
    line3, = ax1.plot(noise_rate[subset_ids], influence_abs_errs[subset_ids], time_color_map[influence_label]+':', linewidth=lw, markersize=mw)
    
#     ax1.legend([line1, line2], [provenance_approach + ":" + abs_dist, influence_approach + ":" + abs_dist], loc="best", prop={'size': 15})

    
#     ax1.yscale('log')
    
    ax1.set_ylabel('distance', fontsize=25, weight = 'bold')
    
    ax1.set_yscale("log")
    
    ax1.set_xlabel('deletion rate', fontsize=25, weight = 'bold')
    
    ax1.tick_params(axis='both', labelsize = 25)
    
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
    
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('cosine similarity', fontsize=25, weight = 'bold')  # we already handled the x-label with ax1
#     line4, = ax2.plot(noise_rate[subset_ids], prov_rel_errs[subset_ids], time_color_map[provenance_label] + '-', linewidth=lw, markersize=mw)
    line5, = ax2.plot(noise_rate[subset_ids], prov_opt_rel_errs[subset_ids], time_color_map[provenance_opt_label]+'-', linewidth=lw, markersize=mw)
    line6, = ax2.plot(noise_rate[subset_ids], influence_rel_errs[subset_ids], time_color_map[influence_label]+'-', linewidth=lw, markersize=mw)
    ax2.tick_params(axis='y', labelsize = 25)
    
    plt.setp(ax2.get_yticklabels(), fontweight="bold")
    
    lns = []
    
#     lns.append(line1)
    
    lns.append(line2)
    
    lns.append(line3)
    
#     lns.append(line4)
    
    lns.append(line5)
    
    lns.append(line6)
    
    labs = [l.get_label() for l in lns]
    
    ax1.legend(lns, [provenance_opt_approach + ":" + abs_dist, influence_approach + ":" + abs_dist, provenance_opt_approach + ":" + angle, influence_approach + ":" + angle], bbox_to_anchor=(0.05, 0.5), loc = 6, prop={'size': 25, 'weight':'bold'})
    
#     ax2.legend([line3, line4], [provenance_approach + ":" + angle, influence_approach + ":" + angle], loc="best", prop={'size': 15})
    
#     ax1.xlabel('error_rate', fontsize = 15)
#     
#     ax1.ylabel('absolute distance', fontsize = 15)
    
#     ax = ax1.gca()
#     ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
#     ax.tick_params(axis = 'both', which = 'minor', labelsize = 2)
#     plt.figure(figsize=(3,4))
    
    plt.savefig(output_file_name, quality = 50, dpi = 300)
    
    
    if show_or_not:
        plt.show()


def draw_distance_figure_sparse(excel_name, output_file_name, show_or_not, subset_ids):

    df1 = pd.read_excel(excel_name + '.xlsx', sheet_name='abs_err')
    
    df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='relative_err')
        
    noise_rate = df1['noise_rate']
    
    prov_abs_errs = df1[provenance_label]
    
#     prov_opt_abs_errs = df1[provenance_opt_label]
    
#     influence_abs_errs = df1[influence_label]
    
    
    
        
    prov_rel_errs = df2[provenance_label]
    
#     prov_opt_rel_errs = df2[provenance_opt_label]
    
#     influence_rel_errs = df2[influence_label]
    
    
    
    
    
    lw = 8
    
    mw = 25
    
    print(noise_rate[1:])
    
#     print(prov_abs_errs[1:])
    
#     print(influence_abs_errs[1:])
    
    fig, ax1 = plt.subplots(figsize=(14,10))
    
#     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    
    line1, = ax1.plot(noise_rate[subset_ids], prov_abs_errs[subset_ids], time_color_map[provenance_label] + ':', linewidth=lw, markersize=mw)
    
#     line2, = ax1.plot(noise_rate[subset_ids], prov_opt_abs_errs[subset_ids], time_color_map[provenance_opt_label]+':', linewidth=lw, markersize=mw)
    
#     line3, = ax1.plot(noise_rate[subset_ids], influence_abs_errs[subset_ids], time_color_map[influence_label]+':', linewidth=lw, markersize=mw)
    
#     ax1.legend([line1, line2], [provenance_approach + ":" + abs_dist, influence_approach + ":" + abs_dist], loc="best", prop={'size': 15})

    
#     ax1.yscale('log')
    
    ax1.set_ylabel('distance', fontsize=25, weight = 'bold')
    
    ax1.set_yscale("log")
    
    ax1.set_xlabel('deletion rate', fontsize=25, weight = 'bold')
    
    ax1.tick_params(axis='both', labelsize = 25)
    
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
    
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('cosine similarity', fontsize=25, weight = 'bold')  # we already handled the x-label with ax1
    line4, = ax2.plot(noise_rate[subset_ids], prov_rel_errs[subset_ids], time_color_map[provenance_label] + '-', linewidth=lw, markersize=mw)
#     line5, = ax2.plot(noise_rate[subset_ids], prov_opt_rel_errs[subset_ids], time_color_map[provenance_opt_label]+'-', linewidth=lw, markersize=mw)
#     line6, = ax2.plot(noise_rate[subset_ids], influence_rel_errs[subset_ids], time_color_map[influence_label]+'-', linewidth=lw, markersize=mw)
    ax2.tick_params(axis='y', labelsize = 25)
    
    plt.setp(ax2.get_yticklabels(), fontweight="bold")
    
    lns = []
    
    lns.append(line1)
    
#     lns.append(line2)
#     
#     lns.append(line3)
    
    lns.append(line4)
    
#     lns.append(line5)
#     
#     lns.append(line6)
    
    labs = [l.get_label() for l in lns]
    
    ax1.legend(lns, [provenance_approach + ":" + abs_dist, provenance_approach + ":" + angle], loc = 'best', prop={'size': 25, 'weight':'bold'})
    
#     ax2.legend([line3, line4], [provenance_approach + ":" + angle, influence_approach + ":" + angle], loc="best", prop={'size': 15})
    
#     ax1.xlabel('error_rate', fontsize = 15)
#     
#     ax1.ylabel('absolute distance', fontsize = 15)
    
#     ax = ax1.gca()
#     ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
#     ax.tick_params(axis = 'both', which = 'minor', labelsize = 2)
#     plt.figure(figsize=(3,4))
    
    plt.savefig(output_file_name, quality = 50, dpi = 300)
    
    
    if show_or_not:
        plt.show()


# def draw_distance_figure_sparse(excel_name, output_file_name, show_or_not, subset_ids):
# 
#     df1 = pd.read_excel(excel_name + '.xlsx', sheet_name='abs_err')
#     
#     df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='relative_err')
#         
#     noise_rate = df1['noise_rate']
#     
#     prov_abs_errs = df1[provenance_label]
#     
# #     prov_opt_abs_errs = df1[provenance_opt_label]
#     
# #     influence_abs_errs = df1[influence_label]
#     
#     
#     
#         
#     prov_rel_errs = df2[provenance_label]
#     
# #     prov_opt_rel_errs = df2[provenance_opt_label]
#     
#     influence_rel_errs = df2[influence_label]
#     
#     
#     
#     
#     
#     lw = 8
#     
#     mw = 25
#     
#     print(noise_rate[1:])
#     
# #     print(prov_abs_errs[1:])
#     
# #     print(influence_abs_errs[1:])
#     
#     fig, ax1 = plt.subplots(figsize=(14,10))
#     
# #     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
# 
#     
#     line1, = ax1.plot(noise_rate[subset_ids], prov_abs_errs[subset_ids], time_color_map[provenance_label] + ':', linewidth=lw, markersize=mw)
#     
# #     line2, = ax1.plot(noise_rate[subset_ids], prov_opt_abs_errs[subset_ids], time_color_map[provenance_opt_label]+':', linewidth=lw, markersize=mw)
#     
#     line3, = ax1.plot(noise_rate[subset_ids], influence_abs_errs[subset_ids], time_color_map[influence_label]+':', linewidth=lw, markersize=mw)
#     
# #     ax1.legend([line1, line2], [provenance_approach + ":" + abs_dist, influence_approach + ":" + abs_dist], loc="best", prop={'size': 15})
# 
#     
# #     ax1.yscale('log')
#     
#     ax1.set_ylabel('distance', fontsize=25, weight = 'bold')
#     
#     ax1.set_yscale("log")
#     
#     ax1.set_xlabel('deletion rate', fontsize=25, weight = 'bold')
#     
#     ax1.tick_params(axis='both', labelsize = 25)
#     
#     plt.setp(ax1.get_xticklabels(), fontweight="bold")
#     
#     plt.setp(ax1.get_yticklabels(), fontweight="bold")
#     
#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# 
#     ax2.set_ylabel('cosine similarity', fontsize=25, weight = 'bold')  # we already handled the x-label with ax1
# #     line4, = ax2.plot(noise_rate[subset_ids], prov_rel_errs[subset_ids], time_color_map[provenance_label] + '-', linewidth=lw, markersize=mw)
#     line5, = ax2.plot(noise_rate[subset_ids], prov_opt_rel_errs[subset_ids], time_color_map[provenance_opt_label]+'-', linewidth=lw, markersize=mw)
#     line6, = ax2.plot(noise_rate[subset_ids], influence_rel_errs[subset_ids], time_color_map[influence_label]+'-', linewidth=lw, markersize=mw)
#     ax2.tick_params(axis='y', labelsize = 25)
#     
#     plt.setp(ax2.get_yticklabels(), fontweight="bold")
#     
#     lns = []
#     
# #     lns.append(line1)
#     
#     lns.append(line2)
#     
#     lns.append(line3)
#     
# #     lns.append(line4)
#     
#     lns.append(line5)
#     
#     lns.append(line6)
#     
#     labs = [l.get_label() for l in lns]
#     
#     ax1.legend(lns, [provenance_opt_approach + ":" + abs_dist, influence_approach + ":" + abs_dist, provenance_opt_approach + ":" + angle, influence_approach + ":" + angle], bbox_to_anchor=(0.05, 0.5), loc = 6, prop={'size': 25, 'weight':'bold'})
#     
# #     ax2.legend([line3, line4], [provenance_approach + ":" + angle, influence_approach + ":" + angle], loc="best", prop={'size': 15})
#     
# #     ax1.xlabel('error_rate', fontsize = 15)
# #     
# #     ax1.ylabel('absolute distance', fontsize = 15)
#     
# #     ax = ax1.gca()
# #     ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
# #     ax.tick_params(axis = 'both', which = 'minor', labelsize = 2)
# #     plt.figure(figsize=(3,4))
#     
#     plt.savefig(output_file_name, quality = 50, dpi = 300)
#     
#     
#     if show_or_not:
#         plt.show()


def draw_compare_pro_and_prov_opt_linear_regression(excel_name, output_file_name, show_or_not, subset_ids):

    df1 = pd.read_excel(excel_name + '.xlsx', sheet_name='training_time')
    
    df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='test_accuracy')
        
    noise_rate = df1['noise_rate']
    
    prov_abs_errs = df1[provenance_label]
    
    prov_opt_abs_errs = df1[provenance_opt_label]
    
#     influence_abs_errs = df1[influence_label]
    
    
#     test_acc_std = df2[standard_lib_label]
    
#     test_acc_iter = df2[iteration_label]
    
    test_acc_prov = df2[provenance_label]
    
    test_acc_prov_opt = df2[provenance_opt_label]
    
#     test_acc_inf = df2[influence_label]
    
    
#     test_acc_origin = df2[origin_label]
        
    
    
    
    
    
    
    lw = 10
    
    mw = 25
    
    print(noise_rate[1:])
    
    print(prov_abs_errs[1:])
    
#     print(influence_abs_errs[1:])
    
    fig, ax1 = plt.subplots(figsize=(24,12))
    
#     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

 
    line1, = ax1.plot(noise_rate[subset_ids], prov_abs_errs[subset_ids], time_color_map[provenance_label] + '-', linewidth=lw, markersize=mw)
    
    line2, = ax1.plot(noise_rate[subset_ids], prov_opt_abs_errs[subset_ids], time_color_map[provenance_opt_label] + '-', linewidth=lw, markersize=mw)

        
#     line3, = ax1.plot(noise_rate[subset_ids], influence_abs_errs[subset_ids], time_color_map[influence_label] + '-', linewidth=lw, markersize=mw)
    
#     ax1.legend([line1, line2], [provenance_approach + ":" + abs_dist, influence_approach + ":" + abs_dist], loc="best", prop={'size': 15})

    
#     ax1.yscale('log')
    
    ax1.set_ylabel('update time (second)', fontsize=40, weight = 'bold')
    
#     ax1.set_yscale("log")

    ax1.set_xlabel('deletion rate', fontsize=40, weight = 'bold')
    
    ax1.tick_params(axis='both', labelsize = 30)
    
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
    
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Mean Square Error', fontsize=40, weight = 'bold')  # we already handled the x-label with ax1
    
    line6, = ax2.plot(noise_rate[subset_ids], test_acc_prov[subset_ids], time_color_map[provenance_label] + ':', linewidth=lw, markersize=mw)
    line7, = ax2.plot(noise_rate[subset_ids], test_acc_prov_opt[subset_ids], time_color_map[provenance_opt_label] + ':', linewidth=lw, markersize=mw)


#     line4, = ax2.plot(noise_rate[subset_ids], test_acc_std[subset_ids], time_color_map[standard_lib_label] + ':', linewidth=lw, markersize=mw)
#     line5, = ax2.plot(noise_rate[subset_ids], test_acc_iter[subset_ids], 'c<-.', linewidth=lw, markersize=mw)
#     line8, = ax2.plot(noise_rate[subset_ids], test_acc_inf[subset_ids], time_color_map[influence_label] + ':', linewidth=lw, markersize=mw)
#     line9, = ax2.plot(noise_rate[subset_ids], test_acc_origin[subset_ids], 'rH-', linewidth=lw, markersize=mw)
    ax2.tick_params(axis='y', labelsize = 30)
    
    ax2.set_ylim([0,0.0015])
    
    plt.setp(ax2.get_yticklabels(), fontweight="bold")
    
    lns = []

    
    lns.append(line1)
    
    lns.append(line2)
    
        
    lns.append(line6)
    
    lns.append(line7)
    
#     lns.append(line3)
    
#     lns.append(line4)
    
#     lns.append(line5)
    

    
#     lns.append(line8)
    
#     lns.append(line9)
    
    labs = []
    
    mse_label = 'validation accuracy'
    
    labs.append(provenance_approach + ":update_time")
    
    labs.append(provenance_opt_approach + ":update_time")
    
    labs.append(provenance_approach + ':' + mse_label)
    
    labs.append(provenance_opt_approach + ':' + mse_label)
    

    
#     labs.append(influence_approach + ":" + abs_dist)
    
#     labs.append(standard_lib_approach + ':' + mse_label)
    
#     labs.append(iteration_approach + ':' + test_acc)
    

    
#     labs.append(influence_approach + ':' + mse_label)
    
#     labs.append(origin + ':' + test_acc)
    
#     ax1.legend(lns, labs, bbox_to_anchor=(0.05, 0.6), loc = 6, prop={'size': 15, 'weight':'bold'})
    
    ax1.legend(lns, labs, loc = 'center right', prop={'size': 40, 'weight':'bold'})
    
    ax2.legend(lns, labs, loc = 'center right', prop={'size': 40, 'weight':'bold'})
    
#     ax2.legend([line3, line4], [provenance_approach + ":" + angle, influence_approach + ":" + angle], loc="best", prop={'size': 15})
    
#     ax1.xlabel('error_rate', fontsize = 15)
#     
#     ax1.ylabel('absolute distance', fontsize = 15)
    
#     ax = ax1.gca()
#     ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
#     ax.tick_params(axis = 'both', which = 'minor', labelsize = 2)
#     plt.figure(figsize=(3,4))
    
    plt.savefig(output_file_name, quality = 50, dpi = 300)
    
    
    if show_or_not:
        plt.show()


def draw_distance_figure_linear_regression2(excel_name, output_file_name, show_or_not, subset_ids):

    df1 = pd.read_excel(excel_name + '.xlsx', sheet_name='abs_err')
    
    df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='test_accuracy')
        
    noise_rate = df1['noise_rate']
    
    prov_abs_errs = df1[provenance_label]
    
    prov_opt_abs_errs = df1[provenance_opt_label]
    
#     influence_abs_errs = df1[influence_label]
    
    
    test_acc_std = df2[standard_lib_label]
    
    test_acc_iter = df2[iteration_label]
    
    test_acc_prov = df2[provenance_label]
    
    test_acc_prov_opt = df2[provenance_opt_label]
    
#     test_acc_inf = df2[influence_label]
    
    
    test_acc_origin = df2[origin_label]
        
    
    
    
    
    
    
    lw = 10
    
    mw = 25
    
    print(noise_rate[1:])
    
    print(prov_abs_errs[1:])
    
#     print(influence_abs_errs[1:])
    
    fig, ax1 = plt.subplots(figsize=(24,12))
    
#     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    
#     line1, = ax1.plot(noise_rate[subset_ids], prov_abs_errs[subset_ids], 'go:', linewidth=lw, markersize=mw)
    
#     line2, = ax1.plot(noise_rate[subset_ids], prov_opt_abs_errs[subset_ids], time_color_map[provenance_opt_label] + '-', linewidth=lw, markersize=mw)
#     
#     line3, = ax1.plot(noise_rate[subset_ids], influence_abs_errs[subset_ids], time_color_map[influence_label] + '-', linewidth=lw, markersize=mw)
    line4, = ax1.plot(noise_rate[subset_ids], test_acc_std[subset_ids], time_color_map[iteration_label] + '-', linewidth=lw, markersize=mw)
#     line5, = ax2.plot(noise_rate[subset_ids], test_acc_iter[subset_ids], 'c<-.', linewidth=lw, markersize=mw)
#     line6, = ax2.plot(noise_rate[subset_ids], test_acc_prov[subset_ids], time_color_map[provenance_label], linewidth=lw, markersize=mw)
    line7, = ax1.plot(noise_rate[subset_ids], test_acc_prov_opt[subset_ids], time_color_map[provenance_opt_label] + '-', linewidth=lw, markersize=mw)
#     line8, = ax1.plot(noise_rate[subset_ids], test_acc_inf[subset_ids], time_color_map[influence_label] + '-', linewidth=lw, markersize=mw)
   
#     ax1.legend([line1, line2], [provenance_approach + ":" + abs_dist, influence_approach + ":" + abs_dist], loc="best", prop={'size': 15})

    
#     ax1.yscale('log')
    
    ax1.set_ylabel('Mean Square Error', fontsize=40, weight = 'bold')
    
#     ax1.set_yscale("log")
    
    ax1.set_xlabel('deletion rate', fontsize=40, weight = 'bold')
    
    ax1.tick_params(axis='both', labelsize = 30)
    
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
    
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# 
#     ax2.set_ylabel('Mean Square Error', fontsize=40, weight = 'bold')  # we already handled the x-label with ax1
# #     line9, = ax2.plot(noise_rate[subset_ids], test_acc_origin[subset_ids], 'rH-', linewidth=lw, markersize=mw)
#     ax2.tick_params(axis='y', labelsize = 30)
#     
#     plt.setp(ax2.get_yticklabels(), fontweight="bold")
    
    lns = []
    
#     lns.append(line1)
    
#     lns.append(line2)
#     
#     lns.append(line3)
    
    lns.append(line4)
    
#     lns.append(line5)
    
#     lns.append(line6)
    
    lns.append(line7)
    
#     lns.append(line8)
    
#     lns.append(line9)
    
    labs = []
    
    mse_label = 'validation accuracy'
    
#     labs.append(provenance_approach + ":" + abs_dist)
    
#     labs.append(provenance_opt_approach + ":" + abs_dist)
#     
#     labs.append(influence_approach + ":" + abs_dist)
    
    labs.append(iteration_approach)
    
#     labs.append(iteration_approach + ':' + test_acc)
    
#     labs.append(provenance_approach + ':' + mse_label)
    
    labs.append(provenance_opt_approach)
    
    labs.append(influence_approach)
    
#     labs.append(origin + ':' + test_acc)
    
#     ax1.legend(lns, labs, bbox_to_anchor=(0.05, 0.6), loc = 6, prop={'size': 15, 'weight':'bold'})
    
    ax1.legend(lns, labs, loc = 'best', prop={'size': 40, 'weight':'bold'})
    
#     ax2.legend([line3, line4], [provenance_approach + ":" + angle, influence_approach + ":" + angle], loc="best", prop={'size': 15})
    
#     ax1.xlabel('error_rate', fontsize = 15)
#     
#     ax1.ylabel('absolute distance', fontsize = 15)
    
#     ax = ax1.gca()
#     ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
#     ax.tick_params(axis = 'both', which = 'minor', labelsize = 2)
#     plt.figure(figsize=(3,4))
    
    plt.savefig(output_file_name, quality = 50, dpi = 300)
    
    
    if show_or_not:
        plt.show()


def draw_distance_figure_linear_regression1(excel_name, output_file_name, show_or_not, subset_ids):

    df1 = pd.read_excel(excel_name + '.xlsx', sheet_name='abs_err')
    
    df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='test_accuracy')
        
    noise_rate = df1['noise_rate']
    
    prov_abs_errs = df1[provenance_label]
    
    prov_opt_abs_errs = df1[provenance_opt_label]
    
#     influence_abs_errs = df1[influence_label]
    
    
    test_acc_std = df2[standard_lib_label]
    
#     test_acc_iter = df2[iteration_label]

    

    
    test_acc_prov = df2[provenance_label]
    
    test_acc_prov_opt = df2[provenance_opt_label]
    
#     test_acc_inf = df2[influence_label]
    
    
    test_acc_origin = df2[origin_label]
        
    
    
    
    
    
    
    lw = 10
    
    mw = 25
    
    print(noise_rate[1:])
    
    print(prov_abs_errs[1:])
    
#     print(influence_abs_errs[1:])
    
    fig, ax1 = plt.subplots(figsize=(24,12))
    
#     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    
#     line1, = ax1.plot(noise_rate[subset_ids], prov_abs_errs[subset_ids], 'go:', linewidth=lw, markersize=mw)
    
    line2, = ax1.plot(noise_rate[subset_ids], prov_opt_abs_errs[subset_ids], time_color_map[provenance_opt_label] + '-', linewidth=lw, markersize=mw)
    
#     line3, = ax1.plot(noise_rate[subset_ids], influence_abs_errs[subset_ids], time_color_map[influence_label] + '-', linewidth=lw, markersize=mw)
    
#     ax1.legend([line1, line2], [provenance_approach + ":" + abs_dist, influence_approach + ":" + abs_dist], loc="best", prop={'size': 15})

    
#     ax1.yscale('log')
    
    ax1.set_ylabel('distance', fontsize=40, weight = 'bold')
    
    ax1.set_yscale("log")
    
    ax1.set_xlabel('deletion rate', fontsize=40, weight = 'bold')
    
    ax1.tick_params(axis='both', labelsize = 30)
    
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
    
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# 
#     ax2.set_ylabel('Mean Square Error', fontsize=40, weight = 'bold')  # we already handled the x-label with ax1
#     line4, = ax2.plot(noise_rate[subset_ids], test_acc_std[subset_ids], time_color_map[standard_lib_label] + ':', linewidth=lw, markersize=mw)
# #     line5, = ax2.plot(noise_rate[subset_ids], test_acc_iter[subset_ids], 'c<-.', linewidth=lw, markersize=mw)
# #     line6, = ax2.plot(noise_rate[subset_ids], test_acc_prov[subset_ids], time_color_map[provenance_label], linewidth=lw, markersize=mw)
#     line7, = ax2.plot(noise_rate[subset_ids], test_acc_prov_opt[subset_ids], time_color_map[provenance_opt_label] + ':', linewidth=lw, markersize=mw)
#     line8, = ax2.plot(noise_rate[subset_ids], test_acc_inf[subset_ids], time_color_map[influence_label] + ':', linewidth=lw, markersize=mw)
# #     line9, = ax2.plot(noise_rate[subset_ids], test_acc_origin[subset_ids], 'rH-', linewidth=lw, markersize=mw)
#     ax2.tick_params(axis='y', labelsize = 30)
#     
#     plt.setp(ax2.get_yticklabels(), fontweight="bold")
    
    lns = []
    
#     lns.append(line1)
    
    lns.append(line2)
    
#     lns.append(line3)
    
#     lns.append(line4)
#     
# #     lns.append(line5)
#     
# #     lns.append(line6)
#     
#     lns.append(line7)
#     
#     lns.append(line8)
    
#     lns.append(line9)
    
    labs = []
    
    mse_label = 'validation accuracy'
    
#     labs.append(provenance_approach + ":" + abs_dist)
    
    labs.append(provenance_opt_approach)
    
    labs.append(influence_approach)
    
#     labs.append(standard_lib_approach + ':' + mse_label)
#     
# #     labs.append(iteration_approach + ':' + test_acc)
#     
# #     labs.append(provenance_approach + ':' + mse_label)
#     
#     labs.append(provenance_opt_approach + ':' + mse_label)
#     
#     labs.append(influence_approach + ':' + mse_label)
    
#     labs.append(origin + ':' + test_acc)
    
#     ax1.legend(lns, labs, bbox_to_anchor=(0.05, 0.6), loc = 6, prop={'size': 15, 'weight':'bold'})
    
    ax1.legend(lns, labs, loc = 'best', prop={'size': 40, 'weight':'bold'})
    
#     ax2.legend([line3, line4], [provenance_approach + ":" + angle, influence_approach + ":" + angle], loc="best", prop={'size': 15})
    
#     ax1.xlabel('error_rate', fontsize = 15)
#     
#     ax1.ylabel('absolute distance', fontsize = 15)
    
#     ax = ax1.gca()
#     ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
#     ax.tick_params(axis = 'both', which = 'minor', labelsize = 2)
#     plt.figure(figsize=(3,4))
    
    plt.savefig(output_file_name, quality = 50, dpi = 300)
    
    
    if show_or_not:
        plt.show()



def draw_distance_figure_linear_regression_with_batch(excel_name, output_file_name, show_or_not, subset_ids):

    df1 = pd.read_excel(excel_name + '.xlsx', sheet_name='abs_err')
    
    df2 = pd.read_excel(excel_name + '.xlsx', sheet_name='test_accuracy')
        
    noise_rate = df1['noise_rate']
    
    prov_abs_errs = df1[provenance_label]
    
    prov_opt_abs_errs = df1[provenance_opt_label]
    
    influence_abs_errs = df1[influence_label]
    
    iteration_batch_abs_errs1 = df1[iteration_batch_label_1]
    
    iteration_batch_abs_errs2 = df1[iteration_batch_label_2]
    
    iteration_batch_abs_errs3 = df1[iteration_batch_label_3]
    
    
    
    test_acc_std = df2[standard_lib_label]
    
    test_acc_iter = df2[iteration_label]
    
    test_acc_prov = df2[provenance_label]
    
    test_acc_prov_opt = df2[provenance_opt_label]
    
    test_acc_inf = df2[influence_label]
    
    test_acc_iteration_batch_1 = df2[iteration_batch_label_1]
    
    test_acc_iteration_batch_2 = df2[iteration_batch_label_2]
    
    test_acc_iteration_batch_3 = df2[iteration_batch_label_3]
    
    test_acc_origin = df2[origin_label]
        
    
    
    
    
    
    
    lw = 4
    
    mw = 15
    
    print(noise_rate[1:])
    
    print(prov_abs_errs[1:])
    
    print(influence_abs_errs[1:])
    
    fig, ax1 = plt.subplots(figsize=(20,16))
    
    cmap = plt.get_cmap('jet_r')

    
#     figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    
    line1, = ax1.plot(noise_rate[subset_ids], prov_abs_errs[subset_ids], 'go:', linewidth=lw, markersize=mw)
    
    line2, = ax1.plot(noise_rate[subset_ids], prov_opt_abs_errs[subset_ids], 'k*:', linewidth=lw, markersize=mw)
    
    line3, = ax1.plot(noise_rate[subset_ids], influence_abs_errs[subset_ids], 'b^--', linewidth=lw, markersize=mw)
    
#     line4, = ax1.plot(noise_rate[subset_ids], iteration_batch_abs_errs1[subset_ids], color = cmap(10), marker = 'H', linewidth=lw, markersize=mw)
#     
#     line5, = ax1.plot(noise_rate[subset_ids], iteration_batch_abs_errs2[subset_ids], color = cmap(20), marker = 'H', linewidth=lw, markersize=mw)
#     
#     ax1.legend([line1, line2], [provenance_approach + ":" + abs_dist, influence_approach + ":" + abs_dist], loc="best", prop={'size': 15})

    
#     ax1.yscale('log')
    
    ax1.set_ylabel('distance', fontsize=30, weight = 'bold')
    
    ax1.set_yscale("log")
    
    ax1.set_xlabel('deletion rate', fontsize=30, weight = 'bold')
    
    ax1.tick_params(axis='both', labelsize = 25)
    
    plt.setp(ax1.get_xticklabels(), fontweight="bold")
    
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Mean Square Error', fontsize=30, weight = 'bold')  # we already handled the x-label with ax1
    line4, = ax2.plot(noise_rate[subset_ids], test_acc_std[subset_ids], 'rD-.', linewidth=lw, markersize=mw)
#     line5, = ax2.plot(noise_rate[subset_ids], test_acc_iter[subset_ids], 'c<-.', linewidth=lw, markersize=mw)
    line6, = ax2.plot(noise_rate[subset_ids], test_acc_prov[subset_ids], 'v-', linewidth=lw, markersize=mw)
    line7, = ax2.plot(noise_rate[subset_ids], test_acc_prov_opt[subset_ids], 'mx-', linewidth=lw, markersize=mw)
    line8, = ax2.plot(noise_rate[subset_ids], test_acc_inf[subset_ids], 'c>-', linewidth=lw, markersize=mw)
#     line9, = ax2.plot(noise_rate[subset_ids], test_acc_origin[subset_ids], 'rH-', linewidth=lw, markersize=mw)
    ax2.tick_params(axis='y', labelsize = 25)
    
    plt.setp(ax2.get_yticklabels(), fontweight="bold")
    
    lns = []
    
    lns.append(line1)
    
    lns.append(line2)
    
    lns.append(line3)
    
    lns.append(line4)
    
#     lns.append(line5)
    
    lns.append(line6)
    
    lns.append(line7)
    
    lns.append(line8)
    
#     lns.append(line9)
    
    labs = []
    
    mse_label = 'MSE'
    
    labs.append(provenance_approach + ":" + abs_dist)
    
    labs.append(provenance_opt_approach + ":" + abs_dist)
    
    labs.append(influence_approach + ":" + abs_dist)
    
    labs.append(standard_lib_approach + ':' + mse_label)
    
#     labs.append(iteration_approach + ':' + test_acc)
    
    labs.append(provenance_approach + ':' + mse_label)
    
    labs.append(provenance_opt_approach + ':' + mse_label)
    
    labs.append(influence_approach + ':' + mse_label)
    
#     labs.append(origin + ':' + test_acc)
    
#     ax1.legend(lns, labs, bbox_to_anchor=(0.05, 0.6), loc = 6, prop={'size': 15, 'weight':'bold'})
    
    ax1.legend(lns, labs, loc = 'best', prop={'size': 30, 'weight':'bold'})
    
#     ax2.legend([line3, line4], [provenance_approach + ":" + angle, influence_approach + ":" + angle], loc="best", prop={'size': 15})
    
#     ax1.xlabel('error_rate', fontsize = 15)
#     
#     ax1.ylabel('absolute distance', fontsize = 15)
    
#     ax = ax1.gca()
#     ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
#     ax.tick_params(axis = 'both', which = 'minor', labelsize = 2)
#     plt.figure(figsize=(3,4))
    
    plt.savefig(output_file_name, quality = 50, dpi = 300)
    
    
    if show_or_not:
        plt.show()



directory = '../../scripts/'

random_flipping = 'random_flipping'

change_data_value = 'change_data_value'

add_noise = 'add_noise'

epoch_num = 'epoch_num'

curr_type = change_data_value

# curr_type = change_data_value

# output_name = 'sgemm'
output_name = 'covtype'
# output_name = 'rcv'
# output_name = 'higgs'
# output_name = 'linear_regression_less'
# output_name = 'output_small_batch2'
# output_name = 'heart'

# output_name = 'sgemm'
# excel_name = directory + 'output_higgs_' + curr_type
# excel_name = directory + 'output_covtype_binary_' + curr_type
# excel_name = directory + 'skin/logistic_regression_sensitivity_analysis_skin_' + curr_type
# excel_name = directory + 'multi_logistic_regression_sensitivity_analysis_heart_' + curr_type 
# excel_name = directory + 'linear_regression_sensitivity_analysis_sgemm_' + curr_type
# excel_name = directory + 'output_' + curr_type 
excel_name = directory + 'output_cov_small_batch2_' + curr_type

# excel_name = directory + 'output_heartbeat_feature_errors_' + curr_type
# excel_name = directory + 'output_rcv1_' + curr_type
# excel_name = directory + 'output_higgs_' + curr_type
# excel_name = directory + 'output_linear_regression_less_features_' + curr_type


ids = list(range(9))

# draw_distance_figure_linear_regression1(excel_name, directory + output_name + '_'+ curr_type +'_distance.jpg', False, ids)
#      
# draw_distance_figure_linear_regression2(excel_name, directory + output_name + '_'+ curr_type +'_accuracy.jpg', False, ids)
# 
# 
# draw_time_figure_linear_regression(excel_name, directory + output_name + '_' + curr_type +'_time.jpg', False, ids)
# 
# draw_compare_pro_and_prov_opt_linear_regression(excel_name, directory + output_name + '_'+ curr_type +'_prov_opt_trade_off.jpg', False, ids)
    
# draw_distance_figure(excel_name, directory + output_name + '_'+ curr_type +'_distance.jpg', False, ids)    
# #          
# draw_accuracy_figure(excel_name, directory + output_name + '_' + curr_type +'_accuracy.jpg', False, ids)
# #    
draw_time_figure(excel_name, directory + output_name + '_' + curr_type +'_time.eps', False, ids)



# draw_distance_figure_sparse(excel_name, directory + output_name + '_'+ curr_type +'_distance.jpg', False, ids)    
#             
# draw_accuracy_figure_sparse(excel_name, directory + output_name + '_' + curr_type +'_accuracy.jpg', False, ids)
#        
# draw_time_figure_sparse(excel_name, directory + output_name + '_' + curr_type +'_time.jpg', False, ids)




curr_type = epoch_num

# excel_name = directory + 'logistic_regression_sensitivity_analysis_skin_change_data_value'
# excel_name = directory + 'linear_regression_sensitivity_analysis_sgemm_' + curr_type
# excel_name = directory + 'linear_regression_sensitivity_analysis_sgemm_' + curr_type 


# draw_time_figure_with_varied_epochs(excel_name, directory + 'skin_' + curr_type +'_time_varied_epochs.jpg', False, ids)

# draw_distance_figure(excel_name, directory + 'skin_change_data_value_distance.jpg', False)    
#      
# draw_accuracy_figure(excel_name, directory + 'skin_change_data_value_accuracy.jpg', False)
#  
# draw_time_figure(excel_name, directory + 'skin_change_data_time.jpg', False)





