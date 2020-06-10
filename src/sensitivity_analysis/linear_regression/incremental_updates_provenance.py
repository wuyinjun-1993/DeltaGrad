'''
Created on Mar 15, 2019

'''

import torch

import sys, os

from scipy.stats import t

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from sensitivity_analysis.linear_regression.Linear_regression import * 
    from data_IO.Load_data import *
    from sensitivity_analysis.linear_regression.utils import *
    from sensitivity_analysis.linear_regression.evaluating_test_samples import *
except ImportError:
    from Linear_regression import * 
    from Load_data import *
    from utils import *
    from evaluating_test_samples import *
    
    
    

def statistic_significance_test(model_standard_lib, model_provenance, update_X, update_Y, update_X_prod_inverse):
    
    sse = torch.pow(torch.mm(update_X, model_standard_lib) - update_Y, 2)
    
    sse = torch.sum(sse, 0)
    
    s_square = sse.view(1, update_Y.shape[1])/(update_X.shape[0] - (update_X.shape[1] + 1))
    
    v_list = torch.diag(update_X_prod_inverse).view(update_X.shape[1], 1)
    
    df = (update_X.shape[0] - (update_X.shape[1] + 1))
    
    t_coeff = t.ppf(0.9999, df)
    
    model_upper_bound = model_standard_lib + t_coeff * torch.sqrt(v_list*s_square)
    
    model_lower_bound = model_standard_lib - t_coeff * torch.sqrt(v_list*s_square)
    
    print(model_provenance < model_upper_bound)
    
    print(model_provenance > model_lower_bound)
    






if __name__ == '__main__':
    
    
    configs = load_config_data(config_file)
    
    git_ignore_folder = configs['git_ignore_folder']
    
    
    sys_args = sys.argv

    
    alpha = torch.load(git_ignore_folder + 'alpha')
    
    beta = torch.load(git_ignore_folder + 'beta')
    
    opt = bool(int(sys_args[1]))
    
    X = torch.load(git_ignore_folder+'noise_X')
    
    Y = torch.load(git_ignore_folder+'noise_Y')
    
    num_of_output = Y.shape[1]
    
    dim = X.shape
    
    max_epoch = torch.load(git_ignore_folder+'epoch')
    
    delta_data_ids = torch.load(git_ignore_folder+'noise_data_ids')
    
    delta_data_ids = delta_data_ids.type(torch.long)
    
    X_Y_mult = torch.load(git_ignore_folder+'X_Y_mult')
    
    X_prod = torch.load(git_ignore_folder + 'X_prod')
    
    
    M = torch.load(git_ignore_folder + 'eigen_vectors')
    
    M_inverse = torch.load(git_ignore_folder + 'eigen_vectors_inverse')
    
    s = torch.load(git_ignore_folder + 'eigen_values')
    

#     delta_data_ids = torch.load(git_ignore_folder+'delta_data_ids')
    
    print(delta_data_ids.shape[0])
    
    print(X.shape[0])
    
    update_X, selected_rows = get_subset_training_data(X, X.shape, delta_data_ids)
    
    update_Y, s_rows = get_subset_training_data(Y, Y.shape, delta_data_ids)
    
    init_theta = initialize(update_X.shape, num_of_output)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)
    
    M_inverse_times_theta = torch.mm(M_inverse, init_theta.theta)
    
    t1 = time.time()
    
    delta_X = Variable(X[delta_data_ids])
     
    delta_Y = Variable(Y[delta_data_ids])

    
#     update_x_sum_by_class = compute_x_sum_by_class(update_X, update_Y, num_class, update_X.shape)
#     update_X_Y_mult = update_X.mul(update_Y)
    #     res1 = update_model_parameters_from_the_scratch(update_X, update_Y)dim, theta,  X, Y, X_sum_by_class, num_class
    if opt:
#         res2 = linear_regression_provenance(X_prod - torch.mm(torch.t(delta_X), delta_X), X_Y_mult - torch.mm(torch.t(delta_X), delta_Y), update_X.shape, init_theta, max_epoch, alpha, beta)
#         res2, expected_s, exp_res1, exp_res2 = linear_regression_provenance_opt(s, M, M_inverse, X_prod,X, X_prod - torch.mm(torch.t(delta_X), delta_X), X_Y_mult - torch.mm(torch.t(delta_X), delta_Y), update_X.shape, init_theta, max_epoch, alpha, beta)
        
        res2 = linear_regression_provenance_opt2(s, M, M_inverse, X_prod, X, delta_X, X_Y_mult - torch.mm(torch.t(delta_X), delta_Y), update_X.shape, init_theta, max_epoch, alpha, beta, M_inverse_times_theta)
    else:
        res2 = linear_regression_iteration(X_prod - torch.mm(torch.t(delta_X), delta_X), X_Y_mult - torch.mm(torch.t(delta_X), delta_Y), update_X.shape, init_theta, max_epoch, alpha, beta)#(X.shape, init_theta, update_X_Y_mult, max_epoch)        
#     res2 = linear_regression_provenance(torch.mm(torch.t(update_X), update_X), torch.mm(torch.t(update_X), update_Y), dim, init_theta, max_epoch, alpha, beta)#(X.shape, init_theta, update_X_Y_mult, max_epoch)


    
    t2 = time.time()
    
    model_standard_lib = torch.load(git_ignore_folder+'model_standard_lib')
    
    torch.save(res2, git_ignore_folder+'model_provenance')
    
    print('training_time_provenance::', t2 - t1)
    
    print(torch.norm(model_standard_lib - res2))
    
    
    test_X = torch.load(git_ignore_folder+'test_X')
    
    test_Y = torch.load(git_ignore_folder+'test_Y')
    
    print('training_accuracy::', compute_accuracy2(update_X, update_Y, res2))
    
    print('test_accuracy::', compute_accuracy2(test_X, test_Y, res2))
    
    
    error = torch.norm(res2 - model_standard_lib)/torch.norm(model_standard_lib)
#     cos_sim = torch.dot(torch.reshape(res2, [-1]), torch.reshape(model_standard_lib, [-1]))/(torch.norm(torch.reshape(res2, [-1]))*torch.norm(torch.reshape(model_standard_lib, [-1])))
    
    print('absolute_error::', torch.norm(res2 - model_standard_lib))
    
    print('relative_error::', error)
    
#     print('angle::', cos_sim)

    print('angle::', torch.dot(res2.view(-1), model_standard_lib.view(-1))/(torch.norm(res2.view(-1))*torch.norm(model_standard_lib.view(-1))))


    update_X_prod_inverse = torch.inverse(torch.mm(torch.t(update_X), update_X))

#     statistic_significance_test(model_standard_lib, res2, update_X, update_Y, update_X_prod_inverse)
    
#     print(model_standard_lib)
    
#     print(torch.sum(torch.nonzero((res2 > 0) == (model_standard_lib > 0))))
#     
#     print((res2 < 0) == (model_standard_lib < 0) )
#     
#     get_relative_change(res2, model_standard_lib)
    
    print(torch.nonzero(res2*model_standard_lib > 0).shape)
    
    get_relative_change(res2, model_standard_lib)
    