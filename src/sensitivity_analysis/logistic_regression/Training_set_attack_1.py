import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.sparse import csc_matrix
from scipy.optimize import BFGS
from scipy.optimize import SR1
from sensitivity_analysis.Load_data import *
from sensitivity_analysis.logistic_regression.Logistic_regression import *
import sys
import random


def sigmoid_func(x):
    return 1/(1+np.exp(-x))

def sigmoid_func_der(x):
    return np.exp(-x)/((1+np.exp(-1))**2)


def sigmoid_func_second_der(x):
    return (np.exp(-2*x) - np.exp(-x))/((1+np.exp(-x))**3)


def loss_function(x):
    return np.log(1 + np.exp(-x))




def generate_data():
    X1 = np.random.multivariate_normal(mean = [1, 2], cov = [[0.23, 0],[0, 0.54]], size = 10)
    
    Y1 = np.zeros(10)
    
    Y1[:] = 1
    
    X2 = np.random.multivariate_normal(mean = [-1, 2], cov = [[0.03, 0],[0, 0.12]], size = 15)
    
    Y2 = np.zeros(15)
    
    Y2[:] = -1
    
#     print(X1)
#     
#     print(X2)
    
    X = np.concatenate((X1, X2), axis = 0)
    
    Y = np.concatenate((Y1, Y2), axis = 0)
    
    return X, Y


def compute_accuracy(X, Y, validate_X, validate_Y):
    x0 = np.zeros(X.shape[1])
    
    print('start computing accuracy')
    
    def lost_function_der1(w):
        
        theta = w
        
        prod = np.matmul(X, theta)*Y
        
        x_y_prod = X*Y.reshape(-1,1)
        
        vecfunc = np.vectorize(sigmoid_func)
        
        der = np.sum(x_y_prod*(1 - vecfunc(prod)).reshape(-1,1), axis = 0)
        
        
        
#         for i in range(validate_X.shape[0]):
#             der[0:dim] += x_y_prod[i]*(1-sigmoid_func(prod[i]))
            
            
    #     x_der = np.zeros(x.shape)
    
    #     return np.concatenate((-res, x_der), axis = 0).reshape(-1)
        return -der + beta*theta
    
    def lost_function1(w):
        
        res = 0
        
        dim = X.shape[1]
        
        theta = w[0:dim]
        
        x = w[(dim):]
        
        
        prod = np.matmul(X, theta)*Y
        
        vecfunc = np.vectorize(loss_function)
        
        res = np.sum(vecfunc(prod)) + beta/2*np.dot(theta, theta)
        
        
#         for i in range(X.shape[0]):
#             res += np.log(1 + np.exp(-Y[i]*np.dot(X[i], theta)))
        print(lost_function_der1(w))
            
        print('loss::', res)
        
        return res
    
    
#     res = minimize(lost_function1, x0, method='nelder-mead',
#                    options={'xtol': 0.001, 'maxiter': 100, 'disp': True})
#     res = minimize(lost_function1, x0, method='BFGS', jac=lost_function_der1,
#                options={'disp': True, 'gtol': 0.001, 'maxiter': 100})
    res = minimize(lost_function1, x0, method='trust-constr',  jac=lost_function_der1, hess=SR1(), options={'verbose': 1, 'xtol': 0.001, 'gtol': 0.001, 'barrier_tol': 0.001, 'maxiter': 200})

    
    
    print(res.x)
    
    acc = np.sum(np.matmul(validate_X, res.x)*validate_Y > 0)*1.0/validate_X.shape[0]
    
    
    print('accuracy::', acc)
    
    print('derivative::', lost_function_der1(res.x))
    
    return acc
    
    
    
# X, Y = generate_data()

file_name = sys.argv[1]

X, Y = load_data_numpy(True, file_name)

X = extended_by_constant_terms_numpy(X)


Y = Y.reshape(-1)

# X = normalize(X)

validate_X = np.copy(X)

validate_Y = np.copy(Y)


entired_X = np.copy(X)

entired_Y = np.copy(Y)



noise_data_num = int(X.shape[0]*0.01)


print('noise_data_num::', noise_data_num)

print(X.shape[1])

unique_y =np.unique(Y)


expected_label = 0

min_count = Y.shape[0]


for i in range(unique_y.shape[0]):
    count = np.sum(Y == unique_y[i])
    if count < min_count:
        min_count = count
        
        expected_label = unique_y[i]
        
    
    




compute_accuracy(validate_X, validate_Y, validate_X, validate_Y)

batch_num = 1


for i in range(1):
# for i in range(1):    
    
    y = np.zeros(batch_num)
    
    
    for i in range(batch_num):
    
        id = random.randint(0, unique_y.shape[0]-1)
        y[i] = expected_label
#         y[i] = expected_label
    
    # y = np.array([1,-1,1,-1])
    
    def lost_function(w):
        
        res = 0
        
        dim = validate_X.shape[1]
        
        theta = w[0:dim]
        
        x = w[(dim):].reshape(-1,dim)
        
        prod = np.matmul(validate_X, theta)*validate_Y
        
        
        vecfunc = np.vectorize(loss_function)
        
        res = np.sum(vecfunc(prod)) + beta/2*np.dot(theta, theta)
        
        
#         for i in range(validate_X.shape[0]):
#             res += np.log(1 + np.exp(-prod[i]))
        print('training_loss::', -res)
            
        
        
        return -res
    
    
    def lost_function0(w):
        res = 0
        
        dim = validate_X.shape[1]
        
        theta = w[0:dim]
        
        x = w[(dim):].reshape(-1,dim)
        
        prod = np.matmul(validate_X, theta)*validate_Y
        
        
        res = np.sum(prod>=0)
        
        
#         vecfunc = np.vectorize(loss_function)
#         
#         res = np.sum(vecfunc(prod))/validate_X.shape[0] + beta/2*np.dot(theta, theta)
        
        
#         for i in range(validate_X.shape[0]):
#             res += np.log(1 + np.exp(-prod[i]))
        print('training_loss::', res)
            
        
        
        return res
        
        
    
    def lost_function_der(w):
        
        dim = X.shape[1]
        
        theta = w[0:dim]
        
        x = w[(dim):].reshape(-1,dim)
        
        der = np.zeros_like(w)
        
        prod = np.matmul(validate_X, theta)*validate_Y
        
        x_y_prod = validate_X*validate_Y.reshape(-1,1)
        
        vecfunc = np.vectorize(sigmoid_func)
        
        der[0:dim] += np.sum(x_y_prod*(1 - vecfunc(prod)).reshape(-1,1), axis = 0) - beta*theta
        
        
        
#         for i in range(validate_X.shape[0]):
#             der[0:dim] += x_y_prod[i]*(1-sigmoid_func(prod[i]))
            
            
    #     x_der = np.zeros(x.shape)
    
    #     return np.concatenate((-res, x_der), axis = 0).reshape(-1)
        return der
    
    
    def lost_function_hessian(w):
         
        dim = X.shape[1]
        
        theta = w[0:dim]
        
        x = w[(dim):] 
        
        res = np.zeros([w.shape[0], w.shape[0]])
        
        
        exponet = np.matmul(validate_X, theta)*validate_Y
        
        vecfunc = np.vectorize(sigmoid_func_der)
        
        sigmoid_der_res = vecfunc(exponet)
     
     
        for i in range(validate_X.shape[0]):
            
            res[0:theta.shape[0], 0:theta.shape[0]] -= sigmoid_der_res[i]*(np.matmul(validate_X[i].reshape(dim, 1), validate_X[i].reshape(1, dim)))
            
        
        
            
        return res
            
    
    def constrain(w):
        dim = X.shape[1]
        
        theta = w[0:dim]
        
        x = w[dim:]
        
        x = x.reshape(-1, dim)
        
        
        
        curr_X = np.concatenate((entired_X, x), axis = 0)
        
        curr_Y = np.concatenate((entired_Y, y), axis = 0)
        
    #     print(curr_X.shape)
    #     
    #     print(curr_Y.shape)
        
        X_Y_theta_prod = np.matmul(curr_X, theta)*curr_Y
        
        X_Y_prod = np.multiply(curr_X, curr_Y.reshape(-1,1))
        
        res = np.zeros(dim)
        
        
        for i in range(curr_X.shape[0]):
            res += X_Y_prod[i]*(1-sigmoid_func(X_Y_theta_prod[i]))
        
        
        return res.reshape(-1) - beta*theta
    
    
    def constrain_der(w):
        dim = X.shape[1]
        
        theta = w[0:dim]
        
        x = w[(dim):]
        
        x = x.reshape(-1, dim)
        
        
        
        curr_X = np.concatenate((entired_X, x), axis = 0)
        
    #     print(theta.shape)
    #     
    #     print(w.shape)
        
        curr_Y = np.concatenate((entired_Y, y), axis = 0)
        
        res = np.zeros((theta.shape[0], w.shape[0]))
        
        exponet = np.matmul(curr_X, theta)*curr_Y
        
        
        for i in range(curr_X.shape[0]):
            res[0:theta.shape[0], 0:theta.shape[0]] += np.exp(exponet[i])*(sigmoid_func(exponet[i]))*(sigmoid_func(exponet[i]))*(np.matmul(curr_X[i].reshape(dim, 1), curr_X[i].reshape(1, dim)))
            
#         res[0:theta.shape[0], 0:theta.shape[0]] -beta*np.eye(theta.shape[0])
            
        for i in range(x.shape[0]):    
    #         res[0:theta.shape[0], theta.shape[0]*i:theta.shape[0]*(i+1)] +=  np.exp(exponet[i + X.shape[0]])*(sigmoid_func(exponet[i + X.shape[0]]))*(sigmoid_func(exponet[i + X.shape[0]]))*(np.matmul(x[i].reshape(dim, 1), x[i].reshape(1, dim)))
    #         res[0:theta.shape[0], theta.shape[0]*(i+1):theta.shape[0]*(i+2)] += y[i]*np.eye(dim)*(1-sigmoid_func(exponet[i + X.shape[0]])) - np.matmul(np.matmul(x[i].reshape(dim, 1), theta.reshape[1, dim]), np.exp(-exponet[i+X.shape[0]]))/(sigmoid_func(exponet[i+X.shape[0]])*sigmoid_func(exponet[i+X.shape[0]]))
            res[0:theta.shape[0], theta.shape[0]*(i+1):theta.shape[0]*(i+2)] = y[i]*(1-sigmoid_func(exponet[i]))*np.eye(theta.shape[0]) - sigmoid_func_der(exponet[i])*np.matmul(x[i].reshape(-1,1), theta.reshape(1,-1))
            
            
            
            
            
            
        return res
    
    
    def constrain_hessian(w, v):
        dim = X.shape[1]
        
        theta = w[0:dim]
        
        x = w[(dim):]
        
        x = x.reshape(-1, dim)
        
        
        
        curr_X = np.concatenate((entired_X, x), axis = 0)
        
        curr_Y = np.concatenate((entired_Y, y), axis = 0)
        
        res = np.zeros(dim, w.shape[0], w.shape[0])
        
        exponet = np.matmul(curr_X, theta)*curr_Y
        
        for j in range(dim):
            for i in range(curr_X.shape[0]):
    #             res[j, 0:theta.shape[0], 0:theta.shape[0]] -= curr_X[i][j]*(np.exp(-exponet[i])*(-curr_Y[i])*(1+np.exp(-exponet[i])) + 2*np.exp(-2*exponet[i])*curr_Y[i])/(1+np.exp())
                res[j, 0:theta.shape[0], 0:theta.shape[0]] -= curr_X[i][j]*curr_Y[i]*sigmoid_func_second_der(exponet[i])*np.matmul(curr_X[i].reshape(dim, 1), curr_X[i].reshape(1, dim))#(np.exp(-exponet[i])*(-curr_Y[i])*(1+np.exp(-exponet[i])) + 2*np.exp(-2*exponet[i])*curr_Y[i])/(1+np.exp())  
        
        for j in range(dim):
            one_id_vector = np.zeros(dim)
            one_id_vector[j] = 1
            for i in range(x.shape[0]):
                res[j, 0:theta.shape[0], theta.shape[0]*(i+1):theta.shape[0]*(i+2)] += x[i][j]*y[i]*sigmoid_func_second_der(exponet[i])*np.matmul(theta.reshape(dim, 1), x[i].reshape(1, dim))
                res[j, 0:theta.shape[0], theta.shape[0]*(i+1):theta.shape[0]*(i+2)] += x[i][j]*sigmoid_func_der(exponet[i])*np.eye(theta.shape[0])
                
                res[j, 0:theta.shape[0], theta.shape[0]*(i+1):theta.shape[0]*(i+2)] += sigmoid_func_der(exponet[i]) * np.matmul(one_id_vector.reshape(-1,1), x.reshape(1,-1)) 
                
        
        for j in range(dim):
            one_id_vector = np.zeros(dim)
            one_id_vector[j] = 1
            for i in range(x.shape[0]):
                
                
                
                res[j, theta.shape[0]*(i+1):theta.shape[0]*(i+2), 0:theta.shape[0]] -= sigmoid_func_der(exponet[i])*np.matmul(x.reshape(-1,1), one_id_vector.reshape(1, -1))
                res[j, theta.shape[0]*(i+1):theta.shape[0]*(i+2), 0:theta.shape[0]] -= sigmoid_func_second_der(exponet[i])*y[i]*x[i][j]*np.matmul(x[i].reshape(-1,1), theta.reshape(1,-1))
                res[j, theta.shape[0]*(i+1):theta.shape[0]*(i+2), 0:theta.shape[0]] -= sigmoid_func_der(exponet[i])*x[i][j]*np.eye(theta.shape[0])
        
        
        
        for j in range(dim):
            one_id_vector = np.zeros(dim)
            one_id_vector[j] = 1
            for i in range(x.shape[0]):
                res[j, theta.shape[0]*(i+1):theta.shape[0]*(i+2), theta.shape[0]*(i+1):theta.shape[0]*(i+2)] -= sigmoid_func_der(exponet[i])*np.matmul(theta.reshape(-1,1), one_id_vector.reshape(1,-1))
                res[j, theta.shape[0]*(i+1):theta.shape[0]*(i+2), theta.shape[0]*(i+1):theta.shape[0]*(i+2)] -= sigmoid_func_second_der(exponet[i])*y[i]*x[i][j]*np.matmul(theta.reshape(-1,1), theta.reshape(1,-1))
                res[j, theta.shape[0]*(i+1):theta.shape[0]*(i+2), theta.shape[0]*(i+1):theta.shape[0]*(i+2)] -= sigmoid_func_der(exponet[i])*x[i][j]*np.eye(theta.shape[0])
                
                
                
                
        
        
        final_res = np.zeros(res.shape[1], res.shape[1])
        
        for i in range(res.shape[0]):
            final_res += v[i]*res[i]
            
        return final_res
    
    x0 = np.zeros(X.shape[1]*(batch_num + 1))
    
    
    
    x0[X.shape[1]:] = 10
    
    lbs = np.zeros(X.shape[1]*(batch_num + 1))
    
    ubs = np.zeros(X.shape[1]*(batch_num + 1))
    
    
    lbs[0:X.shape[1]] = -np.inf
    
    lbs[X.shape[1]:] = 0
    
    ubs[0:X.shape[1]] = np.inf
    
    ubs[X.shape[1]:] = 1
    
    
    
    # bounds = Bounds([-np.inf, -np.inf, 0, 0,0,0,0,0,0,0], [np.inf, np.inf, 1, 1,1 ,1,1,1,1,1])
    
    bounds = Bounds(lbs, ubs)
        
#     nonlinear_constraint = NonlinearConstraint(constrain, 0, 0, jac=constrain_der, hess=BFGS())
    nonlinear_constraint = NonlinearConstraint(constrain, 0, 0, jac='2-point', hess=BFGS())

    
#     res = minimize(lost_function, x0, method='trust-constr',  jac=lost_function_der, hess=lost_function_hessian,
#                    constraints=[nonlinear_constraint],
#                    options={'verbose': 1}, bounds=bounds)
    
    
    res = minimize(lost_function, x0, method='trust-constr',  jac='2-point', hess=SR1(),
               constraints=[nonlinear_constraint],
               options={'verbose': 1, 'xtol': 1e-8, 'gtol': 1e-8, 'barrier_tol': 1e-8, 'maxiter': 200})
    
    
    # res = minimize(lost_function, x0, method='trust-constr', jac=lost_function_der, hess=lost_function_hessian,
    #                constraints=[nonlinear_constraint],
    #                options={'verbose': 1}, bounds=bounds)
    
    
    # res = minimize(lost_function, x0, method='BFGS', jac=lost_function_der,  options={'disp': True})
    
    print(res.x)
    
    
    print(constrain(res.x))


    theta = res.x[0:X.shape[1]]
     
    x = res.x[X.shape[1]:].reshape(-1, X.shape[1])
#     
#     for i in range(X.shape[0]):
#         prod = np.dot(theta, X[i])
#         
#         print(prod*Y[i] > 0)
    
    copied_x = np.repeat(x, 1, axis = 0)
    
    copied_y = np.repeat(y, 1, axis = 0)
    
    
    
    curr_X = np.concatenate((X, copied_x), axis = 0)
    
    x0 = np.zeros(X.shape[1])
    
    
    curr_Y = np.concatenate((Y,copied_y), axis = 0)
    
    
    print(curr_X.shape, curr_Y.shape, entired_X.shape, entired_Y.shape)
    
    print('new_added_data::')
    
    print(x)
    
    
    compute_accuracy(curr_X, curr_Y, validate_X, validate_Y)
    
    
#     def lost_function2(w):
#         
#         res = 0
#         
#         dim = X.shape[1]
#         
#         theta = w[0:dim]
#         
#         x = w[(dim):]
#         
#         
#         for i in range(curr_X.shape[0]):
#             res += np.log(1 + np.exp(-curr_Y[i]*np.dot(curr_X[i], theta)))
#         
#             
#         
#         
#         return res
#     
#     
#     
#     res = minimize(lost_function2, x0, method='nelder-mead',
#                    options={'xtol': 1e-8, 'disp': True})
#     
#     
#     print(res.x)
#     
#     for i in range(X.shape[0]):
#         prod = np.dot(res.x, X[i])
#         
#         print(prod*Y[i] > 0)
        
        
        
    entired_X = np.concatenate((entired_X, x), axis = 0)
    
    entired_Y = np.concatenate((entired_Y, y), axis = 0)
    
    
    
np.save('modified_X', entired_X)

np.save('modified_Y', entired_Y)
# def rosen(x):
#     """The Rosenbrock function"""
#     
#     
#     
#     
#     
#     
#     return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
# 
# 
# 
# def rosen_der(x):
#     xm = x[1:-1]
#     xm_m1 = x[:-2]
#     xm_p1 = x[2:]
#     der = np.zeros_like(x)
#     der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
#     der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
#     der[-1] = 200*(x[-1]-x[-2]**2)
#     return der
# 
# def rosen_hess(x):
#     x = np.asarray(x)
#     H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
#     diagonal = np.zeros_like(x)
#     diagonal[0] = 1200*x[0]**2-400*x[1]+2
#     diagonal[-1] = 200
#     diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
#     H = H + np.diag(diagonal)
#     return H
# 
# def rosen_hess_p(x, p):
#     x = np.asarray(x)
#     Hp = np.zeros_like(x)
#     Hp[0] = (1200*x[0]**2 - 400*x[1] + 2)*p[0] - 400*x[0]*p[1]
#     Hp[1:-1] = -400*x[:-2]*p[:-2]+(202+1200*x[1:-1]**2-400*x[2:])*p[1:-1] \
#                -400*x[1:-1]*p[2:]
#     Hp[-1] = -400*x[-2]*p[-2] + 200*p[-1]
#     return Hp
# 
# def cons_f(x):
#     return [x[0]**2 + x[1], x[0]**2 - x[1], x[2] + x[1]]
# def cons_J(x):
#     return [[2*x[0], 1, 0], [2*x[0], -1, 0], [0, 1, 1]]
# def cons_H(x, v):
#     return v[0]*np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]]) + v[1]*np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]]) + v[2]*np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
# 
# def cons_H_sparse(x, v):
#     return v[0]*csc_matrix([[2, 0], [0, 0]]) + v[1]*csc_matrix([[2, 0], [0, 0]])
# 
# bounds = Bounds([0, -0.5, 0], [1.0, 2.0, 1.5])
# linear_constraint = LinearConstraint([[1, 2, 0], [2, 1, 1]], [-np.inf, 1], [1, 1])
# nonlinear_constraint = NonlinearConstraint(cons_f, [-np.inf, -np.inf, -1], [1,2, 5], jac=cons_J, hess=cons_H)
# # nonlinear_constraint = NonlinearConstraint(cons_f, [-np.inf, -5], [1,2],
# #                                            jac=cons_J, hess=cons_H_sparse)
# 
# 
# x0 = np.array([0.5, 0, 1])
# 
# res = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hess=rosen_hess,
#                constraints=[linear_constraint, nonlinear_constraint],
#                options={'verbose': 1}, bounds=bounds)
# 
# 
# 
# print(res.x)


