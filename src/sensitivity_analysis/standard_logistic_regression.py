'''
Created on Feb 5, 2019

'''
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sensitivity_analysis.Load_data import load_data

# N = 100
# D = 2
# 
# X = np.random.randn(N,D)*2
# 
# print(X)
# 
# # center the first N/2 points at (-2,-2)
# X[:int(N/2),:] = X[:int(N/2),:] - 2*np.ones((int(N/2),D))
# 
# # center the last N/2 points at (2, 2)
# X[int(N/2):,:] = X[int(N/2):,:] + 2*np.ones((int(N/2),D))
# 
# # labels: first N/2 are 0, last N/2 are 1
# T = np.array([0]*(int(N/2)) + [1]*(int(N/2))).reshape(100,1)
# 
# x_data = Variable(torch.Tensor(X))
# y_data = Variable(torch.Tensor(T))

x_data, y_data = load_data(True)

device = torch.device('cpu')

dim = x_data.shape

class logistic_regressor_parameter:
    def __init__(self, theta):
        self.theta = theta

w = torch.zeros(dim[1], 1, device=device, requires_grad=True, dtype = torch.double)
# w2 = torch.randn(H, D_out, device=device, requires_grad=True)

para = logistic_regressor_parameter(w)


def binary_cross_entropy_loss(X, Y, w):
    sigmoid_res = torch.stack(list(map(torch.sigmoid, torch.mm(X, w))))
    
    res = torch.sum(sigmoid_res*Y + (1 - sigmoid_res)*(1-Y))
    
    return res


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(x_data.shape[1], 1).type(torch.DoubleTensor) # 2 in and 1 out
        
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
# Our model    
# model = Model()

# criterion = torch.nn.BCELoss(size_average=True)

lr = 0.0002

# optimizer = torch.optim.SGD(model.parameters(), lr = 0.0002, )
# torch.optim.SGD()

# Training loop
for epoch in range(200):
    # Forward pass: Compute predicted y by passing x to the model
#     y_pred = model(x_data)

    loss = binary_cross_entropy_loss(x_data, y_data, para.theta)
    
    
    # Compute and print loss
#     loss = criterion(y_pred, y_data)
    print(loss)
#     print(epoch, loss, list(model.parameters()))
    
    loss.backward()
    
    with torch.no_grad():
#         print(model.parameters().grad)
#         print(w.grad)
        para.theta -= lr * para.theta.grad
        para.theta.grad.zero_()
    
    # Zero gradients, perform a backward pass, and update the weights.
#     optimizer.zero_grad()
#     optimizer.step()

# for f in model.parameters():
#     print('data is')
#     print(f.data)
#     print(f.grad)

# w = list(model.parameters())
# w0 = w[0].data.numpy()
# w1 = w[1].data.numpy()

import matplotlib.pyplot as plt

print('model parameter', para.theta)
# plot the data and separating line
# plt.scatter(X[:,0], X[:,1], c=T.reshape(N), s=100, alpha=0.5)
# x_axis = np.linspace(-6, 6, 100)
# y_axis = -(w1[0] + x_axis*w0[0][0]) / w0[0][1]
# line_up, = plt.plot(x_axis, y_axis,'r--', label='gradient descent')
# plt.legend(handles=[line_up])
# plt.xlabel('X(1)')
# plt.ylabel('X(2)')
# plt.show()