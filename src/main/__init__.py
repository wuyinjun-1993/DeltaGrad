import time
from torch.autograd import Variable

from torch import nn, optim
import torch

import numpy as np


def poly_horner(A, x):
    p = A[-1]
    i = len(A) - 2
    while i >= 0:
        p = p * x + A[i]
        i -= 1
    return p


x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

x_train = torch.from_numpy(x_train)

y_train = torch.from_numpy(y_train)

X = Variable(x_train)

Y = Variable(y_train)

A =[3,2,1,4,5,2,3]

res = torch.matmul(torch.transpose(X, 0, 1), X)/700


start = time.time()

for i in range(10000):
    torch.nn.functional.sigmoid(res)

end = time.time()
print(end - start)

start = time.time()

t1 = torch.pow(res, 3) 
t2 = torch.pow(t1, 3)

for i in range(10000):
#     torch.pow(res, 10) 
    poly_horner(A, res)
    
     
end = time.time()
print(end - start)

print(torch.pow(res, 9))
print(torch.exp(res))


# start = time.time()
# print("hello")
# end = time.time()
# print(end - start)

# thisset.update(set2)
# 
# 
# import torch
# 
# a = torch.ones(2)
# 
# b = torch.eye(2)
# 
# 
# print(torch.add(a, b))