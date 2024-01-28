##################### SVM : hinge loss q=1  ###################### 
###################################################

import torch
import numpy as np

import os
import sys
current_dir = os.path.dirname(__file__)  # current script's directory
parent_dir = os.path.join(current_dir, '..')  # parent directory
sys.path.append(parent_dir) 
from utils import equals, from_DF_to_DFpqmn, get_DFpqmn

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

decimals=5

######### Hyperparams
n, d, c = 3, 5, 4 #  number of samples, input dim, output dim
beta = 0.7 # temperature
#tau = 0.5 # uniform regularizer strengths (noise)
gamma = 0.5 # l2 regularization
zeta = 1.0 # margin

######### Input, labels, P^* and P
X = torch.randn(n, d, requires_grad=False).to(device) # (n, d)
Y = torch.empty(n, dtype=torch.long).random_(c).to(device) # (n,)
P_start = torch.nn.functional.one_hot(Y, num_classes=c).float().to(device) # (n, c)
#P = tau * P_start + (1-tau)/c #* torch.ones_like(n, c) # (n, c)

######### Model
V = torch.randn(c, d, requires_grad=True, device=device) # (c, d)
######### Model output
hat_Y = beta * (X @ V.T) # (n, c)
hat_Y.retain_grad() # for Grad(Y)

######### Margin
hat_Y_iYi = hat_Y[torch.arange(n), Y].unsqueeze(1) # (n, 1)
margins = hat_Y - hat_Y_iYi + zeta # (n, c)
max_margins = torch.maximum(torch.zeros_like(margins), margins) # (n, c)
max_margins[torch.arange(n), Y] = 0 # (n, c)

######### Loss 
loss = (1/n) * max_margins.sum() + 0.5*gamma*torch.trace(V.T@V)

# OR
# print(loss)
# loss = 0
# for i in range(n) :
#     for j in range(c) :
#         loss += int(j!=Y[i]) *  max(0, hat_Y[i][j] - hat_Y[i][Y[i]] + zeta)
# loss = (1/n) * loss + 0.5*gamma*torch.trace(V.T@V)
# print(loss)

######### Loss gradient
loss.backward(retain_graph=True)
hat_Ygrad = hat_Y.grad.detach()
Vgrad = V.grad.detach()

######### M

# M = torch.zeros_like(margins) # (n, c)
# for i in range(n) :
#     for j in range(c) :
#         if j == Y[i] :
#             M[i][j] = int(hat_Y[i][j] - hat_Y[i][Y[i]] + zeta >= 0)
#             s = 0
#             for k in range(c) : s += int(hat_Y[i][k] - hat_Y[i][Y[i]] + zeta >= 0)
#             M[i][j] = M[i][j] - s
#         else :
#             M[i][j] = int(hat_Y[i][j] - hat_Y[i][Y[i]] + zeta >= 0)

## OR

M = 1.0*(margins >= 0) # (n, c)
M[torch.arange(n), Y] -= torch.sum(M, dim=1) # (n,)

######### Gradient Y

dhat_Y = (1/n) * M # (n, c)
#dhat_Y.retain_grad() # for Hess(Y)
print(equals(hat_Ygrad, dhat_Y, dec=decimals))#, "\n", hat_Ygrad,"\n", dhat_Y,"\n")

######### Gradient V

dV = M.T @ X #  (c, n)x(n, d)

## OR
# dV = torch.zeros_like(V)
# for i in range(n) :
#     for j in range(c) :
#         if j == Y[i] : continue
#         for l in range(c) : dV[l] += (int(j==l) - int(Y[i]==l)) * X[i] * int(hat_Y[i][j] - hat_Y[i][Y[i]] + zeta >= 0)

dV = ( beta/n) * dV + gamma*V
print(equals(Vgrad, dV, dec=decimals))#, "\n", Vgrad,"\n", dV,"\n")

######### Hessian Y
# Hess_Y = torch.zeros(n*c, n*c).to(device) # (nc, nc)
# DFYY1 = from_DF_to_DFpqmn(DF=Hess_Y, m=n, n=c, p=n, q=c) # (n, c, n, c)

# hat_Y.grad.zero_()
# DFYY2 = get_DFpqmn(F=dhat_Y, X=hat_Y, p=n, q=c) # (n, c, n, c)
# print(equals(DFYY1, DFYY2, dec=decimals))#, "\n", DFYY1,"\n", DFYY2,"\n")

######### Hessian V
Icd = torch.eye(c*d).to(device)
Hess_V = gamma*Icd  # (cd, cd)
DFVV1 = from_DF_to_DFpqmn(DF=Hess_V, m=c, n=d, p=c, q=d) # (c, d, c, d)

V.grad.zero_()
DFVV2 = get_DFpqmn(F=dV, X=V, p=c, q=d) # (c, d, c, d)
print(equals(DFVV1, DFVV2, dec=decimals))#, "\n", DFWW1,"\n", DFWW2,"\n")