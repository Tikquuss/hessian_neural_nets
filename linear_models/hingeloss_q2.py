##################### SVM : hinge loss q=2  ###################### 
###################################################

import torch
import numpy as np

import os
import sys
current_dir = os.path.dirname(__file__)  # current script's directory
parent_dir = os.path.join(current_dir, '..')  # parent directory
sys.path.append(parent_dir) 
from utils import equals, from_DF_to_DFpqmn, get_DFpqmn
from utils import torch_commutation_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

decimals=5

######### Hyperparams
n, d, c = 3, 4, 2 #  number of samples, input dim, output dim
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
max_margins = torch.maximum(torch.zeros_like(margins), margins)**2 # (n, c)
max_margins[torch.arange(n), Y] = 0 # (n, c)

######### Loss
loss = (1/(2*n)) * max_margins.sum() + 0.5*gamma*torch.trace(V.T@V)

## OR
# print(loss)
# loss = 0
# for i in range(n) :
#     for j in range(c) :
#         loss += int(j!=Y[i]) *  max(0, hat_Y[i][j] - hat_Y[i][Y[i]] + zeta)**2
# loss = (1/(2*n)) * loss + 0.5*gamma*torch.trace(V.T@V)
# print(loss)

######### Loss gradient
loss.backward(retain_graph=True)
hat_Ygrad = hat_Y.grad.detach()
Vgrad = V.grad.detach()

######### N

# N = torch.zeros_like(margins) # (n, c)
# for i in range(n) :
#     for j in range(c) :
#         if j == Y[i] :
#             N[i][j] = (hat_Y[i][j] - hat_Y[i][Y[i]] + zeta) * int(hat_Y[i][j] - hat_Y[i][Y[i]] + zeta >= 0)
#             s = 0
#             for k in range(c) : s += (hat_Y[i][k] - hat_Y[i][Y[i]] + zeta) * int(hat_Y[i][k] - hat_Y[i][Y[i]] + zeta >= 0)
#             N[i][j] = N[i][j] - s
#         else :
#             N[i][j] = (hat_Y[i][j] - hat_Y[i][Y[i]] + zeta) * int(hat_Y[i][j] - hat_Y[i][Y[i]] + zeta >= 0)

## OR

N = margins*(margins >= 0) # (n, c)
N[torch.arange(n), Y] = margins[torch.arange(n), Y] -  torch.sum(margins*(margins >= 0), dim=1)

######### Gradient Y

dhat_Y = (1/n) * N # (n, c)
dhat_Y.retain_grad() # for Hess(Y)
print(equals(hat_Ygrad, dhat_Y, dec=decimals))#, "\n", hat_Ygrad,"\n", dhat_Y,"\n")

######### Gradient V

dV = N.T @ X #  (c, n)x(n, d)
## OR
# dV = torch.zeros_like(V)
# for i in range(n) :
#     for j in range(c) :
#         if j == Y[i] : continue
#         for l in range(c) : 
#             dV[l] += (int(j==l) - int(Y[i]==l)) * X[i] * (hat_Y[i][j] - hat_Y[i][Y[i]] + zeta) * int(hat_Y[i][j] - hat_Y[i][Y[i]] + zeta >= 0)

dV = (beta/n) * dV + gamma*V
print(equals(Vgrad, dV, dec=decimals))#, "\n", Vgrad,"\n", dV,"\n")

######### M

M = 1.0*(margins >= 0) # (n, c)
M[torch.arange(n), Y] -= torch.sum(M, dim=1) # (n,)

######### bigM_k

def get_bigT_k(k):
    # required : margins, n, M, Y
    bigM_k = torch.zeros_like(M) # (n, c)
    tmp = 1.0*(margins >= 0) # (n, c)
    bigM_k[torch.arange(n), torch.ones_like(Y)*k] = tmp[torch.arange(n), torch.ones_like(Y)*k]
    bigM_k[torch.arange(n), Y] -= torch.sum(bigM_k, dim=1)
    mask = Y.unsqueeze(1) == torch.ones_like(M)*k # (n, c)
    bigT_k = bigM_k - mask * M
    return bigT_k # (n, c)

bigT = torch.stack([get_bigT_k(k) for k in range(c)], dim=0) # (c, n, c)

######### Hessian Y
Hess_YT = torch.zeros(n*c, n*c).to(device) # (nc, nc)
for i in range(n) :
    Hess_YT[i*c:(i+1)*c,i*c:(i+1)*c] = (1/n) * bigT[:,i,:].T
# Hess(Y^T) = K(n,c) @ Hess_(Y) @ K(c,n)
Knc = torch_commutation_matrix(n, c, min_size_csr=10000) + 0.0
Hess_Y = Knc.T @ Hess_YT @ Knc
DFYY1 = from_DF_to_DFpqmn(DF=Hess_Y, m=n, n=c, p=n, q=c) # (n, c, n, c)

hat_Y.grad.zero_()
DFYY2 = get_DFpqmn(F=dhat_Y, X=hat_Y, p=n, q=c) # (n, c, n, c)
print(equals(DFYY1, DFYY2, dec=decimals))#, "\n", DFYY1,"\n", DFYY2,"\n")

######### Hessian V

S = torch.zeros(n*c, c*d) # (n*c, c*d)
for i in range(n) :
    for k in range(c) :
        #S[i*c:(i+1)*c,k*d:(k+1)*d] = torch.row_stack([bigT[k,i,l]*X[i] for l in range(c)]) # (c, d)
        S[i*c:(i+1)*c,k*d:(k+1)*d] = torch.outer(bigT[k][i], X[i]) # (c, d)

Icd = torch.eye(c*d).to(device)
Ic = torch.eye(c).to(device)
Kcd = torch_commutation_matrix(c, d, min_size_csr=10000) + 0.0
H = torch.kron(X, Ic).T @ S @ Kcd
H = (beta**2/n)*H + gamma*Icd   # (cd, cd)
DF1 = from_DF_to_DFpqmn(DF=H, m=c, n=d, p=c, q=d)

V.grad.zero_()
DF2 = get_DFpqmn(F=dV, X=V, p=c, q=d)
print(equals(DF1, DF2, dec=decimals))#, "\n", DF1,"\n", DF2,"\n")
