######## OVA SVM (multiclass classification) ##########
###################################################

q=1
q=2
assert q in {1, 2}

import torch
import numpy as np

import os
import sys
current_dir = os.path.dirname(__file__)  # current script's directory
parent_dir = os.path.join(current_dir, '..')  # parent directory
sys.path.append(parent_dir) 
from utils import equals, from_DF_to_DFpqmn, get_DFpqmn
from utils import torch_commutation_matrix, torch_vec

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
Delta = 2*P_start - 1 # OVA labels

######### Model
V = torch.randn(c, d, requires_grad=True, device=device) # (c, d)
######### Model output
hat_Y = beta * (X @ V.T) # (n, c)
hat_Y.retain_grad() # for Grad(Y)

######### Margin
margins = zeta - hat_Y*Delta # (n, c) 
max_margins = torch.maximum(torch.zeros_like(hat_Y), margins)  # (n, c)     

######### Loss
loss =  (1/(q*n)) * (max_margins**q).sum() + 0.5*gamma*torch.trace(V.T@V)

######### Loss gradient
loss.backward(retain_graph=True)
hat_Ygrad = hat_Y.grad.detach()
Vgrad = V.grad.detach()

######### Gradient Y

dhat_Y = -(1/n) * Delta * (margins>=0 if q==1 else max_margins) # (n, c)
if q==2 : dhat_Y.retain_grad() # for Hess(Y)
print(equals(hat_Ygrad, dhat_Y, dec=decimals))#, "\n", hat_Ygrad,"\n", dhat_Y,"\n")

######### Gradient V

dV = -(beta/n) * ( Delta * (margins>=0 if q==1 else max_margins) ).T @ X + gamma*V
print(equals(Vgrad, dV, dec=decimals))#, "\n", Vgrad,"\n", dV,"\n")

######### Hessian V

Ic = torch.eye(c).to(device)
Hess_V = gamma*torch.eye(c*d).to(device)
if q==2 : Hess_V += (beta**2/n) * torch.kron(X, Ic).T @ torch.diag(torch_vec(1.0*(margins.T>=0))) @ torch.kron(X, Ic)  # (cd, cd)
DF1 = from_DF_to_DFpqmn(DF=Hess_V, m=c, n=d, p=c, q=d)

V.grad.zero_()
DF2 = get_DFpqmn(F=dV, X=V, p=c, q=d)
print(equals(DF1, DF2, dec=decimals))#, "\n", DF1,"\n", DF2,"\n")

######### Hessian Y
if q==1: exit()
Hess_YT = (1/n) * torch.diag(torch_vec(1.0*(margins.T>=0)))
# Hess(Y^T) = K(n,c) @ Hess_(Y) @ K(c,n)
Knc = torch_commutation_matrix(n, c, min_size_csr=10000) + 0.0
Hess_Y = Knc.T @ Hess_YT @ Knc
DFYY1 = from_DF_to_DFpqmn(DF=Hess_Y, m=n, n=c, p=n, q=c) # (n, c, n, c)

hat_Y.grad.zero_()
DFYY2 = get_DFpqmn(F=dhat_Y, X=hat_Y, p=n, q=c) # (n, c, n, c)
print(equals(DFYY1, DFYY2, dec=decimals))#, "\n", DFYY1,"\n", DFYY2,"\n")