######## OVA SVM (binary classification) ##########
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
n, d = 3, 5 #  number of samples, input dim, output dim
beta = 0.7 # temperature
#tau = 0.5 # uniform regularizer strengths (noise)
gamma = 0.5 # l2 regularization
zeta = 1.0 # margin

######### Input, labels, P^* and P
X = torch.randn(n, d, requires_grad=False).to(device) # (n, d)
Y = torch.empty(n, dtype=torch.long).random_(2).to(device) # (n,)
Y = Y.unsqueeze(1) # (n, 1)
Delta = 2*Y - 1 # OVA labels

######### Model
V = torch.randn(1, d, requires_grad=True, device=device) # (c, d)
######### Model output
hat_Y = beta * (X @ V.T) # (n, 1)
hat_Y.retain_grad() # for Grad(Y)

######### Margin
margins = zeta - hat_Y*Delta # (n, 1) 
max_margins = torch.maximum(torch.zeros_like(hat_Y), margins)  # (n, 1)     

######### Loss
loss =  (1/(q*n)) * (max_margins**q).sum() + 0.5*gamma*torch.trace(V.T@V)

######### Loss gradient
loss.backward(retain_graph=True)
hat_Ygrad = hat_Y.grad.detach()
Vgrad = V.grad.detach()

######### Gradient Y

dhat_Y = -(1/n) * Delta * (margins>=0 if q==1 else max_margins) # (n, 1)
if q==2 : dhat_Y.retain_grad() # for Hess(Y)
print(equals(hat_Ygrad, dhat_Y, dec=decimals))#, "\n", hat_Ygrad,"\n", dhat_Y,"\n")

######### Gradient V

dV = -(beta/n) * ( Delta * (margins>=0 if q==1 else max_margins) ).T @ X + gamma*V # (1, d)
print(equals(Vgrad, dV, dec=decimals))#, "\n", Vgrad,"\n", dV,"\n")

######### Hessian V

Hess_V = gamma*torch.eye(d).to(device)
if q==2 : Hess_V += (beta**2/n) * X.T @ torch.diag(torch_vec(1.0*(margins.T>=0))) @ X  # (d, d)
DF1 = from_DF_to_DFpqmn(DF=Hess_V, m=1, n=d, p=1, q=d) # (1, d, 1, d)

V.grad.zero_()
DF2 = get_DFpqmn(F=dV, X=V, p=1, q=d) # (1, d, 1, d)
print(equals(DF1, DF2, dec=decimals))#, "\n", DF1,"\n", DF2,"\n")

######### Hessian Y
if q==1: exit()
Hess_Y = (1/n)*torch.diag(1.0*(margins.squeeze()>=0)) # (n, n)
DFYY1 = from_DF_to_DFpqmn(DF=Hess_Y, m=n, n=1, p=n, q=1) # (n, 1, n, 1)
#DFYY1 =  Hess_Y

hat_Y.grad.zero_()
DFYY2 = get_DFpqmn(F=dhat_Y, X=hat_Y, p=n, q=1) # (n, 1, n, 1)
#DFYY2 = DFYY2.squeeze() # (n, n)
print(equals(DFYY1, DFYY2, dec=decimals))#, "\n", DFYY1,"\n", DFYY2,"\n")