##########  Binary cross entropy loss #############
###################################################

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
# tau = 0.5 # uniform regularizer strengths (noise)
gamma = 0.5 # l2 regularization
zeta = 1.0 # margin

######### Input, labels, P^* and P
X = torch.randn(n, d, requires_grad=False).to(device) # (n, d)
Y = torch.empty(n, dtype=torch.long).random_(2).to(device) # (n,)
Y = Y.float() # expected scalar type Long but found Float
Y = Y.unsqueeze(1) # (n, 1)

######### Model
V = torch.randn(1, d, requires_grad=True, device=device) # (1, d)
######### Model output
hat_Y = beta * (X @ V.T) # (n, 1)
hat_Y.retain_grad() # for Grad(Y)
hat_P = torch.sigmoid(hat_Y) # (n, 1)

######### Loss
loss = torch.nn.BCELoss(reduction='mean')(hat_P, Y) + 0.5*gamma*torch.trace(V.T@V)
loss = -(1/n) * (Y.T @ hat_P.log() + (1-Y).T @ (1-hat_P).log() ).trace() + 0.5*gamma*torch.trace(V.T@V)

######### Loss gradient
loss.backward(retain_graph=True)
hat_Ygrad = hat_Y.grad.detach()
Vgrad = V.grad.detach()

######### Gradient Y
dhat_Y = (1/n) * (hat_P - Y) # (n, c)
dhat_Y.retain_grad() # for Hess(Y)
print(equals(hat_Ygrad, dhat_Y, dec=decimals))#, "\n", hat_Ygrad,"\n", dhat_Y,"\n")

######### Gradient V

dV = beta * dhat_Y.T @ X + gamma*V
print(equals(Vgrad, dV, dec=decimals))#, "\n", Vgrad,"\n", dV,"\n")

######### Hessian Y
Hess_Y = (1/n)*torch.diag( (hat_P * (1-hat_P)).squeeze()) # (n, n)
DFYY1 = from_DF_to_DFpqmn(DF=Hess_Y, m=n, n=1, p=n, q=1) # (n, 1, n, 1)
#DFYY1 =  Hess_Y

hat_Y.grad.zero_()
DFYY2 = get_DFpqmn(F=dhat_Y, X=hat_Y, p=n, q=1) # (n, 1, n, 1)
#DFYY2 = DFYY2.squeeze() # (n, n)
print(equals(DFYY1, DFYY2, dec=decimals))#, "\n", DFYY1,"\n", DFYY2,"\n")

######### Hessian V
Hess_V = (beta**2) * X.T @ Hess_Y @ X + gamma*torch.eye(d).to(device)  # (cd, cd)
DF1 = from_DF_to_DFpqmn(DF=Hess_V, m=1, n=d, p=1, q=d)

V.grad.zero_()
DF2 = get_DFpqmn(F=dV, X=V, p=1, q=d) #
print(equals(DF1, DF2, dec=decimals))#, "\n", DF1,"\n", DF2,"\n")

