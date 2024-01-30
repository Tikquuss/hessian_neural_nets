########### Absolute and quadratic loss ###########
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
tau = 0.5 # uniform regularizer strengths (noise)
gamma = 0.5 # l2 regularization
zeta = 1.0 # margin

######### Input, labels
X = torch.randn(n, d, requires_grad=False).to(device) # (n, d)
Y = torch.randn(n, c, requires_grad=False).to(device) # (n, d)

######### Model
V = torch.randn(c, d, requires_grad=True, device=device) # (c, d)
######### Model output
hat_Y = beta * (X @ V.T) # (n, c)
hat_Y.retain_grad() # for Grad(Y)
Delta_Y = hat_Y - Y

######### Loss
if q==1:
    # L_theta = torch.nn.L1Loss(reduction='mean')(input=hat_Y, target=Y) + 0.5*gamma*torch.trace(V.T@V)
    # L_theta = Delta_Y.abs().mean(dim=1).mean() + 0.5*gamma*torch.trace(V.T@V)
    # L_theta = (1/(n*c))*Delta_Y.abs().sum(dim=1).sum() + 0.5*gamma*torch.trace(V.T@V)
    L_theta = (1/(n*c))*torch.trace(Delta_Y.abs() @ torch.ones(c, n)) + 0.5*gamma*torch.trace(V.T@V)
elif q==2:
    # L_theta = (1/2)*torch.nn.MSELoss(reduction="mean")(input=hat_Y, target=Y) + 0.5*gamma*torch.trace(V.T@V)
    # L_theta = (1/2)*(Delta_Y**2).mean(dim=1).mean() + 0.5*gamma*torch.trace(V.T@V)
    # L_theta = (1/(2*n*c))*(Delta_Y**2).sum(dim=1).sum() + 0.5*gamma*torch.trace(V.T@V)
    L_theta = (1/(2*n*c))*torch.trace(Delta_Y.T @ Delta_Y) + 0.5*gamma*torch.trace(V.T@V)


######### Loss gradient
L_theta.backward(retain_graph=True)
hat_Ygrad = hat_Y.grad.detach()
Vgrad = V.grad.detach()

######### Gradient Y
dhat_Y = (1/(n*c)) * (Delta_Y.sign() if q==1 else Delta_Y)  # (n, c)
dhat_Y.retain_grad() # for Hess(Y)
print(equals(hat_Ygrad, dhat_Y, dec=decimals))#, "\n", hat_Ygrad,"\n", dhat_Y,"\n")

######### Gradient V
# 
dV = beta * dhat_Y.T @ X + gamma*V
# OR
Sigma_X = ((beta**2)/(n*c)) * X.T@X # (d, d)
dV = V @ (Sigma_X + gamma * torch.eye(d)) - (beta/(n*c)) * Y.T@X

print(equals(Vgrad, dV, dec=decimals))#, "\n", Vgrad,"\n", dV,"\n")

######### Hessian Y
Hess_Y = (1/(n*c)) * (torch.zeros(n*c, n*c) if q==1 else torch.eye(n*c)) # (nc, nc)
DFYY1 = from_DF_to_DFpqmn(DF=Hess_Y, m=n, n=c, p=n, q=c) # (n, c, n, c)

hat_Y.grad.zero_()
DFYY2 = get_DFpqmn(F=dhat_Y, X=hat_Y, p=n, q=c) # (n, c, n, c)
print(equals(DFYY1, DFYY2, dec=decimals))#, "\n", DFYY1,"\n", DFYY2,"\n")

######### Hessian V
# Hess(Y^T) = K(n,c) @ Hess_(Y) @ K(c,n)
Knc = torch_commutation_matrix(n, c, min_size_csr=10000) + 0.0
Hess_YT = Knc @ Hess_Y @ Knc.T
Ic = torch.eye(c).to(device)
Hess_V = (beta**2) * torch.kron(X, Ic).T @ Hess_YT @ torch.kron(X, Ic) + gamma*torch.eye(c*d).to(device)  # (cd, cd)
DF1 = from_DF_to_DFpqmn(DF=Hess_V, m=c, n=d, p=c, q=d)

V.grad.zero_()
DF2 = get_DFpqmn(F=dV, X=V, p=c, q=d) #
print(equals(DF1, DF2, dec=decimals))#, "\n", DF1,"\n", DF2,"\n")

