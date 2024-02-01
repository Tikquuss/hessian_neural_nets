################# Classification : square loss ##############
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
decimals=3

######### Hyperparams
n, d, c = 2, 2, 2 #  number of samples, input dim, output dim
beta = 0.7 # temperature
tau = 0.5 # uniform regularizer strengths (noise)
gamma = 0.5 # l2 regularization
zeta = 1.0 # margin

######### Input, labels, P^* and P
X = torch.randn(n, d, requires_grad=False).to(device) # (n, d)
Y = torch.empty(n, dtype=torch.long).random_(c).to(device) # (n,)
P_start = torch.nn.functional.one_hot(Y, num_classes=c).float().to(device) # (n, c)
P = tau * P_start + (1-tau)/c #* torch.ones_like(n, c) # (n, c)

######### Model
V = torch.randn(c, d, requires_grad=True, device=device) # (c, d)
######### Model output
hat_Y = beta * (X @ V.T) # (n, c)
hat_Y.retain_grad() # for Grad(Y)
#hat_P = torch.softmax(hat_Y, dim=1)
One_cc = torch.ones(c, c).to(device) # (c, c)
hat_P = hat_Y.exp() / (hat_Y.exp() @ One_cc) # (n, c)
hat_P.retain_grad() # for Grad(P)
Delta_P = hat_P-P # (n, c)

######### Loss
# L_theta = (1/2)*torch.nn.MSELoss(reduction="mean")(input=hat_P, target=P) + 0.5*gamma*torch.trace(V.T@V)
# L_theta = (1/2)*(Delta_P**2).mean(dim=1).mean() + 0.5*gamma*torch.trace(V.T@V)
# L_theta = (1/(2*n*c))*(Delta_P**2).sum(dim=1).sum() + 0.5*gamma*torch.trace(V.T@V)
L_theta = (1/(2*n*c))*torch.trace(Delta_P.T @ Delta_P) + 0.5*gamma*torch.trace(V.T@V)

######### Loss gradient
L_theta.backward(retain_graph=True)
hat_Ygrad = hat_Y.grad.detach()
Vgrad = V.grad.detach()

######### Gradient Y
dhat_Y = (1/(n*c)) * hat_P * ( Delta_P - (Delta_P*hat_P)@One_cc ) # (n, c)
dhat_Y.retain_grad() # for Hess(Y)
print(equals(hat_Ygrad, dhat_Y, dec=decimals))#, "\n", hat_Ygrad,"\n", dhat_Y,"\n")

######### Gradient V

dV = beta * dhat_Y.T @ X + gamma*V
print(equals(Vgrad, dV, dec=decimals))#, "\n", Vgrad,"\n", dV,"\n")

######### Gradient PT wrt to YT (B)
Inc = torch.eye(n*c).to(device)
In = torch.eye(n).to(device)
diag_vec_hatPT = torch.diag(torch_vec(hat_P.T)) 
B = diag_vec_hatPT @ ( Inc - torch.kron(In, One_cc) @ diag_vec_hatPT ) # dhatPT_dhatYT
# dhatPT_dhatYT = K(n,c) @ dhatP_dhatY @ K(c,n)
Knc = torch_commutation_matrix(n, c, min_size_csr=10000) + 0.0
dhatP_dhatY = Knc.T @ B @ Knc
DFPY1 = from_DF_to_DFpqmn(DF=dhatP_dhatY, m=n, n=c, p=n, q=c) # (n, c, n, c)

hat_Y.grad.zero_()
DFPY2 = get_DFpqmn(F=hat_P, X=hat_Y, p=n, q=c) # (n, c, n, c)
print(equals(DFPY1, DFPY2, dec=decimals))#, "\n", DFPY1,"\n", DFPY2,"\n")

######### Hessian Y
# H1 = torch.diag(  torch_vec( (Delta_P - (Delta_P * hat_P)@One_cc).T )  ) @ B # (nc, nc) x (nc, nc) = (n, c)
# H2 = diag_vec_hatPT @ ( B - torch.kron(In, One_cc) @ ( diag_vec_hatPT @ B + torch.diag(torch_vec(Delta_P.T)) @ B )  )
# Hess_YT = (1/(n*c)) * (H1 + H2)

H1 = torch.diag(  torch_vec( (Delta_P - (Delta_P * hat_P)@One_cc).T )  )
H2 = diag_vec_hatPT @ torch.kron(In, One_cc) @ torch.diag(torch_vec(Delta_P.T))
Hess_YT = (1/(n*c)) * (H1 + B - H2) @ B

# Hess(Y^T) = K(n,c) @ Hess_(Y) @ K(c,n)
Hess_Y = Knc.T @ Hess_YT @ Knc
DFYY1 = from_DF_to_DFpqmn(DF=Hess_Y, m=n, n=c, p=n, q=c) # (n, c, n, c)

hat_Y.grad.zero_()
DFYY2 = get_DFpqmn(F=dhat_Y, X=hat_Y, p=n, q=c) # (n, c, n, c)
print(equals(DFYY1, DFYY2, dec=decimals))#, "\n", DFYY1,"\n", DFYY2,"\n")

######### Hessian V
Ic = torch.eye(c).to(device)
Hess_V = (beta**2) * torch.kron(X, Ic).T @ Hess_YT @ torch.kron(X, Ic) + gamma*torch.eye(c*d).to(device)  # (cd, cd)
DF1 = from_DF_to_DFpqmn(DF=Hess_V, m=c, n=d, p=c, q=d)

V.grad.zero_()
DF2 = get_DFpqmn(F=dV, X=V, p=c, q=d) #
#print(equals(DF1, DF2, dec=decimals))#, "\n", DF1,"\n", DF2,"\n")