import torch
import numpy as np

from utils import equals, from_DF_to_DFpqmn

###################### Linear model  ###################### 
###########################################################

# decimals = 2
# n, d, c = 44760, 16, 3
# n, d, c = 1000, 16, 3
# n, d, c = 2, 2, 2
# beta = 0.5
# tau = 1
# gamma = 0.5

# One_cc = torch.ones(c, c).to(device)

# ###########

# X = torch.randn(n, d+1, requires_grad=False).to(device)
# W = torch.randn(c, d+1, requires_grad=True, device=device)

# Y = torch.empty(n, dtype=torch.long).random_(c).to(device)
# P = torch.nn.functional.one_hot(Y, num_classes=c).float().to(device)
# P = tau * P + (1-tau)/c #* torch.ones_like(n, c)

# Y_hat = X @ W.T # (n, c)
# #P_hat = torch.softmax(beta*Y_hat, dim=1)
# P_hat = (beta * Y_hat).exp() / ( (beta * Y_hat).exp() @ One_cc)

# #loss = torch.nn.CrossEntropyLoss(reduction="mean")(Y_hat, Y) + 0.5*gamma*torch.trace(W.T@W)
# loss = - (1/n) * (P.T @ (P_hat.log())).trace() + 0.5*gamma*torch.trace(W.T@W)

# ######### Gradient

# loss.backward(retain_graph=True)
# Wgrad = W.grad.detach()
# dW = (beta/n) * (P_hat-P).T @ X + gamma*W
# print(equals(Wgrad, dW, dec=decimals))#, "\n", Wgrad,"\n", dW,"\n")

# ######### Hessian

# def eyes(n, c) :
#     M = torch.zeros(n, c)
#     for i in range(min(n, c)) : M[i,i] = 1
#     return M


# In = torch.eye(n).to(device)
# Inc = torch.eye(n*c).to(device)
# Icd1 = torch.eye(c*(d+1)).to(device)
# Ic = torch.eye(c).to(device)


# #M = torch.diag(torch_vec(1/(One_cc @ (beta*Y_hat.T).exp()))) @ torch.kron(In, One_cc) @  torch.diag(torch_vec((beta*Y_hat.T).exp()))
# M =  torch.kron(In, One_cc) @ torch.diag(torch_vec(P_hat.T))
# N = torch.diag(torch_vec(P_hat.T)) 
# Q = N @ (Inc -  M)
# T = torch.kron(X, Ic).T @ Q @ torch.kron(X, Ic)  

# # H = (beta**2/n) * T + gamma*Icd1  # (c(d+1), c(d+1))
# # DFWW1 = from_DF_to_DFpqmn(DF=H, m=c, n=d+1, p=c, q=d+1)

# # W.grad.zero_()
# # DFWW2 = get_DFpqmn(F=dW, X=W, p=c, q=d+1)

# # print(equals(DFWW1, DFWW2, dec=decimals))#, "\n", DFWW1,"\n", DFWW2,"\n")