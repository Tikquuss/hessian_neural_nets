# ###################### SVM2  ###################### 
# ###########################################################

import torch
import numpy as np

from utils import equals, from_DF_to_DFpqmn


decimals = 5
n, d, c = 44760, 16, 3
n, d, c = 1000, 16, 3
n, d, c = 3, 1, 4
beta = 0.5
tau = 1
gamma = 0.5
Delta = 10

###########

X = torch.randn(n, d+1, requires_grad=False).to(device)
W = torch.randn(c, d+1, requires_grad=True, device=device)

Y = torch.empty(n, dtype=torch.long).random_(c).to(device)
P = torch.nn.functional.one_hot(Y, num_classes=c).float().to(device)
P = tau * P + (1-tau)/c #* torch.ones_like(n, c)

Y_hat = X @ W.T # (n, c)

######## LOSS 1

Y_hat_iyi = Y_hat[torch.arange(n), Y].unsqueeze(1) # (n, 1)
margins = Y_hat - Y_hat_iyi + Delta # (n, c)
max_margins = torch.maximum(torch.zeros_like(margins), margins)**2 # (n, c)
max_margins[torch.arange(n), Y] = 0
loss = (1/(2*n)) * max_margins.sum() + 0.5*gamma*torch.trace(W.T@W)

# print(loss)
# loss = 0
# for i in range(n) :
#     for j in range(c) :
#         if j == Y[i] : continue
#         loss += max(0, Y_hat[i][j] - Y_hat[i][Y[i]] + Delta)**2
# loss = (1/n) * loss + 0.5*gamma*torch.trace(W.T@W)
# print(loss)

######### Gradient

loss.backward(retain_graph=True)
Wgrad = W.grad.detach()

X_mask = margins*(margins >= 0)
count = torch.sum(X_mask, dim=1) # (n,)
X_mask[torch.arange(n), Y] -= count
dW = X_mask.T @ X #  (c, n)x(n, d+1)
dW = (1/n) * dW + gamma*W

# ##
# Q = margins*(margins >= 0)
# Q[torch.arange(n), Y] = margins[torch.arange(n), Y] -  torch.sum(margins*(margins >= 0), dim=1)
# ##
# Q = torch.zeros_like(margins)
# for i in range(n) :
#     for j in range(c) :
#         if j == Y[i] :
#             Q[i][j] = (Y_hat[i][j] - Y_hat[i][Y[i]] + Delta) * int(Y_hat[i][j] - Y_hat[i][Y[i]] + Delta >= 0)
#             s = 0
#             for k in range(c) : s += (Y_hat[i][k] - Y_hat[i][Y[i]] + Delta) * int(Y_hat[i][k] - Y_hat[i][Y[i]] + Delta >= 0)
#             Q[i][j] = Q[i][j] - s
#         else :
#             Q[i][j] = (Y_hat[i][j] - Y_hat[i][Y[i]] + Delta) * int(Y_hat[i][j] - Y_hat[i][Y[i]] + Delta >= 0)

# print(X_mask)
# print(Q)
# print(equals(X_mask, Q, dec=decimals))

# # dW = torch.zeros_like(W)
# # for i in range(n) :
# #     for j in range(c) :
# #         #if j == Y[i] : continue
# #         for l in range(c) : dW[l] += (int(j==l) - int(Y[i]==l)) * X[i] * (Y_hat[i][j] - Y_hat[i][Y[i]] + Delta) * int(Y_hat[i][j] - Y_hat[i][Y[i]] + Delta >= 0)
# # dW = (2/n) * dW + gamma*W

# print(Wgrad)
# print(dW)
# print(equals(Wgrad, dW, dec=decimals))#, "\n", Wgrad,"\n", dW,"\n")

######### Hessian

M = 1.0*(margins >= 0)
M[torch.arange(n), Y] -= torch.sum(M, dim=1)

print(Y==0)

def get_deltakM(k):
    tmp = 1.0*(margins >= 0)
    deltakM = torch.zeros_like(M)
    deltakM[torch.arange(n), torch.ones_like(Y)*k] = tmp[torch.arange(n), torch.ones_like(Y)*k]
    deltakM[torch.arange(n), Y] -= torch.sum(deltakM, dim=1)
    mask = Y.unsqueeze(1) == torch.ones_like(M)*k
    deltakM = deltakM - mask * M
    return deltakM

def get_Nil_Wk(i, l, k):
    return get_deltakM(k)[i][l]*X[i]
    return (get_deltakM(k)[i][l] - int(k == Y[i]) * M[i][l])*X[i]
    s = int(k==l) * int(Y_hat[i][l] - Y_hat[i][Y[i]] + Delta >= 0)
    s -= int(l == Y[i]) * sum([int(k==j) * int(Y_hat[i][j] - Y_hat[i][Y[i]] + Delta >= 0) for j in range(c)])
    s -= int(k == Y[i]) * M[i][l]
    s = s * X[i]
    return s

all_deltakM = {k : get_deltakM(k) for k in range(c)}
T = torch.zeros(n*c, c*d+c)

print(n*c, c*d+c)
i_start = 0
for i in range(n) :
    i_end = i_start + c
    k_start = 0
    for k in range(c) :
        k_end = k_start + (d+1)
        nabla_Ni_Wk = torch.row_stack([get_Nil_Wk(i, l, k) for l in range(c)]) # (c, d+1) 
        #nabla_Ni_Wk = ( all_deltakM[k][i] - int(k == Y[i]) * M[i]).unsqueeze(1) @ X[i].unsqueeze(1).T
        #nabla_Ni_Wk = torch.outer(all_deltakM[k][i] - int(k == Y[i]) * M[i], X[i])
        nabla_Ni_Wk = torch.outer(all_deltakM[k][i], X[i])
        #print((i_start, i_end), (k_start, k_end), nabla_Ni_Wk.shape, T[i_start:i_end,k_start:k_end].shape)
        T[i_start:i_end,k_start:k_end] = nabla_Ni_Wk
        k_start = k_end
    i_start = i_end

Icd1 = torch.eye(c*(d+1)).to(device)
Ic = torch.eye(c).to(device)
Kcd1 = torch_commutation_matrix(c, d+1, min_size_csr=10000) + 0.0
H = torch.kron(X, Ic).T @ T @ Kcd1
H = (1/n)*H + gamma*Icd1  # (c(d+1), c(d+1))
print(H.shape, c*(d+1))
DF1 = from_DF_to_DFpqmn(DF=H, m=c, n=d+1, p=c, q=d+1)
W.grad.zero_()
DF2 = get_DFpqmn(F=dW, X=W, p=c, q=d+1)
print(equals(DF1, DF2, dec=decimals))#, "\n", DF1,"\n", DF2,"\n")
