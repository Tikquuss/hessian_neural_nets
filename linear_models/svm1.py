##################### SVM1  ###################### 
###################################################

import torch
import numpy as np

from utils import equals, from_DF_to_DFpqmn

decimals=2

n, d, c = 3, 1, 4
beta = 1.0
tau = 1.0
gamma = 0.5 # l2 regularization
Delta = 10 # margin


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# Input and labels
X = torch.randn(n, d, requires_grad=False).to(device) # (n, d)
Y = torch.empty(n, dtype=torch.long).random_(c).to(device) # (n,)

# P^*
P_start = torch.nn.functional.one_hot(Y, num_classes=c).float().to(device) # (n, c)
# P^*
P = tau * P_start + (1-tau)/c #* torch.ones_like(n, c) # (n, c)

# Model
V = torch.randn(c, d, requires_grad=True, device=device) # (c, d)
# Model output
Y_hat = X @ V.T # (n, c)

######### Gradient

loss.backward(retain_graph=True)
Vgrad = V.grad.detach()

# Grandiant 1

#X_mask = torch.zeros_like(margins) # (n, c)
#X_mask[margins >= 0] = 1
X_mask = 1.0*(margins >= 0)
count = torch.sum(X_mask, dim=1) # (n,)
X_mask[torch.arange(n), Y] -= count

# X_mask = torch.zeros_like(margins)
# for i in range(n) :
#     for j in range(c) :
#         if j == Y[i] :
#             X_mask[i][j] = int(Y_hat[i][j] - Y_hat[i][Y[i]] + Delta >= 0)
#             s = 0
#             for k in range(c) : s += int(Y_hat[i][k] - Y_hat[i][Y[i]] + Delta >= 0)
#             X_mask[i][j] = X_mask[i][j] - s
#         else :
#             X_mask[i][j] = int(Y_hat[i][j] - Y_hat[i][Y[i]] + Delta >= 0)

dV = X_mask.T @ X #  (c, n)x(n, d)
dV = (1/n) * dV + gamma*V

print(equals(Vgrad, dV, dec=decimals))#, "\n", Vgrad,"\n", dV,"\n")

# Grandiant 2
dV = torch.zeros_like(V)
for i in range(n) :
    for j in range(c) :
        if j == Y[i] : continue
        for l in range(c) : dV[l] += (int(j==l) - int(Y[i]==l)) * X[i] * int(Y_hat[i][j] - Y_hat[i][Y[i]] + Delta >= 0)
dV = (1/n) * dV + gamma*V

print(equals(Vgrad, dV, dec=decimals))#, "\n", Vgrad,"\n", dV,"\n")

# Hessian
Icd = torch.eye(c*d).to(device)
H = gamma*Icd  # (cd, cd)

DFWW1 = from_DF_to_DFpqmn(DF=H, m=c, n=d+1, p=c, q=d+1)

W.grad.zero_()
DFWW2 = get_DFpqmn(F=dW, X=W, p=c, q=d+1)
print(equals(DFWW1, DFWW2, dec=decimals))#, "\n", DFWW1,"\n", DFWW2,"\n")