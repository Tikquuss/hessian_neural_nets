import torch

from utils import equals, torch_vec, torch_unvec, torch_commutation_matrix
from utils import get_DFpqmn, from_DFpqmn_to_DF, from_dict_to_matrix
from utils import from_dict_to_matrix, get_hessian_loop
from utils import MIN_SIZE_CSR, activations_functions

####################################################################
# Forward
###################################################################

def forward(V, W, X, g, alpha, beta):
    """
    V : (c, d)
    W : (d, p)
    X : (n, p)
    """
    H = alpha * (X @ W.T) # (n, p) x (p, d) = (n, d)
    A = g(H) # (n, d)
    hat_Y = beta * (A @ V.T) # (n, d) x (d, c) = (n, c)
    #hat_Y = hat_Y.squeeze() # (n, c) or (n,) if c=1
    return hat_Y, A, H

def forward_T(V, WT, X, g, alpha, beta):
    """
    V : (c, d)
    WT : (p, d)
    X : (n, p)
    """
    H = alpha * (X @ WT) # (n, p) x (p, d) = (n, d)
    A = g(H) # (n, d)
    hat_Y = beta * (A @ V.T) # (n, d) x (d, c) = (n, c)
    #hat_Y = hat_Y.squeeze() # (n, c) or (n,) if c=1
    return hat_Y, A, H


####################################################################
# Gradient
###################################################################

def get_gradients(require_grad, V, X, W, Psi, A, Delta_P, beta, gamma_V=0, gamma_W=0, gamma_X=0):
    """
    V : (c, d)
    W : (d, p)
    X : (n, p)
    A : (n, d)
    Delta_P : (n, c)
    """ 
    grad = {"dV":None, "dW":None, "dX":None}
    if require_grad.get("V", False) :
        n, d = Psi.shape
        c, _ = V.shape
        grad["dV"] = (beta/n) * Delta_P.T @ A + gamma_V * V
    if require_grad.get("W", False) or require_grad.get("X", False) :
        if require_grad.get("W", False) :
            grad["dW"] = Psi.T @ X + gamma_W * W
        if require_grad.get("X", False) :
            grad["dX"] = Psi @ W + gamma_X * X
    return grad

def get_gradients_T(require_grad, V, X, WT, Psi, A, Delta_P, beta, gamma_V=0, gamma_W=0, gamma_X=0):
    """
    V : (c, d)
    WT : (p, d)
    X : (n, p)
    A : (n, d)
    Delta_P : (n, c)
    """ 
    grad = {}
    if require_grad.get("V", False) :
        n, d = Psi.shape
        c, _ = V.shape
        grad["dV"] = (beta/n) * Delta_P.T @ A + gamma_V * V
    if require_grad.get("W", False) or require_grad.get("X", False) :
        if require_grad.get("W", False) :
            grad["dWT"] = X.T @ Psi + gamma_W * WT
        if require_grad.get("X", False) :
            grad["dX"] = Psi @ WT.T + gamma_X * X
    return grad

####################################################################
# Hessian
###################################################################

def get_hessian(require_grad, dV, dW, dX, V, W, X, Psi, Delta_Y, dg, ddg, alpha, beta):
    """
    V, dV: (c, d)
    W, dW : (d, p)
    X, dX : (n, p)
    Psi : (n, d)
    Delta_Y : (n, c)
    """
    c, d = V.shape
    p = W.shape[1]
    n = X.shape[0]

    hess_dict = {}

    Id = torch.eye(d)
    if require_grad["W"] or require_grad["V"] :
        Kdn = torch_commutation_matrix(d, n, min_size_csr=MIN_SIZE_CSR)+0.0
        Kdn_X_Id = Kdn @ torch.kron(X, Id) # (cd, dp)

    In = torch.eye(n)
    if require_grad["X"] :
        W_In = torch.kron(W, In)

    if require_grad["X"] and require_grad["W"] :
        Ip = torch.eye(p)

    i, j = 0, 0
    # HVV & HVW & HVX
    if require_grad["V"]  :
        Sigma_A = ((beta * beta)/(n*c)) * A.T@A # (d, d)
        Ic = torch.eye(c)
        hess_dict["HVV"] = torch.kron(Sigma_A + gamma_V * Id, Ic) # (cd, cd)
        if require_grad["W"] or require_grad["X"] :
            Kdc = torch_commutation_matrix(d, c, min_size_csr=MIN_SIZE_CSR)+0.0
            AT = A.T @ In
            M =  ((beta * alpha)/(n*c)) * (
                beta * Kdc @ torch.kron(V, AT) + torch.kron(Id, Delta_Y).T
            ) @ torch.diag(torch_vec(dg(H)))
            if require_grad["W"] :
                hess_dict["HVW"] = M @ Kdn_X_Id # (cd, dp)
            if require_grad["X"] :
                hess_dict["HVX"] = M @ W_In  # (cd, np)

    if require_grad["W"] or require_grad["X"] :
        tmp = torch.diag(torch_vec(dg(H)))
        N =  ((beta * alpha * alpha)/(n*c)) * (
                torch.diag( torch_vec( ddg(H) * (Delta_Y@V) ) ) + 
                beta * tmp @ torch.kron(V.T@V, In) @ tmp ) 

    # HWV & HWW & HWX
    if require_grad["W"] :
        if require_grad["V"] :
            if "HVW" in hess_dict : 
                hess_dict["HWV"] = hess_dict["HVW"].T
            else :
                hess_dict["HWV"] = Kdn_X_Id.T @ M.T
        hess_dict["HWW"] =  Kdn_X_Id.T @ N @ Kdn_X_Id + gamma_W*torch.eye(d*p)
        if require_grad["X"] :
            hess_dict["HWX"] = torch.kron(Ip, Psi).T + Kdn_X_Id.T @ N @ W_In

    # HXV & HXW & HXX
    if require_grad["X"] :
        if require_grad["V"] :
            if "HVX" in hess_dict : 
                hess_dict["HXV"] = hess_dict["HVX"].T
            else :
                hess_dict["HXV"] = torch.kron(W, In).T @ M.T

        if require_grad["W"] :
            if "HWX" in hess_dict : 
                hess_dict["HXW"] = hess_dict["HWX"].T
            else :
                hess_dict["HXW"] = torch.kron(Ip, Psi) + W_In.T @ N @ Kdn_X_Id

        hess_dict["HXX"] = W_In.T @ N @ W_In + gamma_X*torch.eye(n*p)

    return hess_dict

def get_hessian_T(require_grad, dV, dWT, dX, V, WT, X, Psi, Delta_Y, dg, ddg, alpha, beta):
    """
    V, dV: (c, d)
    WT, dWT : (p, d)
    X, dX : (n, p)
    Psi : (n, d)
    Delta_Y : (n, c)
    """
    c, d = V.shape
    p = W.shape[1]
    n = X.shape[0]

    hess_dict = {}

    Id = torch.eye(d)
    if require_grad["W"] or require_grad["V"] :
        X_Id = torch.kron(X, Id) # (cd, dp)

    In = torch.eye(n)
    if require_grad["X"] :
        W_In = torch.kron(W, In)

    if require_grad["X"] and require_grad["W"] :
        Ip = torch.eye(p)

    i, j = 0, 0
    # HVV & HVW & HVX
    if require_grad["V"]  :
        Sigma_A = ((beta * beta)/(n*c)) * A.T@A # (d, d)
        Ic = torch.eye(c)
        hess_dict["HVV"] = torch.kron(Sigma_A + gamma_V * Id, Ic) # (cd, cd)
        if require_grad["W"] or require_grad["X"] :
            Kdc = torch_commutation_matrix(d, c, min_size_csr=MIN_SIZE_CSR)+0.0
            AT = A.T @ In
            M =  ((beta * alpha)/(n*c)) * (
                beta * Kdc @ torch.kron(V, AT) + torch.kron(Id, Delta_Y).T
            ) @ torch.diag(torch_vec(dg(H)))
            if require_grad["W"] :
                hess_dict["HVW"] = M @ X_Id # (cd, dp)
            if require_grad["X"] :
                hess_dict["HVX"] = M @ W_In  # (cd, np)

    if require_grad["W"] or require_grad["X"] :
        tmp = torch.diag(torch_vec(dg(H)))
        N =  ((beta * alpha * alpha)/(n*c)) * (
                torch.diag( torch_vec( ddg(H) * (Delta_Y@V) ) ) + 
                beta * tmp @ torch.kron(V.T@V, In) @ tmp ) 

    # HWV & HWW & HWX
    if require_grad["W"] :
        if require_grad["V"] :
            if "HVW" in hess_dict : 
                hess_dict["HWV"] = hess_dict["HVW"].T
            else :
                hess_dict["HWV"] = X_Id.T @ M.T
        hess_dict["HWW"] =  X_Id.T @ N @ X_Id + gamma_W*torch.eye(d*p)
        if require_grad["X"] :
            hess_dict["HWX"] = torch.kron(Ip, Psi).T + X_Id.T @ N @ W_In

    # HXV & HXW & HXX
    if require_grad["X"] :
        if require_grad["V"] :
            if "HVX" in hess_dict : 
                hess_dict["HXV"] = hess_dict["HVX"].T
            else :
                hess_dict["HXV"] = torch.kron(W, In).T @ M.T

        if require_grad["W"] :
            if "HWX" in hess_dict : 
                hess_dict["HXW"] = hess_dict["HWX"].T
            else :
                hess_dict["HXW"] = torch.kron(Ip, Psi) + W_In.T @ N @ Kdn_X_Id

        hess_dict["HXX"] = W_In.T @ N @ W_In + gamma_X*torch.eye(n*p)

    return hess_dict


# if __name__ == "__main__":
#     print("")

act = activations_functions["id"]
# act = activations_functions["relu"]
# act = activations_functions["tanh"]
# act = activations_functions["sigmoid"]
# act = activations_functions["sin"]
# act = activations_functions["x2"]

g, dg, ddg = act["g"], act["dg"], act["ddg"]

decimals = 4
n, p, d, c = 5, 4, 3, 2
alpha = .7
beta = .8
tau = 1.0
gamma_V, gamma_W, gamma_X = 1.0, 1.0, 1.0
require_grad = {"V":True, "W":True, "X":True}

# Data
X = torch.randn(n, p, requires_grad=True)
Y = torch.empty(n, dtype=torch.long).random_(c)
P = torch.nn.functional.one_hot(Y, num_classes=c).float() # (n, c)
P = tau * P + (1-tau)/c #* torch.ones_like(n, c)

########################################################

# Parameters
W = torch.randn(d, p, requires_grad=True)
V = torch.randn(c, d, requires_grad=True)

# Forward
hat_Y, A, H = forward(V, W, X, g, alpha, beta)
hat_Y.retain_grad()
#hat_P = torch.softmax(hat_Y, dim=1) # (n, c)
One_cc = torch.ones(c, c) # (c, c)
hat_P = hat_Y.exp() / (hat_Y.exp() @ One_cc) # (n, c)

Delta_P = hat_P-P # (n, c)
Psi = ((beta * alpha)/n) * (dg(H) * (Delta_P@V))

# Gradient
grad = get_gradients(require_grad, V, X, W, Psi, A, Delta_P, beta, gamma_V, gamma_W, gamma_X)

#L_theta = torch.nn.CrossEntropyLoss(reduction="mean")(hat_Y, Y)
L_theta = - (1/n) * (P.T @ (hat_P.log())).trace()

L_theta += 0.5*(gamma_V*torch.trace(V.T@V) + gamma_W*torch.trace(W.T@W) + gamma_X*torch.trace(X.T@X))
L_theta.backward(retain_graph=True)
hat_Ygrad = hat_Y.grad.detach()
Vgrad = V.grad.detach()
Wgrad = W.grad.detach()
Xgrad = X.grad.detach()

# print(equals(hat_Ygrad, Delta_P/n, dec=decimals), "\n", hat_Ygrad,"\n", Delta_P/n,"\n")
print(equals(Vgrad, grad["dV"], dec=decimals), "\n", Vgrad,"\n", grad["dV"],"\n")
print(equals(Wgrad, grad["dW"], dec=decimals), "\n", Wgrad,"\n", grad["dW"],"\n")
print(equals(Xgrad, grad["dX"], dec=decimals), "\n", Xgrad, "\n", grad["dX"],"\n")

# Hessian

# hess_dict_1 = get_hessian_loop(require_grad, grad["dV"], grad["dW"], grad["dX"], V, W, X)
# Hess1 = from_dict_to_matrix(require_grad, hess_dict_1, n, p, c, d, device=W.device, dtype=torch.float64)

# hess_dict_2 = get_hessian(require_grad, grad["dV"], grad["dW"], grad["dX"], V, W, X, Psi, Delta_Y, dg, ddg, alpha, beta)
# Hess2 = from_dict_to_matrix(require_grad, hess_dict_2, n, p, c, d, device=W.device, dtype=torch.float64)

# for key in hess_dict_1 :
#     flag = equals(hess_dict_1[key], hess_dict_2[key], dec=decimals)
#     print(key, flag)
#     if not flag : print(hess_dict_1[key], "\n", hess_dict_2[key])
# print(equals(Hess1, Hess2, dec=decimals))
# #print(Hess1, "\n", Hess2)

########################################################
# TODO
# # Parameters
# WT = torch.randn(p, d, requires_grad=True)
# V = torch.randn(c, d, requires_grad=True)

# # Forward
# hat_Y, A, H = forward_T(V, WT, X, g, alpha, beta)

# Delta_Y = hat_Y-Y # (n, c)
# Sigma_A = ((beta**2)/(n*c)) * A.T@A # (d, d)
# Psi = ((beta * alpha)/(n*c)) * (dg(H) * (Delta_Y@V))

# # Gradient
# grad = get_gradients_T(require_grad, Y, V, X, WT, Sigma_A, Psi, beta, gamma_V, gamma_W, gamma_X)

# L_theta = (1/(2*n*c))*torch.trace(Delta_Y.T @ Delta_Y) + 0.5*(gamma_V*torch.trace(V.T@V) + gamma_W*torch.trace(WT@WT.T) + gamma_X*torch.trace(X.T@X))
# L_theta.backward(retain_graph=True)
# Vgrad = V.grad.detach()
# WTgrad = WT.grad.detach()
# Xgrad = X.grad.detach()

# print(equals(Vgrad, grad["dV"], dec=decimals), "\n", Vgrad,"\n", grad["dV"],"\n")
# print(equals(WTgrad, grad["dWT"], dec=decimals), "\n", WTgrad,"\n", grad["dWT"],"\n")
# print(equals(Xgrad, grad["dX"], dec=decimals), "\n", Xgrad, "\n", grad["dX"],"\n")

# # Hessian

# hess_dict_1 = get_hessian_loop(require_grad, grad["dV"], grad["dWT"], grad["dX"], V, WT, X)
# Hess1 = from_dict_to_matrix(require_grad, hess_dict_1, n, p, c, d, device=W.device, dtype=torch.float64)

# hess_dict_2 = get_hessian(require_grad, grad["dV"], grad["dWT"], grad["dX"], V, WT, X, Psi, Delta_Y, dg, ddg, alpha, beta)
# Hess2 = from_dict_to_matrix(require_grad, hess_dict_2, n, p, c, d, device=W.device, dtype=torch.float64)

# for key in hess_dict_1 :
#     flag = equals(hess_dict_1[key], hess_dict_2[key], dec=decimals)
#     print(key, flag)
#     if not flag : print(hess_dict_1[key], "\n", hess_dict_2[key])
# print(equals(Hess1, Hess2, dec=decimals))
# #print(Hess1, "\n", Hess2)

########################################################

# L_, V_ = torch.linalg.eig(Hess1.detach())
# _ = plot_cdf(L_.real)
# plt.show()