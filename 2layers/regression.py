import torch
import os
import sys

current_dir = os.path.dirname(__file__)  # current script's directory
parent_dir = os.path.join(current_dir, '..')  # parent directory
sys.path.append(parent_dir) 
from utils import MIN_SIZE_CSR, activations_functions
from utils import equals, torch_vec, torch_unvec, torch_commutation_matrix

from utils2layers import from_dict_to_matrix, get_hessian_loop

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

def get_gradients(require_grad, Y, V, X, W, Sigma_A, Psi, beta, gamma_V=0, gamma_W=0, gamma_X=0):
    """
    Y : (n, c)
    V : (c, d)
    W : (d, p)
    X : (n, p)
    Sigma_A : (d, d)
    Psi : (n, d)
    """ 
    grad = {"dV":None, "dW":None, "dX":None}
    if require_grad.get("V", False) :
        n, d = Psi.shape
        c, _ = V.shape
        grad["dV"] = V @ (Sigma_A + gamma_V * torch.eye(d)) - (beta/(n*c)) * Y.T@A
    if require_grad.get("W", False) or require_grad.get("X", False) :
        if require_grad.get("W", False) :
            grad["dW"] = Psi.T @ X + gamma_W * W
        if require_grad.get("X", False) :
            grad["dX"] = Psi @ W + gamma_X * X
    return grad

def get_gradients_T(require_grad, Y, V, X, WT, Sigma_A, Psi, beta, gamma_V=0, gamma_W=0, gamma_X=0):
    """
    Y : (n, c)
    V : (c, d)
    WT : (p, d)
    X : (n, p)
    Sigma_A : (d, d)
    Psi : (n, d)
    """ 
    grad = {}
    if require_grad.get("V", False) :
        n, d = Psi.shape
        c, _ = V.shape
        grad["dV"] = V @ (Sigma_A + gamma_V * torch.eye(d)) - (beta/(n*c)) * Y.T@A
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


# BEGIN TODO
import numpy as np
from scipy.linalg import det as scipy_det, pinv as scipy_pinv
from scipy.sparse import identity as scipy_eyes, diags as scipy_diag, kron as scipy_kron
from scipy.sparse.linalg import inv as scipy_inv, eigs as scipy_eigs
from scipy.sparse import csr_matrix

from utils import np_vec, np_unvec, np_commutation_matrix

def get_hessian_np(require_grad, P, P_start, A, H, V, W, X, beta, dg, ddg, gamma_V, gamma_W, dtype=np.float64):
    """
    P_start, P : (n, c)
    A, H : (n, d)
    V : (c, d)
    W : (d, p)
    X : (n, p)
    """
    c, d = V.shape
    p = W.shape[1]
    n = X.shape[0]

    device = W.device
    P = P.detach().cpu().numpy() 
    P_start = P_start.detach().cpu().numpy() 
    A = A.detach().cpu().numpy() 
    H = H.detach().cpu().numpy() 
    V = V.detach().cpu().numpy() 
    W = W.detach().cpu().numpy() 
    X = X.detach().cpu().numpy() 

    ####################
    q=c*d*require_grad.get("V", False) + d*p*require_grad.get("W", False) 
    Hess = np.empty((q, q), dtype=dtype)
    ####################

    One_cc = np.ones((c, c))
    In = scipy_eyes(n)
    Inc = scipy_eyes(n*c)
    Ic = scipy_eyes(c)
    Icd = scipy_eyes(c*d)
    Id = scipy_eyes(d)

    Delta_P = P-P_start # (n, c)
    S_tilde = Inc -  scipy_kron(In, One_cc) @ scipy_diag(np_vec(P.T))
    R_tilde = scipy_diag(np_vec(P.T)) @ S_tilde
    S = Inc -  scipy_kron(One_cc, In) @ scipy_diag(np_vec(P))
    R = scipy_diag(np_vec(P)) @ S
    Kdc = np_commutation_matrix(d, c, min_size_csr=MIN_SIZE_CSR)+0.0
    T = (beta/(n*(d**0.5)))*(scipy_kron(Id, Delta_P).T + beta* Kdc @ scipy_kron(Ic, A).T @ R @ scipy_kron(V, In)) @ scipy_diag(np_vec(dg(H)))
    tmp = scipy_kron(V, In) @ scipy_diag(np_vec(dg(H)))
    N = (beta/(n*d))*(  scipy_diag( np_vec ( (Delta_P@V) * ddg(H) )  )  + beta * tmp.T@R@tmp )
    
    Kdp = np_commutation_matrix(d, p, min_size_csr=MIN_SIZE_CSR)
    IdXKdp = scipy_kron(Id, X) @ Kdp
    
    i, j = 0, 0
    # HVV & HVW
    if require_grad["V"] :
        tmp = scipy_kron(A, Ic)
        Hess[i:i+c*d,j:j+c*d] = ((beta**2/n) * tmp.T @ R_tilde @ tmp  + gamma_V*Icd).toarray()  # (cd, cd)
        j+=c*d
        if require_grad["W"] :
            Hess[i:i+c*d,j:j+d*p] = T @ IdXKdp  # (cd, dp)
            j+=d*p
        i+=c*d
    
    # HWV & HWW 
    j=0
    if require_grad["W"] :
        if require_grad["V"] :
            Kpd = np_commutation_matrix(p, d, min_size_csr=MIN_SIZE_CSR)+0.0
            Hess[i:i+d*p,j:j+c*d] = Kpd @ scipy_kron (Id, X).T @ T.T # (dp, cd)
            j+=c*d
        Hess[i:i+d*p,j:j+d*p] = IdXKdp.T @ N @ IdXKdp + gamma_W * scipy_eyes(d*p)  # (dp, dp)

    return torch.from_numpy(Hess).to(device)

# END TODO

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
#n, p, d, c = 1, 4, 3, 2
n, p, d, c = 1, 31*2, 200, 31
alpha = 0.7
beta = 0.5
gamma_V, gamma_W, gamma_X = 1.0, 1.0, 1.0
require_grad = {"V":True, "W":True, "X":True}

# Data
X = torch.randn(n, p, requires_grad=True)
Y = torch.randn(n, c, requires_grad=False)

########################################################

# Parameters
W = torch.randn(d, p, requires_grad=True)
V = torch.randn(c, d, requires_grad=True)

# Forward
hat_Y, A, H = forward(V, W, X, g, alpha, beta)

Delta_Y = hat_Y-Y # (n, c)
Sigma_A = ((beta**2)/(n*c)) * A.T@A # (d, d)
Psi = ((beta * alpha)/(n*c)) * (dg(H) * (Delta_Y@V))



##################################################################
##################################################################

Inc = torch.eye(n*c)
Ic = torch.eye(c)
In = torch.eye(n)
Id = torch.eye(d)

Hess_YT = (1/(n*c)) * Inc
QV = beta * torch.kron(A, Ic)
QW = alpha * beta * torch.kron(In, V) @ torch.diag(torch_vec(dg(H.T))) @ torch.kron(X, Id)

print(QV.shape, QW.shape, Hess_YT.shape)

GVV = QV.T @ Hess_YT @ QV # (cd, nc)(nc, nc)(nc, cd) = (cd, cd)
GWV = QW.T @ Hess_YT @ QV # (pd, nc)(nc, nc)(nc, cd) = (pd, cd)
GWW = QW.T @ Hess_YT @ QW # (pd, nc)(nc, nc)(nc, pd) = (pd, pd)

print(GVV.shape, GWV.shape, GWW.shape)


from plotter import plot_cdf
import matplotlib.pyplot as plt
Lambda = torch.linalg.eigvalsh(GVV.detach())
plt.hist(Lambda)
#_ = plot_cdf(Lambda)
print(Lambda)
plt.show()

exit()

##################################################################
##################################################################

# Gradient
grad = get_gradients(require_grad, Y, V, X, W, Sigma_A, Psi, beta, gamma_V, gamma_W, gamma_X)

#L_theta = torch.nn.MSELoss(reduction="mean")(input=hat_Y, target=Y)/2
#L_theta = ((hat_Y - Y)**2).mean(dim=1).mean()/2
#L_theta = ((hat_Y - Y)**2).sum(dim=1).sum()/(2*c*n)
L_theta = (1/(2*n*c))*torch.trace(Delta_Y.T @ Delta_Y)

L_theta += 0.5*(gamma_V*torch.trace(V.T@V) + gamma_W*torch.trace(W.T@W) + gamma_X*torch.trace(X.T@X))
L_theta.backward(retain_graph=True)
Vgrad = V.grad.detach()
Wgrad = W.grad.detach()
Xgrad = X.grad.detach()

print(equals(Vgrad, grad["dV"], dec=decimals), "\n", Vgrad,"\n", grad["dV"],"\n")
print(equals(Wgrad, grad["dW"], dec=decimals), "\n", Wgrad,"\n", grad["dW"],"\n")
print(equals(Xgrad, grad["dX"], dec=decimals), "\n", Xgrad, "\n", grad["dX"],"\n")

# Hessian

hess_dict_1 = get_hessian_loop(require_grad, grad["dV"], grad["dW"], grad["dX"], V, W, X)
Hess1 = from_dict_to_matrix(require_grad, hess_dict_1, n, p, c, d, device=W.device, dtype=torch.float64)

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