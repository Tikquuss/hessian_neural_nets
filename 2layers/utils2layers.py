import torch
import os
import sys

current_dir = os.path.dirname(__file__)  # current script's directory
parent_dir = os.path.join(current_dir, '..')  # parent directory
sys.path.append(parent_dir) 

from utils import torch_commutation_matrix, get_DFpqmn, from_DFpqmn_to_DF


###################### Usefull derivatives ###################### 

def get_DvecGvecX(n, p, d, c):
    Ip = torch.eye(p)
    Id = torch.eye(d)
    Inp = torch.eye(n*p)
    Kndpd = torch_commutation_matrix(n*d, p*d, min_size_csr=n*d*p*d+1)+0.0
    Kdn = torch_commutation_matrix(d, n, min_size_csr=d*n+1)+0.0
    return Kndpd @ torch.kron(torch.kron(Ip, Kdn), Id) @ torch.kron(Inp, torch_vec(Id).unsqueeze(1))

def get_DvecGvecW(n, p, d, c):
    Ip = torch.eye(p)
    In = torch.eye(n)
    Ipd = torch.eye(p*d)
    Kndnp = torch_commutation_matrix(n*d, n*p, min_size_csr=n*d*n*p+1)+0.0
    Knd = torch_commutation_matrix(n, d, min_size_csr=n*d+1)+0.0
    return Kndnp @ torch.kron(torch.kron(Ip, Knd), In) @ torch.kron(Ipd, torch_vec(In).unsqueeze(1))

####################################################################
# Hessian
###################################################################

def from_dict_to_matrix(require_grad, hess_dict, n, p, c, d, device, dtype=torch.float64):
    """
    V : (c, d)
    W : (d, p)
    X : (n, p)
    """
    ####################
    q=c*d*require_grad.get("V", False) + d*p*require_grad.get("W", False) + p*n*require_grad.get("X", False)
    Hess = torch.empty(q, q, dtype=dtype, device=device)
    ####################

    i, j = 0, 0
    # HVV & HVW & HVX
    if require_grad["V"] :
        Hess[i:i+c*d,j:j+c*d] = hess_dict["HVV"]
        j+=c*d
        if require_grad["W"] :
            Hess[i:i+c*d,j:j+d*p] = hess_dict["HVW"]
            j+=d*p
        if require_grad["X"] :
            Hess[i:i+c*d,j:j+p*n] = hess_dict["HVX"]
            j+=p*n
        i+=c*d

    # HWV & HWW & HWX
    j=0
    if require_grad["W"] :
        if require_grad["V"] :
            Hess[i:i+d*p,j:j+c*d] = hess_dict["HWV"]
            j+=c*d
        Hess[i:i+d*p,j:j+d*p] = hess_dict["HWW"]
        j+=d*p
        if require_grad["X"] :
            Hess[i:i+d*p,j:j+p*n] = hess_dict["HWX"]
            j+=p*n
        i+=d*p

    # HXV & HXW & HXX
    j=0
    if require_grad["X"] :
        if require_grad["V"] :
            Hess[i:i+n*p,j:j+c*d] = hess_dict["HXV"]
            j+=c*d
        if require_grad["W"] :
            Hess[i:i+n*p,j:j+d*p] = hess_dict["HXW"]
            j+=d*p
        Hess[i:i+n*p,j:j+p*n] = hess_dict["HXX"]
        j+=p*n
        i+=d*p

    return Hess

def get_hessian_loop(require_grad, dV, dW, dX, V, W, X):
    """
    V, dV : (c, d)
    W, dW : (d, p)
    X, dX : (n, p)
    """
    c, d = V.shape
    p = W.shape[1]
    n = X.shape[0]

    hess_dict = {}

    # HVV & HVW & HVX
    if require_grad["V"] :
        V.grad.zero_()
        hess_dict["HVV"] = from_DFpqmn_to_DF(get_DFpqmn(F=dV, X=V, p=c, q=d))
        if require_grad["W"] :
            W.grad.zero_()
            hess_dict["HVW"] = from_DFpqmn_to_DF(get_DFpqmn(F=dV, X=W, p=c, q=d))
        if require_grad["X"] :
            X.grad.zero_()
            hess_dict["HVX"] = from_DFpqmn_to_DF(get_DFpqmn(F=dV, X=X, p=c, q=d))

    # HWV & HWW & HWX
    if require_grad["W"] :
        if require_grad["V"] :
            V.grad.zero_()
            hess_dict["HWV"] = from_DFpqmn_to_DF(get_DFpqmn(F=dW, X=V, p=d, q=p))
        W.grad.zero_()
        hess_dict["HWW"] = from_DFpqmn_to_DF(get_DFpqmn(F=dW, X=W, p=d, q=p))
        if require_grad["X"] :
            X.grad.zero_()
            hess_dict["HWX"] = from_DFpqmn_to_DF(get_DFpqmn(F=dW, X=X, p=d, q=p))

    # HXV & HXW & HXX
    if require_grad["X"] :
        if require_grad["V"] :
            V.grad.zero_()
            hess_dict["HXV"] = from_DFpqmn_to_DF(get_DFpqmn(F=dX, X=V, p=n, q=p))
        if require_grad["W"] :
            W.grad.zero_()
            hess_dict["HXW"] = from_DFpqmn_to_DF(get_DFpqmn(F=dX, X=W, p=n, q=p))
        X.grad.zero_()
        hess_dict["HXX"] = from_DFpqmn_to_DF(get_DFpqmn(F=dX, X=X, p=n, q=p))

    return hess_dict