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

####################################################################
# Utils
####################################################################

g, dg, ddg = torch.tanh, lambda x : 1 - torch.tanh(x)**2, lambda x : - 2*torch.tanh(x)*(1 - torch.tanh(x)**2)
npg, npdg, npddg = np.tanh, lambda x : 1 - np.tanh(x)**2, lambda x : - 2*np.tanh(x)*(1 - np.tanh(x)**2)

# g, dg, ddg = torch.relu, lambda x : 1.0*(x>=0) + 0.0*(x<0), lambda x : torch.zeros_like(x)
# npg, npdg, npddg = lambda x : np.maximum(0, x), lambda x : 1.0*(x>=0) + 0.0*(x<0), lambda x : np.zeros_like(x)

# g, dg, ddg = lambda x:x, lambda x : torch.ones_like(x) , lambda x : torch.zeros_like(x)
# npg, npdg, npddg = lambda x:x, lambda x : np.ones_like(x) , lambda x : np.zeros_like(x) 

MIN_SIZE_CSR = 10000

####################################################################
# Gradients AND Hessian check 1
####################################################################

n, p, d, c = 5, 4, 3, 2
#n, p, d, c = 2, 2, 2, 2
decimals = 4
gamma_V, gamma_W = 0.1, 1.0
beta = 1.0 + 1.1

X = torch.randn(n, p, requires_grad=False)
W = torch.randn(d, p, requires_grad=True)
V = torch.randn(c, d, requires_grad=True)

Y_start = torch.empty(n, dtype=torch.long).random_(c)
P_start = torch.nn.functional.one_hot(Y_start, num_classes=c).float() # (n, c)

# forward

H = (1/d**0.5) * X @ W.T # (n, d)
A = g(H) # (n, d)
Y = A @ V.T # (n, c)
P = (beta * Y).exp() / ( (beta * Y).exp() @ torch.ones(c, c)) # (n, c)

Delta_P = P-P_start # (n, c)
Psi = (beta/(n*(d**0.5))) * (dg(H) * (Delta_P@V))
dV = (beta/n) * Delta_P.T @ A + gamma_V*V
dW = Psi.T @ X + gamma_W * W

H1 = get_hessian_loop({"V":True, "W":True}, dV, dW, V, W)
H2 = get_hessian({"V":True, "W":True}, P, P_start, A, H, V, W, X, beta, dg, ddg, gamma_V=gamma_V, gamma_W=gamma_W)

np_relu = lambda x : np.maximum(0, x)
H3 = get_hessian_np({"V":True, "W":True}, P, P_start, A, H, V, W, X, beta, npdg, npddg, gamma_V=gamma_V, gamma_W=gamma_W)

print(equals(H1, H2, dec=decimals))#, "\n", H1,"\n", H2,"\n")
print(equals(H1, H3, dec=decimals))

####################################################################
# Gradients AND Hessian check 1
####################################################################

# n, p, d, c = 5, 4, 3, 2
# #n, p, d, c = 2, 2, 2, 2
# decimals = 4
# gamma_V, gamma_W = 0.1, 1.0
# beta = 1.0 + 1.1

# X = torch.randn(n, p, requires_grad=False)
# W = torch.randn(d, p, requires_grad=True)
# V = torch.randn(c, d, requires_grad=True)

# Y_start = torch.empty(n, dtype=torch.long).random_(c)
# P_start = torch.nn.functional.one_hot(Y_start, num_classes=c).float() # (n, c)

# # forward

# H = (1/d**0.5) * X @ W.T # (n, d)
# A = g(H) # (n, d)
# Y = A @ V.T # (n, c)
# P = (beta * Y).exp() / ( (beta * Y).exp() @ torch.ones(c, c)) # (n, c)

# Delta_P = P-P_start # (n, c)

# # Gradient
# # L_theta = - (1/n) * (P_start.T @ (P.log())).trace() + 0.5*(gamma_V*torch.trace(V.T@V) + gamma_W*torch.trace(W.T@W) + gamma_X*torch.trace(X.T@X))
# # L_theta.backward(retain_graph=True)
# # Vgrad = V.grad.detach()
# # Wgrad = W.grad.detach()

# Psi = (beta/(n*(d**0.5))) * (dg(H) * (Delta_P@V))
# dV = (beta/n) * Delta_P.T @ A + gamma_V*V
# dW = Psi.T @ X + gamma_W * W

# # print(equals(Vgrad, dV, dec=decimals), "\n", Vgrad,"\n", dV,"\n")
# # print(equals(Wgrad, dW, dec=decimals), "\n", Wgrad,"\n", dW,"\n")

# ####################################################################
# # Hessian V
# ####################################################################

# One_cc = torch.ones(c, c)
# In = torch.eye(n)
# Inc = torch.eye(n*c)
# Ic = torch.eye(c)
# Icd = torch.eye(c*d)
# Id = torch.eye(d)

# S_tilde = Inc -  torch.kron(In, One_cc) @ torch.diag(torch_vec(P.T))
# R_tilde = torch.diag(torch_vec(P.T)) @ S_tilde
# S = Inc -  torch.kron(One_cc, In) @ torch.diag(torch_vec(P))
# R = torch.diag(torch_vec(P)) @ S

# # tmp1 = torch_commutation_matrix(n, c, MIN_SIZE_CSR) @ S @ torch_commutation_matrix(c, n, MIN_SIZE_CSR)
# # tmp2 = torch_commutation_matrix(n, c, MIN_SIZE_CSR) @ R @ torch_commutation_matrix(c, n, MIN_SIZE_CSR)
# # print(equals(tmp1, S_tilde, dec=decimals))#, "\n", tmp1,"\n", S_tilde,"\n")
# # print(equals(tmp2, R_tilde, dec=decimals))#, "\n", tmp2,"\n", R_tilde,"\n")

# HVV =  (beta**2/n) * torch.kron(A, Ic).T @ R_tilde @ torch.kron(A, Ic)  + gamma_V*Icd  # (cd, cd)
# #Kcn = torch_commutation_matrix(c, n, min_size_csr=MIN_SIZE_CSR)+0.0
# #HVV =  (beta**2/n) * (Kcn @ torch.kron(A, Ic)).T @ R @ (Kcn @ torch.kron(A, Ic))  + gamma_V*Icd  # (cd, cd)

# # DFcdcdVV1 = from_DF_to_DFpqmn(DF=HVV, m=c, n=d, p=c, q=d)
# # DFcdcdVV2 = get_DFpqmn(F=dV, X=V, p=c, q=d)
# # print(equals(DFcdcdVV1, DFcdcdVV2, dec=decimals))#, "\n", DFcdcdVV1,"\n", DFcdcdVV2,"\n")

# ########

# Knc = torch_commutation_matrix(n, c, min_size_csr=MIN_SIZE_CSR)+0.0
# T1 = (beta/(n*(d**0.5))) * (torch.kron(Id, Delta_P).T +  beta * torch.kron(A, Ic).T @ R_tilde @ Knc @ torch.kron(V, In)) @ torch.diag(torch_vec(dg(H)))
# Kdc = torch_commutation_matrix(d, c, min_size_csr=MIN_SIZE_CSR)+0.0
# T2 = (beta/(n*(d**0.5)))*(torch.kron(Id, Delta_P).T + beta* Kdc @ torch.kron(Ic, A).T @ R @ torch.kron(V, In)) @ torch.diag(torch_vec(dg(H)))

# print(equals(T1, T2, dec=decimals))#, "\n", T1,"\n", T2,"\n")

# # Kdp = torch_commutation_matrix(d, p, min_size_csr=MIN_SIZE_CSR)+0.0
# # HVW = T1 @ torch.kron(Id, X) @ Kdp # (cd, dp)
# # DFcdcdVW1 = from_DF_to_DFpqmn(DF=HVW, m=d, n=p, p=c, q=d)
# # #V.grad.zero_()
# # #W.grad.zero_()
# # DFcdcdVW2 = get_DFpqmn(F=dV, X=W, p=c, q=d)
# # print(equals(DFcdcdVW1, DFcdcdVW2, dec=decimals))#, "\n", DFcdcdVW1,"\n", DFcdcdVW2,"\n")

# T_tilde = (beta/(n*(d**0.5))) * (Kdc @ torch.kron(Delta_P, Id).T + beta * torch.kron(A, Ic).T @ R_tilde @ torch.kron(In, V)) @ torch.diag(torch_vec(dg(H.T)))
# # tmp3 = T_tilde @ torch_commutation_matrix(n, d, MIN_SIZE_CSR)
# # print(equals(tmp3, T1, dec=decimals))

# # Kpd = torch_commutation_matrix(p, d, min_size_csr=MIN_SIZE_CSR)+0.0
# # HWV = Kpd @ torch.kron(Id, X).T @ T1.T
# # HWV = torch.kron(X, Id).T @ T_tilde.T 
# # DFHWV1  = from_DF_to_DFpqmn(DF=HWV, m=c, n=d, p=d, q=p)
# # DFHWV2 = get_DFpqmn(F=dW, X=V, p=d, q=p)
# # # V.grad.zero_()
# # # W.grad.zero_()
# # print(equals(DFHWV1, DFHWV2, dec=decimals))#, "\n", DFHWV1,"\n", DFHWV2,"\n")


# ############################################################

# tmp = torch.kron(V, In) @ torch.diag(torch_vec(dg(H)))
# N = (beta/(n*d))*(  torch.diag(torch_vec(Delta_P@V)) @  torch.diag(torch_vec(ddg(H)))  + beta * tmp.T@R@tmp  )
# N = (beta/(n*d))*(  torch.diag( torch_vec(Delta_P@V) * torch_vec(ddg(H)) )  + beta * tmp.T@R@tmp )
# N = (beta/(n*d))*(  torch.diag( torch_vec ( (Delta_P@V) * ddg(H) )  )  + beta * tmp.T@R@tmp )
# Kdp = torch_commutation_matrix(d, p, min_size_csr=MIN_SIZE_CSR)
# tmp = torch.kron(Id, X) @ Kdp
# HWW = tmp.T @ N @ tmp + gamma_W * torch.eye(d*p)
# DFHWW1  = from_DF_to_DFpqmn(DF=HWW, m=d, n=p, p=d, q=p)
# DFHWW2 = get_DFpqmn(F=dW, X=W, p=d, q=p)
# print(equals(DFHWW1, DFHWW2, dec=decimals))#, "\n", DFHWW1,"\n", DFHWW2,"\n")


      

