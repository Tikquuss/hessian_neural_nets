import torch
import numpy as np

MIN_SIZE_CSR = 10000

###################### Activation functions ###################### 

activations_functions = {
    "id" : {"g" : lambda x:x, "dg" : lambda x : torch.ones_like(x), "ddg" : lambda x : torch.zeros_like(x)},
    "relu" : {"g" : torch.relu, "dg" : lambda x : 1.0*(x>=0) + 0.0*(x<0), "ddg" : lambda x : torch.zeros_like(x)},
    "tanh" : {"g" : torch.tanh, "dg" : lambda x : 1 - torch.tanh(x)**2, "ddg" : lambda x : - 2*torch.tanh(x)*(1 - torch.tanh(x)**2)},
    "sigmoid" : {"g" : torch.sigmoid, "dg" : lambda x : torch.sigmoid(x) * (1 - torch.sigmoid(x)), 
                 "ddg" : lambda x : torch.sigmoid(x) * (1 - torch.sigmoid(x)) * (1 - 2*torch.sigmoid(x))},
    "sin" : {"g" : torch.sin, "dg" : torch.cos, "ddg" : lambda x : -torch.sin(x)},
    "x2" : {"g" : lambda x:x*x, "dg" : lambda x:2*x, "ddg" : lambda x:2*torch.ones_like(x)},
}

###################### Equal and round randn ###################### 

def equals(m1, m2, dec=10) : return torch.equal(m1.round(decimals=dec), m2.round(decimals=dec))

def rand_int(m, n, a=0.0, b=1.0, requires_grad=False) :
  return torch.randn(m, n).uniform_(a, b).round(decimals=0).requires_grad_(requires_grad)

###################### Vectorization ###################### 

def np_vec(A): return A.reshape(A.shape[0]*A.shape[1], order='F')
def np_unvec(x, m, n):
    """Inverse of the vectorization operator : mn ---> (m, n)
    https://math.stackexchange.com/a/3122442/1020794"""
    Im, In = np.eye(m), np.eye(n)
    return np.kron(np_vec(In).T, Im) @ np.kron(In, np.expand_dims(x, 1))

def torch_vec(A): return A.T.flatten()#.unsqueeze(1) # A.T.reshape(A.shape[0]*A.shape[1])
def torch_unvec(x, m, n):
    """Inverse of the vectorization operator : mn ---> (m, n)
    https://math.stackexchange.com/a/3122442/1020794"""
    Im, In = torch.eye(m), torch.eye(n)
    return torch.kron(torch_vec(In).T, Im) @ torch.kron(In, x.unsqueeze(1))

###################### Commutation matrix ###################### 

def np_commutation_matrix(m, n, min_size_csr=1):
    # https://stackoverflow.com/a/60680132/11814682
    row  = np.arange(m*n)
    col  = row.reshape((m, n), order='F').ravel()
    data = np.ones(m*n, dtype=np.int8)
    K = csr_matrix((data, (row, col)), shape=(m*n, m*n))
    # If the size is small, no need for 'Compressed Sparse Row' representation
    if n*m < min_size_csr : return K+np.zeros(m*n, dtype=int)
    return K

def torch_commutation_matrix(m, n, min_size_csr=1):

    # coo = np_commutation_matrix(m, n, min_size_csr=0).tocoo()
    # row, col = coo.row, coo.col
    # #values = torch.FloatTensor(coo.data)
    # values = torch.LongTensor(coo.data.astype(np.int32))

    row  = np.arange(m*n)
    col  = row.reshape((m, n), order='F').ravel()
    values =  torch.ones(m*n, dtype=torch.int8)

    #indices = torch.LongTensor([row.tolist(), col.tolist()])
    indices = torch.LongTensor(np.vstack((row, col)))

    K=torch.sparse.FloatTensor(indices, values)
    #K=torch.sparse.LongTensor(indices, values)
    if n*m < min_size_csr : return K.to_dense() + 0.0
    return K

###################### For derivatives & gradients ###################### 

def get_DF(vecF, vecX, p, q):
    DF_for = []
    for i in range(p*q) :
        vecF[i].backward(retain_graph=True)
        #vecF[i].backward(retain_graph=(i!=p*q-1))
        DF_for.append(vecX.grad.tolist())
        vecX.grad.zero_()
    return torch.tensor(DF_for)

def get_DFpqmn(F, X, p, q):
    DFpqmnGrad = []
    for i in range(p) :
        tmp = []
        for j in range(q) :
            F[i][j].backward(retain_graph=True)
            #F[i][j].backward(retain_graph=((i!=p-1) or (j!=q-1)))
            tmp.append(X.grad.tolist())
            X.grad.zero_()
        DFpqmnGrad.append(tmp)
    return torch.tensor(DFpqmnGrad)

def from_DF_to_DFpqmn(DF, m, n, p, q):
    DFpqmnCompare = []
    for i in range(p) :
        tmp1 = []
        for j in range(q) :
            tmp2 = []
            for k in range(m) :
                tmp3 = []
                for l in range(n) : tmp3.append(DF[i+j*p, k+l*m].item())
                tmp2.append(tmp3)
            tmp1.append(tmp2)
        DFpqmnCompare.append(tmp1)
    return torch.tensor(DFpqmnCompare)

def from_DFpqmn_to_DF(DFpqmn):
    p, q, m, n = DFpqmn.shape
    DF = torch.zeros(p*q, m*n)
    for i in range(p) :
        for j in range(q) :
            for k in range(m) :
                for l in range(n) : DF[i+j*p, k+l*m] = DFpqmn[i,j,k,l]#.item()
    return DF

def from_DF_to_DFpmqn(DF, m, n, p, q):
    DFpqmnCompare = []
    for i in range(p) :
        tmp1 = []
        for j in range(m) :
            tmp2 = []
            for k in range(q) :
                tmp3 = []
                for l in range(n) : tmp3.append(DF[i+k*p, j+l*m].item())
                tmp2.append(tmp3)
            tmp1.append(tmp2)
        DFpqmnCompare.append(tmp1)
    return torch.tensor(DFpqmnCompare)

def from_DFpmqn_to_DF(DFpmqn):
    p, m, q, n = DFpmqn.shape
    DF = torch.zeros(p*q, m*n)
    for i in range(p) :
        for j in range(m) :
            for k in range(q) :
                for l in range(n) : DF[i+k*p, j+l*m] = DFpmqn[i,j,k,l]#.item()
    return DF

def from_DFpqmn_to_DFpmqn(DFpqmn): return DFpqmn.transpose(1, 2)
def from_DFpmqn_to_DFpqmn(DFpmqn): return DFpmqn.transpose(1, 2)

def from_DF_p_q_m_n_TO_p_q_mn(DF_p_q_m_n):
    return DF_p_q_m_n.transpose(2, 3).flatten(start_dim=2, end_dim=3) # (p, q, m*n)

def from_DF_p_q_mn_TO_p_q_m_n(DF_p_q_mn, m, n):
    return DF_p_q_mn.unflatten(dim=-1, sizes=(n, m)).transpose(2, 3) # (p, q, m, n)

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