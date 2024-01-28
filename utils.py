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

def np_sigmoid(x) : return 1 / (1 + np.exp(-x))
activations_functions_np = {
    "id" : {"g" : lambda x:x, "dg" : lambda x : np.ones_like(x), "ddg" : lambda x : np.zeros_like(x)},
    "relu" : {"g" : lambda x:np.maximum(0, x), "dg" : lambda x : 1.0*(x>=0) + 0.0*(x<0), "ddg" : lambda x : np.zeros_like(x)},
    "tanh" : {"g" : np.tanh, "dg" : lambda x : 1 - np.tanh(x)**2, "ddg" : lambda x : - 2*np.tanh(x)*(1 - np.tanh(x)**2)},
    "sigmoid" : {"g" : np_sigmoid, "dg" : lambda x : np_sigmoid(x) * (1 - np_sigmoid(x)),
                 "ddg" : lambda x : np_sigmoid(x) * (1 - np_sigmoid(x)) * (1 - 2*np_sigmoid(x))},
    "sin" : {"g" : np.sin, "dg" : np.cos, "ddg" : lambda x : -np.sin(x)},
    "x2" : {"g" : lambda x:x*x, "dg" : lambda x:2*x, "ddg" : lambda x:2*np.ones_like(x)},
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

