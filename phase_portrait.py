import numpy as np 
import matplotlib.pyplot as plt 
import torch

def plot_flow(X, Y, dXdt, dYdt, S=None):
    fig = plt.figure(figsize = (6, 6)) 
    ax = fig.add_subplot(111)
    #color = "k"
    color = None
    lw=0.5
    #speed = np.sqrt(dXdt**2 + dYdt**2)
    ax.streamplot(X, Y, dXdt, dYdt, density=1.5, color=color, linewidth=lw, cmap='autumn')
    ax.set_xlabel(u'x') 
    ax.set_ylabel(u'y')
    S = {} if S is None else S
    for (x, y) in S :
        ax.plot([x], [y], marker="o", markersize=5, label = f"({x}, {y})",
                #markeredgecolor="red", markerfacecolor="green"
                )
    ax.grid()
    ax.legend()
    plt.show()

def flow_1(X, Y) :
    # https://stackoverflow.com/q/62752013/11814682
    dXdt = Y
    dYdt = 1/4.*(X**2 + Y**2)**2 - (X**2 + Y**2) - X
    S = {(0, 0)} # and more x**3 - 4x - 4 = 0
    return dXdt, dYdt, S

def flow_2(X, Y) :
    dXdt = (X-2*Y)*X
    dYdt = (X-2)*Y
    S = {(0, 0), (2, 1)}
    return dXdt, dYdt, S

def flow_3(X, Y) :
    dXdt = X - Y - X**2 + X*Y
    dYdt = - X**2 - Y
    S = {(0, 0), (1, -1), (-1, -1)}
    return dXdt, dYdt, S


activations_functions = {
    "id" : {"g" : lambda x:x, "dg" : lambda x : torch.ones_like(x), "ddg" : lambda x : torch.zeros_like(x)},
    "relu" : {"g" : torch.relu, "dg" : lambda x : 1.0*(x>=0) + 0.0*(x<0), "ddg" : lambda x : torch.zeros_like(x)},
    "tanh" : {"g" : torch.tanh, "dg" : lambda x : 1 - torch.tanh(x)**2, "ddg" : lambda x : - 2*torch.tanh(x)*(1 - torch.tanh(x)**2)},
    "sigmoid" : {"g" : torch.sigmoid, "dg" : lambda x : torch.sigmoid(x) * (1 - torch.sigmoid(x)), 
                 "ddg" : lambda x : torch.sigmoid(x) * (1 - torch.sigmoid(x)) * (1 - 2*torch.sigmoid(x))},
    "sin" : {"g" : torch.sin, "dg" : torch.cos, "ddg" : lambda x : -torch.sin(x)},
    "x2" : {"g" : lambda x : x**2, "dg" : lambda x : 2*x, "ddg" : lambda x : 2*torch.ones_like(x)},
}

act = activations_functions["id"]
act = activations_functions["relu"]
act = activations_functions["tanh"]
act = activations_functions["sigmoid"]
act = activations_functions["sin"]
act = activations_functions["x2"]

g, dg = act["g"], act["dg"]
alpha = .1
beta = 1.0
tau = 1.0
gamma_W = 1.0
gamma_V = 1.0
penalty = "l2"

def nn2layers(X, P, W, V) :
    """
    X : (n, p)
    P : (n, c)
    W : (d, p)
    V : (c, d)
    """
    n = X.shape[0]

    # c, d = V.shape
    # p = W.shape[1]
    # H = alpha * (X @ W.T) # (n, p) x (p, d) = (n, d)
    # A = g(H) # (n, d)    
    # Y_hat = beta * (A @ V.T) # (n, d) x (d, c) = (n, c)
    # #P_hat = torch.softmax(Y_hat, dim=1) # (n, c)
    # One_cc = torch.ones(c, c) # (c, c)
    # P_hat = Y_hat.exp() / (Y_hat.exp() @ One_cc) # (n, c)
    # Delta_P = P_hat-P # (n, c)
    # Psi = (beta * alpha/n) * (dg(H) * (Delta_P@V))
    # dWdt = Psi.T @ X + gamma_W * (W if penalty == "l2" else 1.0*(W >= 0)) # (d, p)
    # dVdt = (beta / n) * Delta_P.T @ A + gamma_V*(V if penalty == "l2" else 1.0*(V >= 0)) # (c, d)

    p, d, c = 1, 1, 1
    H = alpha * (X * W) # (n,) x (1,) = (n,)
    A = g(H) # (n,)    
    Y_hat = beta * (A * V) # (n,) x (1,) = (n,)
    #P_hat = torch.softmax(Y_hat, dim=1) # (n, c)
    One_cc = torch.ones(c, c) # (c, c)
    P_hat = Y_hat.exp().unsqueeze(dim=1) / (Y_hat.exp().unsqueeze(dim=1) @ One_cc) # (n, 1)
    P_hat = P_hat.squeeze() # (n,)
    Delta_P = P_hat-P # (n,)
    Psi = (beta * alpha/n) * (dg(H) * (Delta_P*V)) # (n,)    
    Psi = Psi.unsqueeze(dim=1) # (n, 1)
    dWdt = Psi.T @ X + gamma_W * (W if penalty == "l2" else 1.0*(W >= 0)) # (d, p)
    Delta_P = Delta_P.unsqueeze(dim=1) # (n, 1)
    dVdt = (beta / n) * Delta_P.T @ A + gamma_V*(V if penalty == "l2" else 1.0*(V >= 0)) # (c, d)
    dWdt = dWdt.squeeze()
    dVdt = dVdt.squeeze()
    
    return dWdt, dVdt



def flow_4(W, V) :
    W = torch.from_numpy(W)
    V = torch.from_numpy(V)

    #n, p, d, c = 2, 2, 4, 4
    n, p, d, c = 1000, 1, 1, 1
    #W = torch.randn(d, p, requires_grad=False).squeeze() # (d, p)
    #V = torch.randn(c, d, requires_grad=False).squeeze() # (c, d)
    X = torch.randn(n, p, requires_grad=False).squeeze() # (n, p)
    Y = torch.empty(n, dtype=torch.long).random_(c) # (n,)
    P = torch.nn.functional.one_hot(Y, num_classes=c).float().squeeze() # (n, c)
    P = tau * P + (1-tau)/c #* torch.ones_like(n, c)

    #dWdt, dVdt = nn2layers(X, P, W, V)

    m1, m2 = W.shape
    dWdt = torch.zeros_like(W)
    dVdt = torch.zeros_like(V)
    for i in range(m1) :
        for j in range(m2) :
            dWdt[i][j], dVdt[i][j] = nn2layers(X, P, W[i][j], V[i][j])

    S = {(0, 0)}
    return dWdt.numpy(), dVdt.numpy(), S

a, b = -4, 4
a, b = -10, 10
a, b = -100, 100
m = 100j
m = 200j
#m = 1000j
# define the grid on which to plot the vectorfield here: a grid of m x m points between a and b
dY, dX = np.mgrid[a:b:m, a:b:m] # (m, m) and (m, m)

dXdt, dYdt, S = flow_3(dX, dY)
plot_flow(dX, dY, dXdt, dYdt, S=S)
