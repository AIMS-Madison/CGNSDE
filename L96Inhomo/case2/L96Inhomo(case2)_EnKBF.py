import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torchdiffeq
import time

device = "cpu"
torch.manual_seed(0)
np.random.seed(0)

mpl.use("Qt5Agg")
plt.rcParams["agg.path.chunksize"] = 10000
plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \boldmath"

######################################
########## Data Generation ###########
######################################

c_lst = 2 + 1.5*np.sin(2*np.pi*np.arange(36)/36)

F = 8
sigma = 0.5

I = 36
Lt = 300
dt = 0.001
Nt = int(Lt/dt) + 1
t = np.linspace(0, Lt, Nt)
u = np.zeros((Nt, I))

for n in range(Nt-1):
    for i in range(I):
        u_dot = -c_lst[i]*u[n, i] + u[n,(i+1)%I]*u[n,i-1] - u[n,i-2]*u[n,i-1] + F
        u[n+1, i] = u[n, i] + u_dot*dt + sigma*np.sqrt(dt)*np.random.randn()

# Sub-sampling
u = u[::10]
dt = 0.01
Nt = int(Lt/dt) + 1
t = np.linspace(0, Lt, Nt)
u_dot = np.diff(u, axis=0)/dt

# Split data in to train and test
u_dot = torch.tensor(u_dot, dtype=torch.float32)
u = torch.tensor(u[:-1], dtype=torch.float32)
t = torch.tensor(t[:-1], dtype=torch.float32)

Ntrain = 10000
Ntest = 20000
train_u = u[:Ntrain]
train_u_dot = u_dot[:Ntrain]
train_t = t[:Ntrain]
test_u = u[-Ntest:]
test_u_dot = u_dot[-Ntest:]
test_t = t[-Ntest:]

# Indices of u1 and u2
indices_u1 = np.arange(0, 36, 2)
indices_u2 = np.arange(1, 36, 2)
dim_u1 = len(indices_u1)
dim_u2 = len(indices_u2)


##########################################
################# EnKBF  #################
##########################################

def avg_neg_log_likehood(x, mu, R):
    # x, mu are in matrix form, e.g. (t, x, 1)
    d = x.shape[1]
    neg_log_likehood = 1/2*(d*np.log(2*np.pi) + torch.log(torch.linalg.det(R)) + ((x-mu).permute(0,2,1)@torch.linalg.inv(R)@(x-mu)).flatten())
    return torch.mean(neg_log_likehood)

u = test_u.numpy()
def cross_cov(X, Y):
    n = X.shape[0]
    assert n == Y.shape[0]
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    cross_cov_matrix = np.dot(X_centered.T, Y_centered) / (n - 1)
    return cross_cov_matrix

u1 = u[:, indices_u1]
SIG1 = np.diag([sigma**2]*dim_u1)
SIG2 = np.diag([sigma**2]*dim_u2)
sig1 = np.diag([sigma]*dim_u1)
sig2 = np.diag([sigma]*dim_u2)

J = 100
p = dim_u2
u2_ens = np.zeros((J, Ntest, p))
err_lst = []
nll_lst = []
for _ in range(100):
    for n in range(1, Ntest):
        u1_repeat = np.tile(u1[n-1], (J, 1))
        u_ens = np.stack([u1_repeat, u2_ens[:, n-1]]).transpose(1,2,0).reshape(-1, 36)
        u_dot_ens = np.zeros_like(u_ens)
        for i in range(I):
            u_dot_ens[:, i] = -c_lst[i]*u_ens[:, i] + u_ens[:,(i+1)%I]*u_ens[:,i-1] - u_ens[:,i-2]*u_ens[:,i-1] + F
        g = u_dot_ens[:, ::2]
        f = u_dot_ens[:, 1::2]

        g_bar = np.mean(g, axis=0)
        CCOV = cross_cov(u2_ens[:, n-1], g)
        Sys_term = f*dt + np.random.randn(J,p) @ sig2 * np.sqrt(dt)
        DA_term = -0.5*((g+g_bar)*dt-2*(u1[n]-u1[n-1])) @ (CCOV@np.linalg.inv(SIG1)).T

        u2_ens[:, n, :] = u2_ens[:, n-1, :] + Sys_term + DA_term

    mu_trace = np.mean(u2_ens, axis=0).reshape(u2_ens.shape[1], u2_ens.shape[2], 1)
    R_trace = np.zeros((u2_ens.shape[1], u2_ens.shape[2], u2_ens.shape[2]))
    for i in range(u2_ens.shape[1]):
        R_trace[i] = np.cov(u2_ens[:, i, :].T)

    nll = avg_neg_log_likehood(torch.tensor(u[:,indices_u2].reshape(20000, 18 ,1)[1:]),
                             torch.tensor(mu_trace)[1:],
                             torch.tensor(R_trace)[1:]).item()

    err = np.mean( (u[:, indices_u2] - mu_trace.squeeze(2) )**2 )

    err_lst.append(err)
    nll_lst.append(nll)
    print(_, err, nll)


np.mean(err_lst)
np.mean(nll_lst)



