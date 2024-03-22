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
indices_u1 = np.array([i for i in range(36) if i % 3 != 2])
indices_u2 = np.array([i for i in range(36) if i % 3 == 2])
dim_u1 = len(indices_u1)
dim_u2 = len(indices_u2)


################################
########## CG-Filter ###########
################################
u = test_u.numpy()
Nt = u.shape[0]
indices_u1 = np.array([i for i in range(36) if i % 3 != 2])
indices_u2 = np.array([i for i in range(36) if i % 3 == 2])

dim_u1 = len(indices_u1)
dim_u2 = len(indices_u2)
mu_trace = np.zeros((Nt, dim_u2, 1))
R_trace = np.zeros((Nt, dim_u2, dim_u2))
mu_trace[0] = np.zeros((dim_u2, 1))
R_trace[0] = np.eye(dim_u2)*0.01
mu0 = mu_trace[0]
R0 = R_trace[0]

for n in range(1, Nt):
    du1 = (u[n, indices_u1] - u[n-1, indices_u1]).reshape(-1, 1)

    f1 = F - (c_lst[indices_u1]*u[n-1, indices_u1]).reshape(-1, 1)
    g1 = np.zeros((dim_u1, dim_u2))
    for j in range(dim_u2):
        g1[np.arange(2*j+1, 2*j+4)%dim_u1, j] = [u[n-1, 3*j], u[n-1, (3*j+4)%I]-u[n-1, 3*j+1], -u[n-1, (3*j+3)%I]]
    s1 = np.diag([sigma]*dim_u1)
    f2 = np.zeros((dim_u2, 1))
    for i in range(dim_u2):
        f2[i] = F + (u[n-1, (3*i+3)%I] - u[n-1, 3*i]) * u[n-1, 3*i+1]
    g2 = np.diag(-c_lst[indices_u2])
    s2 = np.diag([sigma]*dim_u2)
    invs1os1 = np.linalg.inv(s1@s1.T)
    s2os2 = s2@s2.T

    mu1 = mu0 + (f2+g2@mu0)*dt + (R0@g1.T) @ invs1os1 @ (du1 -(f1+g1@mu0)*dt)
    R1 = R0 + ( g2@R0 + R0@g2.T + s2os2 - R0@g1.T@ invs1os1 @ g1@R0 )*dt
    mu_trace[n] = mu1
    R_trace[n] = R1
    mu0 = mu1
    R0 = R1

np.mean( (u[:, indices_u2] - mu_trace.squeeze(2) )**2 )


def avg_neg_log_likehood(x, mu, R):
    # x, mu are in matrix form, e.g. (t, x, 1)
    d = x.shape[1]
    neg_log_likehood = 1/2*(d*np.log(2*np.pi) + torch.log(torch.linalg.det(R)) + ((x-mu).permute(0,2,1)@torch.linalg.inv(R)@(x-mu)).flatten())
    return torch.mean(neg_log_likehood)
avg_neg_log_likehood(torch.tensor(u[:,indices_u2].reshape(20000, 12 ,1)),
                     torch.tensor(mu_trace),
                     torch.tensor(R_trace))


plt.plot(u[:, indices_u2][:, 0])
plt.plot( mu_trace.squeeze(2)[:, 0] )
plt.show()

