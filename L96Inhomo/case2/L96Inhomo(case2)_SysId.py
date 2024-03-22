import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

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

# # Indices of u1 and u2
# indices_u1 = np.array([i for i in range(36) if i % 3 != 2])
# indices_u2 = np.array([i for i in range(36) if i % 3 == 2])
# dim_u1 = len(indices_u1)
# dim_u2 = len(indices_u2)


############################################
########## System Identifycation ###########
############################################

u = train_u.numpy()
u_dot = train_u_dot.numpy()

def cem(A, B):
    """
    :param A: numpy.array(Nt, Na); Basis Functions
    :param B: numpy.array(Nt, Nb); Dynamics
    :return: numpy.array(Nb, Na); Causation Entropy Matrix C(X->Y|Z)
    """
    Na = A.shape[1]
    Nb = B.shape[1]
    CEM = np.zeros((Nb, Na))
    for i in range(Nb):
        XYZ = np.concatenate([A, B[:, [i]]], axis=1)
        RXYZ = np.cov(XYZ.T)
        RXYZ_det = np.linalg.det(RXYZ)
        RXZ = RXYZ[:-1, :-1]
        RXZ_det = np.linalg.det(RXZ)
        for j in range(Na):
            RYZ = np.delete(np.delete(RXYZ, j, axis=0), j, axis=1)
            RYZ_det = np.linalg.det(RYZ)
            RZ = RYZ[:-1, :-1]
            RZ_det = np.linalg.det(RZ)
            CEM[i, j] = 1/2 * np.log(RYZ_det) - 1/2*np.log(RZ_det) - 1/2*np.log(RXYZ_det) + 1/2*np.log(RXZ_det)
    return CEM

def basisCG1(x):
    # x: shape(N, x)
    out = np.stack([x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4],
                    x[:, 0]**2, x[:, 2]**2, x[:, 4]**2,
                    x[:,0]*x[:,1], x[:,0]*x[:,2], x[:,0]*x[:,3], x[:,0]*x[:,4],
                    x[:,1]*x[:,2], x[:,1]*x[:,4],
                    x[:,2]*x[:,3], x[:,2]*x[:,4],
                    x[:,3]*x[:,4]]).T
    return out
def basisCG2(x):
    out = np.stack([x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4],
                    x[:, 1]**2, x[:, 3]**2,
                    x[:,0]*x[:,1], x[:,0]*x[:,3],
                    x[:,1]*x[:,2], x[:,1]*x[:,3], x[:,1]*x[:,4],
                    x[:,2]*x[:,3],
                    x[:,3]*x[:,4]]).T
    return out


LibCG1 = np.concatenate([basisCG1( u[:, [i-2, i-1, i, i+1, (i+2)%I]] ) for i in indices_u1], axis=0)
u_dot1_stacked = u_dot[:, indices_u1].T.reshape(-1, 1)
np.where( ( cem(LibCG1, u_dot1_stacked)*1e7 > 1e5).flatten() )
cem(LibCG1, u_dot1_stacked).round(3)



LibCG2 = np.concatenate([basisCG2( u[:, [i-2, i-1, i, (i+1)%I, (i+2)%I]] ) for i in indices_u2], axis=0)
u_dot2_stacked = u_dot[:, indices_u2].T.reshape(-1, 1)
np.where( (cem(LibCG2, u_dot2_stacked)*1e7 > 1e5).flatten() )
cem(LibCG2, u_dot2_stacked).round(3)

