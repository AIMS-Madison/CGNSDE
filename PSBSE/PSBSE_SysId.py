import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

device = "cpu"
torch.manual_seed(0)
np.random.seed(0)


###################################################
################# Data Geneartion #################
###################################################
beta_x, beta_y, beta_z = 0.2, -0.3, -0.5
sigma_x, sigma_y, sigma_z = 0.3, 1, 1
alpha = 5


Lt = 600
dt = 0.001
Nt = int(Lt/dt) + 1
t = np.linspace(0, Lt, Nt)
u = np.zeros((Nt, 3))
u[0] = np.ones(3)
for n in range(Nt-1):
    u[n + 1, 0] = u[n, 0] + (beta_x * u[n,0] + alpha * u[n,0] * u[n,1] + alpha * u[n,1] * u[n,2]) * dt + sigma_x * np.sqrt(dt) * np.random.randn()
    u[n + 1, 1] = u[n, 1] + (beta_y * u[n,1] - alpha * u[n,0] ** 2 + 2 * alpha * u[n,0] * u[n,2]) * dt + sigma_y * np.sqrt(dt) * np.random.randn()
    u[n + 1, 2] = u[n, 2] + (beta_z * u[n,2] - 3 * alpha * u[n,0] * u[n,1]) * dt + sigma_z * np.sqrt(dt) * np.random.randn()

# Sub-sampling
u = u[::10]
dt = 0.01
Nt = int(Lt/dt) + 1
t = np.linspace(0, Lt, Nt)

# Split data in to train and test
u = torch.tensor(u[:-1], dtype=torch.float32)
t = torch.tensor(t[:-1], dtype=torch.float32)
Ntrain = 10000
Ntest = 50000
train_u = u[:Ntrain]
train_t = t[:Ntrain]
test_u = u[-Ntest:]
test_t = t[-Ntest:]



########################################################
################# System Identification ################
########################################################

# The CG Library [x, y, z, x^2, xy, xz]
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
train_u_dot = torch.diff(train_u, dim=0)/dt
train_LibCG = torch.stack([train_u[:, 0], train_u[:, 1], train_u[:, 2],
                           train_u[:, 0]**2, train_u[:, 0]*train_u[:, 1], train_u[:, 0]*train_u[:, 2]]).T[:-1]
CEM = cem(train_LibCG.numpy(), train_u_dot.numpy())

CEM.round(3)


