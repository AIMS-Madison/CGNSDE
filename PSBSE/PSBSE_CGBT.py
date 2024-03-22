import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
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


###################################################
################# Data Geneartion #################
###################################################

beta_x, beta_y, beta_z = 0.2, -0.5, -1
sigma_x, sigma_y, sigma_z = 0.3, 1., 1.
alpha = 1.2

Lt = 100
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


########## DA ###########
dim_u2 = 2
mu_trace = np.zeros((Nt, dim_u2, 1))
R_trace = np.zeros((Nt, dim_u2, dim_u2))
mu_trace[0] = np.zeros((dim_u2, 1))
R_trace[0] = np.eye(dim_u2)*0.01
mu0 = mu_trace[0]
R0 = R_trace[0]
for n in range(1, Nt):
    x0 = u[n-1, 0]
    x1 = u[n, 0]
    y1 = u[n, 1]
    f1 = np.array([[beta_x*x0]])
    g1 = np.array([[alpha*x0, 0]])
    # g1 = np.array([[alpha*x0, alpha*y1]])
    S1 = np.array([[sigma_x]])
    f2 = np.array([[-alpha*x0**2], [0]])
    g2 = np.array([beta_y, 2*alpha*x0, -3*alpha*x0, beta_z]).reshape(2,2)
    S2 = np.diag([sigma_y**2, sigma_z**2])
    invS1oS1 = np.linalg.inv(S1@S1.T)
    S2oS2 = S2*S2
    du1 = np.array([[x1 - x0]])
    mu1 = mu0 + (f2+g2@mu0)*dt + (R0@g1.T) @ invS1oS1 @ (du1 -(f1+g1@mu0)*dt)
    R1 = R0 + ( g2@R0 + R0@g2.T + S2oS2 - R0@g1.T@ invS1oS1 @ g1@R0 )*dt
    mu_trace[n] = mu1
    R_trace[n] = R1
    mu0 = mu1
    R0 = R1
mu_trace = mu_trace.reshape(mu_trace.shape[0], mu_trace.shape[1])

np.mean( ( u[:, 1:] - mu_trace)**2 )



# fig = plt.figure(figsize=(16, 10))
# axs = fig.subplots(2, 1)
# axs[0].plot(t, u[:, 1], linewidth=3, label="Original State")
# axs[0].plot(t, mu_trace[:, 0], linewidth=2, linestyle="dashed", label="DA Mean")
# axs[0].set_ylabel(r"\unboldmath$y$", fontsize=30, rotation=0)
# axs[0].tick_params(labelsize=30)
# axs[0].set_title(r"\textbf{NonCG: DA without $\alpha xy$}", fontsize=40)
# axs[1].plot(t, u[:, 2], linewidth=3)
# axs[1].plot(t, mu_trace[:, 1], linewidth=2, linestyle="dashed")
# axs[1].set_ylabel(r"\unboldmath$z$", fontsize=30, rotation=0)
# axs[1].tick_params(labelsize=30)
# fig.tight_layout()
# plt.show()
