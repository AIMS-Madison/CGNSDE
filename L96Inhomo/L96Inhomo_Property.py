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


#########################################
########## Data Visualization ###########
#########################################
u = u.numpy()
t = t.numpy()
# 100s - 200s
u_part = u[10000:20000]
t_part = t[10000:20000]


fig = plt.figure(figsize=(12, 6))
# Pcolor
ax00 = plt.subplot2grid((1, 4), (0, 0), colspan=4)
c = ax00.pcolor(t_part, np.arange(I), u_part.T, cmap="magma")
cbar = fig.colorbar(c, ax=ax00, location='top')
cbar.ax.tick_params(labelsize=30, length=5, width=3)
ax00.set_title(r"\textbf{Hovmoller diagram}", fontsize=35, pad=130)
ax00.set_xlabel(r"$t$", fontsize=35)
ax00.set_ylabel(r"$i$", fontsize=35, rotation=0, labelpad=20)
ax00.set_yticks(np.array([0, 18, 35]))
ax00.set_xticks(np.arange(100, 150+10, 10))
for ax in fig.get_axes():
    ax.tick_params(labelsize=30, length=7, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
fig.tight_layout()
# fig.subplots_adjust(wspace=1.0)
plt.show()
