import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
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
        u_dot = -u[n, i] + u[n,(i+1)%I]*u[n,i-1] - u[n,i-2]*u[n,i-1] + F
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
# 100s - 150s
u_part = u[10000:15000]
t_part = t[10000:15000]

def acf(x, lag=500):
    i = np.arange(0, lag+1)
    v = np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  for i in range(1, lag+1)])
    return (i, v)
t_lags = np.linspace(0, 5, 501)



plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
fig = plt.figure(figsize=(14, 16))
# Pcolor
ax00 = plt.subplot2grid((3, 4), (0, 0), colspan=4)
c = ax00.pcolor(t_part[::10], np.arange(1, I+1), u_part[::10].T, cmap="magma")
cbar = fig.colorbar(c, ax=ax00, location='top')
cbar.ax.tick_params(labelsize=30, length=5, width=3)
ax00.set_title(r"(a) Hovmoller diagram", fontsize=35, pad=100)
ax00.set_xlabel(r"$t$", fontsize=35)
ax00.set_ylabel(r"$i$", fontsize=35, rotation=0, labelpad=20)
ax00.set_yticks(np.array([1, 18, 36]))
ax00.set_xticks(np.arange(100, 150+10, 10))
# Time Series
ax10 = plt.subplot2grid((3, 4), (1, 0), colspan=4)
ax10.plot(t_part, u_part[:, 0], linewidth=3, color="blue")
ax10.set_title(r"(b) True signal", fontsize=35)
ax10.set_xlabel(r"$t$", fontsize=35)
ax10.set_ylabel(r"$x_1$", fontsize=35, rotation=0)
ax10.set_xlim([100, 150])
# PDF & ACF
ax20 = plt.subplot2grid((3, 4), (2, 0), colspan=2)
sns.kdeplot(u[:, 0], ax=ax20, linewidth=3, bw_adjust=2, color="blue")
ax20.set_ylabel("")
ax20.set_title(r"(c) PDF", fontsize=35)
ax21 = plt.subplot2grid((3, 4), (2, 2), colspan=2)
ax21.plot(t_lags,  acf(u[:, 0])[1], linewidth=3, color="blue")
ax21.set_xlabel(r"$t$", fontsize=35)
ax21.set_yticks(np.arange(0, 1+0.5, 0.5))
ax21.set_xticks(np.linspace(0, 5, 6))
ax21.set_title(r"(d) ACF", fontsize=35)
ax21.set_xlim([0, 5])
for ax in fig.get_axes():
    ax.tick_params(labelsize=30, length=8, width=1, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(1)
ax00.tick_params(labelsize=30, length=10, width=1, direction="out")
cbar.ax.tick_params(labelsize=30, length=10, width=1, direction="out")
fig.tight_layout()
fig.subplots_adjust(wspace=1.5)
ax10.set_ylim([np.min(u[:,0]), np.max(u[:,0])])
ax20.set_xlim([np.min(u[:,0]), np.max(u[:,0])])
ax10.set_yticks([-5, 0, 5, 10])
ax20.set_xticks([-5, 0, 5, 10])
plt.show()


