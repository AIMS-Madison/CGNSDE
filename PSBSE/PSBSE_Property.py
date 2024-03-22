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


#########################################
########## Data Visualization ###########
#########################################
#########################################
########## Data Visualization ###########
#########################################
u = u.numpy()
t = t.numpy()

def acf(x, lag=500):
    i = np.arange(0, lag+1)
    v = np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  for i in range(1, lag+1)])
    return (i, v)
t_lags = np.linspace(0, 5, 501)


plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

fig = plt.figure(figsize=(20, 10))
# x dynamics
ax00 = plt.subplot2grid((3, 5), (0, 0), colspan=3)
ax00.plot(t, u[:, 0], linewidth=3, color="blue")
ax00.set_xlim([100, 200])
ax00.set_ylim( [np.min(u[:,0]), np.max(u[:,0])] )
ax00.set_ylabel(r"$x$", fontsize=35, rotation=0, labelpad=15)
ax00.set_title(r"{(a) True signal", fontsize=35)
ax01 = plt.subplot2grid((3, 5), (0, 3))
sns.kdeplot(u[:, 0], ax=ax01, linewidth=3, bw_adjust=2, color="blue")
ax01.set_ylabel("")
ax01.set_xlim( [np.min(u[:,0]), np.max(u[:,0])] )
ax01.set_title(r"(b) PDF", fontsize=35)
ax02 = plt.subplot2grid((3, 5), (0, 4))
ax02.plot(t_lags, acf(u[:, 0])[1], linewidth=3, color="blue")
ax02.set_title(r"(c) ACF", fontsize=35)
ax02.set_yticks(np.arange(0, 1+0.5, 0.5))
ax02.set_xticks(np.linspace(0, 5, 6))
# y dynamics
ax10 = plt.subplot2grid((3, 5), (1, 0), colspan=3)
ax10.plot(t, u[:, 1], linewidth=3, color="blue")
ax10.set_xlim([100, 200])
ax10.set_ylim( [np.min(u[:,1]), np.max(u[:,1])] )
ax10.set_ylabel(r"$y$", fontsize=35, rotation=0, labelpad=30)
ax11 = plt.subplot2grid((3, 5), (1, 3))
ax11.set_xlim( [np.min(u[:,1]), np.max(u[:,1])] )
sns.kdeplot(u[:, 1], ax=ax11, linewidth=3, bw_adjust=2, color="blue")
ax11.set_ylabel("")
ax12 = plt.subplot2grid((3, 5), (1, 4))
ax12.plot(t_lags,  acf(u[:, 1])[1], linewidth=3, color="blue")
ax12.set_yticks(np.arange(0, 1+0.5, 0.5))
ax12.set_xticks(np.linspace(0, 5, 6))
# z dynamics
ax20 = plt.subplot2grid((3, 5), (2, 0), colspan=3)
ax20.plot(t, u[:, 2], linewidth=3, color="blue")
ax20.set_xlim([100, 200])
ax20.set_ylim( [np.min(u[:,2]), np.max(u[:,2])] )
ax20.set_ylabel(r"$z$", fontsize=35, rotation=0, labelpad=15)
ax20.set_xlabel(r"$t$", fontsize=35)
ax21 = plt.subplot2grid((3, 5), (2, 3))
sns.kdeplot(u[:, 2], ax=ax21, linewidth=3, bw_adjust=2, color="blue")
ax21.set_xlim( [np.min(u[:,2]), np.max(u[:,2])] )
ax21.set_ylabel("")
ax22 = plt.subplot2grid((3, 5), (2, 4))
ax22.plot(t_lags,  acf(u[:, 2])[1], linewidth=3, color="blue")
ax22.set_xlabel(r"$t$", fontsize=35)
ax22.set_yticks(np.arange(0, 1+0.5, 0.5))
ax22.set_xticks(np.linspace(0, 5, 6))
ax02.set_xlim([0, 5])
ax12.set_xlim([0, 5])
ax22.set_xlim([0, 5])
ax00.set_yticks([-1.5, 0, 1.5])
ax10.set_yticks([-4, -2, 0, 2])
ax20.set_yticks([-1.5, 0, 1.5])
ax01.set_xticks([-1.5, 0, 1.5])
ax11.set_xticks([-4, -2, 0, 2])
ax21.set_xticks([-1.5, 0, 1.5])
ax01.set_yticks([0., 1., 2.])
ax11.set_yticks([0, 0.25, 0.5])
ax21.set_yticks([0, 0.5, 1])
for ax in fig.get_axes():
    ax.tick_params(labelsize=30, length=8, width=1, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(1)
fig.tight_layout()
# fig.subplots_adjust(top=1.0)
plt.show()


