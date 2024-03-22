import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
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


######################################
########## Data Generation ###########
######################################
a, b, f, g = 1/4, 4, 8, 1
sigma_x, sigma_y, sigma_z = 1., 0.05, 0.05

Lt = 250
dt = 0.001
Nt = int(Lt/dt) + 1
t = np.linspace(0, Lt, Nt)
u = np.zeros((Nt, 3))
u[0] = np.ones(3)

for n in range(Nt-1):
    u[n+1, 0] = u[n, 0] + (a * f - a * u[n, 0] - u[n, 1] ** 2 - u[n, 2] ** 2) * dt + sigma_x * np.sqrt(dt) * np.random.randn()
    u[n+1, 1] = u[n, 1] + (g + u[n, 0] * u[n, 1] - u[n, 1] - b * u[n, 0] * u[n, 2]) * dt + sigma_y * np.sqrt(dt) * np.random.randn()
    u[n+1, 2] = u[n, 2] + (b * u[n, 0] * u[n, 1] + u[n, 0] * u[n, 2] - u[n, 2]) * dt + sigma_z * np.sqrt(dt) * np.random.randn()


# Split data in to train and test
u = torch.tensor(u[:-1], dtype=torch.float32)
t = torch.tensor(t[:-1], dtype=torch.float32)

Ntrain = 50000
Ntest = 200000
train_u = u[:Ntrain]
train_t = t[:Ntrain]
test_u = u[-Ntest:]
test_t = t[-Ntest:]



# CGF for True System
test_u = test_u.numpy()
Nt = test_u.shape[0]
dim_u2 = 1
mu_trace = np.zeros((Nt, dim_u2, 1))
R_trace = np.zeros((Nt, dim_u2, dim_u2))
mu_trace[0] = np.zeros((dim_u2, 1))
R_trace[0] = np.eye(dim_u2)*0.01
mu0 = mu_trace[0]
R0 = R_trace[0]
for n in range(1, Nt):
    y0 = test_u[n-1, 1]
    z0 = test_u[n-1, 2]
    y1 = test_u[n, 1]
    z1 = test_u[n, 2]
    f1 = np.array([[g-y0], [-z0]])
    g1 = np.array([[y0-b*z0], [b*y0+z0]])
    s1 = np.diag([sigma_y, sigma_z])
    f2 = np.array([[a*f - (y0**2+z0**2)]])
    g2 = np.array([[-a]])
    s2 = np.array([[sigma_x]])
    invs1os1 = np.linalg.inv(s1@s1.T)
    s2os2 = s2@s2.T
    du1 = np.array([[y1-y0], [z1-z0]])
    mu1 = mu0 + (f2+g2@mu0)*dt + (R0@g1.T) @ invs1os1 @ (du1 -(f1+g1@mu0)*dt)
    R1 = R0 + ( g2@R0 + R0@g2.T + s2os2 - R0@g1.T@ invs1os1 @ g1@R0 )*dt
    mu_trace[n] = mu1
    R_trace[n] = R1
    mu0 = mu1
    R0 = R1

np.mean( (test_u[:,0] - mu_trace.flatten())**2 )


def avg_neg_log_likehood(x, mu, R):
    # x, mu are in matrix form, e.g. (t, x, 1)
    d = x.shape[1]
    neg_log_likehood = 1/2*(d*np.log(2*np.pi) + torch.log(torch.linalg.det(R)) + ((x-mu).permute(0,2,1)@torch.linalg.inv(R)@(x-mu)).flatten())
    return torch.mean(neg_log_likehood)

avg_neg_log_likehood(
    torch.tensor(test_u[:, [0]]).unsqueeze(2),
    torch.tensor(mu_trace),
    torch.tensor(R_trace))


# CGF for Models
test_u = torch.from_numpy(test_u)
class RegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.reg0 = nn.Linear(2, 1)
        self.reg1 = nn.Linear(1, 1)
        self.reg2 = nn.Linear(2, 1)

    def forward(self, t, u):
        basis_x = torch.stack([u[:,0], u[:,2]**2]).T
        basis_y = torch.stack([u[:,1]]).T
        basis_z = torch.stack([u[:,2], u[:,0]*u[:,2]]).T
        x_dyn = self.reg0(basis_x)
        y_dyn = self.reg1(basis_y)
        z_dyn = self.reg2(basis_z)
        self.out = torch.cat([x_dyn, y_dyn, z_dyn], dim=1)
        return self.out
class CGNN(nn.Module):
    def __init__(self, input_size=2, output_size=6):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, 5), nn.ReLU(),
                                 nn.Linear(5, 12), nn.ReLU(),
                                 nn.Linear(12, 15), nn.ReLU(),
                                 nn.Linear(15, 10), nn.ReLU(),
                                 nn.Linear(10, output_size))
    def forward(self, x):
        out = self.net(x)
        return out
class MixModel(nn.Module):
    def __init__(self, cgnn):
        super().__init__()
        self.outnet = None
        self.out = None
        self.reg0 = nn.Linear(2, 1)
        self.reg1 = nn.Linear(1, 1)
        self.reg2 = nn.Linear(2, 1)
        self.net = cgnn

    def forward(self, t, u):
        u1 = u[:, 1:]
        self.outnet = self.net(u1)

        basis_x = torch.stack([u[:,0], u[:,2]**2]).T
        basis_y = torch.stack([u[:,1]]).T
        basis_z = torch.stack([u[:,2], u[:,0]*u[:,2]]).T

        x_dyn = self.reg0(basis_x) + self.outnet[:, [0]] + self.outnet[:, [3]]*u[:, [0]]
        y_dyn = self.reg1(basis_y) + self.outnet[:, [1]] + self.outnet[:, [4]]*u[:, [0]]
        z_dyn = self.reg2(basis_z) + self.outnet[:, [2]] + self.outnet[:, [5]]*u[:, [0]]
        self.out = torch.cat([x_dyn, y_dyn, z_dyn], dim=1)
        return self.out

def CGFilter_RegModel(regmodel, u1, mu0, R0, cut_point, sigma_lst):
    # u1, mu0 are in col-matrix form, e.g. (t, x, 1)
    device = u1.device
    sigma_x, sigma_y, sigma_z = sigma_lst

    a0 = regmodel.reg0.bias[:]
    a1 = regmodel.reg0.weight[:, 0]
    a2 = regmodel.reg0.weight[:, 1]
    b0 = regmodel.reg1.bias[:]
    b1 = regmodel.reg1.weight[:, 0]
    c0 = regmodel.reg2.bias[:]
    c1 = regmodel.reg2.weight[:, 0]
    c2 = regmodel.reg2.weight[:, 1]

    Nt = u1.shape[0]
    dim_u2 = mu0.shape[0]
    mu_trace = torch.zeros((Nt, dim_u2, 1)).to(device)
    R_trace = torch.zeros((Nt, dim_u2, dim_u2)).to(device)
    mu_trace[0] = mu0
    R_trace[0] = R0
    for n in range(1, Nt):
        y0 = u1[n-1, 0].flatten()
        z0 = u1[n, 1].flatten()
        du1 = u1[n] - u1[n-1]

        f1 = torch.cat([b0+b1*y0, c0+c1*z0]).reshape(-1, 1)
        g1 = torch.cat([torch.zeros(1), c2*z0]).reshape(-1, 1)
        s1 = torch.diag(torch.tensor([sigma_y, sigma_z]))
        f2 = (a0+a2*z0**2).reshape(1,1)
        g2 = (a1).reshape(1,1)
        s2 = torch.tensor([[sigma_x]])

        invs1os1 = torch.linalg.inv(s1@s1.T)
        s2os2 = s2@s2
        mu1 = mu0 + (f2+g2@mu0)*dt + (R0@g1.T) @ invs1os1 @ (du1 -(f1+g1@mu0)*dt)
        R1 = R0 + (g2@R0 + R0@g2.T + s2os2 - R0@g1.T@ invs1os1 @ g1@R0 )*dt
        mu_trace[n] = mu1
        R_trace[n] = R1
        mu0 = mu1
        R0 = R1
    return (mu_trace[cut_point:], R_trace[cut_point:])
def CGFilter_MixModel(mixmodel, u1, mu0, R0, cut_point, sigma_lst):
    # u1, mu0 are in col-matrix form, e.g. (t, x, 1)
    device = u1.device
    sigma_x, sigma_y, sigma_z = sigma_lst

    a0 = mixmodel.reg0.bias[:]
    a1 = mixmodel.reg0.weight[:, 0]
    a2 = mixmodel.reg0.weight[:, 1]
    b0 = mixmodel.reg1.bias[:]
    b1 = mixmodel.reg1.weight[:, 0]
    c0 = mixmodel.reg2.bias[:]
    c1 = mixmodel.reg2.weight[:, 0]
    c2 = mixmodel.reg2.weight[:, 1]

    Nt = u1.shape[0]
    dim_u2 = mu0.shape[0]
    mu_trace = torch.zeros((Nt, dim_u2, 1)).to(device)
    R_trace = torch.zeros((Nt, dim_u2, dim_u2)).to(device)
    mu_trace[0] = mu0
    R_trace[0] = R0
    for n in range(1, Nt):
        y0 = u1[n-1, 0].flatten()
        z0 = u1[n, 1].flatten()
        du1 = u1[n] - u1[n-1]
        outnet = mixmodel.net(u1[n-1].T).T

        f1 = torch.cat([b0+b1*y0+outnet[1], c0+c1*z0+outnet[2]]).reshape(-1, 1)
        g1 = torch.cat([outnet[4], c2*z0+outnet[5]]).reshape(-1, 1)
        s1 = torch.diag(torch.tensor([sigma_y, sigma_z]))
        f2 = (a0+a2*z0**2+outnet[0]).reshape(1,1)
        g2 = (a1+outnet[3]).reshape(1,1)
        s2 = torch.tensor([[sigma_x]])

        invs1os1 = torch.linalg.inv(s1@s1.T)
        s2os2 = s2@s2
        mu1 = mu0 + (f2+g2@mu0)*dt + (R0@g1.T) @ invs1os1 @ (du1 -(f1+g1@mu0)*dt)
        R1 = R0 + (g2@R0 + R0@g2.T + s2os2 - R0@g1.T@ invs1os1 @ g1@R0 )*dt
        mu_trace[n] = mu1
        R_trace[n] = R1
        mu0 = mu1
        R0 = R1
    return (mu_trace[cut_point:], R_trace[cut_point:])


model1 = RegModel()
model2 = MixModel(CGNN())
model3 = MixModel(CGNN())
model1.load_state_dict(torch.load("/home/cc/CodeProjects/CGNSDE/L84/L84_Model/L84_regmodel.pt"))
model2.load_state_dict(torch.load("/home/cc/CodeProjects/CGNSDE/L84/L84_Model/L84_mixmodel1_fc.pt"))
model3.load_state_dict(torch.load("/home/cc/CodeProjects/CGNSDE/L84/L84_Model/L84_mixmodel2_fc_ep500.pt"))


# sigma estimation
train_u_dot = torch.diff(train_u, dim=0)/dt
with torch.no_grad():
    train_u_dot_pred1 = model1(None, train_u[:-1])
    train_u_dot_pred2 = model2(None, train_u[:-1])
sigma_hat1 = torch.sqrt( dt*torch.mean( (train_u_dot - train_u_dot_pred1)**2, dim=0 ) ).tolist()
sigma_hat2 = torch.sqrt( dt*torch.mean( (train_u_dot - train_u_dot_pred2)**2, dim=0 ) ).tolist()

with torch.no_grad():
    mu_preds1, R_preds1 = CGFilter_RegModel(model1, u1=test_u[:, 1:].reshape(-1, 2, 1), mu0=torch.zeros(1, 1).to(device), R0=0.01*torch.eye(1).to(device), cut_point=0, sigma_lst=sigma_hat1)
    mu_preds2, R_preds2 = CGFilter_MixModel(model2, u1=test_u[:, 1:].reshape(-1, 2, 1), mu0=torch.zeros(1, 1).to(device), R0=0.01*torch.eye(1).to(device), cut_point=0, sigma_lst=sigma_hat2)
    mu_preds3, R_preds3 = CGFilter_MixModel(model3, u1=test_u[:, 1:].reshape(-1, 2, 1), mu0=torch.zeros(1, 1).to(device), R0=0.01*torch.eye(1).to(device), cut_point=0, sigma_lst=sigma_hat2)
F.mse_loss(test_u[:,[0]], mu_preds1.reshape(-1, 1))
F.mse_loss(test_u[:,[0]], mu_preds2.reshape(-1, 1))
F.mse_loss(test_u[:,[0]], mu_preds3.reshape(-1, 1))
avg_neg_log_likehood(test_u[:,[0]].unsqueeze(2), mu_preds1, R_preds1)
avg_neg_log_likehood(test_u[:,[0]].unsqueeze(2), mu_preds2, R_preds2)
avg_neg_log_likehood(test_u[:,[0]].unsqueeze(2), mu_preds3, R_preds3)


# Visualizaton
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

fig = plt.figure(figsize=(20, 10))
axs = fig.subplots(2, 2)
for ax in fig.get_axes():
    ax.plot(test_t, test_u[:, 0], linewidth=3.5, color="blue")
    ax.set_xlim([100, 120])
    ax.tick_params(labelsize=30, length=8, width=1, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(1)
axs[0,0].set_ylabel(r"$x$", fontsize=35, rotation=0)
axs[1,0].set_ylabel(r"$x$", fontsize=35, rotation=0)
axs[1,0].set_xlabel(r"$t$", fontsize=35)
axs[1,1].set_xlabel(r"$t$", fontsize=35)
axs[0,0].plot(test_t, test_u[:, 0], linewidth=3.5, color="blue", label="True signal")
axs[0,0].plot(test_t, mu_trace.flatten(),  linewidth=2.5, color="red", label="Posterior mean")
axs[0,1].plot(test_t, mu_preds1.flatten(), linewidth=2.5, color="red")
axs[1,0].plot(test_t, mu_preds2.flatten(), linewidth=2.5, color="red")
axs[1,1].plot(test_t, mu_preds3.flatten(), linewidth=2.5, color="red")
axs[0,0].fill_between(test_t, mu_trace[:, 0, 0]-2*np.sqrt(R_trace[:, 0, 0]), mu_trace[:, 0, 0]+2*np.sqrt(R_trace[:, 0, 0]), color='grey', alpha=0.8, label=r"Uncertainty")
axs[0,1].fill_between(test_t, mu_preds1[:, 0, 0]-2*torch.sqrt(R_preds1[:, 0, 0]), mu_preds1[:, 0, 0]+2*torch.sqrt(R_preds1[:, 0, 0]), color='grey', alpha=0.8)
axs[1,0].fill_between(test_t, mu_preds2[:, 0, 0]-2*torch.sqrt(R_preds2[:, 0, 0]), mu_preds2[:, 0, 0]+2*torch.sqrt(R_preds2[:, 0, 0]), color='grey', alpha=0.8)
axs[1,1].fill_between(test_t, mu_preds3[:, 0, 0]-2*torch.sqrt(R_preds3[:, 0, 0]), mu_preds3[:, 0, 0]+2*torch.sqrt(R_preds3[:, 0, 0]), color='grey', alpha=0.8)
axs[0,0].set_title(r"(a) True model", fontsize=35)
axs[0,1].set_title(r"(b) Physics-based regression model", fontsize=35)
axs[1,0].set_title(r"(c) CGNSDE without DA loss", fontsize=35)
axs[1,1].set_title(r"(d) CGNSDE with DA loss", fontsize=35)
axs[0,0].set_yticks(np.arange(-2, 5, 2))
axs[0,1].set_yticks(np.arange(-10, 7, 4))
axs[1,0].set_yticks(np.arange(-2, 5, 2))
axs[1,1].set_yticks(np.arange(-2, 5, 2))
lege = fig.legend(fontsize=35, loc="upper center", ncol=3, fancybox=False, edgecolor="black", bbox_to_anchor=(0.53, 1))
lege.get_frame().set_linewidth(1)
fig.tight_layout()
fig.subplots_adjust(top=0.8, hspace=0.4)
plt.show()
