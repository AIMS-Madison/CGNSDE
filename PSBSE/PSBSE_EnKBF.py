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
u = u[:-1]
t = t[:-1]
Ntrain = 10000
Ntest = 50000
train_u = u[:Ntrain]
train_t = t[:Ntrain]
test_u = u[-Ntest:]
test_t = t[-Ntest:]



###############################################################
################# Ensemble Kalman-Bucy Filter #################
###############################################################

def avg_neg_log_likehood(x, mu, R):
    # x, mu are in matrix form, e.g. (t, x, 1)
    d = x.shape[1]
    neg_log_likehood = 1/2*(d*np.log(2*np.pi) + torch.log(torch.linalg.det(R)) + ((x-mu).permute(0,2,1)@torch.linalg.inv(R)@(x-mu)).flatten())
    return torch.mean(neg_log_likehood)

def cross_cov(X, Y):
    n = X.shape[0]
    assert n == Y.shape[0]
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    X_centered = X_centered + np.random.randn(X_centered.shape[0], X_centered.shape[1]) * 0.01
    Y_centered = Y_centered + np.random.randn(Y_centered.shape[0], Y_centered.shape[1]) * 0.01
    cross_cov_matrix = np.dot(X_centered.T, Y_centered) / (n - 1)
    return cross_cov_matrix

sigma_x = sigma_x/2
sigma_y = sigma_y/3
sigma_z = sigma_z/3

u1 = test_u[:,[0]]
SIG1 = np.array([[sigma_x**2]])
SIG2 = np.diag([sigma_y**2, sigma_z**2])
sig1 = np.array([[sigma_x]])
sig2 = np.diag([sigma_y, sigma_z])

J = 100
p = 2
u2_ens = np.zeros((J, Ntest, p))

err_lst = []
nll_lst = []
for _ in range(100):
    for n in range(1, Ntest):
        f1 = beta_y*u2_ens[:, n-1, 0] - alpha*u1[n-1]**2 + 2*alpha*u1[n-1]*u2_ens[:, n-1, 1]
        f2 = beta_z*u2_ens[:, n-1, 1] - 3*alpha*u1[n-1]*u2_ens[:, n-1, 0]
        f = np.stack([f1, f2]).T
        Sys_term = f*dt + np.random.randn(J,p) @ sig2 * np.sqrt(dt)

        g1 = beta_x*u1[n-1] + alpha*u1[n-1]*u2_ens[:, n-1, 0] + alpha*u2_ens[:,n-1, 0]*u2_ens[:, n-1, 1]
        g = np.stack([g1]).T
        g_bar = np.mean(g, axis=0)
        CCOV = cross_cov(u2_ens[:, n-1], g)
        DA_term = -0.5*((g+g_bar)*dt-2*(u1[n]-u1[n-1])) @ (CCOV@np.linalg.inv(SIG1)).T

        u2_ens[:, n, :] = u2_ens[:, n-1, :] + Sys_term + DA_term

    mu_trace = np.mean(u2_ens, axis=0).reshape(u2_ens.shape[1], u2_ens.shape[2], 1)
    R_trace = np.zeros((u2_ens.shape[1], u2_ens.shape[2], u2_ens.shape[2]))
    for i in range(u2_ens.shape[1]):
        R_trace[i] = np.cov(u2_ens[:, i, :].T)


    nll = avg_neg_log_likehood(torch.tensor(test_u[:,1:].reshape(50000, 2, 1)[1:]),
                               torch.tensor(mu_trace)[1:],
                               torch.tensor(R_trace)[1:]).item()

    err = np.mean( (test_u[:, 1:] - mu_trace.squeeze(2) )**2 )

    err_lst.append(err)
    nll_lst.append(nll)
    print(_, err, nll)

np.mean(err_lst)
np.mean(nll_lst)



###########################################
#######
###########################################
torch.manual_seed(0)
np.random.seed(0)

for n in range(1, Ntest):
    f1 = beta_y*u2_ens[:, n-1, 0] - alpha*u1[n-1]**2 + 2*alpha*u1[n-1]*u2_ens[:, n-1, 1]
    f2 = beta_z*u2_ens[:, n-1, 1] - 3*alpha*u1[n-1]*u2_ens[:, n-1, 0]
    f = np.stack([f1, f2]).T
    Sys_term = f*dt + np.random.randn(J,p) @ sig2 * np.sqrt(dt)

    g1 = beta_x*u1[n-1] + alpha*u1[n-1]*u2_ens[:, n-1, 0] + alpha*u2_ens[:,n-1, 0]*u2_ens[:, n-1, 1]
    g = np.stack([g1]).T
    g_bar = np.mean(g, axis=0)
    CCOV = cross_cov(u2_ens[:, n-1], g)
    DA_term = -0.5*((g+g_bar)*dt-2*(u1[n]-u1[n-1])) @ (CCOV@np.linalg.inv(SIG1)).T

    u2_ens[:, n, :] = u2_ens[:, n-1, :] + Sys_term + DA_term

mu_trace = np.mean(u2_ens, axis=0)
R_trace = np.zeros((50000, 2, 2))
for i in range(50000):
    R_trace[i] = np.cov(u2_ens[:, i, :].T)


avg_neg_log_likehood(torch.tensor(test_u[:,1:]).unsqueeze(2)[1:],
                     torch.tensor(mu_trace).unsqueeze(2)[1:],
                     torch.tensor(R_trace)[1:])



# CGF for Models
train_u = torch.tensor(train_u, dtype=torch.float32)
test_u = torch.tensor(test_u, dtype=torch.float32)

class RegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.reg0 = nn.Linear(2, 1, bias=False)
        self.reg1 = nn.Linear(2, 1, bias=False)
        self.reg2 = nn.Linear(1, 1, bias=False)

    def forward(self, t, u):
        basis_x = torch.stack([u[:,2], u[:,0]*u[:,1]]).T
        basis_y = torch.stack([u[:,0]**2, u[:,0]*u[:,2]]).T
        basis_z = torch.stack([u[:,0]*u[:,1]]).T
        x_dyn = self.reg0(basis_x)
        y_dyn = self.reg1(basis_y)
        z_dyn = self.reg2(basis_z)
        self.out = torch.cat([x_dyn, y_dyn, z_dyn], dim=1)
        return self.out

class CGNN(nn.Module):
    def __init__(self, input_size=1, output_size=9):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, 5), nn.ReLU(),
                                 nn.Linear(5, 10), nn.ReLU(),
                                 nn.Linear(10, 20), nn.ReLU(),
                                 nn.Linear(20, 5), nn.ReLU(),
                                 nn.Linear(5, output_size))
    def forward(self, x):
        out = self.net(x)
        return out

class MixModel(nn.Module):
    def __init__(self, cgnn):
        super().__init__()
        self.outnet = None
        self.out = None
        self.reg0 = nn.Linear(2, 1, bias=False)
        self.reg1 = nn.Linear(2, 1, bias=False)
        self.reg2 = nn.Linear(1, 1, bias=False)
        self.net = cgnn

    def forward(self, t, u):
        u1 = u[:, [0]]
        self.outnet = self.net(u1)

        basis_x = torch.stack([u[:,2], u[:,0]*u[:,1]]).T
        basis_y = torch.stack([u[:,0]**2, u[:,0]*u[:,2]]).T
        basis_z = torch.stack([u[:,0]*u[:,1]]).T

        x_dyn = self.reg0(basis_x) + self.outnet[:, [0]] + self.outnet[:, [3]]*u[:,[1]] + self.outnet[:, [4]]*u[:,[2]]
        y_dyn = self.reg1(basis_y) + self.outnet[:, [1]] + self.outnet[:, [5]]*u[:,[1]] + self.outnet[:, [6]]*u[:,[2]]
        z_dyn = self.reg2(basis_z) + self.outnet[:, [2]] + self.outnet[:, [7]]*u[:,[1]] + self.outnet[:, [8]]*u[:,[1]]
        self.out = torch.cat([x_dyn, y_dyn, z_dyn], dim=1)
        return self.out

def CGFilter_RegModel(regmodel, u1, mu0, R0, cut_point, sigma_lst):
    # u1, mu0 are in col-matrix form, e.g. (t, x, 1)
    device = u1.device
    sigma_x, sigma_y, sigma_z = sigma_lst

    a1 = regmodel.reg0.weight[:, 0]
    a2 = regmodel.reg0.weight[:, 1]
    b1 = regmodel.reg1.weight[:, 0]
    b2 = regmodel.reg1.weight[:, 1]
    c1 = regmodel.reg2.weight[:, 0]

    Nt = u1.shape[0]
    dim_u2 = mu0.shape[0]
    mu_trace = torch.zeros((Nt, dim_u2, 1)).to(device)
    R_trace = torch.zeros((Nt, dim_u2, dim_u2)).to(device)
    mu_trace[0] = mu0
    R_trace[0] = R0
    for n in range(1, Nt):
        x0 = u1[n-1].flatten()
        x1 = u1[n].flatten()
        du1 = (x1 - x0).reshape(-1, 1)

        f1 = torch.zeros(1,1)
        g1 = torch.cat([a2*x0, a1]).reshape(1, 2)
        s1 = torch.tensor([[sigma_x]]).to(device)
        f2 = torch.cat([b1*x0**2, torch.zeros(1)]).reshape(2, 1)
        g2 = torch.cat([torch.zeros(1), b2*x0, c1*x0, torch.zeros(1)]).reshape(2, 2)
        s2 = torch.diag(torch.tensor([sigma_y, sigma_z])).to(device)


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

    a1 = mixmodel.reg0.weight[:, 0]
    a2 = mixmodel.reg0.weight[:, 1]
    b1 = mixmodel.reg1.weight[:, 0]
    b2 = mixmodel.reg1.weight[:, 1]
    c1 = mixmodel.reg2.weight[:, 0]

    Nt = u1.shape[0]
    dim_u2 = mu0.shape[0]
    mu_trace = torch.zeros((Nt, dim_u2, 1)).to(device)
    R_trace = torch.zeros((Nt, dim_u2, dim_u2)).to(device)
    mu_trace[0] = mu0
    R_trace[0] = R0
    for n in range(1, Nt):
        x0 = u1[n-1].flatten()
        x1 = u1[n].flatten()  # for dynamics of x0
        du1 = (x1 - x0).reshape(-1, 1)
        outnet = mixmodel.net(x0.unsqueeze(0)).T

        f1 = (outnet[0]).reshape(-1, 1)
        g1 = torch.cat([a2*x0+outnet[3], a1+outnet[4]]).reshape(1, 2)
        s1 = torch.tensor([[sigma_x]]).to(device)
        f2 = torch.cat([b1*x0**2+outnet[1], outnet[2]]).reshape(2, 1)
        g2 = torch.cat([outnet[5], b2*x0+outnet[6], c1*x0+outnet[7], outnet[8]]).reshape(2, 2)
        s2 = torch.diag(torch.tensor([sigma_y, sigma_z])).to(device)

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
model1.load_state_dict(torch.load("/home/cc/CodeProjects/CGNSDE/NonCG/NonCG_Model/NonCG_regmodel.pt"))
model2.load_state_dict(torch.load("/home/cc/CodeProjects/CGNSDE/NonCG/NonCG_Model/NonCG_mixmodel1.pt"))
model3.load_state_dict(torch.load("/home/cc/CodeProjects/CGNSDE/NonCG/NonCG_Model/NonCG_mixmodel2.pt"))

# sigma estimation
train_u_dot = torch.diff(train_u, dim=0)/dt
with torch.no_grad():
    train_u_dot_pred1 = model1(None, train_u[:-1])
    train_u_dot_pred2 = model2(None, train_u[:-1])
sigma_hat1 = torch.sqrt( dt*torch.mean( (train_u_dot - train_u_dot_pred1)**2, dim=0 ) ).tolist()
sigma_hat2 = torch.sqrt( dt*torch.mean( (train_u_dot - train_u_dot_pred2)**2, dim=0 ) ).tolist()

with torch.no_grad():
    mu_preds1, R_preds1 = CGFilter_RegModel(model1, u1=test_u[:, [0]].reshape(-1, 1, 1), mu0=torch.zeros(2, 1).to(device), R0=0.01*torch.eye(2).to(device), cut_point=0, sigma_lst=sigma_hat1)
    mu_preds2, R_preds2 = CGFilter_MixModel(model2, u1=test_u[:, [0]].reshape(-1, 1, 1), mu0=torch.zeros(2, 1).to(device), R0=0.01*torch.eye(2).to(device), cut_point=0, sigma_lst=sigma_hat2)
    mu_preds3, R_preds3 = CGFilter_MixModel(model3, u1=test_u[:, [0]].reshape(-1, 1, 1), mu0=torch.zeros(2, 1).to(device), R0=0.01*torch.eye(2).to(device), cut_point=0, sigma_lst=sigma_hat2)
F.mse_loss(test_u[:,1:], mu_preds1.reshape(-1, 2))
F.mse_loss(test_u[:,1:], mu_preds2.reshape(-1, 2))
F.mse_loss(test_u[:,1:], mu_preds3.reshape(-1, 2))
avg_neg_log_likehood(test_u[:,1:].unsqueeze(2), mu_preds1, R_preds1)
avg_neg_log_likehood(test_u[:,1:].unsqueeze(2), mu_preds2, R_preds2)
avg_neg_log_likehood(test_u[:,1:].unsqueeze(2), mu_preds3, R_preds3)



# Visualizaton
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

fig = plt.figure(figsize=(20, 15))
subfigs = fig.subfigures(2, 2, hspace=-0.1, wspace=-0.1)
# EnKBF
axs0 = subfigs[0,0].subplots(2, 1, sharex=True)
axs0[0].plot(test_t, test_u[:, 1], linewidth=3.5, label="True signal", color="blue")
axs0[0].plot(test_t, mu_trace[:, 0], linewidth=2.5, label="Posterior mean", color="red")
axs0[0].fill_between(test_t, mu_trace[:, 0]-2*np.sqrt(R_trace[:, 0, 0]), mu_trace[:, 0]+2*np.sqrt(R_trace[:, 0, 0]), color='grey', alpha=0.5, label=r"Uncertainty")
axs0[0].set_ylabel(r"$y$", fontsize=35, rotation=0)
axs0[0].set_title(r"(a) True model", fontsize=35, rotation=0)
axs0[1].plot(test_t, test_u[:, 2], linewidth=3.5, color="blue")
axs0[1].plot(test_t, mu_trace[:, 1], linewidth=3.5, color="red")
axs0[1].fill_between(test_t, mu_trace[:, 1]-2*np.sqrt(R_trace[:, 1, 1]), mu_trace[:, 1]+2*np.sqrt(R_trace[:, 1, 1]), color='grey', alpha=0.5)
axs0[1].set_ylabel(r"$z$", fontsize=35, rotation=0)
axs0[1].set_xlabel(r"$t$", fontsize=35)
# PRM
axs1 = subfigs[0,1].subplots(2, 1, sharex=True)
axs1[0].plot(test_t, test_u[:, 1], linewidth=3.5, color="blue")
axs1[0].plot(test_t, mu_preds1[:, 0, 0], linewidth=2.5, color="red")
axs1[0].fill_between(test_t, mu_preds1[:, 0, 0]-2*torch.sqrt(R_preds1[:, 0, 0]), mu_preds1[:, 0, 0]+2*torch.sqrt(R_preds1[:, 0, 0]), color='grey', alpha=0.5)
axs1[0].set_ylabel(r"$y$", fontsize=35, rotation=0)
axs1[0].set_title(r"(b) Physics-based regression model", fontsize=35, rotation=0)
axs1[1].plot(test_t, test_u[:, 2], linewidth=3.5, color="blue")
axs1[1].plot(test_t, mu_preds1[:, 1, 0], linewidth=2.5, color="red")
axs1[1].fill_between(test_t, mu_preds1[:, 1, 0]-2*torch.sqrt(R_preds1[:, 1, 1]), mu_preds1[:, 1, 0]+2*torch.sqrt(R_preds1[:, 1, 1]), color='grey', alpha=0.5)
axs1[1].set_ylabel(r"$z$", fontsize=35, rotation=0)
axs1[1].set_xlabel(r"$t$", fontsize=35)
# CGNSDE without DA loss
axs2 = subfigs[1,0].subplots(2, 1, sharex=True)
axs2[0].plot(test_t, test_u[:, 1], linewidth=3.5, color="blue")
axs2[0].plot(test_t, mu_preds2[:, 0, 0], linewidth=2.5, color="red")
axs2[0].fill_between(test_t, mu_preds2[:, 0, 0]-2*torch.sqrt(R_preds2[:, 0, 0]), mu_preds2[:, 0, 0]+2*torch.sqrt(R_preds2[:, 0, 0]), color='grey', alpha=0.5)
axs2[0].set_ylabel(r"$y$", fontsize=35, rotation=0)
axs2[0].set_title(r"(b) CGNSDE without DA loss", fontsize=35, rotation=0)
axs2[1].plot(test_t, test_u[:, 2], linewidth=3.5, color="blue")
axs2[1].plot(test_t, mu_preds2[:, 1, 0], linewidth=2.5, color="red")
axs2[1].fill_between(test_t, mu_preds2[:, 1, 0]-2*torch.sqrt(R_preds2[:, 1, 1]), mu_preds2[:, 1, 0]+2*torch.sqrt(R_preds2[:, 1, 1]), color='grey', alpha=0.5)
axs2[1].set_ylabel(r"$z$", fontsize=35, rotation=0)
axs2[1].set_xlabel(r"$t$", fontsize=35)
# CGNSDE with DA
axs3 = subfigs[1,1].subplots(2, 1, sharex=True)
axs3[0].plot(test_t, test_u[:, 1], linewidth=3.5, color="blue")
axs3[0].plot(test_t, mu_preds3[:, 0, 0], linewidth=2.5, color="red")
axs3[0].fill_between(test_t, mu_preds3[:, 0, 0]-2*torch.sqrt(R_preds3[:, 0, 0]), mu_preds3[:, 0, 0]+2*torch.sqrt(R_preds3[:, 0, 0]), color='grey', alpha=0.5)
axs3[0].set_ylabel(r"$y$", fontsize=35, rotation=0)
axs3[0].set_title(r"(b) CGNSDE with DA loss", fontsize=35, rotation=0)
axs3[1].plot(test_t, test_u[:, 2], linewidth=3.5, color="blue")
axs3[1].plot(test_t, mu_preds3[:, 1, 0], linewidth=2.5, color="red")
axs3[1].fill_between(test_t, mu_preds3[:, 1, 0]-2*torch.sqrt(R_preds3[:, 1, 1]), mu_preds3[:, 1, 0]+2*torch.sqrt(R_preds3[:, 1, 1]), color='grey', alpha=0.5)
axs3[1].set_ylabel(r"$z$", fontsize=35, rotation=0)
axs3[1].set_xlabel(r"$t$", fontsize=35)
for sf in subfigs.flatten():
    for ax in sf.get_axes():
        ax.set_xlim([300, 310])
        ax.tick_params(labelsize=35, length=8, width=1, direction="in")
        for spine in ax.spines.values():
            spine.set_linewidth(1)
lege = subfigs[0,0].legend(fontsize=35, loc="upper center", ncol=3, fancybox=False, edgecolor="black", bbox_to_anchor=(0.94, 1))
lege.get_frame().set_linewidth(1)
fig.subplots_adjust(top=0.74) #
axs0[0].set_ylim([-5.5, 3.5])
axs1[0].set_ylim([-5.5, 3.5])
axs2[0].set_ylim([-5.5, 3.5])
axs3[0].set_ylim([-5.5, 3.5])
axs0[0].set_yticks([-4, -2, 0, 2])
axs1[0].set_yticks([-4, -2, 0, 2])
axs2[0].set_yticks([-4, -2, 0, 2])
axs3[0].set_yticks([-4, -2, 0, 2])
axs0[1].set_ylim([-2.5, 2.5])
axs1[1].set_ylim([-2.5, 2.5])
axs2[1].set_ylim([-2.5, 2.5])
axs3[1].set_ylim([-2.5, 2.5])
axs0[1].set_yticks([-2, 0, 2])
axs1[1].set_yticks([-2, 0, 2])
axs2[1].set_yticks([-2, 0, 2])
axs3[1].set_yticks([-2, 0, 2])

