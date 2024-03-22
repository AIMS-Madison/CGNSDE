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
u = torch.tensor(u[:-1], dtype=torch.float32)
t = torch.tensor(t[:-1], dtype=torch.float32)
Ntrain = 10000
Ntest = 50000
train_u = u[:Ntrain]
train_t = t[:Ntrain]
test_u = u[-Ntest:]
test_t = t[-Ntest:]


####################################################
################# CGNN & MixModel  #################
####################################################

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

def ODESolver(model, u0, steps, dt):
    # u0 is in vector form, e.g. (x)
    dim = u0.shape[0]
    u_pred = torch.zeros(steps, dim)
    u_pred[0] = u0
    for n in range(0, steps-1):
        u_dot_pred = model(None, u_pred[n].unsqueeze(0)).squeeze(0)
        u_pred[n+1] = u_pred[n]+u_dot_pred*dt
    return u_pred

############################################################
################# Train MixModel (Stage1)  #################
############################################################
short_steps = int(0.5/dt)

# Stage1: Train mixmodel with forecast loss
epochs = 10000
train_loss_history = []
train_loss_da_history = []

cgnn = CGNN()
mixmodel = MixModel(cgnn).to(device)
optimizer = torch.optim.Adam(mixmodel.parameters(), lr=1e-3)
for ep in range(1, epochs+1):
    start_time = time.time()
    head_idx_short = torch.from_numpy(np.random.choice(Ntrain-short_steps+1, size=1))
    u_short = u[head_idx_short:head_idx_short + short_steps].to(device)
    t_short = t[head_idx_short:head_idx_short + short_steps].to(device)

    optimizer.zero_grad()

    out = torchdiffeq.odeint(mixmodel, u_short[[0]], t_short)[:,0,:]
    loss = F.mse_loss(u_short, out)

    loss.backward()
    optimizer.step()
    train_loss_history.append(loss.item())
    end_time = time.time()
    print(ep, " loss: ", loss.item(), " time: ", end_time-start_time)

# torch.save(mixmodel.state_dict(), r"NonCG_mixmodel1.pt")

# mixmodel.load_state_dict(torch.load("/home/cc/CodeProjects/CGNSDE/NonCG/NonCG_Model/NonCG_mixmodel1.pt"))

##########################################################
################# Estimate sigma  & CGF  #################
##########################################################
train_u_dot = torch.diff(train_u, dim=0)/dt
with torch.no_grad():
    train_u_dot_pred = mixmodel(None, train_u[:-1])
sigma_hat = torch.sqrt( dt*torch.mean( (train_u_dot - train_u_dot_pred)**2, dim=0 ) ).tolist()


def CGFilter(mixmodel, u1, mu0, R0, cut_point, sigma_lst):
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

def SDESolver(model, u0, steps, dt, sigma_lst):
    # u0 is in vector form, e.g. (x)
    dim = u0.shape[0]
    sigma = torch.tensor(sigma_lst)
    u_simu = torch.zeros(steps, dim)
    u_simu[0] = u0
    for n in range(0, steps-1):
        u_dot_pred = model(None, u_simu[n].unsqueeze(0)).squeeze(0)
        u_simu[n+1] = u_simu[n] + u_dot_pred*dt + sigma*np.sqrt(dt)*torch.randn(3)
    return u_simu

############################################################
################# Train MixModel (Stage2)  #################
############################################################
# Stage 2: Train mixmodel with forcast loss + DA loss

def avg_neg_log_likehood(x, mu, R):
    # x, mu are in matrix form, e.g. (t, x, 1)
    d = x.shape[1]
    neg_log_likehood = 1/2*(d*np.log(2*np.pi) + torch.log(torch.linalg.det(R)) + ((x-mu).permute(0,2,1)@torch.linalg.inv(R)@(x-mu)))
    return torch.mean(neg_log_likehood)

long_steps = int(100/dt)
cut_point = int(10/dt)

epochs = 500
train_loss_history = []
train_loss_da_history = []
optimizer = torch.optim.Adam(mixmodel.parameters(), lr=1e-3)
for ep in range(1, epochs+1):
    start_time = time.time()

    head_idx_short = torch.from_numpy(np.random.choice(Ntrain - short_steps + 1, size=1))
    u_short = u[head_idx_short:head_idx_short + short_steps].to(device)
    t_short = t[head_idx_short:head_idx_short + short_steps].to(device)

    head_idx_long = torch.from_numpy( np.random.choice(Ntrain-long_steps+1, size=1) )
    u_long = u[head_idx_long:head_idx_long + long_steps].to(device)
    t_long = t[head_idx_long:head_idx_long + long_steps].to(device)

    optimizer.zero_grad()

    out = torchdiffeq.odeint(mixmodel, u_short[[0]], t_short)[:,0,:]
    loss = F.mse_loss(u_short, out)

    out_da = CGFilter(mixmodel, u1=u_long[:, [0]].reshape(-1, 1, 1), mu0=torch.zeros(2,1).to(device), R0=0.01*torch.eye(2).to(device), cut_point=cut_point, sigma_lst=sigma_hat)[0]
    loss_da = F.mse_loss(u_long[cut_point:, 1:], out_da.squeeze(2))

    # out_da, out_R = CGFilter(mixmodel, u1=u_long[:, [0]].reshape(-1, 1, 1), mu0=torch.zeros(2,1).to(device), R0=0.01*torch.eye(2).to(device), cut_point=cut_point, sigma_lst=sigma_hat)
    # loss_da = avg_neg_log_likehood(u_long[cut_point:, 1:].unsqueeze(2), out_da, out_R)

    total_loss = loss + loss_da
    total_loss.backward()
    optimizer.step()
    train_loss_history.append(loss.item())
    train_loss_da_history.append(loss_da.item())

    end_time = time.time()
    print(ep, "time:", end_time-start_time, " loss:", loss.item(), " loss da:", loss_da.item())

# torch.save(mixmodel.state_dict(), r"NonCG_mixmodel2.pt")

#################################################
################# Test MixModel #################
#################################################

# cgnn = CGNN()
# mixmodel = MixModel(cgnn)
# mixmodel.load_state_dict(torch.load("/home/cc/CodeProjects/CGNSDE/NonCG/NonCG_Model/NonCG_mixmodel2.pt"))

# Short-term Prediction
def integrate_batch(t, u, model, batch_steps):
    # u is in vector form, e.g. (t, x)
    device = u.device
    Nt = u.shape[0]
    num_batchs = int(Nt / batch_steps)
    error_abs = 0
    # error_rel = 0
    u_pred = torch.tensor([]).to(device)
    for i in range(num_batchs):
        u_batch = u[i*batch_steps: (i+1)*batch_steps]
        with torch.no_grad():
            u_batch_pred = torchdiffeq.odeint(model, u_batch[[0]], t[:batch_steps])[:,0,:]
        u_pred = torch.cat([u_pred, u_batch_pred])
        error_abs += torch.mean( (u_batch - u_batch_pred)**2 ).item()
        # error_rel += torch.mean( torch.norm(stt_batch - stt_pred_batch, 2, 1) / (torch.norm(stt_batch, 2, 1)) ).item()
    error_abs /= num_batchs
    # error_rel /= num_batch
    return [u_pred, error_abs]
u_shortPreds, error_abs = integrate_batch(test_t, test_u, mixmodel, short_steps)

fig = plt.figure(figsize=(12, 10))
axs = fig.subplots(3, 1, sharex=True)
axs[0].plot(test_t, test_u[:, 0], linewidth=3)
axs[0].plot(test_t, u_shortPreds[:, 0],linestyle="dashed", linewidth=2)
axs[0].set_ylabel(r"$x$", fontsize=25, rotation=0)
axs[0].set_title(r"\textbf{Short-term Prediction by MixModel-DA}", fontsize=30)
axs[1].plot(test_t, test_u[:, 1], linewidth=3)
axs[1].plot(test_t, u_shortPreds[:, 1],linestyle="dashed", linewidth=2)
axs[1].set_ylabel(r"$y$", fontsize=25, rotation=0)
axs[2].plot(test_t, test_u[:, 2], linewidth=3)
axs[2].plot(test_t, u_shortPreds[:, 2],linestyle="dashed", linewidth=2)
axs[2].set_ylabel(r"$z$", fontsize=25, rotation=0)
axs[2].set_xlabel(r"$t$", fontsize=25)
for ax in fig.get_axes():
    ax.tick_params(labelsize=25, length=7, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
fig.tight_layout()
plt.show()


# Data Assimilation
with torch.no_grad():
    mu_preds, R_preds = CGFilter(mixmodel, u1=test_u[:, [0]].reshape(-1, 1, 1), mu0=torch.zeros(2, 1).to(device), R0=0.01*torch.eye(2).to(device), cut_point=0, sigma_lst=sigma_hat)
F.mse_loss(test_u[:,1:], mu_preds.reshape(-1, 2))
avg_neg_log_likehood(test_u[:,1:].unsqueeze(2), mu_preds, R_preds)

fig = plt.figure(figsize=(12, 8))
axs = fig.subplots(2, 1, sharex=True)
axs[0].plot(test_t, test_u[:, 1], linewidth=3, label=r"\textbf{True System}")
axs[0].plot(test_t, mu_preds[:, 0, 0], linewidth=3, linestyle="dashed", label=r"\textbf{DA Mean}")
axs[0].fill_between(test_t, mu_preds[:, 0, 0]-2*torch.sqrt(R_preds[:, 0, 0]), mu_preds[:, 0, 0]+2*torch.sqrt(R_preds[:, 0, 0]), color='C1', alpha=0.3, label=r"\textbf{Uncertainty}")
axs[0].set_ylabel(r"$y$", fontsize=30, rotation=0)
axs[0].set_title(r"\textbf{MixModel}", fontsize=35)
axs[0].tick_params(labelsize=30)
axs[1].plot(test_t, test_u[:, 2], linewidth=3)
axs[1].plot(test_t, mu_preds[:, 1, 0], linewidth=3, linestyle="dashed")
axs[1].fill_between(test_t, mu_preds[:, 1, 0]-2*torch.sqrt(R_preds[:, 1, 1]), mu_preds[:, 1, 0]+2*torch.sqrt(R_preds[:, 1, 1]), color='C1', alpha=0.3, label="Uncertainty")
axs[1].set_ylabel(r"$z$", fontsize=30, rotation=0)
axs[1].set_xlabel(r"$t$", fontsize=30)
axs[1].tick_params(labelsize=30)
for ax in fig.get_axes():
    ax.tick_params(labelsize=25, length=7, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
fig.tight_layout()
plt.show()





# Long-term Simulation
torch.manual_seed(0)
np.random.seed(0)
with torch.no_grad():
    u_longSimu = SDESolver(mixmodel, test_u[0], steps=500000, dt=0.001, sigma_lst=sigma_hat)[::10]

def acf(x, lag=500):
    i = np.arange(0, lag+1)
    v = np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  for i in range(1, lag+1)])
    return (i, v)
t_lags = np.linspace(0, 5, 501)

test_u = test_u.numpy()
u_longSimu = u_longSimu.numpy()

plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} "

fig = plt.figure(figsize=(22, 12))
# x dynamics
ax00 = plt.subplot2grid((3, 5), (0, 0), colspan=3)
ax00.plot(test_t, test_u[:, 0], linewidth=2, label="True signal", color="blue")
ax00.plot(test_t, u_longSimu[:, 0], linewidth=2, label="CGNSDE with DA loss", color="red")
ax00.set_ylim( [min(np.min(test_u[:,0]),np.min(u_longSimu[:,0])), max(np.max(test_u[:,0]), np.max(u_longSimu[:,0]))] )
ax00.set_ylabel(r"$x$", fontsize=35, rotation=0)
ax00.set_title(r"{(a) Signal", fontsize=35)
ax01 = plt.subplot2grid((3, 5), (0, 3))
sns.kdeplot(test_u[:, 0], ax=ax01, linewidth=3, bw_adjust=2, color="blue")
sns.kdeplot(u_longSimu[:, 0], ax=ax01, linewidth=3, bw_adjust=2, color="red")
ax01.set_ylabel("")
ax01.set_xlim( [min(np.min(test_u[:,0]),np.min(u_longSimu[:,0])), max(np.max(test_u[:,0]), np.max(u_longSimu[:,0]))] )
ax01.set_title(r"(b) PDF", fontsize=35)
ax02 = plt.subplot2grid((3, 5), (0, 4))
ax02.plot(t_lags, acf(test_u[:, 0])[1], linewidth=3, color="blue")
ax02.plot(t_lags, acf(u_longSimu[:, 0])[1], linewidth=3, color="red")
ax02.set_title(r"(c) ACF", fontsize=35)
ax02.set_yticks(np.arange(0, 1+0.5, 0.5))
ax02.set_xticks(np.linspace(0, 5, 6))
# y dynamics
ax10 = plt.subplot2grid((3, 5), (1, 0), colspan=3)
ax10.plot(test_t, test_u[:, 1], linewidth=2, color="blue")
ax10.plot(test_t, u_longSimu[:, 1], linewidth=2, color="red")
ax10.set_ylim( [min(np.min(test_u[:,1]),np.min(u_longSimu[:,1])), max(np.max(test_u[:,1]), np.max(u_longSimu[:,1]))] )
ax10.set_ylabel(r"$y$", fontsize=35, rotation=0, labelpad=25)
ax11 = plt.subplot2grid((3, 5), (1, 3))
sns.kdeplot(test_u[:, 1], ax=ax11, linewidth=3, bw_adjust=2, color="blue")
sns.kdeplot(u_longSimu[:, 1], ax=ax11, linewidth=3, bw_adjust=2, color="red")
ax11.set_xlim( [min(np.min(test_u[:,1]),np.min(u_longSimu[:,1])), max(np.max(test_u[:,1]), np.max(u_longSimu[:,1]))] )
ax11.set_ylabel("")
ax12 = plt.subplot2grid((3, 5), (1, 4))
ax12.plot(t_lags, acf(test_u[:, 1])[1], linewidth=3, color="blue")
ax12.plot(t_lags, acf(u_longSimu[:, 1])[1], linewidth=3, color="red")
ax12.set_yticks(np.arange(0, 1+0.5, 0.5))
ax12.set_xticks(np.linspace(0, 5, 6))
# z dynamics
ax20 = plt.subplot2grid((3, 5), (2, 0), colspan=3)
ax20.plot(test_t, test_u[:, 2], linewidth=2, color="blue")
ax20.plot(test_t, u_longSimu[:, 2], linewidth=2, color="red")
ax20.set_ylim( [min(np.min(test_u[:,2]),np.min(u_longSimu[:,2]))-0.2, max(np.max(test_u[:,2]), np.max(u_longSimu[:,2]))] )
ax20.set_ylabel(r"$z$", fontsize=35, rotation=0, labelpad=25)
ax20.set_xlabel(r"$t$", fontsize=35)
ax21 = plt.subplot2grid((3, 5), (2, 3))
sns.kdeplot(test_u[:, 2], ax=ax21, linewidth=3, bw_adjust=2, color="blue")
sns.kdeplot(u_longSimu[:, 2], ax=ax21, linewidth=3, bw_adjust=2, color="red")
ax21.set_xlim( [min(np.min(test_u[:,2]),np.min(u_longSimu[:,2])), max(np.max(test_u[:,2]), np.max(u_longSimu[:,2]))] )
ax21.set_ylabel("")
ax22 = plt.subplot2grid((3, 5), (2, 4))
ax22.plot(t_lags, acf(test_u[:, 2])[1], linewidth=3, color="blue")
ax22.plot(t_lags, acf(u_longSimu[:, 2])[1], linewidth=3, color="red")
ax22.set_xlabel(r"$t$", fontsize=35)
ax22.set_yticks(np.arange(0, 1+0.5, 0.5))
ax22.set_xticks(np.linspace(0, 5, 6))
for ax in fig.get_axes():
    ax.tick_params(labelsize=30, length=8, width=1, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(1)
ax00.set_xlim([100, 600])
ax10.set_xlim([100, 600])
ax20.set_xlim([100, 600])
ax02.set_xlim([0, 5])
ax12.set_xlim([0, 5])
ax22.set_xlim([0, 5])
ax00.set_yticks([-1.5, 0, 1.5])
ax10.set_yticks([-4, -2, 0])
ax20.set_yticks([-2, 0, 2])
ax01.set_xticks([-1.5, 0, 1.5])
ax11.set_xticks([-4, -2, 0])
ax21.set_xticks([-2, 0, 2])
lege = fig.legend(fontsize=35, loc="upper center", ncol=2, fancybox=False, edgecolor="black", bbox_to_anchor=(0.53, 1))
lege.get_frame().set_linewidth(1)
fig.tight_layout()
fig.subplots_adjust(top=0.84)
plt.show()

