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

# Indices of u1 and u2
indices_u1 = np.arange(0, 36, 2)
indices_u2 = np.arange(1, 36, 2)
dim_u1 = len(indices_u1)
dim_u2 = len(indices_u2)


####################################################
################# CGNN & MixModel  #################
####################################################
class UnitNet1(nn.Module):
    def __init__(self, input_size=3, output_size=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, 5), nn.ReLU(),
                                 nn.Linear(5, 10), nn.ReLU(),
                                 nn.Linear(10, 15), nn.ReLU(),
                                 nn.Linear(15, 10), nn.ReLU(),
                                 nn.Linear(10, output_size))


    def forward(self, x):
        out = self.net(x)
        return out
class UnitNet2(nn.Module):
    def __init__(self, input_size=2, output_size=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, 4), nn.ReLU(),
                                 nn.Linear(4, 6), nn.ReLU(),
                                 nn.Linear(6, output_size))

    def forward(self, x):
        out = self.net(x)
        return out
class CGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.unitnet1 = UnitNet1()
        self.unitnet2 = UnitNet2()

    def forward(self, u1):
        N, dim_u1 = u1.shape
        x1 = torch.stack([u1[:, [i-1, i, (i+1)%dim_u1]] for i in range(18)], dim=1)
        x2 = torch.stack([u1[:, [i, (i+1)%dim_u1]] for i in range(18)], dim=1)
        out1 = self.unitnet1(x1)
        out2 = self.unitnet2(x2)
        return (out1, out2)

class MixModel(nn.Module):
    def __init__(self, cgnn):
        super().__init__()
        self.outreg = None
        self.outnet = None
        self.out = None
        self.reg1 = nn.Parameter(torch.randn(6))
        self.reg2 = nn.Parameter(torch.randn(4))
        self.net = cgnn

    def forward(self, t, u):
        indices_u1 = np.arange(0, 36, 2)
        indices_u2 = np.arange(1, 36, 2)
        dim_u1 = len(indices_u1)
        dim_u2 = len(indices_u2)
        dim_u = dim_u1 + dim_u2

        self.outreg = torch.zeros_like(u)
        x1 = torch.stack([torch.stack([u[:,i], u[:,i+1], u[:,i-2]*u[:,i-1], u[:,i-1]*u[:,(i+2)%dim_u], u[:,i]*u[:,i+1]]) for i in indices_u1]).permute(2,0,1)
        x2 = torch.stack([torch.stack([u[:,i], u[:,i-2]*u[:,i-1], u[:,i-1]*u[:,(i+1)%dim_u]]) for i in indices_u2]).permute(2,0,1)
        self.outreg[:, indices_u1] = torch.einsum("nij,j->ni", x1, self.reg1[1:]) + self.reg1[0]
        self.outreg[:, indices_u2] = torch.einsum("nij,j->ni", x2, self.reg2[1:]) + self.reg2[0]

        outnet_dynamics = torch.zeros_like(u)
        u1 = u[:, indices_u1]
        self.outnet = self.net(u1)
        outnet_dynamics[:, indices_u1] = self.outnet[0][:,:,0] + self.outnet[0][:,:,1]*u[:,torch.arange(-1,34,2)] + self.outnet[0][:,:,2]*u[:,torch.arange(1,36,2)]
        outnet_dynamics[:, indices_u2] = self.outnet[1][:,:,0] + self.outnet[1][:,:,1]*u[:,torch.arange(-1,34,2)] + self.outnet[1][:,:,2]*u[:,torch.arange(1,36,2)] + self.outnet[1][:,:,3]*u[:,torch.arange(3,38,2)%dim_u]

        self.out = outnet_dynamics + self.outreg
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

# # Stage1: Train mixmodel with forecast loss (One-Step)
epochs = 500
batch_size = 200
train_tensor = torch.utils.data.TensorDataset(train_u, train_u_dot)
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
train_num_batches = len(train_loader)
train_loss_history = []

cgnn = CGNN()
mixmodel = MixModel(cgnn)
optimizer = torch.optim.Adam(mixmodel.parameters(), lr=1e-3)
for ep in range(1, epochs+1):
    start_time = time.time()
    train_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        out = mixmodel(None, x)
        loss = nnF.mse_loss(y, out)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= train_num_batches
    train_loss_history.append(train_loss)
    end_time = time.time()
    print(ep, " loss: ", loss.item(), " time: ", end_time-start_time)

# torch.save(mixmodel.state_dict(), r"/home/cc/CodeProjects/CGNN/L96/case2/L96(case2)_Model/L96(case2)_mixmodel1.pt")
# np.save(r"/home/cc/CodeProjects/CGNN/L96/case2/L96(case2)_Model/L96(case2)_mixmodel1_train_loss_history.npy", train_loss_history)

mixmodel.load_state_dict(torch.load("/home/cc/CodeProjects/CGNSDE/L96/case2/L96(case2)_Model/L96(case2)_mixmodel1.pt"))


# # Stage1: Train mixmodel with forecast loss (Multi-Steps)
short_steps = int(0.05/dt)
# epochs = 10000
# train_loss_history = []
#
# cgnn = CGNN()
# mixmodel = MixModel(cgnn).to(device)
# optimizer = torch.optim.Adam(mixmodel.parameters(), lr=1e-3)
# for ep in range(1, epochs+1):
#     start_time = time.time()
#     head_idx_short = torch.from_numpy(np.random.choice(Ntrain-short_steps+1, size=1))
#     u_short = u[head_idx_short:head_idx_short + short_steps].to(device)
#     t_short = t[head_idx_short:head_idx_short + short_steps].to(device)
#
#     optimizer.zero_grad()
#
#     out = torchdiffeq.odeint(mixmodel, u_short[[0]], t_short, method="rk4", options={"step_size":0.005})[:,0,:]
#     # out = ODESolver(mixmodel, u_short[0], short_steps, dt)
#     loss = nnF.mse_loss(u_short, out)
#
#     loss.backward()
#     optimizer.step()
#     train_loss_history.append(loss.item())
#     end_time = time.time()
#     print(ep, " loss: ", loss.item(), " time: ", end_time-start_time)
#
# torch.save(mixmodel.state_dict(), r"/home/cc/CodeProjects/CGNN/L96/case1/L96(case1)_Model/L96(case1)_mixmodel1.pt")
# np.save(r"/home/cc/CodeProjects/CGNN/L96/case1/L96(case1)_Model/L96(case1)_mixmodel1_train_loss_history.npy", train_loss_history)


##########################################################
################# Estimate sigma  & CGF  #################
##########################################################
with torch.no_grad():
    train_u_dot_pred = mixmodel(None, train_u)
sigma_hat = torch.sqrt( dt*torch.mean( (train_u_dot - train_u_dot_pred)**2, dim=0 ) ).tolist()


# mixmodel
# u1 = train_u[:, indices_u1].reshape(10000, 18, 1)
# mu0 = torch.zeros(dim_u2, 1)
# R0 = 0.01*torch.eye(dim_u2)
# cut_point = 0
# sigma_lst = sigma_hat
def CGFilter(mixmodel, u1, mu0, R0, cut_point, sigma_lst):
    # u1, mu0 are in col-matrix form, e.g. (t, x, 1)
    device = u1.device
    sigma_tsr = torch.tensor(sigma_lst)

    indices_u1 = np.arange(0, 36, 2)
    indices_u2 = np.arange(1, 36, 2)
    dim_u1 = len(indices_u1)
    dim_u2 = len(indices_u2)
    dim_u = dim_u1 + dim_u2

    a = mixmodel.reg1
    b = mixmodel.reg2

    Nt = u1.shape[0]
    mu_trace = torch.zeros((Nt, dim_u2, 1)).to(device)
    R_trace = torch.zeros((Nt, dim_u2, dim_u2)).to(device)
    mu_trace[0] = mu0
    R_trace[0] = R0
    for n in range(1, Nt):
        du1 = u1[n] - u1[n-1]
        outnet = mixmodel.net(u1[n-1].T) # A tuple with outputs of 2 NNs
        outnet1 = outnet[0].squeeze(0)  # (18, 3)
        outnet2 = outnet[1].squeeze(0)  # (18, 4)

        f1 = a[0] + a[1]*u1[n-1] + outnet1[:,[0]]
        g1 = torch.zeros(dim_u1, dim_u2)
        g1[torch.arange(18).unsqueeze(dim=1), torch.stack([torch.arange(18)-1, torch.arange(18)]).T] = \
            torch.cat([ a[3]*u1[n-1][torch.arange(-1, 17)]+a[4]*u1[n-1][torch.arange(1,19)%dim_u1]+outnet1[:,[1]],
                          a[2]+a[5]*u1[n-1]+outnet1[:,[2]] ], dim=1)
        s1 = torch.diag(sigma_tsr[indices_u1])
        f2 = b[0] + outnet2[:, [0]] + b[3]*u1[n-1]*u1[n-1][torch.arange(1, 19)%dim_u1]
        g2 = torch.zeros(dim_u2, dim_u2)
        g2[torch.arange(18).unsqueeze(dim=1), torch.stack([torch.arange(-1, 17), torch.arange(0, 18), torch.arange(1, 19)%18]).T] = \
            torch.cat([ b[2]*u1[n-1]+outnet2[:,[1]],
                        b[1]+outnet2[:, [2]],
                        outnet2[:, [3]] ], dim=1)
        s2 = torch.diag(sigma_tsr[indices_u2])
        invs1os1 = torch.linalg.inv(s1@s1.T)
        s2os2 = s2@s2.T
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
        u_simu[n+1] = torch.clamp(u_simu[n] + u_dot_pred*dt + sigma*np.sqrt(dt)*torch.randn(dim), min=torch.min(train_u), max=torch.max(train_u))
    return u_simu


############################################################
################# Train MixModel (Stage2)  #################
############################################################
# Stage 2: Train mixmodel with forcast loss + DA loss

def avg_neg_log_likehood(x, mu, R):
    # x, mu are in matrix form, e.g. (t, x, 1)
    d = x.shape[1]
    neg_log_likehood = 1/2*(d*np.log(2*np.pi) + torch.log(torch.linalg.det(R)) + ((x-mu).permute(0,2,1)@torch.linalg.inv(R)@(x-mu)).flatten())
    return torch.mean(neg_log_likehood)

long_steps = int(100/dt)
cut_point = int(5/dt)

epochs = 1000
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
    loss = nnF.mse_loss(u_short, out)

    out_da = CGFilter(mixmodel, u1=u_long[:, indices_u1].unsqueeze(2), mu0=torch.zeros(dim_u2,1).to(device), R0=0.01*torch.eye(dim_u2).to(device), cut_point=cut_point, sigma_lst=sigma_hat)[0]
    loss_da = nnF.mse_loss(u_long[cut_point:, indices_u2], out_da.squeeze(2))

    total_loss = loss + loss_da
    total_loss.backward()
    optimizer.step()
    train_loss_history.append(loss.item())
    train_loss_da_history.append(loss_da.item())

    end_time = time.time()
    print(ep, "time:", end_time-start_time, " loss:", loss.item(), " loss da:", loss_da.item())

    # if ep % 100 == 0:
    #     torch.save(mixmodel.state_dict(), r"/home/cc/CodeProjects/CGNN/L96/case2/L96(case2)_Model/L96(case2)_mixmodel2_ep"+str(ep) + ".pt")

# np.save(r"/home/cc/CodeProjects/CGNN/L96/case2/L96(case2)_Model/L96(case2)_mixmodel2_train_loss_history.npy", train_loss_history)
# np.save(r"/home/cc/CodeProjects/CGNN/L96/case2/L96(case2)_Model/L96(case2)_mixmodel2_train_loss_da_history.npy", train_loss_da_history)

#################################################
################# Test MixModel #################
#################################################

# cgnn = CGNN()
# mixmodel = MixModel(cgnn)
# mixmodel.load_state_dict(torch.load("/home/cc/CodeProjects/CGNSDE/L96/case2/L96(case2)_Model/L96(case2)_mixmodel2_ep200.pt"))


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
u_shortPreds, error_abs = integrate_batch(test_t, test_u, mixmodel, batch_steps=20)

fig = plt.figure(figsize=(12, 8))
axs = fig.subplots(2, 1, sharex=True)
axs[0].plot(test_t, test_u[:, 0], linewidth=3)
axs[0].plot(test_t, u_shortPreds[:, 0],linestyle="dashed", linewidth=2)
axs[0].set_ylabel(r"$x_0$", fontsize=25, rotation=0)
axs[0].set_title(r"\textbf{Short-term Prediction by MixModel-DA}", fontsize=30)
axs[1].plot(test_t, test_u[:, 1], linewidth=3)
axs[1].plot(test_t, u_shortPreds[:, 1],linestyle="dashed", linewidth=2)
axs[1].set_ylabel(r"$x_1$", fontsize=25, rotation=0)
axs[1].set_xlabel(r"$t$", fontsize=25)
for ax in fig.get_axes():
    ax.tick_params(labelsize=25, length=7, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
fig.tight_layout()
plt.show()


# Data Assimilation
with torch.no_grad():
    mu_preds, R_preds = CGFilter(mixmodel, u1=test_u[:, indices_u1].unsqueeze(2), mu0=torch.zeros(dim_u2, 1).to(device), R0=0.01*torch.eye(dim_u2).to(device), cut_point=0, sigma_lst=sigma_hat)
nnF.mse_loss(test_u[:,indices_u2], mu_preds.squeeze(2))
avg_neg_log_likehood(test_u[:,indices_u2].unsqueeze(2), mu_preds, R_preds)

# fig = plt.figure(figsize=(10, 4))
# ax = fig.subplots(1, 1)
# ax.plot(test_t, test_u[:, 1], linewidth=3, label=r"\textbf{True System}")
# ax.plot(test_t, mu_preds[:, 0, 0], linewidth=2, linestyle="dashed", label=r"\textbf{DA Mean}")
# ax.fill_between(test_t, mu_preds[:, 0, 0]-2*torch.sqrt(R_preds[:, 0, 0]), mu_preds[:, 0, 0]+2*torch.sqrt(R_preds[:, 0, 0]), color='C1', alpha=0.2, label=r"\textbf{Uncertainty}")
# ax.set_ylabel(r"$x_1$", fontsize=30, rotation=0)
# ax.set_title(r"\textbf{MixModel}", fontsize=30)
# ax.tick_params(labelsize=30)
# for ax in fig.get_axes():
#     ax.tick_params(labelsize=25, length=7, width=2)
#     for spine in ax.spines.values():
#         spine.set_linewidth(2)
# fig.tight_layout()
# plt.show()



# Long-term Simulation
torch.manual_seed(0)
np.random.seed(0)
with torch.no_grad():
    u_longSimu = SDESolver(mixmodel, test_u[0], steps=Ntest, dt=0.01, sigma_lst=sigma_hat)

test_u = test_u.numpy()
u_longSimu = u_longSimu.numpy()

def acf(x, lag=200):
    i = np.arange(0, lag+1)
    v = np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  for i in range(1, lag+1)])
    return (i, v)
t_lags = np.linspace(0, 2, 201)


# Pcolor
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
fig = plt.figure(figsize=(14, 6))
ax = fig.subplots()
c = ax.pcolor(t[10000:15000:10], np.arange(1, I+1), u_longSimu[:5000:10].T, cmap="magma")
cbar = fig.colorbar(c, ax=ax, location='top')
cbar.ax.tick_params(labelsize=30, length=8, width=1, direction="out")
ax.set_title(r"(a) Hovmoller diagram by CGNSDE with DA loss", fontsize=35, pad=110)
ax.set_xlabel(r"$t$", fontsize=35)
ax.set_ylabel(r"$i$", fontsize=35, rotation=0, labelpad=20)
ax.set_yticks(np.array([1, 18, 36]))
ax.set_xticks(np.arange(100, 150+10, 10))
ax.tick_params(labelsize=30, length=12, width=1, direction="out")
for spine in ax.spines.values():
    spine.set_linewidth(1)
fig.tight_layout()



# Time Series
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
fig = plt.figure(figsize=(22, 9))
# x1 dynamics
ax00 = plt.subplot2grid((2, 5), (0, 0), colspan=3)
ax00.plot(test_t, test_u[:, 0], linewidth=2, label="True signal", color="blue")
ax00.plot(test_t, u_longSimu[:, 0], linewidth=2, label="CGNSDE with DA loss", color="red")
ax00.set_ylim( [min(np.min(test_u[:,0]),np.min(u_longSimu[:,0])), max(np.max(test_u[:,0]), np.max(u_longSimu[:,0]))] )
ax00.set_ylabel(r"$x_1$", fontsize=35, rotation=0, labelpad=8)
ax00.set_title(r"{(b) Signal", fontsize=35)
ax01 = plt.subplot2grid((2, 5), (0, 3))
sns.kdeplot(test_u[:, 0], ax=ax01, linewidth=3, bw_adjust=2, color="blue")
sns.kdeplot(u_longSimu[:, 0], ax=ax01, linewidth=3, bw_adjust=2, color="red")
ax01.set_ylabel("")
ax01.set_xlim( [min(np.min(test_u[:,0]),np.min(u_longSimu[:,0])), max(np.max(test_u[:,0]), np.max(u_longSimu[:,0]))] )
ax01.set_title(r"(c) PDF", fontsize=35)
ax02 = plt.subplot2grid((2, 5), (0, 4))
ax02.plot(t_lags, acf(test_u[:, 0])[1], linewidth=3, color="blue")
ax02.plot(t_lags, acf(u_longSimu[:, 0])[1], linewidth=3, color="red")
ax02.set_title(r"(d) ACF", fontsize=35)
ax02.set_yticks(np.arange(0, 1+0.5, 0.5))
ax02.set_xticks(np.linspace(0, 2, 3))
# x2 dynamics
ax10 = plt.subplot2grid((2, 5), (1, 0), colspan=3)
ax10.plot(test_t, test_u[:, 1], linewidth=2, color="blue")
ax10.plot(test_t, u_longSimu[:, 1], linewidth=2, color="red")
ax10.set_ylim( [min(np.min(test_u[:,1]),np.min(u_longSimu[:,1])), max(np.max(test_u[:,1]), np.max(u_longSimu[:,1]))] )
ax10.set_ylabel(r"$x_2$", fontsize=35, rotation=0, labelpad=8)
ax11 = plt.subplot2grid((2, 5), (1, 3))
sns.kdeplot(test_u[:, 1], ax=ax11, linewidth=3, bw_adjust=2, color="blue")
sns.kdeplot(u_longSimu[:, 1], ax=ax11, linewidth=3, bw_adjust=2, color="red")
ax11.set_xlim( [min(np.min(test_u[:,1]),np.min(u_longSimu[:,1])), max(np.max(test_u[:,1]), np.max(u_longSimu[:,1]))] )
ax11.set_ylabel("")
ax12 = plt.subplot2grid((2, 5), (1, 4))
ax12.plot(t_lags, acf(test_u[:, 1])[1], linewidth=3, color="blue")
ax12.plot(t_lags, acf(u_longSimu[:, 1])[1], linewidth=3, color="red")
ax12.set_yticks(np.arange(0, 1+0.5, 0.5))
ax12.set_xticks(np.linspace(0, 2, 3))
for ax in fig.get_axes():
    ax.tick_params(labelsize=30, length=8, width=1, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(1)
ax00.set_xlim([100, 300])
ax10.set_xlim([100, 300])
ax02.set_xlim([0, 2])
ax12.set_xlim([0, 2])
ax10.set_xlabel("$t$", fontsize=35)
ax12.set_xlabel("$t$", fontsize=35)
lege = fig.legend(fontsize=35, loc="upper center", ncol=2, fancybox=False, edgecolor="black") #, bbox_to_anchor=(0.53, 1)
lege.get_frame().set_linewidth(1)
fig.tight_layout()
fig.subplots_adjust(top=0.77)
# ax00.set_ylim([-4.5, 4.5])
# ax10.set_ylim([-4.5, 4.5])
# ax20.set_ylim([-4.5, 4.5])
# ax01.set_xlim([-4, 4])
# ax11.set_xlim([-4, 4])
# ax21.set_xlim([-4, 4])
ax00.set_yticks([-10, -5, 0, 5, 10])
ax10.set_yticks([-10, -5, 0, 5, 10])
ax01.set_xticks([-10, -5, 0, 5, 10])
ax11.set_xticks([-10, -5, 0, 5, 10])
plt.show()

