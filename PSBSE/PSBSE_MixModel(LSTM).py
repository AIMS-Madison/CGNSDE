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

Lt = 200
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
Ntest = 10000
train_u = u[:Ntrain]
train_t = t[:Ntrain]
test_u = u[-Ntest:]
test_t = t[-Ntest:]

####################################################
################# CGNN & MixModel  #################
####################################################

class CGNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=5, output_size=9, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out_lstm, _ = self.lstm(x)
        out = self.fc(out_lstm[:, -1, :])
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

    def forward(self, u_history):
        # u_history shape (N, t, x)
        u1_history = u_history[:, :, [0]]
        self.outnet = self.net(u1_history)

        u = u_history[:, -1, :]
        basis_x = torch.stack([u[:,2], u[:,0]*u[:,1]]).T
        basis_y = torch.stack([u[:,0]**2, u[:,0]*u[:,2]]).T
        basis_z = torch.stack([u[:,0]*u[:,1]]).T

        x_dyn = self.reg0(basis_x) + self.outnet[:, [0]] + self.outnet[:, [3]]*u[:,[1]] + self.outnet[:, [4]]*u[:,[2]]
        y_dyn = self.reg1(basis_y) + self.outnet[:, [1]] + self.outnet[:, [5]]*u[:,[1]] + self.outnet[:, [6]]*u[:,[2]]
        z_dyn = self.reg2(basis_z) + self.outnet[:, [2]] + self.outnet[:, [7]]*u[:,[1]] + self.outnet[:, [8]]*u[:,[1]]
        self.out = torch.cat([x_dyn, y_dyn, z_dyn], dim=1)
        return self.out

def ODESolver(mixmodel, u_history, steps, dt):
    # u_history is in vector form, e.g. (t, x)
    dim = u_history.shape[1]
    u_pred = torch.zeros(steps, dim)
    u_pred[0] = u_history[-1]
    for n in range(0, steps-1):
        u_dot_pred = mixmodel(u_history.unsqueeze(0)).squeeze(0)
        u_pred[n+1] = u_pred[n] + u_dot_pred * dt
        u_history = torch.cat([u_history[:-1], u_pred[n+1].unsqueeze(0)])
    return u_pred

############################################################
################# Train MixModel (Stage1)  #################
############################################################
short_steps = int(0.5/dt)
memory_steps = int(0.2/dt)

# Stage1: Train mixmodel with forecast loss
epochs = 10000
train_loss_history = []
train_loss_da_history = []

cgnn = CGNN()
mixmodel = MixModel(cgnn).to(device)
optimizer = torch.optim.Adam(mixmodel.parameters(), lr=1e-3)
for ep in range(1, epochs+1):
    start_time = time.time()

    head_idx_short = torch.from_numpy(np.random.choice(np.arange(memory_steps-1, Ntrain-short_steps+1), size=1))
    u_history = u[head_idx_short+1-memory_steps: head_idx_short+1].to(device)
    u_short = u[head_idx_short:head_idx_short+short_steps].to(device)

    optimizer.zero_grad()

    out = ODESolver(mixmodel, u_history, short_steps, dt)
    loss = F.mse_loss(u_short, out)

    loss.backward()
    optimizer.step()
    train_loss_history.append(loss.item())
    end_time = time.time()
    print(ep, " loss: ", loss.item(), " time: ", end_time-start_time)

torch.save(mixmodel.state_dict(), r"/home/cc/CodeProjects/CGNN/NonCG/NonCG_Model/NonCG_mixmodel1_lstm.pt")
np.save(r"/home/cc/CodeProjects/CGNN/NonCG/NonCG_Model/NonCG_mixmodel1_lstm_train_loss.npy", train_loss_history)



##########################################################
################# Estimate sigma  & CGF  #################
##########################################################
def sliding_batches(u, batch_size, step=1):
    # u is in vector form, e.g. (t, x)
    # u_batched is in vector form, e.g. (N, t, x)
    num_batches = (u.shape[0] - batch_size) // step + 1
    u_batched = torch.zeros((num_batches, batch_size, u.shape[1]))
    for i in range(num_batches):
        start_index = i * step
        end_index = start_index + batch_size
        u_batched[i] = u[start_index:end_index]
    return u_batched
train_u_dot = torch.diff(train_u, dim=0)/dt
train_u_dot = train_u_dot[memory_steps-1:]

train_u_batched = sliding_batches(train_u, memory_steps)
with torch.no_grad():
    train_u_dot_pred = mixmodel(train_u_batched[:-1])

sigma_hat = torch.sqrt( dt*torch.mean( (train_u_dot - train_u_dot_pred)**2, dim=0 ) ).tolist()

def CGFilter(mixmodel, u1, mu0, R0, cut_point, sigma_lst, memory_steps):
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
    mu_trace = torch.zeros((Nt-memory_steps+1, dim_u2, 1)).to(device)
    R_trace = torch.zeros((Nt-memory_steps+1, dim_u2, dim_u2)).to(device)
    mu_trace[0] = mu0
    R_trace[0] = R0
    for n in range(1, Nt-memory_steps+1):
        start_idx = n-1
        end_idx = n-1+memory_steps
        u1_history = u1[start_idx:end_idx]
        x0 = u1[end_idx-1].flatten()
        x1 = u1[end_idx].flatten()
        du1 = (x1 - x0).reshape(-1, 1)
        outnet = mixmodel.net(u1_history.reshape(1, memory_steps, 1)).T

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

def SDESolver(mixmodel, u_history, steps, dt, sigma_lst):
    # u_history is in vector form, e.g. (t, x)
    dim = u_history.shape[1]
    sigma = torch.tensor(sigma_lst)
    u_simu = torch.zeros(steps, dim)
    u_simu[0] = u_history[-1]
    for n in range(0, steps-1):
        u_dot_pred = mixmodel(u_history.unsqueeze(0)).squeeze(0)
        u_simu[n+1] = u_simu[n]+u_dot_pred*dt + sigma*np.sqrt(dt)*torch.randn(3)
        u_history = torch.cat([u_history[:-1], u_simu[n+1].unsqueeze(0)])
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

    out = ODESolver(mixmodel, u_history, short_steps, dt)
    loss = F.mse_loss(u_short, out)

    out_da = CGFilter(mixmodel, u1=u_long[:, [0]].reshape(-1, 1, 1), mu0=torch.zeros(2,1).to(device), R0=0.01*torch.eye(2).to(device), cut_point=cut_point, sigma_lst=sigma_hat, memory_steps=memory_steps)[0]
    loss_da = F.mse_loss(u_long[memory_steps-1+cut_point:, 1:], out_da.reshape(-1, 2))

    total_loss = loss + loss_da
    total_loss.backward()
    optimizer.step()
    train_loss_history.append(loss.item())
    train_loss_da_history.append(loss_da.item())

    end_time = time.time()
    print(ep, "time:", end_time-start_time, " loss:", loss.item(), " loss da:", loss_da.item())

    if ep % 100 == 0:
        torch.save(mixmodel.state_dict(), r"/home/cc/CodeProjects/CGNN/NonCG/NonCG_Model/NonCG_mixmodel2_lstm_ep"+str(ep)+".pt")

np.save(r"/home/cc/CodeProjects/CGNN/NonCG/NonCG_Model/NonCG_mixmodel2_lstm_train_loss_history.npy", train_loss_history)
np.save(r"/home/cc/CodeProjects/CGNN/NonCG/NonCG_Model/NonCG_mixmodel2_lstm_train_loss_da_history.npy", train_loss_da_history)



# cgnn = CGNN()
# mixmodel = MixModel(cgnn)
# mixmodel.load_state_dict(torch.load("/home/cc/CodeProjects/CGNN/NonCG/NonCG_Model/NonCG_mixmodel2_fc_ep500.pt"))


#################################################
################# Test MixModel #################
#################################################

# Regression Coefficients
torch.cat([mixmodel.reg0.weight.flatten(),
           mixmodel.reg1.weight.flatten(),
           mixmodel.reg2.weight.flatten()])

# Short-term Prediction
def integrate_batch(u, model, batch_steps, memory_steps):
    # u is in vector form: (t, x)
    device = u.device
    Nt = u.shape[0]
    num_batchs = int((Nt-memory_steps+1) / batch_steps)
    error_abs = 0
    # error_rel = 0
    u_pred = torch.tensor([]).to(device)
    for i in range(num_batchs):
        u_batch = u[i*batch_steps+memory_steps-1: (i+1)*batch_steps+memory_steps-1]
        u_history = u[i*batch_steps: i*batch_steps+memory_steps]
        with torch.no_grad():
            u_batch_pred = ODESolver(model, u_history, batch_steps, dt)
        u_pred = torch.cat([u_pred, u_batch_pred])
        error_abs += torch.mean( (u_batch - u_batch_pred)**2 ).item()
        # error_rel += torch.mean( torch.norm(stt_batch - stt_pred_batch, 2, 1) / (torch.norm(stt_batch, 2, 1)) ).item()
    error_abs /= num_batchs
    # error_rel /= num_batch
    return [u_pred, error_abs]
u_shortPred, error_abs = integrate_batch(train_u, mixmodel, short_steps, memory_steps)

fig = plt.figure(figsize=(20, 10))
axs = fig.subplots(3, 1)
axs[0].plot(train_t, train_u[:, 0], linewidth=5)
axs[0].plot(train_t, u_shortPred[:, 0], linewidth=5)
axs[0].set_ylabel(r"$x$", fontsize=30)
axs[0].set_title(r"\textbf{CGNN}", fontsize=40)
axs[0].tick_params(labelsize=30)
axs[1].plot(train_t, train_u[:, 1], linewidth=5)
axs[1].plot(train_t, u_shortPred[:, 1], linewidth=5)
axs[1].set_ylabel(r"$y$", fontsize=30)
axs[1].tick_params(labelsize=30)
axs[2].plot(train_t, train_u[:, 2], linewidth=5)
axs[2].plot(train_t, u_shortPred[:, 2], linewidth=5)
axs[2].set_ylabel(r"$z$", fontsize=30)
axs[2].set_xlabel(r"$t$", fontsize=30)
axs[2].tick_params(labelsize=30)
fig.tight_layout()
plt.show()



# DA Integration
start = 0
end = Ntrain
with torch.no_grad():
    mu_pred = CGFilter(mixmodel, u1=train_u[:, [0]].reshape(-1, 1, 1), mu0=torch.zeros(2, 1).to(device), R0=0.01 * torch.eye(2).to(device), cut_point=0, sigma_lst=sigma_hat)[0]
F.mse_loss(train_u[:,1:], mu_pred.reshape(-1, 2))

fig = plt.figure(figsize=(16, 10))
axs = fig.subplots(2, 1)
axs[0].plot(train_t, train_u[:, 1], linewidth=3, label="Original State")
axs[0].plot(train_t, mu_pred[:, 0, 0], linewidth=3, linestyle="dashed", label="DA Mean")
axs[0].set_ylabel(r"\unboldmath$y$", fontsize=30, rotation=0)
axs[0].tick_params(labelsize=30)
axs[0].set_title(r"\textbf{CGNN}", fontsize=40)
axs[1].plot(train_t, train_u[:, 2], linewidth=3)
axs[1].plot(train_t, mu_pred[:, 1, 0], linewidth=3, linestyle="dashed")
axs[1].set_ylabel(r"\unboldmath$z$", fontsize=30, rotation=0)
axs[1].tick_params(labelsize=30)
fig.tight_layout()
plt.show()



# Long-term Prediction
start = 0
end = Ntrain
with torch.no_grad():
    u_longPred = ODESolver(mixmodel, train_u[0], Ntrain, dt)

fig = plt.figure(figsize=(20, 10))
axs = fig.subplots(3, 1)
axs[0].plot(train_t[start:end], train_u[start:end, 0], linewidth=5)
axs[0].plot(train_t[start:end], u_longPred[:, 0], linewidth=5)
axs[0].set_ylabel(r"$x$", fontsize=30)
axs[0].set_title(r"\textbf{CGNN}", fontsize=40)
axs[0].tick_params(labelsize=30)
axs[1].plot(train_t[start:end], train_u[start:end, 1], linewidth=5)
axs[1].plot(train_t[start:end], u_longPred[:, 1], linewidth=5)
axs[1].set_ylabel(r"$y$", fontsize=30)
axs[1].tick_params(labelsize=30)
axs[2].plot(train_t[start:end], train_u[start:end, 2], linewidth=5)
axs[2].plot(train_t[start:end], u_longPred[:, 2], linewidth=5)
axs[2].set_ylabel(r"$z$", fontsize=30)
axs[2].set_xlabel(r"$t$", fontsize=30)
axs[2].tick_params(labelsize=30)
fig.tight_layout()
plt.show()




# Long-term Simulation
start = 0
end = Ntrain
with torch.no_grad():
    u_longSimu = SDESolver(mixmodel, train_u[0], Ntrain, dt, sigma_hat)

fig = plt.figure(figsize=(20, 10))
axs = fig.subplots(3, 1)
axs[0].plot(train_t[start:end], train_u[start:end, 0], linewidth=3)
axs[0].plot(train_t[start:end], u_longSimu[:, 0], linewidth=3)
axs[0].set_ylabel(r"$x$", fontsize=30)
axs[0].set_title(r"\textbf{CGNN: Long-Term Simulation with Noise}", fontsize=40)
axs[0].tick_params(labelsize=30)
axs[1].plot(train_t[start:end], train_u[start:end, 1], linewidth=3)
axs[1].plot(train_t[start:end], u_longSimu[:, 1], linewidth=3)
axs[1].set_ylabel(r"$y$", fontsize=30)
axs[1].tick_params(labelsize=30)
axs[2].plot(train_t[start:end], train_u[start:end, 2], linewidth=3)
axs[2].plot(train_t[start:end], u_longSimu[:, 2], linewidth=3)
axs[2].set_ylabel(r"$z$", fontsize=30)
axs[2].set_xlabel(r"$t$", fontsize=30)
axs[2].tick_params(labelsize=30)
fig.tight_layout()
plt.show()





# Test test data

u_shortPreds, error_abs = integrate_batch(test_t, test_u, mixmodel, short_steps)

fig = plt.figure(figsize=(20, 10))
axs = fig.subplots(3, 1)
axs[0].plot(test_t, test_u[:, 0], linewidth=5)
axs[0].plot(test_t, u_shortPreds[:, 0], linewidth=5)
axs[0].set_ylabel(r"$x$", fontsize=30)
axs[0].set_title(r"\textbf{CGNN}", fontsize=40)
axs[0].tick_params(labelsize=30)
axs[1].plot(test_t, test_u[:, 1], linewidth=5)
axs[1].plot(test_t, u_shortPreds[:, 1], linewidth=5)
axs[1].set_ylabel(r"$y$", fontsize=30)
axs[1].tick_params(labelsize=30)
axs[2].plot(test_t, test_u[:, 2], linewidth=5)
axs[2].plot(test_t, u_shortPreds[:, 2], linewidth=5)
axs[2].set_ylabel(r"$z$", fontsize=30)
axs[2].set_xlabel(r"$t$", fontsize=30)
axs[2].tick_params(labelsize=30)
fig.tight_layout()
plt.show()



start = 0
end = Ntest
with torch.no_grad():
    mu_preds = CGFilter(mixmodel, u1=test_u[:, [0]].reshape(-1, 1, 1), mu0=torch.zeros(2, 1).to(device), R0=0.01 * torch.eye(2).to(device), cut_point=0, sigma_lst=sigma_hat)[0]
F.mse_loss(test_u[:,1:], mu_preds.reshape(-1, 2))

fig = plt.figure(figsize=(16, 10))
axs = fig.subplots(2, 1)
axs[0].plot(test_t, test_u[:, 1], linewidth=3, label="Original State")
axs[0].plot(test_t, mu_preds[:, 0, 0], linewidth=3, linestyle="dashed", label="DA Mean")
axs[0].set_ylabel(r"\unboldmath$y$", fontsize=30, rotation=0)
axs[0].tick_params(labelsize=30)
axs[0].set_title(r"\textbf{CGNN}", fontsize=40)
axs[1].plot(test_t, test_u[:, 2], linewidth=3)
axs[1].plot(test_t, mu_preds[:, 1, 0], linewidth=3, linestyle="dashed")
axs[1].set_ylabel(r"\unboldmath$z$", fontsize=30, rotation=0)
axs[1].tick_params(labelsize=30)
fig.tight_layout()
plt.show()




start = 0
end = Ntest
with torch.no_grad():
    u_longPreds = torchdiffeq.odeint(mixmodel, test_u[[start]], test_t[start:end])[:, 0, :]

fig = plt.figure(figsize=(20, 10))
axs = fig.subplots(3, 1)
axs[0].plot(test_t[start:end], test_u[start:end, 0], linewidth=5)
axs[0].plot(test_t[start:end], u_longPreds[:, 0], linewidth=5)
axs[0].set_ylabel(r"$x$", fontsize=30)
axs[0].set_title(r"\textbf{CGNN}", fontsize=40)
axs[0].tick_params(labelsize=30)
axs[1].plot(test_t[start:end], test_u[start:end, 1], linewidth=5)
axs[1].plot(test_t[start:end], u_longPreds[:, 1], linewidth=5)
axs[1].set_ylabel(r"$y$", fontsize=30)
axs[1].tick_params(labelsize=30)
axs[2].plot(test_t[start:end], test_u[start:end, 2], linewidth=5)
axs[2].plot(test_t[start:end], u_longPreds[:, 2], linewidth=5)
axs[2].set_ylabel(r"$z$", fontsize=30)
axs[2].set_xlabel(r"$t$", fontsize=30)
axs[2].tick_params(labelsize=30)
fig.tight_layout()
plt.show()


# Long-term Simulation with Random Noise
start = 0
end = Ntest
u_longSimu = torch.zeros(Ntest, 3)
u_longSimu[0] = test_u[start]
for n in range(start, end-1):
    with torch.no_grad():
        u_dot = mixmodel(None, u_longSimu[[n], :])
    u_longSimu[n + 1, 0] = u_longSimu[n, 0] + (u_dot[0,0]) * dt + sigma_x * np.sqrt(dt) * np.random.randn()
    u_longSimu[n + 1, 1] = u_longSimu[n, 1] + (u_dot[0,1]) * dt + sigma_y * np.sqrt(dt) * np.random.randn()
    u_longSimu[n + 1, 2] = u_longSimu[n, 2] + (u_dot[0,2]) * dt + sigma_z * np.sqrt(dt) * np.random.randn()

fig = plt.figure(figsize=(20, 10))
axs = fig.subplots(3, 1)
axs[0].plot(test_t[start:end], test_u[start:end, 0], linewidth=3)
axs[0].plot(test_t[start:end], u_longSimu[:, 0], linewidth=3)
axs[0].set_ylabel(r"$x$", fontsize=30)
axs[0].set_title(r"\textbf{CGNN: Long-Term Simulation with Noise}", fontsize=40)
axs[0].tick_params(labelsize=30)
axs[1].plot(test_t[start:end], test_u[start:end, 1], linewidth=3)
axs[1].plot(test_t[start:end], u_longSimu[:, 1], linewidth=3)
axs[1].set_ylabel(r"$y$", fontsize=30)
axs[1].tick_params(labelsize=30)
axs[2].plot(test_t[start:end], test_u[start:end, 2], linewidth=3)
axs[2].plot(test_t[start:end], u_longSimu[:, 2], linewidth=3)
axs[2].set_ylabel(r"$z$", fontsize=30)
axs[2].set_xlabel(r"$t$", fontsize=30)
axs[2].tick_params(labelsize=30)
fig.tight_layout()
plt.show()
