import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.preprocessing import StandardScaler;
import os;
# ----------------------------------
#          Part A
# ----------------------------------

np.random.seed(1140);

# Load full LT Spice simulated dataset ~2000 points -> skip 775 points until diode is active
dataset = np.loadtxt("../lt_spice/project_2_lt_spice.txt", skiprows=775);


N = 50;
idx = np.arange(dataset.shape[0]);
np.random.shuffle( idx );

# Trim dataset to N points
dataset = dataset[idx[0:N:], :];

# Introduce measurement error - Using project example
v_r = dataset[:,2];
v_s = dataset[:,1];

# Clip to 4 decimal palces
v_r = (np.round(v_r*100000))/100000;

# Create noiseM - Normal distribtuion clipped between +- 0.05% = 0.0005
noiseM_scale = np.random.normal( loc=0, scale=0.0005, size=v_r.shape);
# Create noiseQ - unitform distribution -0.0002 -> + 0.0002
noiseQ = np.random.randint(-2, 2, size=v_r.shape)/10000;

# Add noise to voltage measurement
v_r = (v_r*(1+noiseM_scale)) + (noiseQ);

# Compute Diode current
R = 15; # Resistor Value
i_d = v_r/R;

# Compute diode voltage
v_d = v_s - v_r;


plt.figure();
plt.scatter(v_d,1000* i_d );
#plt.scatter(v_d, np.log(i_d) );
plt.xlabel("V_d (V)");
plt.ylabel("I_d (mA)");

dataset = np.empty((N,3));
dataset[:,0] = v_d;
dataset[:,1] = i_d;
dataset[:,2] = np.log(i_d);

# ----------------------------------
#          Part B
# ----------------------------------
N = 50;
idx = np.arange(N);
np.random.shuffle(idx);

train_idx = idx[:int(N*0.8)];
val_idx   = idx[int(N*0.8):];

x_train, y_train = dataset[train_idx, 0], dataset[train_idx, 2];
x_val,   y_val   = dataset[  val_idx, 0], dataset[  val_idx, 2];
x_train = x_train.reshape(-1,1);
y_train = y_train.reshape(-1,1);
x_val = x_val.reshape(-1,1);
y_val = y_val.reshape(-1,1);

ss_x = StandardScaler();
ss_x.fit( x_train );
x_train = ss_x.transform(x_train)
x_val   = ss_x.transform(x_val  )


#plt.figure();
#plt.scatter(v_d, i_d );
#plt.scatter(x_train, 1000*i_d[val_idx].reshape(-1,1));



# ----------------------------------
#          Part C
# ----------------------------------

lr = 0.01;
n_epochs= 500;
loss = np.empty((500,));
b = np.random.randn(1);
w = np.random.randn(1);
for epoch in range(n_epochs):
    y_hat = b + w * x_train;
    err = (y_hat - y_train);
    loss[epoch] = (err**2).mean();
    
    b_grad = 2*err.mean();
    w_grad = 2*(x_train*err).mean();
    
    b = b - lr*b_grad;
    w = w - lr*w_grad;

print(f"[Epoch {n_epochs}] Loss={loss[n_epochs-1]}")

plt.figure();
plt.plot(range(n_epochs), loss );
plt.ylabel("Loss (MSE)");
plt.xlabel("Epoch");
#plt.scatter(x_train, 1000*i_d[val_idx].reshape(-1,1));
# ----------------------------------
#          Part D
# ----------------------------------

y_pred = b + w * x_val;
err = (y_pred - y_val);
loss = (err**2).mean();
print(f"Fianl b={b[0]:10.6f} w={w[0]:10.6f} loss={loss:.7f}")

# ----------------------------------
#          Part E
# ----------------------------------
# Unnormalize b and w
mu = ss_x.mean_[0];
sigma = np.sqrt(ss_x.var_[0]);
w_plt = w / sigma;
b_plt = b - (w * mu / sigma);

x_lr = np.linspace( min(ss_x.inverse_transform(x_train)), max(ss_x.inverse_transform(x_train)), 10 );
y_lr = b_plt + (x_lr*w_plt);
plt.figure();
# plt.scatter(ss_x.inverse_transform(x_train), y_train, label="Training Data" );
plt.scatter(ss_x.inverse_transform(x_train), 1000*np.e**y_train, color="blue", label="Training Data" );
plt.scatter(ss_x.inverse_transform(x_val), 1000*np.e**y_val, color="red", label="Validation Data" );
plt.yscale("log");
plt.plot( x_lr, 1000*np.e**y_lr, '--', color='black', label='Linear Regression', linewidth=1 );
plt.legend();
plt.xlabel("V_d (V)");
plt.ylabel("I_d (mA)");


print(f"Manual LR:\n  Normalized:   b: {b[0]:10.6f} w:{w[0]:10.6f}\n  Unnormalized: b: {b_plt[0]:10.6f} w: {w_plt[0]:10.6f}\n")

# ----------------------------------
#          Part F
# ----------------------------------

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

class LR_Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1140)
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.w = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    def forward(self, x):
        return self.b + self.w * x
    

x_train_tensor = torch.as_tensor(x_train).float()
y_train_tensor = torch.as_tensor(y_train).float()

model = LR_Model()
optimizer=optim.SGD(model.parameters(), lr=lr)
loss_fn=nn.MSELoss(reduction='mean')

for epoch in range(n_epochs):
    model.train()
    yhat = model(x_train_tensor)
    loss = loss_fn(yhat, y_train_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"[Epoch {n_epochs}] Loss={loss}")
P=list(model.parameters())
bt=float(P[0].detach())
wt=float(P[1].detach())
yt_pred = model(torch.as_tensor(x_val).float());
losst = loss_fn(yt_pred, torch.as_tensor(y_val).float())
print(f"Fianl b={bt:10.6f} w={wt:10.6f} loss={losst:.7f}")

wt_plt = wt / sigma;
bt_plt = bt - (w * mu / sigma);
print(f"Torch LR:\n  Normalized:   b: {bt:10.6f} w:{wt:10.6f}\n  Unnormalized: b: {bt_plt[0]:10.6f} w: {wt_plt:10.6f}")