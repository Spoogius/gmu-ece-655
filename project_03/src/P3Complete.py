import numpy as np
from sklearn.linear_model import LinearRegression

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from sklearn.preprocessing import StandardScaler;
# from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

np.random.seed(1140)
# ----------------------------------
#          Model Config
# ----------------------------------

def make_train_step_fn(model, loss_fn, optimizer):
     # Builds function that performs a step in the train loop
     def perform_train_step_fn(x, y):
         # Sets model to TRAIN mode
         model.train()
         # Step 1 - Computes model's predictions - forward pass
         yhat = model(x)
         # Step 2 - Computes the loss
         loss = loss_fn(yhat, y.reshape(y.shape[0],1))
         # Step 3 - Computes gradients for "b" and "w" parameters
         loss.backward()
         # Step 4 - Updates parameters using gradients and
         # the learning rate
         optimizer.step()
         optimizer.zero_grad()
         # Returns the loss
         return loss.item()
     # Returns the function that will be called inside the train loop
     return perform_train_step_fn


def make_val_step_fn(model, loss_fn):
     # Builds function that performs a step in the validation loop
     def perform_val_step_fn(x, y):
         # Sets model to EVAL mode
         model.eval()
         # Step 1 - Computes model's predictions - forward pass
         yhat = model(x)
         # Step 2 - Computes the loss
         loss = loss_fn(yhat, y.reshape(y.shape[0],1))
         # Do not calculate the gradients. Forward pass is all we need
         # Returns the loss
         return loss.item()
     # Returns the function that will be called inside the train loop
     return perform_val_step_fn
 
def mini_batch(device, data_loader, step_fn):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        mini_batch_loss = step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)
    loss = np.mean(mini_batch_losses)
    return loss

def plot_loss_curve(losses, val_losses, title='Training and Validation Loss per Epoch'):
    epochs = np.arange(1, len(losses)+1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, color='blue', label='Training Loss')
    plt.plot(epochs, val_losses, color='red', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()  

# ----------------------------------
#          Part A
# ----------------------------------
dataset = np.loadtxt("../real-estate-price-prediction/Real estate.csv", delimiter=',', skiprows=1);
dataset = np.delete(dataset,0,axis=1); # Drop index column
x = dataset[:, 0:5:];
y = dataset[:,6];

N = dataset.shape[0];
# Shuffles the indices
idx = np.arange(N)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:int(N*.8)]
# Uses the remaining indices for validation
val_idx = idx[int(N*.8):]

# Generates train and validation sets in numpy
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]


num_features = x_train.shape[1]
# ---- Training set figure ----
fig1, axs1 = plt.subplots(2, 3, figsize=(20, 12))
fig1.suptitle("Training Data: Features vs Target (y)", fontsize=14)

for i in range(2):
    for j in range(3):
        axs1[i][j].scatter(x_train[:, 2*i+j], y_train, color='b')
        axs1[i][j].set_xlabel(f"Feature {3*i+j}")
        axs1[i][j].set_ylabel("y")
        axs1[i][j].set_title(f"x[{3*i + j}] vs y")



# ---- Validation set figure ----
fig2, axs2 = plt.subplots(2, 3, figsize=(20, 12))
fig2.suptitle("Validation Data: Features vs Target (y)", fontsize=14)

for i in range(2):
    for j in range(3):
        axs2[i][j].scatter(x_val[:, 2*i+j], y_val, color='r')
        axs2[i][j].set_xlabel(f"Feature {3*i+j}")
        axs2[i][j].set_ylabel("y")
        axs2[i][j].set_title(f"x[{3*i + j}] vs y")



# ----------------------------------
#          Part B
# ----------------------------------

ss_x = StandardScaler();
ss_x.fit( x );
x_norm = ss_x.transform(x)

x_tensor = torch.as_tensor(x_norm).float()
y_tensor = torch.as_tensor(y).float()

# Builds dataset containing ALL data points
dataset = TensorDataset(x_tensor, y_tensor)

# Performs the split
ratio = .8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train


train_data, val_data = random_split(dataset, [n_train, n_val])
# Builds a loader of each set
train_loader = DataLoader(dataset=train_data, batch_size=n_train, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=n_train)


device = 'cpu'
n_epochs = 100
lr = 0.05
model=nn.Sequential(nn.Linear(num_features,1)).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_fn=nn.MSELoss(reduction='mean')
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
val_step_fn   = make_val_step_fn(model, loss_fn)

losses = np.empty(n_epochs);
val_losses = np.empty(n_epochs);
for epoch in range(n_epochs):
    loss = mini_batch(device, train_loader, train_step_fn)
    losses[epoch] = loss;
    # VALIDATION - no gradients
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step_fn)
        val_losses[epoch] = val_loss;

plot_loss_curve(losses,val_losses,title='Loss Curve (Loss=MSE, Batch Size=Full, Optimizer=SGD)');

# ----------------------------------
#          Part C
# ----------------------------------

n_epochs = 100
lr = 0.05
model=nn.Sequential(nn.Linear(num_features,1)).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_fn=nn.L1Loss(reduction='mean')
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
val_step_fn   = make_val_step_fn(model, loss_fn)

losses = np.empty(n_epochs);
val_losses = np.empty(n_epochs);
for epoch in range(n_epochs):
    loss = mini_batch(device, train_loader, train_step_fn)
    losses[epoch] = loss;
    # VALIDATION - no gradients
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step_fn)
        val_losses[epoch] = val_loss;

plot_loss_curve(losses,val_losses,title='Loss Curve (Loss=L1, Batch Size=Full, Optimizer=SGD)');

# ----------------------------------
#          Part D
# ----------------------------------

n_epochs = 100
lr = 0.05
model=nn.Sequential(nn.Linear(num_features,1)).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn=nn.MSELoss(reduction='mean')
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
val_step_fn   = make_val_step_fn(model, loss_fn)

losses = np.empty(n_epochs);
val_losses = np.empty(n_epochs);
for epoch in range(n_epochs):
    loss = mini_batch(device, train_loader, train_step_fn)
    losses[epoch] = loss;
    # VALIDATION - no gradients
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step_fn)
        val_losses[epoch] = val_loss;

plot_loss_curve(losses,val_losses,title='Loss Curve (Loss=MSE, Batch Size=Full, Optimizer=Adam)');


# %%
# ----------------------------------
#          Part E
# ----------------------------------

n_epochs = 100
lr = 0.05
plt_bs = np.array([4, 8, 16, 32, 64])
plt_final_loss = np.empty(len(plt_bs));
plt_final_val_loss = np.empty(len(plt_bs));

for batch_idx, batch_size in enumerate(plt_bs):
    train_loader = DataLoader(dataset=train_data, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=int(batch_size))
    
    model=nn.Sequential(nn.Linear(num_features,1)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn=nn.MSELoss(reduction='mean')
    train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
    val_step_fn   = make_val_step_fn(model, loss_fn)
    
    losses = np.empty(n_epochs);
    val_losses = np.empty(n_epochs);
    for epoch in range(n_epochs):
        loss = mini_batch(device, train_loader, train_step_fn)
        losses[epoch] = loss;
        # VALIDATION - no gradients
        with torch.no_grad():
            val_loss = mini_batch(device, val_loader, val_step_fn)
            val_losses[epoch] = val_loss;
            
    plt_final_loss[batch_idx] = loss;
    plt_final_loss[batch_idx] = val_loss;

# Create plot
plt.figure(figsize=(8, 5))
width = 0.35
x = np.arange(len(plt_bs))
plt.bar(x - width/2, plt_final_loss, width, color='blue', label='Loss')
plt.bar(x + width/2, plt_final_val_loss, width, color='red', label='Val Loss')

plt.xticks(x, plt_bs)
plt.xlabel('Batch Size')
plt.ylabel('Loss')
plt.title('Final Loss (Loss=MSE, Optimizer=SGD)')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# ----------------------------------
#          Part F
# ----------------------------------
train_loader = DataLoader(dataset=train_data, batch_size=n_train, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=n_train)

epochs_sweep = np.linspace(20,1000,50)
optim_sweep  = [optim.SGD, optim.Adam]
results = np.empty((len(optim_sweep), len(epochs_sweep), 2))

import time

lr = 0.05
for optim_idx, opt in enumerate(optim_sweep):
    for epoch_idx, n_epochs in enumerate(epochs_sweep):
        
        start_ts = time.time()
        
        model=nn.Sequential(nn.Linear(num_features,1)).to(device)
        optimizer = opt(model.parameters(), lr=lr)
        loss_fn=nn.MSELoss(reduction='mean')
        train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
        val_step_fn   = make_val_step_fn(model, loss_fn)

        for epoch in range(int(n_epochs)):
            loss = mini_batch(device, train_loader, train_step_fn)
        
        stop_ts = time.time()
        
        results[optim_idx][epoch_idx][0] = loss;
        results[optim_idx][epoch_idx][1] = stop_ts-start_ts;
        print(f"Finished [{opt.__name__:4}][{int(n_epochs):4}]: Loss={loss:6.3f} Time={results[optim_idx][epoch_idx][1]:9.5f}")
        
# %%
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

ax1.plot(epochs_sweep, results[0][:,0], 'b', label='SGD')
ax1.plot(epochs_sweep, results[1][:,0], 'b--', label='Adam')

ax2.plot(epochs_sweep, results[0][:,1], 'r', label='SGD')
ax2.plot(epochs_sweep, results[1][:,1], 'r--', label='Adam')

plt.xlabel('Epoch')
ax1.set_ylabel('MSE Loss', color='b')
ax2.set_ylabel('Elapsed Time (sec)', color='r')
plt.title('Final Loss and Runtime')

from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color='k', lw=2, linestyle='-'),
    Line2D([0], [0], color='k', lw=2, linestyle='--')
]

# Add a single shared legend for both axes
ax1.legend(custom_lines, ['SGD', 'Adam'], loc='upper center')

plt.show()     


# %%
# ----------------------------------
#          Part F
# ----------------------------------