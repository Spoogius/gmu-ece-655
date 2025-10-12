import numpy as np
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
         # loss = loss_fn(yhat, y.reshape(y.shape[0],1))
         loss = loss_fn(yhat, y)
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
         # loss = loss_fn(yhat, y.reshape(y.shape[0],1))
         loss = loss_fn(yhat, y)
         # Do not calculate the gradients. Forward pass is all we need
         # Returns the loss
         return loss.item()
     # Returns the function that will be called inside the train loop
     return perform_val_step_fn
 
def mini_batch(device, data_loader, step_fn):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
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