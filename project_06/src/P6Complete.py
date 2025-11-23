import random
import time
import numpy as np
from PIL import Image
from copy import deepcopy
import subprocess
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.style.use('fivethirtyeight')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage,\
Resize, CenterCrop, RandomResizedCrop
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torchvision.models as models


def top_1(label, y_pred):
    preds = np.argmax(y_pred, axis=1);
    return (preds == label).astype(int);


def top_5(label, y_pred):

    top5 = np.argsort(y_pred, axis=1)[:, -5:];
    return np.array([1 if label[i] in top5[i] else 0 for i in range(len(label))]);


def test_model( model, train_loader, test_loader, optimizer=None,  device=None, epochs=20 ):
    total_time_start = time.time();
    loss_fn = nn.CrossEntropyLoss()
    if( optimizer == None ):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loss = np.zeros((epochs,));
    test_loss  = np.zeros((epochs,));
    train_acc1 = np.zeros((epochs,));
    test_acc1  = np.zeros((epochs,));
    train_acc5 = np.zeros((epochs,));
    test_acc5  = np.zeros((epochs,));
    train_time = np.zeros((epochs,));
    
    N_training = 50000;
    N_testing  = 10000;
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct1 = 0
        total_correct5 = 0
        model.train();
        start_time = time.time();
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            y_pred = model(batch_x.to(device))
            loss = loss_fn(y_pred, batch_y.to(device))
            loss.backward()
            optimizer.step()
            
            total_correct1 += np.sum(top_1(batch_y.cpu().numpy(), y_pred.detach().cpu().numpy()));
            total_correct5 += np.sum(top_5(batch_y.cpu().numpy(), y_pred.detach().cpu().numpy()));
            total_loss += loss.item()* batch_x.size(0)
            
        stop_time = time.time();
        train_time[epoch] = stop_time - start_time;
        train_loss[epoch] = total_loss/N_training;
        train_acc1[epoch] = total_correct1/N_training;
        train_acc5[epoch] = total_correct5/N_training;
        
        model.eval()
        total_loss = 0.0
        total_correct1 = 0
        total_correct5 = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                y_pred = model(batch_x.to(device))
                loss = loss_fn(y_pred, batch_y.to(device))
                test_loss += loss.item()* batch_x.size(0);
                total_correct1 += np.sum(top_1(batch_y.cpu().numpy(), y_pred.detach().cpu().numpy()));
                total_correct5 += np.sum(top_5(batch_y.cpu().numpy(), y_pred.detach().cpu().numpy()));
            test_loss[epoch] = total_loss/N_testing;
            test_acc1[epoch] = total_correct1/N_testing;
            test_acc5[epoch] = total_correct5/N_testing;
        total_time_stop = time.time();
        print(f"[{total_time_stop-total_time_start:5.2f}] {epoch:2d}/{epochs}: Loss: {train_loss[epoch]:.4f} T1 Acc (Tr): {train_acc1[epoch]:.4f} T5 Acc (Tr): {train_acc5[epoch]:.4f} T1 Acc (Te): {test_acc1[epoch]:.4f} T5 Acc (Te) {test_acc5[epoch]:.4f}")
        
    
    # Return parameters
    rtn = {};
    rtn["train_loss"] = train_loss;
    rtn["test_loss"]  = test_loss;
    rtn["train_acc1"]  = train_acc1;
    rtn["test_acc1"]   = test_acc1;
    rtn["train_acc5"]  = train_acc5;
    rtn["test_acc5"]   = test_acc5;
    
    # Min Val Loss
    rtn["minCELoss"] = np.min(test_loss);
    
    # Num Of parameters
    rtn["P"] = sum(p.numel() for p in model.parameters());
    # Evalutaion time
    rtn["train_time"] = train_time;
    start_time = time.time();
    for ii in range(10):
       model.eval()
       with torch.no_grad():
           for batch_x, batch_y in test_loader:
               y_pred = model(batch_x.to(device))
    stop_time = time.time();
    rtn["eval_time"] = (stop_time - start_time)/10;

    return rtn;


# %% 
# Load EfficientNet-B3 with pre-trained ImageNet weights
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

# Remove last layer 
model.classifier = nn.Identity()

for par in model.parameters():
    par.requires_grad=False

# Save local Copy for later steps
model_base = deepcopy(model);


composer_train = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

composer_test = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the training and testing datasets
train_dataset = datasets.CIFAR100(root='./dataset', train=True,  download=True, transform=composer_train)
test_dataset  = datasets.CIFAR100(root='./dataset', train=False, download=True, transform=composer_test)
train_loader = DataLoader( train_dataset, batch_size=32, shuffle=True  );
test_loader  = DataLoader(  test_dataset, batch_size=32, shuffle=False );

# %% Part A
device = "cuda";

# Add new output layer to model
model.classifier = nn.Sequential( nn.Dropout(0.3), nn.Linear(1536, 100));
model.to(device);
# %%
# N_EPOCHS = 3;
# results_A = test_model( model, train_loader, test_loader, device=device, epochs=N_EPOCHS );

# %% Part B
def PreprocessedDataset(model, loader, device=None):
    model.to(device)
    for i, (x, y) in enumerate(loader):
        model.eval()
        output = model(x.to(device))
        if i == 0:
            features = output.detach().cpu()
            labels = y.cpu()
        else:
            features = torch.cat(
                [features, output.detach().cpu()])
            labels = torch.cat([labels, y.cpu()])
    dataset = TensorDataset(features, labels)
    return dataset

# Preprocess the dataset
model = deepcopy(model_base); #Identify final layer

pp_dataset_train = PreprocessedDataset(model, train_loader, device );
pp_dataset_test  = PreprocessedDataset(model,  test_loader, device );

torch.save(pp_dataset_train.tensors, 'dataset/CIFAR100preproc_train.pth');
torch.save(pp_dataset_test.tensors,  'dataset/CIFAR100preproc_test.pth' );

# %% 
pp_dataset_train_x, pp_dataset_train_y = torch.load('dataset/CIFAR100preproc_train.pth');
pp_dataset_test_x,  pp_dataset_test_y  = torch.load('dataset/CIFAR100preproc_test.pth' );
pp_dataset_train = TensorDataset( pp_dataset_train_x, pp_dataset_train_y );
pp_dataset_test  = TensorDataset( pp_dataset_test_x, pp_dataset_test_y );
pp_loader_train = DataLoader( pp_dataset_train, batch_size=32, shuffle=True  );
pp_loader_test  = DataLoader(  pp_dataset_test, batch_size=32, shuffle=False );
# %% Train final model classifier layers
model_classifier = nn.Sequential(nn.Dropout(.3), nn.Linear(1536, 100, bias=True));
model_classifier.to(device);
N_EPOCHS = 30;
results_B = test_model( model_classifier, pp_loader_train, pp_loader_test, optimizer=optim.Adam(model_classifier[1].parameters(), lr=0.001), device=device, epochs=N_EPOCHS );

# %% Get coarse labels
import os
import pickle
with open(os.path.join('./dataset', 'cifar-100-python', 'train'), 'rb') as f:
    meta_train = pickle.load(f, encoding='latin1')
with open(os.path.join('./dataset', 'cifar-100-python', 'test' ), 'rb') as f:
    meta_test  = pickle.load(f, encoding='latin1')
    
train_dataset = datasets.CIFAR100(root='./dataset', train=True,  download=True, transform=composer_train)
test_dataset  = datasets.CIFAR100(root='./dataset', train=False, download=True, transform=composer_test)
train_dataset.targets = meta_train["coarse_labels"]
test_dataset.targets  = meta_test[ "coarse_labels"]
train_loader = DataLoader( train_dataset, batch_size=32, shuffle=True  );
test_loader  = DataLoader(  test_dataset, batch_size=32, shuffle=False );
    
# Preprocess the dataset
model = deepcopy(model_base); #Identify final layer

pp_dataset_coarse_train = PreprocessedDataset(model, train_loader, device );
pp_dataset_coarse_test  = PreprocessedDataset(model,  test_loader, device );

torch.save(pp_dataset_train.tensors, 'dataset/CIFAR100preproc_coarse_train.pth');
torch.save(pp_dataset_test.tensors,  'dataset/CIFAR100preproc_coarse_test.pth' );

# %%
pp_dataset_coarse_train_x, pp_dataset_coarse_train_y = torch.load('dataset/CIFAR100preproc_coarse_train.pth');
pp_dataset_coarse_test_x,  pp_dataset_coarse_test_y  = torch.load('dataset/CIFAR100preproc_coarse_test.pth' );
pp_dataset_coarse_train = TensorDataset( pp_dataset_coarse_train_x, pp_dataset_coarse_train_y );
pp_dataset_coarse_test  = TensorDataset( pp_dataset_coarse_test_x,  pp_dataset_coarse_test_y  );
pp_loader_coarse_train = DataLoader( pp_dataset_coarse_train, batch_size=32, shuffle=True  );
pp_loader_coarse_test  = DataLoader(  pp_dataset_coarse_test, batch_size=32, shuffle=False );
    

# %% Train final model classifier layers on coarse labels
model_classifier = nn.Sequential(nn.Dropout(.3), nn.Linear(1536, 100, bias=True));
model_classifier.to(device);
N_EPOCHS = 30;
results_B = test_model( model_classifier, pp_loader_coarse_train, pp_loader_coarse_test, optimizer=optim.Adam(model_classifier[1].parameters(), lr=0.001), device=device, epochs=N_EPOCHS );
