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
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    if( optimizer == None ):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loss = np.zeros((epochs,));
    test_loss  = np.zeros((epochs,));
    train_acc1 = np.zeros((epochs,));
    test_acc1  = np.zeros((epochs,));
    train_acc5 = np.zeros((epochs,));
    test_acc5  = np.zeros((epochs,));
    train_time = np.zeros((epochs,));
    test_time  = np.zeros((epochs,));
    
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
        start_time = time.time();
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                y_pred = model(batch_x.to(device))
                loss = loss_fn(y_pred, batch_y.to(device))
                total_loss += loss.item()* batch_x.size(0);
                total_correct1 += np.sum(top_1(batch_y.cpu().numpy(), y_pred.detach().cpu().numpy()));
                total_correct5 += np.sum(top_5(batch_y.cpu().numpy(), y_pred.detach().cpu().numpy()));
            test_loss[epoch] = total_loss/N_testing;
            test_acc1[epoch] = total_correct1/N_testing;
            test_acc5[epoch] = total_correct5/N_testing;
        stop_time = time.time();
        test_time[epoch] = stop_time - start_time;
        total_time_stop = time.time();
        print(f"[{total_time_stop-total_time_start:5.2f}] {epoch:2d}/{epochs}: Loss: {train_loss[epoch]:.4f} T1 Acc (Tr): {train_acc1[epoch]:.4f} T5 Acc (Tr): {train_acc5[epoch]:.4f} T1 Acc (Te): {test_acc1[epoch]:.4f} T5 Acc (Te) {test_acc5[epoch]:.4f}")
        
    
    # Return parameters
    rtn = {};
    rtn["train_loss"] = train_loss;
    rtn["test_loss" ] =  test_loss;
    rtn["train_acc1"] = train_acc1;
    rtn["test_acc1" ] =  test_acc1;
    rtn["train_acc5"] = train_acc5;
    rtn["test_acc5" ] =  test_acc5;
    
    # Min Val Loss
    rtn["minCELoss"] = np.min(test_loss);
    
    # Num Of parameters
    rtn["P"] = sum(p.numel() for p in model.parameters());
    # Evalutaion time
    rtn["train_time"] = train_time;
    rtn["eval_time"] = test_time;

    return rtn;

def plot_accuracy(results):
    N_EPOCHS = int(len(results['test_acc1']))
    
    plt_epoch = np.linspace(1,N_EPOCHS, N_EPOCHS);
    plt.plot(plt_epoch, results['train_acc1'], color="blue", label='Train (Top 1)');
    plt.plot(plt_epoch, results[ 'test_acc1'], color="red", label='Test (Top 1)');
    plt.plot(plt_epoch, results[ 'test_acc5'], color="green", label='Test (Top 5)');
    plt.legend();
    plt.xlabel("Epochs");
    plt.ylabel("Accuracy");
    plt.show();
    # fig, axes = plt.subplots(2, 1)
    # axes[0].plot( plt_epoch, results['train_acc5'], color="blue", label='Training');
    # axes[0].plot( plt_epoch, results['test_acc5' ], color="red", label='Validation');
    # axes[0].set_ylabel('T5 Accuracy')
    # axes[0].set_xticks([]);
    # axes[0].legend()
    
    # axes[1].plot(plt_epoch, results["train_acc1"], color="blue", label="Training");
    # axes[1].plot(plt_epoch, results["test_acc1" ], color='red', label='Validation')
    # axes[1].set_ylabel('T1 Accuracy')
    # axes[1].set_xlabel('Epochs');

    # axes[0].set_title("Accuracy");
    # plt.show();

device = "cuda";
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
    transforms.Resize((300)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

composer_test = transforms.Compose([
    transforms.Resize((300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])




def plot_tensor( images, title="" ):
    if(images.shape[0] == 10 ):
        fig, axes = plt.subplots( 3, 4 );
        for ii, ax in enumerate( axes.flat ):
            do_plt = True;
            if( ii < 8 ):
                img_idx = ii;
            elif( ii == 8 ):
                do_plt = False;
            elif( ii < 11 ):
                img_idx = ii-1;
            else:
                do_plt = False;
                
            if do_plt:
                img = images[img_idx].numpy();
                img = np.moveaxis(img, [0,1,2],[2,0,1])
                ax.imshow( img );
            
            ax.axis=('off');
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([]);
            ax.set_yticks([]);
            ax.label_outer();
        
        
    plt.suptitle(title);
    plt.show()

# train_dataset = datasets.CIFAR100(root='./dataset', train=True,  download=True, transform=composer_tensor)
# test_dataset  = datasets.CIFAR100(root='./dataset', train=False, download=True, transform=composer_tensor)
# train_loader = DataLoader( train_dataset, batch_size=10, shuffle=False  );
# test_loader  = DataLoader(  test_dataset, batch_size=10, shuffle=False );
# for batch_x, batch_y in train_loader:
#     plot_tensor(batch_x, "Train")
#     break;
# for batch_x, batch_y in test_loader:
#    plot_tensor(batch_x, "Test")
#    break;

# Load the training and testing datasets
train_dataset = datasets.CIFAR100(root='./dataset', train=True,  download=True, transform=composer_train)
test_dataset  = datasets.CIFAR100(root='./dataset', train=False, download=True, transform=composer_test)
train_loader = DataLoader( train_dataset, batch_size=32, shuffle=True  );
test_loader  = DataLoader(  test_dataset, batch_size=32, shuffle=False );
# %% Part A

# Add new output layer to model
model.classifier = nn.Sequential(nn.Dropout(.5), nn.Linear(1536, 100, bias=True));
model.to(device);

N_EPOCHS = 3;
results_A = test_model( model, train_loader, test_loader, device=device, epochs=N_EPOCHS );
plot_accuracy(results_A);