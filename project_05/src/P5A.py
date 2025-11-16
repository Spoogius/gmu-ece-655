import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, \
ToPILImage, RandomHorizontalFlip, \
RandomVerticalFlip, Resize, RandomRotation, RandomAffine
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import itertools

random.seed(1140);

LBL_Q = 0;
LBL_M = 1;
LBL_X = 2;
LBL_O = 3;

def label_to_string( lbl ):
    if( lbl == LBL_Q ):
        return "Q";
    if( lbl == LBL_M ):
        return "M";
    if( lbl == LBL_X ):
        return "X";
    if( lbl == LBL_O ):
        return "Other";
    

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
                img = images[img_idx][0].numpy();
                ax.imshow( img, cmap='gray');
            
            ax.axis=('off');
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([]);
            ax.set_yticks([]);
            ax.label_outer();
        
    elif(images.shape[0] == 100 ):
        fig, axes = plt.subplots( 10, 10 );
        for ii, ax in enumerate( axes.flat ):
            img = images[ii][0].numpy();
            ax.imshow( img, cmap='gray');
            
            ax.axis=('off');
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([]);
            ax.set_yticks([]);
            ax.label_outer();
    elif(images.shape[0] == 25 ):
        fig, axes = plt.subplots( 5, 5 );
        for ii, ax in enumerate( axes.flat ):
            img = images[ii][0].numpy();
            ax.imshow( img, cmap='gray');
            
            ax.axis=('off');
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([]);
            ax.set_yticks([]);
            ax.label_outer();
        
    plt.suptitle(title);
    plt.show()

  
def plot_test_results( model, x_test, y_test, title=None ):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        y_pred = np.argmax(y_pred.detach().cpu().numpy(),axis=1);
        
    if( y_pred.size == 80 ):
        fig, axes = plt.subplots( 10, 8, figsize=(30,18) );
    elif( y_pred.size == 25 ):
        fig, axes = plt.subplots( 5, 5, figsize=(25,25) );
    for ii, ax in enumerate( axes.flat ):
        img = x_test[ii][0].cpu().numpy();
        ax.imshow( img, cmap='gray');
        
        ax.axis=('off');
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.label_outer();
        if( y_pred[ii] == y_test[ii].cpu() ):
            ax.set_title(f"Predicted: ({label_to_string(y_pred[ii])}) Truth: ({label_to_string(y_test[ii])})", color="g");
        else:
            ax.set_title(f"Predicted: ({label_to_string(y_pred[ii])}) Truth: ({label_to_string(y_test[ii].cpu())})", color="r");
    if title:
        plt.suptitle(title,size=75);
    
def plot_loss_curve(result, title=None):
    fig, axes = plt.subplots( 2, 1 );
    epochs = np.arange(len(result["train_loss"]))+1;
    axes[0].plot(epochs, result["train_loss"], color="blue", label="Training");
    axes[0].plot(epochs, result["test_loss"], color='red', label='Validation')
    axes[0].set_ylabel('CE Loss')
    axes[0].set_title("Loss");
    axes[0].set_xticks([]);
    axes[0].legend()
    axes[1].plot(epochs, result["train_acc"], color="blue", label="Training");
    axes[1].plot(epochs, result["test_acc"], color='red', label='Validation')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epochs');
    axes[1].set_title("Accuracy");
    
    if title:
        plt.suptitle(title)
    
    plt.show();

def test_model( model, loader, x_test, y_test, epochs=20 ):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    train_loss = np.zeros((epochs,));
    test_loss  = np.zeros((epochs,));
    train_acc  = np.zeros((epochs,));
    test_acc   = np.zeros((epochs,));
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        model.train();
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            total_correct += np.sum(batch_y.cpu().numpy() == np.argmax(y_pred.detach().cpu().numpy(),axis=1));
            total_loss += loss.item()* batch_x.size(0)
        train_loss[epoch] = total_loss/320;
        train_acc[epoch] = total_correct/320;
        
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test)
            loss = loss_fn(y_pred, y_test)
            test_loss[epoch] = loss;
            test_acc[epoch] = np.sum(y_test.cpu().numpy() == np.argmax(y_pred.detach().cpu().numpy(),axis=1))/80;
    
    # Return parameters
    rtn = {};
    rtn["model"] = model;
    rtn["train_loss"] = train_loss;
    rtn["test_loss"]  = test_loss;
    rtn["train_acc"]  = train_acc;
    rtn["test_acc"]   = test_acc;
    
    # Min Val Loss
    rtn["minBC"] = np.min(test_loss);
    # Epochs to reach val accuracy threshold
    for th in [95, 90, 80, 70]:
        above_th = np.where(test_acc >= (th/100))[0];
        if len(above_th):
            rtn[f"E{th}"] = np.min(above_th) + 1;
        else:
            rtn[f"E{th}"] = None;
    # Num Of parameters
    rtn["P"] = sum(p.numel() for p in model.parameters());
    # Evalutaion time
    start_time = time.time();
    for ii in range(100):
        y_pred = model(x_test[0:1:]);
    stop_time = time.time();
    rtn["T"] = (stop_time - start_time)/100;
    # Cost
    rtn["cost"] = rtn["T"] * rtn["P"];
    return rtn;
                
    
# %% Create Upsampled Dataset
Q_images = torch.from_numpy(np.empty((10,1,20,20), dtype=np.float32));
for ii in range(10):
    img = Image.open(f"../dataset/Q/Q{ii}.png");
    Q_images[ii][0] = torch.from_numpy(np.array(img, dtype=np.float32))
    # Q_images[ii][0] = torch.from_numpy(np.abs(np.array(img, dtype=float)-1)*255.0)
    
M_images = torch.from_numpy(np.empty(((10,1,20,20))));
for ii in range(10):
    img = Image.open(f"../dataset/M/M{ii}.png");
    M_images[ii][0] = torch.from_numpy(np.array(img, dtype=np.float32))
    # M_images[ii][0] = torch.from_numpy(np.abs(np.array(img, dtype=float)-1)*255.0)
    
X_images = torch.from_numpy(np.empty((10,1,20,20)));
for ii in range(10):
    img = Image.open(f"../dataset/X/X{ii}.png");
    X_images[ii][0] = torch.from_numpy(np.array(img, dtype=np.float32))
    # X_images[ii][0] = torch.from_numpy(np.abs(np.array(img, dtype=float)-1)*255.0)
    plot_loss_curve
O_images = torch.from_numpy(np.empty((10,1,20,20)));
for ii in range(10):
    img = Image.open(f"../dataset/Other/O{ii}.png");
    O_images[ii][0] = torch.from_numpy(np.array(img, dtype=np.float32))
    # O_images[ii][0] = torch.from_numpy(np.abs(np.array(img, dtype=float)-1)*255.0)
