import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, \
ToPILImage, RandomHorizontalFlip, \
RandomVerticalFlip, Resize, RandomRotation, RandomAffine
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(1140);


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

def plot_images( images, title, dim=0, datatype="np", pred=None, truth=None):

    if( datatype == "PIL" ):
        images_np = np.empty((images.shape[0],1,images.shape[1],images.shape[2]) ) 
        for ii in range(images_np.shape[0]):
            images_np[ii][0] = np.array(images[ii],dtype=float)/255;
        images = images_np;
    if( dim == 0 ):
        dim_row = int(np.ceil(np.sqrt(images.shape[0])));
        dim_col = dim_row;
    else:
        dim_row = int(dim[0]);
        dim_col = int(dim[1]);
        
    fig, axes = plt.subplots(dim_row, dim_col, figsize=(8, 6))
    axes = axes.ravel()

    for ii, ax in enumerate(axes):
        if( ii < images.shape[0] ):
            ax.imshow(images[ii][0], cmap='gray_r', vmin=0, vmax=1)
            if not (pred is None):
                clr = "red"
                if(pred[ii] > 0.5 and truth[ii] == 1) or (pred[ii] <= 0.5 and truth[ii] == 0):
                    clr = "green"
                ax.set_title(f"{pred[ii]:.3f}",color=clr)
        ax.axis('off')
    plt.suptitle(title);
    plt.show()

class TransformedTensorDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
    def __getitem__(self, index):
        x = self.x[index]
        if self.transform:
            x = self.transform(x)
        return x, self.y[index]
    def __len__(self):
        return len(self.x)

def index_splitter(n, splits, seed=13):
    idx = torch.arange(n)
    # Makes the split argument a tensor
    splits_tensor = torch.as_tensor(splits)
    # Finds the correct multiplier, so we don't have
    # to worry about summing up to N (or one)
    multiplier = n / splits_tensor.sum()
    splits_tensor = (multiplier * splits_tensor).long()
    # If there is a difference, throws at the first split
    # so random_split does not complain
    diff = n - splits_tensor.sum()
    splits_tensor[0] += diff
    # Uses PyTorch random_split to split the indices
    torch.manual_seed(seed)
    
    return random_split(idx, splits_tensor)
    

def upsample_dataset( images, N=100 ):
    idx = np.array([ random.randint(0, images.shape[0]-1) for ii in range(int(N)) ], dtype=int);
    rtn_images = np.array([ images[ii] for ii in idx ]);
    return rtn_images


images = np.empty((20,1,20,20));
# Load in images
for ii in range(10):
    img = Image.open(f"../dataset/Q/Q{ii}.png");
    images[ii][0] = np.array(img, dtype=float)

for ii in range(10):
    img = Image.open(f"../dataset/M/M{ii}.png");
    images[ii+10][0] = np.array(img, dtype=float)

#images = np.array([ Image.fromarray(images[ii].squeeze()) for ii in range(images.shape[0]) ]);


plot_images(images[0:10:], ("Preaugmentation Dataset (Q)"), (3,4) )
plot_images(images[10::], ("Preaugmentation Dataset (M)"), (3,4))