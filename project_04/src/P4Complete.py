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


N = 200
upsampled_Q = upsample_dataset( images[0:10:], N/2 )
upsampled_M = upsample_dataset( images[10::],  N/2 )
# augmented_Q = data_augmentation(images[0:10:], N/2);
# plot_images(upsampled_Q, ("Dataset (Q)"), (10,10))
# augmented_M = data_augmentation(images[10::], N/2);
# #plot_images(augmented_M, ("Augmentation Dataset (M)"), (10,10))

images = np.concat((upsampled_Q, upsampled_M), axis=0);
labels = np.empty(N, dtype=float); 
labels[0:int(N/2):] = 0; # 0 = Q
labels[int(N/2)::]  = 1; # 1 = M

# augment_composer = Compose([RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5), RandomAffine(degrees=20, translate=(0.1,0.1))])
augment_composer = Compose([RandomAffine(degrees=(-15,15), translate=(0.1,0.1))])
to_pil = ToPILImage();
augmented_images = np.array( [ augment_composer(to_pil(images[ii][0])) for ii in range(images.shape[0])] )
augmented_images = augmented_images.reshape((N,1,20,20))/255
plot_images(augmented_images[0:int(N/2):], ("Augmented Dataset (Q)"), (10,10))
plot_images(augmented_images[int(N/2)::],  ("Augmented Dataset (M)"), (10,10))

# %%
device = "cuda";
x_tensor = torch.as_tensor(augmented_images).float().to(device)
y_tensor = torch.as_tensor(labels).float().to(device)

train_idx, val_idx = index_splitter(len(x_tensor), [80,20])
# Uses indices to perform the split
x_train_tensor = x_tensor[train_idx]
y_train_tensor = y_tensor[train_idx]
x_val_tensor = x_tensor[val_idx]
y_val_tensor = y_tensor[val_idx]

unnorm_composer = Compose([Normalize(mean=(-1,), std=(2,))])
norm_composer = Compose([Normalize(mean=(.5,), std=(.5,))])

train_dataset = TransformedTensorDataset(
    x_train_tensor, y_train_tensor, transform=norm_composer)
val_dataset = TransformedTensorDataset(
    x_val_tensor, y_val_tensor, transform=norm_composer)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=val_dataset.x.shape[0], shuffle=False)

model = nn.Sequential()
model.add_module('flatten', nn.Flatten())
model.add_module('linear0', nn.Linear(400, 16, bias=True))
model.add_module('ReLU0', nn.ReLU())
model.add_module('linear1', nn.Linear(16, 4, bias=True))
model.add_module('ReLU1', nn.ReLU())
model.add_module('linear2', nn.Linear(4, 1, bias=True))
model.add_module('sigmoid', nn.Sigmoid())
model.to(device);

n_epochs = 50
lr = 0.001

optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn=nn.BCELoss()
losses = np.empty(n_epochs);
val_losses = np.empty(n_epochs);
        
for epoch in range(n_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch.view(-1, 1))
        loss.backward()
        optimizer.step()
        losses[epoch] = loss.cpu().detach().numpy();
        
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch.view(-1, 1))
            val_losses[epoch] = loss.cpu().detach().numpy();
            
plot_loss_curve(losses,val_losses,title='Loss Curve (Loss=BCE, Batch Size=16, Optimizer=Adam)');

# y_pred = y_pred.cpu();
# y_true = y_batch.cpu();


# %%
model.eval();
with torch.no_grad():
    y_pred = y_pred = model(norm_composer(x_val_tensor));
# with torch.no_grad():
#     for x_batch, y_batch in val_loader:
#         y_pred = model(x_batch)
        
y_pred = y_pred.cpu().detach().numpy().squeeze();
y_true = y_val_tensor.cpu().numpy();

plot_images(x_val_tensor.cpu().numpy(), ("ROC"), dim=(7,6), pred=y_pred, truth=y_true)

threshold = 0.5
y_pred_bin = (y_pred >= threshold).astype(int)
TP = np.sum((y_pred_bin == 1) & (y_true == 1))
FP = np.sum((y_pred_bin == 1) & (y_true == 0))
TN = np.sum((y_pred_bin == 0) & (y_true == 0))
FN = np.sum((y_pred_bin == 0) & (y_true == 1))

TPR = TP / (TP + FN) 
FPR = FP / (FP + TN)

print(f"TPR: {TPR:.3f}, FPR: {FPR:.3f}")

N_ROC = 100;
tpr_plt = np.empty(N_ROC);
fpr_plt = np.empty(N_ROC);
for ii, threshold in enumerate(np.linspace(0.00, 1, N_ROC )):
    y_pred_bin = (y_pred >= threshold).astype(int)
    TP = np.sum((y_pred_bin == 1) & (y_true == 1))
    FP = np.sum((y_pred_bin == 1) & (y_true == 0))
    TN = np.sum((y_pred_bin == 0) & (y_true == 0))
    FN = np.sum((y_pred_bin == 0) & (y_true == 1))
    
    tpr_plt[ii] = TP / (TP + FN)
    fpr_plt[ii] = FP / (FP + TN)
    
plt.plot(fpr_plt, tpr_plt )
plt.title("ROC Curve");
plt.xlabel('False Positive Rate' );
plt.ylabel("True Positive Rate" );
plt.show()
# %%

images = np.empty((10,1,20,20));
# Load in images (This set was made usng pinta - requires some extra processing to match the previous dataset)
for ii in range(5):
    img = np.array(Image.open(f"../dataset/test/Q/Q{ii}.png"), dtype=float)[:,:,0];
    img[np.where(img!=255)] = 1
    img[np.where(img==255)] = 0
    images[ii][0] = img

for ii in range(5):
    img = np.array(Image.open(f"../dataset/test/M/M{ii}.png"), dtype=float)[:,:,0];
    img[np.where(img!=255)] = 1
    img[np.where(img==255)] = 0
    images[ii+5][0] = img

test_y = np.array((10,1))
test_y[0:5:] = 0;
test_y[5::] = 1;

test_x_tensor = torch.as_tensor(images).float().to(device)
test_y_tensor = torch.as_tensor(test_y).float().to(device)
test_dataset = TransformedTensorDataset(
    test_x_tensor, test_y_tensor, transform=norm_composer)
test_loader = DataLoader(dataset=test_dataset, batch_size=16)

with torch.no_grad():
    y_pred = model(norm_composer(test_x_tensor))
    
y_true = np.zeros(10)
y_true[5::] =1 
plot_images(images, ("Heldout Test Set"), (2,5), pred=y_pred.cpu().detach().numpy().squeeze(), truth=y_true)





