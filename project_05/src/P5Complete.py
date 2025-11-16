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
    axes[0].plot(epochs, result["train_loss"], color="blue", label="Training Loss");
    axes[0].plot(epochs, result["test_loss"], color='red', label='Validation Loss')
    axes[0].set_ylabel('CE Loss')
    axes[0].set_xticks([]);
    
    axes[1].plot(epochs, result["train_acc"], color="blue", label="Training Accuracy");
    axes[1].plot(epochs, result["test_acc"], color='red', label='Validation Accuracy')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epochs');
    axes[1].set_title("Accuracy");
    if title:
        plt.suptitle(title)
    plt.show();

def test_model( model, loader, x_test, y_test, epochs=20 ):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
    
# Randomly upsample
images = torch.from_numpy(np.empty((400,1,20,20), dtype=np.float32))
images[  0:100:1] = Q_images[np.random.randint(0, 10, 100)];
images[100:200:1] = M_images[np.random.randint(0, 10, 100)];
images[200:300:1] = X_images[np.random.randint(0, 10, 100)];
images[300:400:1] = O_images[np.random.randint(0, 10, 100)];

# %% Agument Dataset
composer = Compose( [RandomAffine(degrees=(-30,30), translate=(0.15,0.15)) ]);
images_aug = torch.from_numpy(np.empty((400,1,20,20), dtype=np.float32))
for ii in range(images.shape[0]):
    images_aug[ii] = composer(images[ii]);

# %% Remap image values to white = 255, black = 0.0
Q_images = torch.from_numpy(np.abs(Q_images.numpy()-1)*255)
M_images = torch.from_numpy(np.abs(M_images.numpy()-1)*255)
X_images = torch.from_numpy(np.abs(X_images.numpy()-1)*255)
O_images = torch.from_numpy(np.abs(O_images.numpy()-1)*255)
images_aug = torch.from_numpy(np.abs(images_aug.numpy()-1)*255)

# %% Generate Dataset figures
plot_tensor(Q_images, "Preagumented Q");
plot_tensor(M_images, "Preagumented M");
plot_tensor(X_images, "Preagumented X");
plot_tensor(O_images, "Preagumented Other");
# %%
plot_tensor(images_aug[  0:100:1], "Augmented Q");
plot_tensor(images_aug[100:200:1], "Augmented M");
plot_tensor(images_aug[200:300:1], "Augmented X");
plot_tensor(images_aug[300:400:1], "Augmented Other");

# %% Create labels

labels = torch.from_numpy(np.empty((400,),dtype=np.long));
labels[  0:100:1] = LBL_Q; 
labels[100:200:1] = LBL_M;
labels[200:300:1] = LBL_X;
labels[300:400:1] = LBL_O;

# %% Normalize the dataset
normalizer = Normalize(mean=(127.5,), std=(127.5,))
images_aug_norm = normalizer(images_aug);
# %% Randomly shuffle and partition into train and test
r_idx = np.random.permutation(400);

x = images_aug_norm[r_idx];
y = labels[r_idx];

x_train = x[  0:320:1];
x_test  = x[320:400:1];
y_train = y[  0:320:1];
y_test  = y[320:400:1];

# %% Move to GPU
device = "cuda";
x_train = x_train.to(device);
y_train = y_train.to(device);
x_test  = x_test.to(device);
y_test  = y_test.to(device);


# %% Part E (Cheap Model)

dataset = TensorDataset(x_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Conv2d(1, 1, kernel_size=5, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(1, 1, kernel_size=3, padding=0),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(36, 4)
).to(device);


# %% Train model
result = test_model( model, loader, x_test, y_test, epochs=50 );

plot_loss_curve(result, title=f"Cheap Model: {result['P']} Parameters Best Acc: {np.max(result['test_acc'])}")

# %% Part F

def get_output_size( model ):
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 20, 20)
        out = model(dummy)
        flat_size = out.shape[1]
    return flat_size;

def create_model( num_features, kernel_size, kernel_depth, num_hidden, hidden_size ):
    model = nn.Sequential();
    for ii in range(kernel_depth):
        model.add_module(f"conv{ii}", nn.Conv2d(get_output_size(model), num_features, kernel_size, padding=0));
        model.add_module(f"ReLU{ii}", nn.ReLU());
        model.add_module(f"maxp{ii}", nn.MaxPool2d(2));
        
    model.add_module("flatten", nn.Flatten());
    for ii in range(num_hidden):
        model.add_module(f"h{ii}", nn.Linear( get_output_size(model), int(hidden_size)));
        model.add_module(f"hReLU{ii}", nn.ReLU());
    model.add_module("output", nn.Linear(get_output_size(model),4));
    model = model.to(device)
    return model;

# %% Create 60 random model definitions
num_features = [2,4,6,8,10,12];
kernel_size = [3,5];
kernel_depth = [1,2];
num_hidden = [0,1,2];
hidden_size = [64,128,256,512];

all_combinations = list(itertools.product(
    num_features,
    kernel_size,
    kernel_depth,
    num_hidden,
    hidden_size
))

random_sets = random.sample(all_combinations, 60)

# %% Train and log model results

results = list();
for model_idx, params in enumerate(random_sets):
    print(params)
    model = create_model( params[0], params[1], params[2], params[3], params[4] );
    
    result = test_model( model, loader, x_test, y_test, epochs=25 );
    
    result["num_kernels"]  = params[0];
    result["kernel_size"]  = params[1];
    result["kernel_depth"] = params[2];
    result["num_hidden"]   = params[3];
    result["hidden_size"]  = params[4];
    result["model_str"]    = str(model);
    results.append(result)
    
    print(f"Finished Model: {model_idx+1}/{60}")

# plot_loss_curve(result, title=f"Model: {result['P']} Parameters Best Acc: {np.max(result['test_acc'])}")

# %% Log Results
best_acc = 0.0;
best_idx = 0;
worst_acc = 1.0;
worst_idx = 0.0;
with open("model_results.csv", "w") as f:
    f.write("model_num, num_kernels, kernel_depth,num_hidden,hidden_size,");
    for k_idx, k in enumerate(results[0].keys()):
        if( k_idx < 5 ):
            continue
        f.write(f"{k},")
    f.write("model_str\n");
    for ii in range(len(results)):
        f.write(f"{ii}, {results[ii]['num_kernels']},{results[ii]['kernel_size']},{results[ii]['kernel_depth']},{results[ii]['num_hidden']},{results[ii]['hidden_size']},");
        for k_idx, k in enumerate(results[0].keys()):
            if( k_idx < 4 ):
                continue
            if( k_idx < len(results[0].keys())-1):
                f.write(f"{results[ii][k]},")
        f.write(f"{results[ii]['model_str'].replace('\n','')}\n")
        if( np.max(results[ii]['test_acc']) > best_acc):
            best_idx = ii;
            best_acc = np.max(results[ii]['test_acc']);
        if( np.max(results[ii]['test_acc']) < worst_acc):
            worst_idx = ii;
            worst_acc = np.max(results[ii]['test_acc']);
        
# %% Best Model
plot_loss_curve(results[best_idx], title=f"Best Model: {results[best_idx]['P']} Parameters Best Acc: {np.max(results[best_idx]['test_acc'])}")
# print(results[best_idx])
plot_test_results(results[best_idx]['model'], x_test, y_test, "Best Model on Validation Set")

# %% Worst Model
plot_loss_curve(results[worst_idx], title=f"Worst Model: {results[worst_idx]['P']} Parameters Best Acc: {np.max(results[worst_idx]['test_acc'])}")
# print(results[best_idx])
plot_test_results(results[worst_idx]['model'], x_test, y_test, "Worse Model on Validation Set")

# %% Part X
test_tensor = torch.empty((25,1,20,20));
tt = ToTensor();
grid_img = Image.open("../dataset/test_set.png");
# Select subimages and convert from PIL to tensor
crop_bounds = np.array([2,2,22,22]);
for col_idx in range(5):
    crop_bounds[0] = 2;
    crop_bounds[2] = 22;
    for row_idx in range(5):
        test_tensor[row_idx+(5*col_idx)] = tt(grid_img.crop(crop_bounds))[0]
        crop_bounds[0] += 22;
        crop_bounds[2] += 22;
    crop_bounds[1] += 22;
    crop_bounds[3] += 22;
    
plot_tensor(test_tensor, "Test Dataset");

test_labels = torch.empty((25,));
test_labels[ 0: 5:] = LBL_Q;
test_labels[ 5:10:] = LBL_M;
test_labels[10:15:] = LBL_X;
test_labels[15:25:] = LBL_O;

test_tensor = test_tensor*255;
test_tensor = test_tensor.to(device);
test_labels = test_labels.to(device);

# %% Test Best and Worse Models on Test Set
norm_test_tensor = normalizer(test_tensor);
plot_test_results(results[best_idx]['model'], norm_test_tensor, test_labels, "Best Model on Test Set")

norm_test_tensor = normalizer(test_tensor);
plot_test_results(results[worst_idx]['model'], norm_test_tensor, test_labels, "Worst Model on Test Set")

        

