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
# %% Part B
def PreprocessedDataset(model_idt, loader, device=None):
    model.to(device)
    for i, (x, y) in enumerate(loader):
        model_idt.eval()
        output = model_idt(x.to(device))
        if i == 0:
            features = output.detach().cpu()
            labels = y.cpu()
        else:
            features = torch.cat(
                [features, output.detach().cpu()])
            labels = torch.cat([labels, y.cpu()])
    dataset = TensorDataset(features, labels)
    return dataset

## Preprocess the dataset
model = deepcopy(model_base); #Identify final layer

#filename ="CIFAR100preproc_class"
filename ="CIFAR100preproc"
#filename ="CIFAR100preproc_gpt"
start_time = time.time();
pp_dataset_train = PreprocessedDataset(model, train_loader, device );
print(f"[{time.time()-start_time:.1f}] Finished Preprocesses Training Dataset");
pp_dataset_test  = PreprocessedDataset(model,  test_loader, device );
print(f"[{time.time()-start_time:.1f}] Finished Preprocesses Testing  Dataset");

torch.save(pp_dataset_train.tensors, f'dataset/{filename}_train.pth');
torch.save(pp_dataset_test.tensors,  f'dataset/{filename}_test.pth' );
print(f"[{time.time()-start_time:.1f}] Finished Writing Dataset");
# %%
pp_dataset_train_x, pp_dataset_train_y = torch.load(f'dataset/{filename}_train.pth');
pp_dataset_test_x,  pp_dataset_test_y  = torch.load(f'dataset/{filename}_test.pth' );
pp_dataset_train = TensorDataset( pp_dataset_train_x, pp_dataset_train_y );
pp_dataset_test  = TensorDataset( pp_dataset_test_x,  pp_dataset_test_y  );
pp_loader_train = DataLoader( pp_dataset_train, batch_size=64, shuffle=True );
pp_loader_test  = DataLoader( pp_dataset_test,  batch_size=64, shuffle=False);

# %% Train final model classifier layers
model_classifier_fine = nn.Sequential(nn.Dropout(0.5), nn.Linear(1536, 100, bias=True));
model_classifier_fine.to(device);
N_EPOCHS = 30;
results_B = test_model( model_classifier_fine, pp_loader_train, pp_loader_test, optimizer=optim.Adam(model_classifier_fine.parameters(), lr=3E-4), device=device, epochs=N_EPOCHS );
plot_accuracy(results_B);
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
# %%
# Preprocess the dataset
model = deepcopy(model_base); #Identify final layer

pp_dataset_coarse_train = PreprocessedDataset(model, train_loader, device );
pp_dataset_coarse_test  = PreprocessedDataset(model,  test_loader, device );

filename ="CIFAR100preproc"
torch.save(pp_dataset_coarse_train.tensors, f'dataset/{filename}_coarse_train.pth');
torch.save(pp_dataset_coarse_test.tensors,  f'dataset/{filename}_coarse_test.pth' );

# %%
pp_dataset_coarse_train_x, pp_dataset_coarse_train_y = torch.load(f'dataset/{filename}_coarse_train.pth');
pp_dataset_coarse_test_x,  pp_dataset_coarse_test_y  = torch.load(f'dataset/{filename}_coarse_test.pth' );
pp_dataset_coarse_train = TensorDataset( pp_dataset_coarse_train_x, pp_dataset_coarse_train_y );
pp_dataset_coarse_test  = TensorDataset( pp_dataset_coarse_test_x,  pp_dataset_coarse_test_y  );
pp_loader_coarse_train = DataLoader( pp_dataset_coarse_train, batch_size=32, shuffle=True  );
pp_loader_coarse_test  = DataLoader(  pp_dataset_coarse_test, batch_size=32, shuffle=False );

# %% Train final model classifier layers on coarse labels
model_classifier_coarse = nn.Sequential(nn.Dropout(.5), nn.Linear(1536, 20, bias=True));
model_classifier_coarse.to(device);
N_EPOCHS = 30;
results_C = test_model( model_classifier_coarse, pp_loader_coarse_train, pp_loader_coarse_test, optimizer=optim.Adam(model_classifier_coarse.parameters(), lr=0.001), device=device, epochs=N_EPOCHS );
plot_accuracy(results_C);
# %%
import json
# https://huggingface.co/datasets/zh-plus/tiny-imagenet/blob/main/classes.py
with open("dataset/tiny-imagenet-classes.json", "r") as f:
    tinyin_classes = json.load(f)

full_dataset = datasets.ImageFolder("dataset/tiny-imagenet/tiny-imagenet-200/train", transform=composer_test)


label_nnum = list(full_dataset.class_to_idx.keys())
classes_key = list();
for label_val in range(len(label_nnum)):
    classes_key.append({'tinyin_label': label_val, 'nnum': label_nnum[label_val], 'tinyin': tinyin_classes[label_nnum[label_val]]})

cifar_coarse_label_strs = [
 'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
 'household electrical devices', 'household furniture', 'insects', 'large carnivores',
 'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
 'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals',
 'trees', 'vehicles 1', 'vehicles 2'
]

for class_key in classes_key:
    print(f"[{class_key['tinyin_label']}] {class_key['tinyin']}")

# Grab some common classes
common_classes = {
    "186": {"cfar_fine": 53 }, # Orange -> Orange
    "185": {"cfar_fine": 51 }, # Mushroom -> Mushroom
    "177": {"cfar_fine": 61 }, # Plate -> Plate
    "8":   {"cfar_fine": 79 }, # Black Widow -> Spider
    "9":   {"cfar_fine": 79 }, # Tarantual -> Spider
    "77":  {"cfar_fine": 15 }, # Snail -> Snail
    "18":  {"cfar_fine": 45 }, # Lobster -> Lobster
    "34":  {"cfar_fine": 43 }, # Lion -> Lion
    "92":  {"cfar_fine": 39 }, # Computer Keyboard -> Keytboard
    "114": {"cfar_fine": 41 }, # Lawn Mower -> Lawn Mower
    "135":  {"cfar_fine": 9 }, # Soda Bottle -> Bottle
    }
# %%
for class_key in classes_key:
    if str(class_key['tinyin_label']) in common_classes:
        class_key['cfar_fine'] = common_classes[str(class_key['tinyin_label'])]['cfar_fine']
        class_key['cfar_fine_str'] = list(train_dataset.class_to_idx.keys())[class_key['cfar_fine']]
        class_key['cfar_coarse'] = meta_train['coarse_labels'][np.where( class_key['cfar_fine'] == np.array(meta_train['fine_labels']))[0][0]]
        class_key['cfar_coarse_str'] = cifar_coarse_label_strs[class_key['cfar_coarse']]
    else:
        class_key['cfar_fine'] = None
        class_key['cfar_fine_str'] = None
        class_key['cfar_coarse'] = None
        class_key['cfar_coarse_str'] = None


full_dataset_idx = np.array([]);
for class_key in classes_key:
    if class_key['cfar_fine'] is None:
        pass;
    else:
        full_dataset_idx = np.append(full_dataset_idx, np.where(np.array(full_dataset.targets) == class_key['tinyin_label'] )[0])
        print(class_key)

tin_data = torch.empty(len(full_dataset_idx),3,300,300, dtype=torch.float32)
tin_label_tin    = torch.empty(len(full_dataset_idx), dtype=torch.int64)
tin_label_fine   = torch.empty(len(full_dataset_idx), dtype=torch.int64)
tin_label_coarse = torch.empty(len(full_dataset_idx), dtype=torch.int64)
for ii, fd_idx in enumerate(full_dataset_idx):
    tin_data[ii]      = full_dataset[int(fd_idx)][0];
    tin_label_tin[ii] = full_dataset[int(fd_idx)][1];
    tin_label_fine[ii]   = classes_key[tin_label_tin[ii]]['cfar_fine'];
    tin_label_coarse[ii] = classes_key[tin_label_tin[ii]]['cfar_coarse'];
    
# %%
class BrainNet:
    def __init__(self, model_common, model_coarse, model_fine, device='cpu'):
        self.model_common_ = model_common;
        self.model_coarse_ = model_coarse;
        self.model_fine_   = model_fine;
        self.model_common_.to(device);
        self.model_coarse_.to(device);
        self.model_fine_.to(  device);
        self.model_common_.eval();
        self.model_coarse_.eval();
        self.model_fine_.eval();
        
    def __call__(self, x ):
        common_output = self.model_common_(x.to(device))
        y_pred_fine   = model_classifier_fine(   common_output );
        y_pred_coarse = model_classifier_coarse( common_output );
        return y_pred_coarse, y_pred_fine
# %%

model_bn = BrainNet( model_base, model_classifier_coarse, model_classifier_fine, device=device)
BS = 64;
start_idx = 0;
total_correct1_coarse = 0;
total_correct5_coarse = 0;
total_correct1_fine = 0;
total_correct5_fine = 0;
total_preds = 0;
pred_coarse = np.empty(len(full_dataset_idx), dtype=np.int64 )
pred_fine   = np.empty(len(full_dataset_idx), dtype=np.int64 )
for ii in range(int(tin_data.size()[0]/BS)):
    y_pred_coarse, y_pred_fine  = model_bn(tin_data[start_idx:start_idx+BS:]);
    
    # common_output = model_base(tin_data[start_idx:start_idx+BS:].to(device))
    
    # y_pred_fine   = model_classifier_fine(   common_output );
    pred_fine[start_idx:start_idx+BS:] = np.argmax(y_pred_fine.detach().cpu().numpy(),axis=1);
    total_correct5_fine += np.sum(top_5(tin_label_fine[start_idx:start_idx+BS:].cpu().numpy(), y_pred_fine.detach().cpu().numpy()))
    total_correct1_fine += np.sum(top_1(tin_label_fine[start_idx:start_idx+BS:].cpu().numpy(), y_pred_fine.detach().cpu().numpy()))
    
    # y_pred_coarse = model_classifier_coarse( common_output );
    pred_coarse[start_idx:start_idx+BS:] = np.argmax(y_pred_coarse.detach().cpu().numpy(),axis=1);
    total_correct5_coarse += np.sum(top_5(tin_label_coarse[start_idx:start_idx+BS:].cpu().numpy(), y_pred_coarse.detach().cpu().numpy()))
    total_correct1_coarse += np.sum(top_1(tin_label_coarse[start_idx:start_idx+BS:].cpu().numpy(), y_pred_coarse.detach().cpu().numpy()))
    
    total_preds += BS;
    
    start_idx += BS;
    
    
print(f"Fine:   T1 Acc: {total_correct1_fine  /total_preds:.2f} - T5 Acc: {total_correct5_fine  /total_preds:.2f}")
print(f"Coarse: T1 Acc: {total_correct1_coarse/total_preds:.2f} - T5 Acc: {total_correct5_coarse/total_preds:.2f}")
    
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


true_labels = [None,
 None,
 None,
 'plate',
 'orange',
 'computer keyboard',
 None,
 None,
 'lion',
 None,
 None,
 'bow tie',
 None,
 'lobster',
 None,
 None,
 None,
 None,
 None,
 'lawn mower']; # Manual clean up names b/c messy
pred_labels = cifar_coarse_label_strs;
# for ck in classes_key:
#     if(ck['cfar_coarse'] is None):
#         continue;
#     else:
#         true_labels[ck['cfar_coarse']] = ck['tinyin'];
#         #pred_labels[ck['cfar_coarse']] = ck['tinyin'];
     
        
cm_coarse = confusion_matrix(tin_label_coarse[0:total_preds:], pred_coarse[0:total_preds:])
fig, ax = plt.subplots(figsize=(10, 10), dpi=200)   # ← Your size NOW works   
plt.rcParams.update({'font.size': 10})
disp = ConfusionMatrixDisplay(confusion_matrix=cm_coarse)
disp.plot(cmap="Blues", xticks_rotation=90, ax=ax, colorbar=False)  # ← Use your axis
ax.set_xticklabels(labels=pred_labels, rotation=90, fontsize=12);
ax.set_yticklabels(labels=true_labels, fontsize=15);
plt.title("Coarse Labels");
plt.tight_layout()
plt.show()

cm_fine = confusion_matrix(tin_label_fine[0:total_preds:], pred_fine[0:total_preds:])
fig, ax = plt.subplots(figsize=(20, 20), dpi=200)   # ← Your size NOW works   
plt.rcParams.update({'font.size': 9})
disp = ConfusionMatrixDisplay(confusion_matrix=cm_fine)
disp.plot(cmap="Blues", xticks_rotation=90, ax=ax, colorbar=False)  # ← Use your axis
plt.title("Fine Labels");
plt.tight_layout()
plt.show()

# %% 100 Random Labels
test_idx = np.random.permutation(tin_data.size()[0])[0:100:];
y_pred_coarse, y_pred_fine  = model_bn(tin_data[test_idx]);
    
# %%
# y_pred_fine   = model_classifier_fine(   common_output );
pred_fine = np.argmax(y_pred_fine.detach().cpu().numpy(),axis=1);
total_correct1_fine = np.sum(top_1(tin_label_fine[test_idx].cpu().numpy(), y_pred_fine.detach().cpu().numpy()))

# y_pred_coarse = model_classifier_coarse( common_output );
pred_coarse = np.argmax(y_pred_coarse.detach().cpu().numpy(),axis=1);
total_correct1_coarse = np.sum(top_1(tin_label_coarse[test_idx].cpu().numpy(), y_pred_coarse.detach().cpu().numpy()))
    
total_preds = 100;

pred_perc_coarse = 100/(1+np.exp(-1*np.max(y_pred_coarse.detach().cpu().numpy(),axis=1)))
pred_perc_fine   = 100/(1+np.exp(-1*np.max(y_pred_fine.detach().cpu().numpy(),axis=1)))

for ii in range(total_preds):
    print(f"{ii:2d}, {classes_key[tin_label_tin[test_idx[ii]]]['tinyin'].replace(',', '')}, {classes_key[tin_label_tin[test_idx[ii]]]['cfar_coarse_str'].replace(',', '')}, {pred_perc_coarse[ii]:.2f}, {int(tin_label_coarse[test_idx[ii]] == pred_coarse[ii])}, {classes_key[tin_label_tin[test_idx[ii]]]['cfar_fine_str'].replace(',', '')}, {pred_perc_fine[ii]:.2f}, {int(tin_label_fine[test_idx[ii]] == pred_fine[ii])}")

