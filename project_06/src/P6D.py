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

