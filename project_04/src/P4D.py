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





