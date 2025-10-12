# ----------------------------------
#          Part B
# ----------------------------------

ss_x = StandardScaler();
ss_x.fit( x );
x_norm = ss_x.transform(x)

x_tensor = torch.as_tensor(x_norm).float()
y_tensor = torch.as_tensor(y).float()

# Builds dataset containing ALL data points
dataset = TensorDataset(x_tensor, y_tensor.unsqueeze(1))

# Performs the split
ratio = .8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train


train_data, val_data = random_split(dataset, [n_train, n_val])
# Builds a loader of each set
train_loader = DataLoader(dataset=train_data, batch_size=n_train, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=n_train)


device = 'cpu'
n_epochs = 100
lr = 0.05
model=nn.Sequential(nn.Linear(num_features,1)).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_fn=nn.MSELoss(reduction='mean')
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
val_step_fn   = make_val_step_fn(model, loss_fn)

losses = np.empty(n_epochs);
val_losses = np.empty(n_epochs);
for epoch in range(n_epochs):
    loss = mini_batch(device, train_loader, train_step_fn)
    losses[epoch] = loss;
    # VALIDATION - no gradients
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step_fn)
        val_losses[epoch] = val_loss;

plot_loss_curve(losses,val_losses,title='Loss Curve (Loss=MSE, Batch Size=Full, Optimizer=SGD)');