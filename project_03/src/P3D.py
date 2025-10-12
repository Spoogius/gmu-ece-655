# ----------------------------------
#          Part D
# ----------------------------------

n_epochs = 100
lr = 0.05
model=nn.Sequential(nn.Linear(num_features,1)).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
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

plot_loss_curve(losses,val_losses,title='Loss Curve (Loss=MSE, Batch Size=Full, Optimizer=Adam)');