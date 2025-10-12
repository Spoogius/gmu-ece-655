model=nn.Sequential(nn.Linear(num_features,1)).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn=nn.L1Loss(reduction='mean')
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
val_step_fn   = make_val_step_fn(model, loss_fn)

for epoch in range(n_epochs):
    loss = mini_batch(device, train_loader, train_step_fn)
    losses[epoch] = loss;
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step_fn)
        val_losses[epoch] = val_loss;