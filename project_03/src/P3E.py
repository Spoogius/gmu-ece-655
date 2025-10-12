# ----------------------------------
#          Part E
# ----------------------------------

n_epochs = 100
lr = 0.05
plt_bs = np.array([4, 8, 16, 32, 64])
plt_final_loss = np.empty(len(plt_bs));
plt_final_val_loss = np.empty(len(plt_bs));

for batch_idx, batch_size in enumerate(plt_bs):
    train_loader = DataLoader(dataset=train_data, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=int(batch_size))
    
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
            
    plt_final_loss[batch_idx] = loss;
    plt_final_loss[batch_idx] = val_loss;

# Create plot
plt.figure(figsize=(8, 5))
width = 0.35
x = np.arange(len(plt_bs))
plt.bar(x - width/2, plt_final_loss, width, color='blue', label='Loss')
plt.bar(x + width/2, plt_final_val_loss, width, color='red', label='Val Loss')

plt.xticks(x, plt_bs)
plt.xlabel('Batch Size')
plt.ylabel('Loss')
plt.title('Final Loss (Loss=MSE, Optimizer=SGD)')
plt.legend()
plt.tight_layout()
plt.show()