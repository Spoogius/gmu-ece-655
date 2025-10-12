device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=nn.Sequential(nn.Linear(num_features,1)).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_fn=nn.MSELoss(reduction='mean')
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)


# train_loader = DataLoader(dataset=train_data, batch_size=n_train, shuffle=True, num_workers=4, pin_memory=True)
train_loader = DataLoader(dataset=train_data, batch_size=n_train, shuffle=True )

epochs_sweep = np.linspace(20,1000,50)
optim_sweep  = [optim.SGD, optim.Adam]
results = np.empty((len(optim_sweep), len(epochs_sweep), 2))

import time

lr = 0.05
for optim_idx, opt in enumerate(optim_sweep):
    for epoch_idx, n_epochs in enumerate(epochs_sweep):
        
        
        
        model=nn.Sequential(nn.Linear(num_features,1)).to(device)
        optimizer = opt(model.parameters(), lr=lr)
        loss_fn=nn.MSELoss(reduction='mean') 
        train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
        val_step_fn   = make_val_step_fn(model, loss_fn)
        
        start_ts = time.time()
        for epoch in range(int(n_epochs)):
            loss = mini_batch(device, train_loader, train_step_fn)
        
        stop_ts = time.time()
        
        del model, optimizer, loss_fn, train_step_fn, val_step_fn
        gc.collect()
        
        results[optim_idx][epoch_idx][0] = loss;
        results[optim_idx][epoch_idx][1] = stop_ts-start_ts;
        print(f"Finished [{opt.__name__:4}][{int(n_epochs):4}]: Loss={loss:6.3f} Time={results[optim_idx][epoch_idx][1]:9.5f}")
        
# %%
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

ax1.plot(epochs_sweep, results[0][:,0], 'b', label='SGD')
ax1.plot(epochs_sweep, results[1][:,0], 'b--', label='Adam')

ax2.plot(epochs_sweep, results[0][:,1], 'r', label='SGD')
ax2.plot(epochs_sweep, results[1][:,1], 'r--', label='Adam')

plt.xlabel('Epoch')
ax1.set_ylabel('MSE Loss', color='b')
ax2.set_ylabel('Elapsed Time (sec)', color='r')
plt.title('[GPU] Final Loss and Runtime')

from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color='k', lw=2, linestyle='-'),
    Line2D([0], [0], color='k', lw=2, linestyle='--')
]

# Add a single shared legend for both axes
ax1.legend(custom_lines, ['SGD', 'Adam'], loc='upper center')

plt.show()  