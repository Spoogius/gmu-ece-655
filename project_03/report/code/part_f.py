epochs_sweep = np.linspace(20,1000,50)
optim_sweep  = [optim.SGD, optim.Adam]
device='cpu'
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