lr = 0.01;
n_epochs= 500;

b = np.random.randn(1);
w = np.random.randn(1);
for epoch in range(n_epochs):
    y_hat = b + w * x_train;
    err = (y_hat - y_train);
    loss = (err**2).mean();
    
    b_grad = 2*err.mean();
    w_grad = 2*(x_train*err).mean();
    
    b = b - lr*b_grad;
    w = w - lr*w_grad;