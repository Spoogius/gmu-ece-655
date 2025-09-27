# ----------------------------------
#          Part C
# ----------------------------------

lr = 0.01;
n_epochs= 500;
loss = np.empty((500,));
b = np.random.randn(1);
w = np.random.randn(1);
for epoch in range(n_epochs):
    y_hat = b + w * x_train;
    err = (y_hat - y_train);
    loss[epoch] = (err**2).mean();
    
    b_grad = 2*err.mean();
    w_grad = 2*(x_train*err).mean();
    
    b = b - lr*b_grad;
    w = w - lr*w_grad;

print(f"[Epoch {n_epochs}] Loss={loss[n_epochs-1]}")

plt.figure();
plt.plot(range(n_epochs), loss );
plt.ylabel("Loss (MSE)");
plt.xlabel("Epoch");
#plt.scatter(x_train, 1000*i_d[val_idx].reshape(-1,1));