iterations = [];
best_mse = float('inf');
N_ITR = 1000;
for itr in range(N_ITR):
    b_rand = np.random.randn();
    w_rand = np.random.randn();
    y_pred = b_rand + x_val*w_rand;
    mse = (( y_pred-y_val)**2).mean()
    if( mse < best_mse ):
        iterations.append([b_rand, w_rand, float(mse)]);
        best_mse = mse;
