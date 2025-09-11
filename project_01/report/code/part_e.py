for N_ITR in [ 1000, 2000, 5000, 10000 ]:
    iterations = [];
    best_mse = float('inf');
    for itr in range(N_ITR):
        b_rand = np.random.randn();
        w_rand = np.random.randn();
        y_pred = b_rand + x_val*w_rand;
        mse = (( y_pred-y_val)**2).mean()
        if( mse < best_mse ):
            iterations.append([b_rand, w_rand, float(mse)]);
            best_mse = mse;

    iterations = np.array(iterations);
    print(f"[Naive-{N_ITR}] b: {iterations[-1][0]:.4}, w: {iterations[-1][1]:.4}, mse: {iterations[-1][2]:.4}\n\tHits: {iterations.shape[0]} Missed: {N_ITR-iterations.shape[0]}");
