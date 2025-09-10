lr = LinearRegression();
lr.fit(x_train, y_train);
b_lr , w_lr = lr.intercept_[0], lr.coef_[0][0]
y_pred = b_lr + x_val*w_lr;
mse = (( y_pred-y_val)**2).mean()