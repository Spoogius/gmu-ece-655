y_pred = b + w * x_val;
err = (y_pred - y_val);
loss = (err**2).mean();