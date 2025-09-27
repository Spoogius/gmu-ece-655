# ----------------------------------
#          Part D
# ----------------------------------

y_pred = b + w * x_val;
err = (y_pred - y_val);
loss = (err**2).mean();
print(f"Fianl b={b[0]:10.6f} w={w[0]:10.6f} loss={loss:.7f}")