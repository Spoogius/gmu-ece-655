# ----------------------------------
#          Part E
# ----------------------------------
# Unnormalize b and w
mu = ss_x.mean_[0];
sigma = np.sqrt(ss_x.var_[0]);
w_plt = w / sigma;
b_plt = b - (w * mu / sigma);

x_lr = np.linspace( min(ss_x.inverse_transform(x_train)), max(ss_x.inverse_transform(x_train)), 10 );
y_lr = b_plt + (x_lr*w_plt);
plt.figure();
# plt.scatter(ss_x.inverse_transform(x_train), y_train, label="Training Data" );
plt.scatter(ss_x.inverse_transform(x_train), 1000*np.e**y_train, color="blue", label="Training Data" );
plt.scatter(ss_x.inverse_transform(x_val), 1000*np.e**y_val, color="red", label="Validation Data" );
plt.yscale("log");
plt.plot( x_lr, 1000*np.e**y_lr, '--', color='black', label='Linear Regression', linewidth=1 );
plt.legend();
plt.xlabel("V_d (V)");
plt.ylabel("I_d (mA)");


print(f"Manual LR:\n  Normalized:   b: {b[0]:10.6f} w:{w[0]:10.6f}\n  Unnormalized: b: {b_plt[0]:10.6f} w: {w_plt[0]:10.6f}\n")
