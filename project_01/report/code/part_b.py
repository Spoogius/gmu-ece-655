b_depth = 3;
w_depth = 3;
b_range = np.linspace( b_lr - b_depth, b_lr + b_depth, 100 );
w_range = np.linspace( w_lr - w_depth, b_lr + w_depth, 100 );
b_surf, w_surf = np.meshgrid( b_range, w_range );
y_surf = np.apply_along_axis( func1d=lambda x: b_surf + w_surf*x,
                              axis=1,
                              arr=x_train );

all_labels = y_train.reshape(-1,1,1);
all_errors = (y_surf-all_labels);
loss_surf = (all_errors**2).mean(axis=0);
