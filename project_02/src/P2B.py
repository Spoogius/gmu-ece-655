# ----------------------------------
#          Part B
# ----------------------------------
N = 50;
idx = np.arange(N);
np.random.shuffle(idx);

train_idx = idx[:int(N*0.8)];
val_idx   = idx[int(N*0.8):];

x_train, y_train = dataset[train_idx, 0], dataset[train_idx, 2];
x_val,   y_val   = dataset[  val_idx, 0], dataset[  val_idx, 2];
x_train = x_train.reshape(-1,1);
y_train = y_train.reshape(-1,1);
x_val = x_val.reshape(-1,1);
y_val = y_val.reshape(-1,1);

ss_x = StandardScaler();
ss_x.fit( x_train );
x_train = ss_x.transform(x_train)
x_val   = ss_x.transform(x_val  )


#plt.figure();
#plt.scatter(v_d, i_d );
#plt.scatter(x_train, 1000*i_d[val_idx].reshape(-1,1));


