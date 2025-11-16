# %% Normalize the dataset
normalizer = Normalize(mean=(127.5,), std=(127.5,))
images_aug_norm = normalizer(images_aug);

# %% Randomly shuffle and partition into train and test
r_idx = np.random.permutation(400);

x = images_aug_norm[r_idx];
y = labels[r_idx];

x_train = x[  0:320:1];
x_test  = x[320:400:1];
y_train = y[  0:320:1];
y_test  = y[320:400:1];
