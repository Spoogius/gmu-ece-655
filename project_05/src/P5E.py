
# %% Create labels

labels = torch.from_numpy(np.empty((400,),dtype=np.long));
labels[  0:100:1] = LBL_Q; 
labels[100:200:1] = LBL_M;
labels[200:300:1] = LBL_X;
labels[300:400:1] = LBL_O;

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

# %% Move to GPU
device = "cuda";
x_train = x_train.to(device);
y_train = y_train.to(device);
x_test  = x_test.to(device);
y_test  = y_test.to(device);


# %% Part E (Cheap Model)

dataset = TensorDataset(x_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Conv2d(1, 1, kernel_size=5, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(4),
    nn.Flatten(),
    nn.Linear(16, 4)
).to(device);



result = test_model( model, loader, x_test, y_test, epochs=50 );

plot_loss_curve(result, title=f"Cheap Model: {result['P']} Parameters Best Acc: {np.max(result['test_acc'])}")
