    
# Randomly upsample
images = torch.from_numpy(np.empty((400,1,20,20), dtype=np.float32))
images[  0:100:1] = Q_images[np.random.randint(0, 10, 100)];
images[100:200:1] = M_images[np.random.randint(0, 10, 100)];
images[200:300:1] = X_images[np.random.randint(0, 10, 100)];
images[300:400:1] = O_images[np.random.randint(0, 10, 100)];

# %% Agument Dataset
composer = Compose( [RandomAffine(degrees=(-30,30), translate=(0.15,0.15)) ]);
images_aug = torch.from_numpy(np.empty((400,1,20,20), dtype=np.float32))
for ii in range(images.shape[0]):
    images_aug[ii] = composer(images[ii]);

# %% Remap image values to white = 255, black = 0.0
Q_images = torch.from_numpy(np.abs(Q_images.numpy()-1)*255)
M_images = torch.from_numpy(np.abs(M_images.numpy()-1)*255)
X_images = torch.from_numpy(np.abs(X_images.numpy()-1)*255)
O_images = torch.from_numpy(np.abs(O_images.numpy()-1)*255)
images_aug = torch.from_numpy(np.abs(images_aug.numpy()-1)*255)
