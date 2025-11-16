# %% Agument Dataset
composer = Compose( [RandomAffine(degrees=(-30,30), translate=(0.15,0.15)) ]);
images_aug = torch.from_numpy(np.empty((400,1,20,20), dtype=np.float32))
for ii in range(images.shape[0]):
    images_aug[ii] = composer(images[ii]);
