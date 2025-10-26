
N = 200
upsampled_Q = upsample_dataset( images[0:10:], N/2 )
upsampled_M = upsample_dataset( images[10::],  N/2 )
# augmented_Q = data_augmentation(images[0:10:], N/2);
# plot_images(upsampled_Q, ("Dataset (Q)"), (10,10))
# augmented_M = data_augmentation(images[10::], N/2);
# #plot_images(augmented_M, ("Augmentation Dataset (M)"), (10,10))

images = np.concat((upsampled_Q, upsampled_M), axis=0);
labels = np.empty(N, dtype=float); 
labels[0:int(N/2):] = 0; # 0 = Q
labels[int(N/2)::]  = 1; # 1 = M

# augment_composer = Compose([RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5), RandomAffine(degrees=20, translate=(0.1,0.1))])
augment_composer = Compose([RandomAffine(degrees=(-15,15), translate=(0.1,0.1))])
to_pil = ToPILImage();
augmented_images = np.array( [ augment_composer(to_pil(images[ii][0])) for ii in range(images.shape[0])] )
augmented_images = augmented_images.reshape((N,1,20,20))/255
plot_images(augmented_images[0:int(N/2):], ("Augmented Dataset (Q)"), (10,10))
plot_images(augmented_images[int(N/2)::],  ("Augmented Dataset (M)"), (10,10))