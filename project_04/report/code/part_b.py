def upsample_dataset( images, N=100 ):
    idx = np.array([ random.randint(0, images.shape[0]-1) for ii in range(int(N)) ], dtype=int);
    rtn_images = np.array([ images[ii] for ii in idx ]);
    return rtn_images

upsampled_Q = upsample_dataset( images[0:10:], N/2 )
upsampled_M = upsample_dataset( images[10::],  N/2 )
images = np.concat((upsampled_Q, upsampled_M), axis=0);

augment_composer = Compose([RandomAffine(degrees=(-15,15), translate=(0.1,0.1))])
to_pil = ToPILImage();
augmented_images = np.array( [ augment_composer(to_pil(images[ii][0])) for ii in range(images.shape[0])] )