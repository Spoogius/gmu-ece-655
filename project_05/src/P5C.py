
# %% Generate Dataset figures
plot_tensor(Q_images, "Preagumented Q");
plot_tensor(M_images, "Preagumented M");
plot_tensor(X_images, "Preagumented X");
plot_tensor(O_images, "Preagumented Other");
# %%
plot_tensor(images_aug[  0:100:1], "Augmented Q");
plot_tensor(images_aug[100:200:1], "Augmented M");
plot_tensor(images_aug[200:300:1], "Augmented X");
plot_tensor(images_aug[300:400:1], "Augmented Other");
