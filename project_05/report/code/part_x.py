test_tensor = torch.empty((25,1,20,20));
tt = ToTensor();
grid_img = Image.open("../dataset/test_set.png");
grid_img.show()
# Select subimages and convert from PIL to tensor
crop_bounds = np.array([2,2,22,22]);
for col_idx in range(5):
    crop_bounds[0] = 2;
    crop_bounds[2] = 22;
    for row_idx in range(5):
        test_tensor[row_idx+(5*col_idx)] = tt(grid_img.crop(crop_bounds))[0]
        crop_bounds[0] += 22;
        crop_bounds[2] += 22;
    crop_bounds[1] += 22;
    crop_bounds[3] += 22;

