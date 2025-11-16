# %% Create 60 random model definitions
num_features = [2,4,6,8,10,12];
kernel_size = [3,5];
kernel_depth = [1,2];
num_hidden = [0,1,2];
hidden_size = [64,128,256,512];

all_combinations = list(itertools.product(
    num_features,
    kernel_size,
    kernel_depth,
    num_hidden,
    hidden_size
))
random_sets = random.sample(all_combinations, 60)


        

