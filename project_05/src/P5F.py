# %% Part F

def get_output_size( model ):
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 20, 20)
        out = model(dummy)
        flat_size = out.shape[1]
    return flat_size;

def create_model( num_features, kernel_size, kernel_depth, num_hidden, hidden_size ):
    model = nn.Sequential();
    for ii in range(kernel_depth):
        model.add_module(f"conv{ii}", nn.Conv2d(get_output_size(model), num_features, kernel_size, padding=0));
        model.add_module(f"ReLU{ii}", nn.ReLU());
        model.add_module(f"maxp{ii}", nn.MaxPool2d(2));
        
    model.add_module("flatten", nn.Flatten());
    for ii in range(num_hidden):
        model.add_module(f"h{ii}", nn.Linear( get_output_size(model), int(hidden_size)));
        model.add_module(f"hReLU{ii}", nn.ReLU());
    model.add_module("output", nn.Linear(get_output_size(model),4));
    model = model.to(device)
    return model;

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

# %% Train and log model results

results = list();
for model_idx, params in enumerate(random_sets):
    print(params)
    model = create_model( params[0], params[1], params[2], params[3], params[4] );
    
    result = test_model( model, loader, x_test, y_test, epochs=25 );
    
    result["num_kernels"]  = params[0];
    result["kernel_size"]  = params[1];
    result["kernel_depth"] = params[2];
    result["num_hidden"]   = params[3];
    result["hidden_size"]  = params[4];
    result["model_str"]    = str(model);
    results.append(result)
    
    print(f"Finished Model: {model_idx+1}/{60}")

# plot_loss_curve(result, title=f"Model: {result['P']} Parameters Best Acc: {np.max(result['test_acc'])}")

# %% Log Results
best_acc = 0.0;
best_idx = 0;
worst_acc = 1.0;
worst_idx = 0.0;
with open("model_results.csv", "w") as f:
    f.write("model_num, num_kernels, kernel_depth,num_hidden,hidden_size,");
    for k_idx, k in enumerate(results[0].keys()):
        if( k_idx < 5 ):
            continue
        f.write(f"{k},")
    f.write("model_str\n");
    for ii in range(len(results)):
        f.write(f"{ii}, {results[ii]['num_kernels']},{results[ii]['kernel_size']},{results[ii]['kernel_depth']},{results[ii]['num_hidden']},{results[ii]['hidden_size']},");
        for k_idx, k in enumerate(results[0].keys()):
            if( k_idx < 4 ):
                continue
            if( k_idx < len(results[0].keys())-1):
                f.write(f"{results[ii][k]},")
        f.write(f"{results[ii]['model_str'].replace('\n','')}\n")
        if( np.max(results[ii]['test_acc']) > best_acc):
            best_idx = ii;
            best_acc = np.max(results[ii]['test_acc']);
        if( np.max(results[ii]['test_acc']) < worst_acc):
            worst_idx = ii;
            worst_acc = np.max(results[ii]['test_acc']);
        
# %% Best Model
plot_loss_curve(results[best_idx], title=f"Best Model: {results[best_idx]['P']} Parameters Best Acc: {np.max(results[best_idx]['test_acc'])}")
# print(results[best_idx])
plot_test_results(results[best_idx]['model'], x_test, y_test, "Best Model on Validation Set")

# %% Worst Model
plot_loss_curve(results[worst_idx], title=f"Worst Model: {results[worst_idx]['P']} Parameters Best Acc: {np.max(results[worst_idx]['test_acc'])}")
# print(results[best_idx])
plot_test_results(results[worst_idx]['model'], x_test, y_test, "Worse Model on Validation Set")

# %% Part X
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
    
plot_tensor(test_tensor, "Test Dataset");

test_labels = torch.empty((25,));
test_labels[ 0: 5:] = LBL_Q;
test_labels[ 5:10:] = LBL_M;
test_labels[10:15:] = LBL_X;
test_labels[15:25:] = LBL_O;

test_tensor = test_tensor*255;
test_tensor = test_tensor.to(device);
test_labels = test_labels.to(device);

# %% Test Best and Worse Models on Test Set
norm_test_tensor = normalizer(test_tensor);
plot_test_results(results[best_idx]['model'], norm_test_tensor, test_labels, "Best Model on Test Set")

norm_test_tensor = normalizer(test_tensor);
plot_test_results(results[worst_idx]['model'], norm_test_tensor, test_labels, "Worst Model on Test Set")

# %% Write Results table for latex
with open("latex_table.txt", "w") as fout:
    for ii in range(len(results)):
        fout.write(f"{ii} & {results[ii]['minBC']:.5f} & {results[ii]['E95']} & {results[ii]['E90']} & {results[ii]['E80']} & {results[ii]['E70']} & {results[ii]['P']} & {results[ii]['T']:.6f} & {results[ii]['cost']:.6f} \\\ \hline\n")
    fout.close();
