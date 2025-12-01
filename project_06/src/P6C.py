# %% Get coarse labels
import os
import pickle
with open(os.path.join('./dataset', 'cifar-100-python', 'train'), 'rb') as f:
    meta_train = pickle.load(f, encoding='latin1')
with open(os.path.join('./dataset', 'cifar-100-python', 'test' ), 'rb') as f:
    meta_test  = pickle.load(f, encoding='latin1')
    
train_dataset = datasets.CIFAR100(root='./dataset', train=True,  download=True, transform=composer_train)
test_dataset  = datasets.CIFAR100(root='./dataset', train=False, download=True, transform=composer_test)
train_dataset.targets = meta_train["coarse_labels"]
test_dataset.targets  = meta_test[ "coarse_labels"]
train_loader = DataLoader( train_dataset, batch_size=32, shuffle=True  );
test_loader  = DataLoader(  test_dataset, batch_size=32, shuffle=False );
# %%
# Preprocess the dataset
model = deepcopy(model_base); #Identify final layer

pp_dataset_coarse_train = PreprocessedDataset(model, train_loader, device );
pp_dataset_coarse_test  = PreprocessedDataset(model,  test_loader, device );

filename ="CIFAR100preproc"
torch.save(pp_dataset_coarse_train.tensors, f'dataset/{filename}_coarse_train.pth');
torch.save(pp_dataset_coarse_test.tensors,  f'dataset/{filename}_coarse_test.pth' );

# %%
pp_dataset_coarse_train_x, pp_dataset_coarse_train_y = torch.load(f'dataset/{filename}_coarse_train.pth');
pp_dataset_coarse_test_x,  pp_dataset_coarse_test_y  = torch.load(f'dataset/{filename}_coarse_test.pth' );
pp_dataset_coarse_train = TensorDataset( pp_dataset_coarse_train_x, pp_dataset_coarse_train_y );
pp_dataset_coarse_test  = TensorDataset( pp_dataset_coarse_test_x,  pp_dataset_coarse_test_y  );
pp_loader_coarse_train = DataLoader( pp_dataset_coarse_train, batch_size=32, shuffle=True  );
pp_loader_coarse_test  = DataLoader(  pp_dataset_coarse_test, batch_size=32, shuffle=False );

# %% Train final model classifier layers on coarse labels
model_classifier_coarse = nn.Sequential(nn.Dropout(.5), nn.Linear(1536, 20, bias=True));
model_classifier_coarse.to(device);
N_EPOCHS = 30;
results_C = test_model( model_classifier_coarse, pp_loader_coarse_train, pp_loader_coarse_test, optimizer=optim.Adam(model_classifier_coarse.parameters(), lr=0.001), device=device, epochs=N_EPOCHS );
plot_accuracy(results_C);