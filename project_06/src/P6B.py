# %% Part B
def PreprocessedDataset(model_idt, loader, device=None):
    model.to(device)
    for i, (x, y) in enumerate(loader):
        model_idt.eval()
        output = model_idt(x.to(device))
        if i == 0:
            features = output.detach().cpu()
            labels = y.cpu()
        else:
            features = torch.cat(
                [features, output.detach().cpu()])
            labels = torch.cat([labels, y.cpu()])
    dataset = TensorDataset(features, labels)
    return dataset

## Preprocess the dataset
model = deepcopy(model_base); #Identify final layer

filename ="CIFAR100preproc"
start_time = time.time();
pp_dataset_train = PreprocessedDataset(model, train_loader, device );
print(f"[{time.time()-start_time:.1f}] Finished Preprocesses Training Dataset");
pp_dataset_test  = PreprocessedDataset(model,  test_loader, device );
print(f"[{time.time()-start_time:.1f}] Finished Preprocesses Testing  Dataset");

torch.save(pp_dataset_train.tensors, f'dataset/{filename}_train.pth');
torch.save(pp_dataset_test.tensors,  f'dataset/{filename}_test.pth' );
print(f"[{time.time()-start_time:.1f}] Finished Writing Dataset");
# %%
pp_dataset_train_x, pp_dataset_train_y = torch.load(f'dataset/{filename}_train.pth');
pp_dataset_test_x,  pp_dataset_test_y  = torch.load(f'dataset/{filename}_test.pth' );
pp_dataset_train = TensorDataset( pp_dataset_train_x, pp_dataset_train_y );
pp_dataset_test  = TensorDataset( pp_dataset_test_x,  pp_dataset_test_y  );
pp_loader_train = DataLoader( pp_dataset_train, batch_size=64, shuffle=True );
pp_loader_test  = DataLoader( pp_dataset_test,  batch_size=64, shuffle=False);

# %% Train final model classifier layers
model_classifier_fine = nn.Sequential(nn.Dropout(0.5), nn.Linear(1536, 100, bias=True));
model_classifier_fine.to(device);
N_EPOCHS = 30;
results_B = test_model( model_classifier_fine, pp_loader_train, pp_loader_test, optimizer=optim.Adam(model_classifier_fine.parameters(), lr=3E-4), device=device, epochs=N_EPOCHS );
plot_accuracy(results_B);