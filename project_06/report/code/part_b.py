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


model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
model.classifier = nn.Identity()
for par in model.parameters():
   par.requires_grad=False

filename ="CIFAR100preproc"
start_time = time.time();
pp_dataset_train = PreprocessedDataset(model, train_loader, device );
print(f"[{time.time()-start_time:.1f}] Finished Preprocesses Training Dataset");
pp_dataset_test  = PreprocessedDataset(model,  test_loader, device );
print(f"[{time.time()-start_time:.1f}] Finished Preprocesses Testing  Dataset");

torch.save(pp_dataset_train.tensors, f'dataset/{filename}_train.pth');
torch.save(pp_dataset_test.tensors,  f'dataset/{filename}_test.pth' );
print(f"[{time.time()-start_time:.1f}] Finished Writing Dataset");