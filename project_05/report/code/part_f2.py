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


        

