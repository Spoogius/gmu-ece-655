model = nn.Sequential(
    nn.Conv2d(1, 1, kernel_size=5, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(4),
    nn.Flatten(),
    nn.Linear(16, 4)
).to(device);