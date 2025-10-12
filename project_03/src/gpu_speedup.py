import torch
import torch.nn as nn
import time
from torch.utils.data import TensorDataset, DataLoader

# Check for GPU
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

print(f"Using GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Synthetic data generation
def generate_data(w, b, num_samples):
    X = torch.randn(num_samples, len(w))
    y = X @ w + b + 0.1 * torch.randn(num_samples)
    return X, y

# Linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.linear(x)

# Training loop
def train(X, y, device, num_epochs=1000, lr=0.03):
    model = LinearRegressionModel(X.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    X = X.to(device)
    y = y.to(device).unsqueeze(1)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()

    for _ in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize() if device.type == 'cuda' else None
    end = time.time()

    return end - start, loss.item()


# Hyperparameters
num_samples = 1_000_000  # Bigger dataset to emphasize GPU
num_features = 50
true_w = torch.randn(num_features)
true_b = 2.0

# Create data once
X, y = generate_data(true_w, true_b, num_samples)

# Run on CPU
cpu_time, cpu_loss = train(X, y, device_cpu)
print(f"\n[CPU] Full Batch Time: {cpu_time:.2f}s | Final Loss: {cpu_loss:.4f}")

# Run on GPU if available
if torch.cuda.is_available():
    gpu_time, gpu_loss = train(X, y, device_gpu)
    print(f"[GPU] Full Batch  Time: {gpu_time:.2f}s | Final Loss: {gpu_loss:.4f}")
else:
    print("\nCUDA not available. Skipping GPU run.")

