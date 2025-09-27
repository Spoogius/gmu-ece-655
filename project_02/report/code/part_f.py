class LR_Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1140)
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.w = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    def forward(self, x):
        return self.b + self.w * x
    

x_train_tensor = torch.as_tensor(x_train).float()
y_train_tensor = torch.as_tensor(y_train).float()

model = LR_Model()
optimizer=optim.SGD(model.parameters(), lr=lr)
loss_fn=nn.MSELoss(reduction='mean')

for epoch in range(n_epochs):
    model.train()
    yhat = model(x_train_tensor)
    loss = loss_fn(yhat, y_train_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()