# ----------------------------------
#          Part F
# ----------------------------------

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

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

print(f"[Epoch {n_epochs}] Loss={loss}")
P=list(model.parameters())
bt=float(P[0].detach())
wt=float(P[1].detach())
yt_pred = model(torch.as_tensor(x_val).float());
losst = loss_fn(yt_pred, torch.as_tensor(y_val).float())
print(f"Fianl b={bt:10.6f} w={wt:10.6f} loss={losst:.7f}")

wt_plt = wt / sigma;
bt_plt = bt - (w * mu / sigma);
print(f"Torch LR:\n  Normalized:   b: {bt:10.6f} w:{wt:10.6f}\n  Unnormalized: b: {bt_plt[0]:10.6f} w: {wt_plt:10.6f}")