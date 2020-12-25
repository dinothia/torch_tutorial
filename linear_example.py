# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import math

swap_in_out = True
# Create training data
N = 10000

A = torch.tensor([
                [4, 0], 
                [0.5, 4]], dtype=torch.float)

D_I = A.shape[0]                
D_O = A.shape[1]                

b = 5
x = torch.rand(N, D_I)
#x = torch.randn(N, D_I)
y = x @ A + b
#y += 0.1 * torch.randn_like(y)

if swap_in_out:
    x_train = y
    y_train = x

    D_I = A.shape[1]                
    D_O = A.shape[0]                
else:
    x_train = x
    y_train = y

# Add to dataloader
bs = 16
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, shuffle=True, batch_size=bs)

lr = 1e-3
model = torch.nn.Sequential(
        torch.nn.Linear(D_I, D_O), 
        #torch.nn.ReLU()
)
        
loss_fn = torch.nn.MSELoss(reduction="sum")
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

loss_arr = []
epochs = 50
for epoch in range(epochs):
    for xb,yb in train_dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    if epoch % 10 == 9:
        loss_arr.append(loss.item())
        print(epoch, loss.item())

linear_layer = model[0]
print(f"bias term = \n{linear_layer.bias.detach().numpy()}")
print(f"actual weight = \n {A.detach().numpy()}")
print(f"inv actual weight = \n {np.linalg.inv(A.detach().numpy())}")
print(f"weights term = \n{linear_layer.weight.detach().numpy().round(2).T}")
print(x_train[-1])

plt.plot(loss_arr)
plt.show()
