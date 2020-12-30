import torch
import numpy as np
from matplotlib import pyplot as plt


D1 = 20 
D2 = 10
N = 1000
ITER = 1000
SWAP = False
NOISE = True
NORMALIZE = True
VAL_PERC = 0.25


x = torch.randn(N, D1)

W_gt = torch.tensor([
    [1.8, 0, 1.8], 
    [0.1, 1, 0.1],
    [1.8, 0, 1.8], 
])
W_gt = torch.randn(D1, D2)
#b_gt = 4.0
y = x @ W_gt# + b_gt
if NOISE:
    y += 0.5 * torch.randn(N, D2)

if SWAP:
    x_copy = x
    x = y
    y = x_copy

x_norm = 1
y_norm = 1
if NORMALIZE:
    x_norm = x.max()
    y_norm = y.max()
    x = x / x_norm
    y = y / y_norm

class LinearRegression(torch.nn.Module): 
    def __init__(self):
        super(LinearRegression, self).__init__() 
        if SWAP:
            self.linear = torch.nn.Linear(D2, D1, bias = True) # bias is default True
        else:
            self.linear = torch.nn.Linear(D1, D2, bias = True) # bias is default True

        self.activation = torch.nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

    def my_loss(self, output, target):
        loss1 = torch.mean((output - target)**2)
        #loss2 = torch.mean((self.linear.weight[0][0] - W_gt[0][0])**2)
        loss2 = 0#torch.mean((self.linear.weight[0][0])**2)
        loss = loss1 + loss2
        return loss

x_train = x[:int(N*(1-VAL_PERC)),None]
y_train = y[:int(N*(1-VAL_PERC)),None]
x_val = x[:int(N*VAL_PERC),None]
y_val = y[:int(N*VAL_PERC),None]

model = LinearRegression()

criterion = torch.nn.MSELoss()
criterion = model.my_loss
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-1)

loss_arr = []
val_arr = []
def trainBuildIn(model, x, y, iter):
    for i in range(iter):
        model.train()
        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()
        
        # get output from the model, given the inputs
        y_pred = model(x_train)
        

        # get loss for the predicted output
        loss = criterion(y_pred, y_train)
        print(loss)
        loss_arr.append(loss.item())
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        print('Iter {} / {}, loss {}'.format(i, iter, loss.item()))

        model.eval()
        y_pred = model(x_val)
        loss = criterion(y_pred, y_val)
        val_arr.append(loss.item())






trainBuildIn(model, x_train, y_train, ITER)

y_pred_bi = model(x_train).data.numpy()

print("----- ----- ----- ----- -----")
print("Prediction:")
#for name, param in model.named_parameters():
#    if param.requires_grad:
#        print(name, param.data)
#print("Ground-truth:")
#print("w_gt = {:.2f}, b_gt = {:.2f}".format(w_gt ,b_gt))

plt.figure()
plt.clf()
plt.plot(y[0], label='True data', alpha=0.5)
plt.plot(y_pred_bi[0][0], label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.figure()
plt.plot(loss_arr)
plt.plot(val_arr)
plt.figure()
plt.subplot(211)
if SWAP:
    plt.title("SWAP-GT")
    plt.imshow(np.linalg.pinv(W_gt).T)
else:
    plt.title("GT")
    plt.imshow(W_gt.T)

plt.subplot(212)
plt.title("Model")
plt.imshow(model.linear.weight.detach())
plt.show()
