import torch
from torch import nn
from torch.nn.functional import normalize
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from numpy import loadtxt
import pandas as pd
import matplotlib.pyplot as plt

file = open('data.csv', 'rb')
data = loadtxt(file,delimiter = ",")
data = data[1:]
file = open('Q_data.csv', 'rb')

Q_data = loadtxt(file,delimiter = ",")
Q_data = Q_data[1:]
q_max = np.amax(Q_data)
q_min = np.amin(Q_data)
newQ = (Q_data - q_min)/(q_max-q_min)

idx = int(len(Q_data)*0.8)
x = torch.Tensor(data[:idx])
y = torch.Tensor(Q_data[:idx])

x_test = torch.Tensor(data[idx:])
y_test = torch.Tensor(Q_data[idx:])

print(x[10:12])
print(x[43])
print(x[22])
print(y[6])
print(y[10])



device = "cpu"
batch_size = int(len(x))

train_dataset = TensorDataset(x, y)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = False)

test_dataset = TensorDataset(x_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
# Define model

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(42, 7)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(42, 7),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(7, 7)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(42, 7),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(7, 7)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
#model = NeuralNetwork().to(device)





def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        non_zero = y.nonzero()
        for i, idx in enumerate(non_zero):
            target = y[i, idx[1]].item()
            y[i] = pred[i]
            y[i, idx[1]] = target
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", len(X))

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    r = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss

def load_checkpoint(model, filename='model.pth'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    model.load_state_dict(torch.load(filename))
    return model




def full_training(train_dataset, test_dataset, epochs, batch_size, load_model = True, model_name = "model2.pth"):
    model = Model3().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9,0.999))
    loss_fn = nn.HuberLoss()
    if load_model:
        model = load_checkpoint(model, model_name)
        torch.save(model.state_dict(),"model2-old.pth")
        optimizer = torch.optim.NAdam(model.parameters(),lr=0.002,weight_decay=0.01)#torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.3)#
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    loss = [test(test_dataloader, model, loss_fn)]
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        l = test(test_dataloader, model, loss_fn)
        loss.append(l)
    return loss, model




def k_fold(k, X, Y, epochs):
    min_loss = float('inf')
    losses = []
    for fold in range(k):
        print(f"FOLD {fold+1}\n\n-------------------------------")
        idx = int(len(X)/k)
        x_train = torch.cat((X[:fold*idx],X[(fold+1)*idx:]), 0)
        y_train = torch.cat((Y[:fold*idx],Y[(fold+1)*idx:]), 0)
        x_test = X[fold*idx: (fold+1)*idx]
        y_test = Y[fold*idx:(fold+1)*idx]
        batch_size = 32
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        l, current_model = full_training(train_dataset, test_dataset, epochs, batch_size)
        print(l[-1])
        losses = losses + l
        if l[-1] < min_loss:
            min_loss = l[-1]
            torch.save(current_model.state_dict(),"best_model.pth")
    model = Model2().to(device)
    model = load_checkpoint(model, "best_model.pth")
    return losses, model

def single_training(k, val_part, X, Y, epochs, saved_file = "model2.pth"):
    idx = int(len(X)*val_part)
    x_train = torch.cat((X[:k*idx],X[(k+1)*idx:]), 0)
    y_train = torch.cat((Y[:k*idx],Y[(k+1)*idx:]), 0)
    x_test = X[k*idx: (k+1)*idx]
    y_test = Y[k*idx:(k+1)*idx]
    batch_size =32
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    l, model = full_training(train_dataset, test_dataset, epochs, batch_size, load_model = False, model_name = saved_file)
    return l, model

X = torch.Tensor(data)
Y = torch.Tensor(Q_data)
model_name = "model2.pth"
epochs = 10

losses, model = single_training(4,0.2, X, Y, epochs)
torch.save(model.state_dict(),"model3.pth")
print("Done!")
plt.plot(losses)
plt.show()

