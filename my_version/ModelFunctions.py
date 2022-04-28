import torch
from torch import nn
from torch.nn.functional import normalize
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from numpy import loadtxt
import pandas as pd
import matplotlib.pyplot as plt
import random

device = "cpu"

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
            nn.Linear(42, 16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(42, 57),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(57, 4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


#model = NeuralNetwork().to(device)

def train_batch(X, Y, model,loss_fn, optimizer, epochs, batch_size, early_stop = False):
    train_loss = []
    test_size = int(len(X)*0.2)
    all_idx = list(range(len(X)))
    test_idx = random.sample(all_idx, test_size)
    train_idx = list(set(all_idx).difference(set(test_idx)))
    train_x = X[train_idx]
    train_y = Y[train_idx]
    test_x = X[test_idx]
    test_y = Y[test_idx]
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    val_loss = []
    min_loss = test(test_dataloader, model, loss_fn)
    val_loss.append(min_loss)
    improvement = False
    epochs_since_improvement = 0
    if early_stop:
        torch.save({'model_state': model.state_dict()},
        'best_model.pth')
    for t in range(epochs):
        epoch_loss = 0
        model.train()
        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.to("cpu"), y.to("cpu")
            if x.shape[0] < 3:
                break
            pred = model(x)
            target = model.target(pred, y)
            loss = loss_fn(pred, target)
            epoch_loss += loss/len(train_dataloader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(epoch_loss)
        v_loss = test(test_dataloader, model, loss_fn)
        val_loss.append(v_loss)
        if early_stop:
            if v_loss <= min_loss:
                epochs_since_improvement = 0
                improvement = True
                torch.save({'model_state': model.state_dict()},
                'best_model.pth')
            else:
                epochs_since_improvement +=1
                if epochs_since_improvement > epochs/2:
                    break
                elif epochs_since_improvement > 3 and improvement:
                    break
        min_loss = min(min_loss, v_loss)
    print("EPOCH=",t, "last improvement: ", t-epochs_since_improvement)
    if not improvement:
        print("ACHTUNG NO IMPROVEMENT")
    print(val_loss)
    if early_stop:
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state'])
    print(min_loss)
    print("LAST LOSS bigger than MIN LOSS: ", val_loss[-1]>min_loss)
    print()
    return model, optimizer, train_loss, val_loss, improvement

def train_one(X, Y, model,loss_fn, optimizer, epochs):
    model.train()
    losses = []
    for t in range(epochs):
        y_pred = model(X)
        loss = loss_fn(y_pred, Y)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, optimizer, losses



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    losses = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        losses.append(loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", len(X))
    return losses

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
            target = model.target(pred, y)
            test_loss += loss_fn(pred, target).item()
    test_loss /= num_batches
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

def train_n_epochs(train_dataloader, test_dataloader, epochs, model, loss_fn, optimizer):
    train_loss=[]
    val_loss = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss_t = train(train_dataloader, model, loss_fn, optimizer)
        val_loss_t = test(test_dataloader, model, loss_fn)
        train_loss.append(np.mean(train_loss_t))
        val_loss.append(val_loss_t)
    return train_loss, val_loss, model

def single_training(k, val_part, X, Y, epochs, batch_size = 32, saved_file = "model3.pth"):
    idx = int(len(X)*val_part)
    x_train = torch.cat((X[:k*idx],X[(k+1)*idx:]), 0)
    y_train = torch.cat((Y[:k*idx],Y[(k+1)*idx:]), 0)
    x_test = X[k*idx: (k+1)*idx]
    y_test = Y[k*idx:(k+1)*idx]
    dim = len(list(y_test.shape))
    if dim  == 1:
        y_train = y_train.unsqueeze(-1)
        y_test = y_test.unsqueeze(-1)
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = Model4().to(device)
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)#torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9,0.999))
    loss_fn = nn.HuberLoss()
    if True:
        #model = load_checkpoint(model, saved_file)
        torch.save(model.state_dict(),"model4-old.pth")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)#torch.optim.NAdam(model.parameters(),lr=0.0005)#
    checkpoint = torch.load('model.pth')
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print(model)
    train_loss,val_loss, model = train_n_epochs(train_dataloader, test_dataloader, epochs, model, loss_fn, optimizer)
    return train_loss,val_loss, model, optimizer


def sampler(k, X, Y, epochs, batch_size, saved_model):
    train_loss = []
    val_loss = []
    min_loss= float('inf')
    for fold in range(k):
        print(f"FOLD {fold+1}\n\n-------------------------------")
        t_loss,v_loss, model, optimizer = single_training(fold, 1/k, X, Y, epochs, 32, saved_model)
        train_loss +=t_loss
        val_loss += v_loss
        print(v_loss[-1])
        if v_loss[-1] < min_loss:
            min_loss = v_loss[-1]
            torch.save(model.state_dict(),"best_model.pth")
    model = Model4().to(device)
    model = load_checkpoint(model, "best_model.pth")
    return train_loss,val_loss, model, optimizer




