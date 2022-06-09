import torch
from torch import nn
from torch.nn.functional import normalize
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class PreProcessLayer():
    def __init__(self):
        self.conv = nn.functional.conv2d
    def process(self, x, filters, stride = 1, padding = 0):
        filters = torch.Tensor(filters).unsqueeze(1)
        return self.conv(x, filters, stride = stride, padding = padding)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(),
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
            nn.Dropout(p=0.5),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Dropout(p=0.5),
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
            nn.Dropout(p=0.6),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(p=0.6),
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
            nn.Dropout(p=0.6),
            nn.Linear(57, 4),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Sparse(nn.Module):
    def __init__(self):
        super(Sparse, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(42, 96),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(96, 72),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(72, 6),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(6, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Big(nn.Module):
    def __init__(self):
        super(Big, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(42, 72),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(72, 12),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(6, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class LinModel(nn.Module):
    def __init__(self):
        super(LinModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(42, 12)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(12, 4)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        print(out)
        print(self.linear3.weight)
        out = self.linear3(out)
        print(out)
        print()
        return out

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.flatten = nn.Flatten()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,4))
        self.relu2 = nn.ReLU()
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.flatten(out)
        out = out.view(-1,32)
        out = self.linear(out)
        return out

    def traceback(self, x):
        out_cnn = self.cnn1(x)
        out_relu = self.relu1(out_cnn)
        out_pool = self.maxpool1(out_relu)
        rev = reverse_lin(self.linear, out_pool)
        rev = reverse_maxpool(self.maxpool1, out_relu, rev.reshape(out_pool.shape))
        rev = reverse_conv(self.cnn1, x, rev)
        return rev


class CNN2Model(nn.Module):
    def __init__(self):
        super(CNN2Model, self).__init__()
        self.flatten = nn.Flatten()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,4))
        self.relu2 = nn.ReLU()
        self.linear1 = nn.Linear(32, 8)
        self.dropout = nn.Dropout(p=0.25)
        self.linear2 = nn.Linear(8, 1)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.flatten(out)
        out = out.view(-1,32)
        out = self.linear1(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out

    def traceback(self, x):
        out_cnn = self.cnn1(x)
        out_relu = self.relu1(out_cnn)
        out_pool = self.maxpool1(out_relu)
        rev = reverse_lin(self.linear, out_pool)
        rev = reverse_maxpool(self.maxpool1, out_relu, rev.reshape(out_pool.shape))
        rev = reverse_conv(self.cnn1, x, rev)
        return rev

class CNN3Model(nn.Module):
    def __init__(self):
        super(CNN3Model, self).__init__()
        self.flatten = nn.Flatten()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1)
        self.relu1 = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=(3,4))
        self.relu2 = nn.ReLU()
        self.linear1 = nn.Linear(32, 8)
        self.dropout = nn.Dropout(p=0.25)
        self.linear2 = nn.Linear(8, 1)

    def forward(self, x):
        out = self.cnn1(x)
        #out = self.relu1(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = out.view(-1,32)
        out = self.linear1(out)
        #out = self.relu2(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out

    def traceback(self, x):
        out_cnn = self.cnn1(x)
        out_relu = self.relu1(out_cnn)
        out_pool = self.avgpool(out_relu)
        rev = reverse_lin(self.linear, out_pool)
        rev = reverse_maxpool(self.maxpool1, out_relu, rev.reshape(out_pool.shape))
        rev = reverse_conv(self.cnn1, x, rev)
        return rev

class CNN4Model(nn.Module):
    def __init__(self):
        super(CNN4Model, self).__init__()
        self.flatten = nn.Flatten()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1)
        self.prelu1 = nn.PReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=(3,4))
        self.prelu2 = nn.PReLU()
        self.linear1 = nn.Linear(32, 8)
        self.dropout = nn.Dropout(p=0.25)
        self.linear2 = nn.Linear(8, 1)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.prelu1(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = out.view(-1,32)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.prelu2(out)
        out = self.linear2(out)
        return out

    def traceback(self, x):
        out_cnn = self.cnn1(x)
        out_relu = self.relu1(out_cnn)
        out_pool = self.avgpool(out_relu)
        rev = reverse_lin(self.linear, out_pool)
        rev = reverse_maxpool(self.maxpool1, out_relu, rev.reshape(out_pool.shape))
        rev = reverse_conv(self.cnn1, x, rev)
        return rev


class CNN5Model(nn.Module):
    def __init__(self):
        super(CNN5Model, self).__init__()
        self.flatten = nn.Flatten()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1)
        self.prelu1 = nn.PReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=(2,2), stride = (1,2))
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1)
        self.prelu2 = nn.PReLU()
        self.linear1 = nn.Linear(32, 8)
        self.dropout = nn.Dropout(p=0.25)
        self.linear2 = nn.Linear(8, 1)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.avgpool(out)
        out = self.prelu1(out)
        out = self.cnn2(out)
        out = self.flatten(out)
        out = out.view(-1,32)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.prelu2(out)
        out = self.linear2(out)
        return out

    def traceback(self, x):
        out_cnn = self.cnn1(x)
        out_relu = self.relu1(out_cnn)
        out_pool = self.avgpool(out_relu)
        rev = reverse_lin(self.linear, out_pool)
        rev = reverse_maxpool(self.maxpool1, out_relu, rev.reshape(out_pool.shape))
        rev = reverse_conv(self.cnn1, x, rev)
        return rev

class CNN6Model(nn.Module):
    def __init__(self):
        super(CNN6Model, self).__init__()
        self.flatten = nn.Flatten()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1)
        self.prelu1 = nn.PReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=(2,2), stride = (1,2))
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1)
        self.prelu2 = nn.PReLU()
        self.linear1 = nn.Linear(32, 8)
        self.dropout = nn.Dropout(p=0.25)
        self.linear2 = nn.Linear(8, 1)

    def target(self, prediction, reward):
        return reward

    def forward(self, x):
        out = self.cnn1(x)
        out = self.prelu1(out)
        out = self.avgpool(out)
        out = self.cnn2(out)
        out = self.flatten(out)
        out = out.view(-1,32)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.prelu2(out)
        out = self.linear2(out)
        return out

    def traceback(self, x):
        out_cnn = self.cnn1(x)
        out_relu = self.relu1(out_cnn)
        out_pool = self.avgpool(out_relu)
        rev = reverse_lin(self.linear, out_pool)
        rev = reverse_maxpool(self.maxpool1, out_relu, rev.reshape(out_pool.shape))
        rev = reverse_conv(self.cnn1, x, rev)
        return rev

class CNN7Model(nn.Module):
    def __init__(self):
        super(CNN7Model, self).__init__()
        self.flatten = nn.Flatten()
        
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1)
        self.prelu1 = nn.PReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=(3,3), stride = 1)
        
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=2, stride=1)
        self.prelu2 = nn.PReLU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=(2,3))
        
        self.dropout = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(64, 8)
        self.prelu3 = nn.PReLU()
        self.linear2 = nn.Linear(8, 1)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.prelu1(out)
        out = self.avgpool1(out)
        out = self.cnn2(out)
        out = self.prelu2(out)
        out = self.avgpool2(out)
        out = self.flatten(out)
        out = out.view(-1,64)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.prelu3(out)
        out = self.linear2(out)
        return out

    def traceback(self, x):
        out_cnn = self.cnn1(x)
        out_relu = self.relu1(out_cnn)
        out_pool = self.avgpool(out_relu)
        rev = reverse_lin(self.linear, out_pool)
        rev = reverse_maxpool(self.maxpool1, out_relu, rev.reshape(out_pool.shape))
        rev = reverse_conv(self.cnn1, x, rev)
        return rev


class CNN8Model(nn.Module):
    def __init__(self):
        super(CNN8Model, self).__init__()
        self.flatten = nn.Flatten()
        
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=2, stride=2, padding = (0, 1))
        self.prelu1 = nn.PReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=(2,2), stride = 1)
        
        self.cnn2 = nn.Conv2d(in_channels=30, out_channels=120, kernel_size=2, stride=1)
        self.prelu2 = nn.PReLU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1,2))
        
        self.dropout = nn.Dropout(p=0.4)
        self.linear1 = nn.Linear(120, 8)
        self.prelu3 = nn.PReLU()
        self.linear2 = nn.Linear(8, 1)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.prelu1(out)
        out = self.avgpool1(out)
        out = self.cnn2(out)
        out = self.prelu2(out)
        out = self.avgpool2(out)
        out = self.flatten(out)
        out = out.view(-1,120)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.prelu3(out)
        out = self.linear2(out)
        return out

    def traceback(self, x):
        out_cnn = self.cnn1(x)
        out_relu = self.relu1(out_cnn)
        out_pool = self.avgpool(out_relu)
        rev = reverse_lin(self.linear, out_pool)
        rev = reverse_maxpool(self.maxpool1, out_relu, rev.reshape(out_pool.shape))
        rev = reverse_conv(self.cnn1, x, rev)
        return rev

class CNN9Model(nn.Module):
    def __init__(self):
        super(CNN9Model, self).__init__()
        self.flatten = nn.Flatten()
        
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=2, stride=2, padding = (0, 1))
        self.prelu1 = nn.PReLU()
        
        self.cnn2 = nn.Conv2d(in_channels=30, out_channels=160, kernel_size=(3,4), stride=1)
        self.prelu2 = nn.PReLU()
        
        self.dropout = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(160, 40)
        self.dropout2 = nn.Dropout(p=0.3)
        self.prelu3 = nn.PReLU()
        self.linear2 = nn.Linear(40, 8)
        self.prelu4 = nn.PReLU()
        self.linear3 = nn.Linear(8, 1)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.prelu1(out)
        out = self.cnn2(out)
        out = self.prelu2(out)
        out = self.flatten(out)
        out = out.view(-1,160)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.prelu3(out)
        out = self.linear2(out)
        out = self.prelu4(out)
        out = self.linear3(out)
        return out

    def traceback(self, x):
        out_cnn = self.cnn1(x)
        out_relu = self.relu1(out_cnn)
        out_pool = self.avgpool(out_relu)
        rev = reverse_lin(self.linear, out_pool)
        rev = reverse_maxpool(self.maxpool1, out_relu, rev.reshape(out_pool.shape))
        rev = reverse_conv(self.cnn1, x, rev)
        return rev


class CNNPreProcessModel(nn.Module):
    def __init__(self):
        super(CNNPreProcessModel, self).__init__()
        self.preprocess = PreProcessLayer()
        diag = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])/4
        diag2 = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])/4
        row = np.array([[1,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]])/4
        row2 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,1]])/4
        col = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]])/4
        col2 = np.array([[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1]])/4
        self.filters = np.array([diag, diag2, row, row2, col, col2])
        self.flatten = nn.Flatten()
        self.cnn1 = nn.Conv2d(in_channels=6, out_channels=120, kernel_size=(1,2), stride=1)
        self.prelu1 = nn.PReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=(2,2), stride = 1)
        self.prelu2 = nn.PReLU()
        self.cnn2 = nn.Conv2d(in_channels=120, out_channels= 32, kernel_size=2, stride=1)
        self.dropout = nn.Dropout(p=0.25)
        self.linear1 = nn.Linear(32, 8)
        self.prelu3 = nn.PReLU()
        self.linear2 = nn.Linear(8, 1)

    def forward(self, x):
        out = self.preprocess.process(x, self.filters)
        out = self.cnn1(out)
        out = self.prelu1(out)
        out = self.avgpool(out)
        out = self.cnn2(out)
        out = self.prelu2(out)
        out = out.view(-1,32)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.prelu3(out)
        out = self.linear2(out)
        return out

def reverse_conv(C, x, future):
    d, h, w = x.shape
    contribution = torch.zeros((h, w))
    n_filt, _, kernel_h, kernel_w = C.weight.shape
    bias = C.bias
    conv = C(x)
    for n, f in enumerate(future):
        for y in range(len(f)):
            for i in range(len(f[y])):
                contr = f[y][i]
                w = C.weight[n]
                b = C.bias[n]
                if conv[n][y][i] != 0:
                    frac = contr/conv[n][y][i]
                    res = w*x.squeeze()[y: y+kernel_h, i : i + kernel_w]*frac
                    contribution[y: y+kernel_h, i : i + kernel_w] += res.reshape((kernel_h, kernel_w))
    return contribution

def reverse_maxpool(pool, x, future):
    out = pool(x)
    idx = x == out
    contribution = idx*future
    return contribution



def reverse_lin(linear, x):
    new_x = x.reshape(linear.weight.shape)
    w = linear.weight
    contribution = w*new_x
    return contribution


class PolicyModel(nn.Module):
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.flatten = nn.Flatten()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride = (1,2))
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=2, stride=1)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(16, 7)
        self.norm = nn.BatchNorm1d(7, affine=False)

    def target(self, prediction, reward):
        reward = reward.reshape(prediction.shape)
        reward = reward.softmax(dim = 1)
        return reward
    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.cnn2(out)
        out = self.flatten(out)
        out = out.view(-1,16)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.norm(out)
        return out

class PolicyModel2(nn.Module):
    def __init__(self):
        super(PolicyModel2, self).__init__()
        self.flatten = nn.Flatten()
        self.cnn1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride = (1,2))
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=2, stride=1)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.0)
        self.linear1 = nn.Linear(16, 7)
        #self.norm = nn.BatchNorm1d(7, affine=False)

    def target(self, prediction, reward):
        reward = reward.reshape(prediction.shape)
        reward = reward.softmax(dim = 1)
        return reward
    def input_tensor(self, x):
        my = x
        his = x.neg().relu()
        if x.shape[0] == 1:
            out = torch.cat((my, his), 0)
            out = out.reshape(2,6,7)
            return out
        else:
            return torch.cat((my,his), 1)

    def forward(self, x):
        out = self.input_tensor(x)
        out = self.cnn1(out)
        #out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.cnn2(out)
        #out = self.batchnorm2(out)
        out = self.flatten(out)
        out = out.view(-1,16)
        out = self.relu2(out)
        #out = self.dropout(out)
        out = self.linear1(out)
        #out = self.norm(out)
        return out

class PolicyModel3(nn.Module):
    def __init__(self):
        super(PolicyModel3, self).__init__()
        self.flatten = nn.Flatten()
        self.cnn1 = nn.Conv2d(in_channels=2, out_channels=49, kernel_size=4, stride=1)
        self.batchnorm1 = nn.BatchNorm2d(49)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(in_channels=49, out_channels=49, kernel_size=(2,2), stride = (1,2))
        self.batchnorm2 = nn.BatchNorm2d(49)
        self.relu2 = nn.ReLU()
        self.cnn3 = nn.Conv2d(in_channels=49, out_channels=49, kernel_size=2, stride = 1)
        self.batchnorm3 = nn.BatchNorm2d(49)
        self.relu3 = nn.ReLU()
        self.linear1 = nn.Linear(49, 7)

    def target(self, prediction, reward):
        reward = reward.reshape(prediction.shape)
        reward = reward.softmax(dim = 1)
        return reward
    def input_tensor(self, x):
        my = x
        his = x.neg().relu()
        if x.shape[0] == 1:
            out = torch.cat((my, his), 0)
            out = out.reshape(1,2,6,7)
            return out
        else:
            batch_size = my.shape[0]
            my = my.reshape((batch_size, 1, 6, 7))
            his = his.reshape((batch_size, 1, 6, 7))
            return torch.cat((my,his), 1)

    def forward(self, x):
        out = self.input_tensor(x)
        out = self.cnn1(out)
        out = self.batchnorm1(out)
        out = self.relu(out)

        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu2(out)

        out = self.cnn3(out)
        out = self.batchnorm3(out)
        out = self.flatten(out)
        out = out.view(-1,49)
        out = self.relu3(out)

        out = self.linear1(out)
        return out

#model = CNNModel()
#checkpoint = torch.load('test-model.pth')
#model.load_state_dict(checkpoint['model_state'])
#x = torch.Tensor(np.array([[[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,-1],[-1,0,1,1,1,1,-1]]]))
#7 10, 12, 23
#self.layers = nn.ModuleList(modules=[nn.Linear(,),  nn.Conv2d(,,,), ..])
#self.shapes = [(,,), (,,,),  ..]

