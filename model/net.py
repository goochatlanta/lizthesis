"""Defines the neural network, loss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import pdb
import math
import torchmetrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor




class CNN40(torch.nn.Module):

    def __init__(self):
        super(CNN40, self).__init__()
        # L1 ImgIn shape=(?, 602, 40, 1)
        # Conv -> (?, 602, 40, 32)
        # Pool -> (?, 301, 20, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.3)
            )
        # L2 ImgIn shape=(?, 48, 11, 125)
        # Conv      ->(?, 14, 14, 64)
        # Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.3)
            )
        # L3 ImgIn shape=(?, 7, 7, 64)
        # Conv ->(?, 7, 7, 128)
        # Pool ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.3)
            )

        # L4 FC 20x2x128 inputs -> 625 outputs
        self.hidden1 = nn.Linear(25*3*128, 5000) 
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # Second hidden layer
        self.hidden2 = nn.Linear(5000, 1000)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # Second hidden layer
        self.hidden3 = nn.Linear(1000, 250)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = nn.ReLU()
        # Third hidden layer
        self.hidden4 = nn.Linear(250, 75)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = nn.ReLU()
        self.hidden5 = nn.Linear(75,1)

    def forward(self, t1, t2):
        X = torch.cat((t1, t2), dim=1)
        X = X[None, :]
        X = X.permute(1,0,2, 3) # (batch size= 64, input_channels=1, signal_length=5)
        out = self.layer1(X)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        #Input to the first hidden layer
        X = self.hidden1(out)
        X = self.act1(X)
        # Second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # Third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # Fourth hidden layer
        X = self.hidden4(X)
        X = self.act4(X)
        # Fourth hidden layer
        X = self.hidden5(X)
        return X


class CNN40_20(torch.nn.Module):

    def __init__(self):
        super(CNN40_20, self).__init__()
        # L1 ImgIn shape=(?, 602, 40, 1)
        # Conv -> (?, 602, 40, 32)
        # Pool -> (?, 301, 20, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.15)
            )
        # L2 ImgIn shape=(?, 48, 11, 125)
        # Conv      ->(?, 14, 14, 64)
        # Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.15)
            )
        # L3 ImgIn shape=(?, 7, 7, 64)
        # Conv ->(?, 7, 7, 128)
        # Pool ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.15)
            )

        # L4 FC 20x2x128 inputs -> 625 outputs
        self.hidden1 = nn.Linear(38*3*128, 5000) 
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # Second hidden layer
        self.hidden2 = nn.Linear(5000, 1000)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # Second hidden layer
        self.hidden3 = nn.Linear(1000, 250)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = nn.ReLU()
        # Third hidden layer
        self.hidden4 = nn.Linear(250, 75)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = nn.ReLU()
        self.hidden5 = nn.Linear(75,1)

    def forward(self, t1, t2):
        X = torch.cat((t1, t2), dim=1)
        X = X[None, :]
        X = X.permute(1,0,2, 3) # (batch size= 64, input_channels=1, signal_length=5)
        out = self.layer1(X)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        #Input to the first hidden layer
        X = self.hidden1(out)
        X = self.act1(X)
        # Second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # Third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # Fourth hidden layer
        X = self.hidden4(X)
        X = self.act4(X)
        # Fourth hidden layer
        X = self.hidden5(X)
        return X


class CNN12(torch.nn.Module):

    def __init__(self):
        super(CNN12, self).__init__()
        # L1 ImgIn shape=(?, 602, 12, 1)
        # Conv -> (?, 602, 12, 32)
        # Pool -> (?, 301, 6, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # torch.nn.Dropout(0.15)
            )
        # L2 ImgIn shape=(?, 48, 11, 125)
        # Conv      ->(?, 14, 14, 64)
        # Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # torch.nn.Dropout(0.15)
            )
        # L3 ImgIn shape=(?, 7, 7, 64)
        # Conv ->(?, 7, 7, 128)
        # Pool ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # torch.nn.Dropout(0.15)
            )

        # L4 FC 20x2x128 inputs -> 625 outputs
        self.hidden1 = nn.Linear(25*1*128, 1000) 
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # Second hidden layer
        self.hidden2 = nn.Linear(1000, 250)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # Third hidden layer
        self.hidden3 = nn.Linear(250, 75)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(75,1)

    def forward(self, t1, t2):
        X = torch.cat((t1, t2), dim=1)
        X = X[None, :]
        X = X.permute(1,0,2, 3) # (batch size= 64, input_channels=1, signal_length=5)
        out = self.layer1(X)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        #Input to the first hidden layer
        X = self.hidden1(out)
        X = self.act1(X)
        # Second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # Third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # Fourth hidden layer
        X = self.hidden4(X)
        return X


class CNN3(torch.nn.Module):

    def __init__(self):
        super(CNN3, self).__init__()
        # L1 ImgIn shape=(?, 602, 12, 1)
        # Conv -> (?, 602, 12, 32)
        # Pool -> (?, 301, 6, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.1)
            )
        # L2 ImgIn shape=(?, 48, 11, 125)
        # Conv      ->(?, 14, 14, 64)
        # Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.1)
            )
        # L3 ImgIn shape=(?, 7, 7, 64)
        # Conv ->(?, 7, 7, 128)
        # Pool ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.1)
            )

        # L4 FC 20x2x128 inputs -> 625 outputs
        self.hidden1 = nn.Linear(151*1*64, 1000) 
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # Second hidden layer
        self.hidden2 = nn.Linear(1000, 250)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # Third hidden layer
        self.hidden3 = nn.Linear(250, 75)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(75,1)

    def forward(self, t1, t2):
        X = torch.cat((t1, t2), dim=1)
        X = X[None, :]
        X = X.permute(1,0,2, 3) # (batch size= 64, input_channels=1, signal_length=5)
        out = self.layer1(X)
        out = self.layer2(out)
        # out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        #Input to the first hidden layer
        X = self.hidden1(out)
        X = self.act1(X)
        # Second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # Third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # Fourth hidden layer
        X = self.hidden4(X)
        return X


class MLP(nn.Module):
    """# Got this MLP from 
    https://python-bloggers.com/2022/05/building-a-pytorch-binary-classification-multi-layer-perceptron-from-the-ground-up/
    """
    def __init__(self):
        super(MLP, self).__init__()
        # First hidden layer, input size is 160 x 94 (batch size (2) x (2 tensors at 40 MFCCs) by 94 frames)
        # when I increase batch_size, then the input size is going to increase
        self.hidden1 = nn.Linear(4824, 1000) 
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # Second hidden layer
        self.hidden2 = nn.Linear(1000, 250)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # Second hidden layer
        self.hidden3 = nn.Linear(250, 75)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = nn.ReLU()
        # Third hidden layer
        self.hidden4 = nn.Linear(75,1)
        # xavier_uniform_(self.hidden4.weight)
        # self.act4 = nn.Sigmoid()
        

    def forward(self, t1, t2):
        # input two tensors that are each 40 mfcc x 94 frames 
        # https://discuss.pytorch.org/t/multiple-numeric-inputs-in-neural-network-for-classification/147795/6
        # concatenate the inputs --> 80 x 94 then reshape to (batch size x 7520)
        X = torch.cat((t1, t2), dim=1)
        X = torch.reshape(X, (t1.shape[0], 2*t1.shape[1]*t1.shape[2]))
        #Input to the first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # Second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # Third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # Fourth hidden layer
        X = self.hidden4(X)
        # Sigmoid
        # X = self.act4(X)
        return X
        

# https://github.com/ldeecke/gmm-torch gmm.GaussianMixture(..)


class LR(nn.Module):
    """https://www.analyticsvidhya.com/blog/2021/07/perform-logistic-regression-with-pytorch-seamlessly/"""
    def __init__(self):
        super(LR,self).__init__()
        self.layer1=nn.Linear(4824,100)
        self.layer2=nn.Linear(100,1)

    def forward(self, t1, t2):
        # input two tensors that are each 13 mfcc x 94 frames 
        # concatenate the inputs --> 188 x 13 then reshape to (batch size x 7520) or 2444 (188 x 13)
        
        # def reduce(signal):
        #     signal = signal[:, :, 0:13]
        #     return signal

        # t1 = reduce(t1) 
        # t2 = reduce(t2)

        X = torch.cat((t1, t2), dim=1)
        X = torch.reshape(X, (t1.shape[0], 2*t1.shape[1]*t1.shape[2]))
        X=self.layer1(X)
        X=self.layer2(X)
        return X



def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    # num_examples = outputs.size()[0]
    # return -torch.sum(outputs[range(num_examples), labels])/num_examples
    criterion = nn.BCEWithLogitsLoss()
    # output_bool = outputs.type(torch.bool)
    return criterion(outputs, labels)

def sigmoid(x):
        return 1 / (1 + math.exp(-x))


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all audio.

    For Logisitic regression this threshold is the only hyperparameter. 

    Args:
        outputs: (np.ndarray) dimension batch_size x 1 - logit output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    predictions = []

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    for output in outputs:
        pred = sigmoid(output)
        if pred > 0.3:
            predictions.append(1.0)
        else: 
            predictions.append(0.0)

    predictions = np.array(predictions)
    return np.sum(predictions==labels)/float(labels.size)
    

def f1(outputs, labels):
    predictions = []

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    for output in outputs:
        pred = sigmoid(output)
        if pred > 0.3:
            predictions.append(1.0)
        else: 
            predictions.append(0.0)

    predictions = np.array(predictions)

    return f1_score(labels, predictions)

def tn(outputs, labels):
    predictions = []

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    for output in outputs:
        pred = sigmoid(output)
        if pred > 0.3:
            predictions.append(1.0)
        else: 
            predictions.append(0.0)

    predictions = np.array(predictions)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    return tn

def fp(outputs, labels):
    predictions = []

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    for output in outputs:
        pred = sigmoid(output)
        if pred > 0.3:
            predictions.append(1.0)
        else: 
            predictions.append(0.0)

    predictions = np.array(predictions)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    return fp

def fn(outputs, labels):
    predictions = []

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    for output in outputs:
        pred = sigmoid(output)
        if pred > 0.3:
            predictions.append(1.0)
        else: 
            predictions.append(0.0)

    predictions = np.array(predictions)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    return fn

def tp(outputs, labels):
    predictions = []

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    for output in outputs:
        pred = sigmoid(output)
        if pred > 0.3:
            predictions.append(1.0)
        else: 
            predictions.append(0.0)

    predictions = np.array(predictions)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    return tp

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    'f1 score': f1,
    'tn': tn,
    'fp': fp,
    'fn': fn,
    'tp': tp
    # could add more metrics such as accuracy for each token type
}
