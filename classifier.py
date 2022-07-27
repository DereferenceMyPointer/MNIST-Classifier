from curses import A_ALTCHARSET
from pickletools import optimize
from re import A
import torch
from torch import nn, sigmoid
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from zmq import device
import numpy as np

"""
A relatively simple demo of deep learning methods.
Compares a convolutional model against a basic dense multilayer perceptron
using the MNIST handwritten letters database. Visualization is of
the convergence of network loss.
"""

# Defines the basic network
# All layers are linear and use ReLU activation
class LinearClassifier(nn.Module):
    def __init__(self, dim):
        super(LinearClassifier, self).__init__()
        self.input_size = dim ** 2
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 27)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)

# Defines the convolutional network
# This iteration uses three convolutions, max pooling, and 4 hidden dense layers.
class ConvolutionalClassifier(nn.Module):
    def __init__(self):
        super(ConvolutionalClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Conv2d(1, 5, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(5, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 27, 3),
            nn.Flatten(),
            nn.Linear(27, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 27),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

# Defines training loop
def train(model, optimizer, data_loader, epochs, loss, device):
    loss_history = []
    for epoch in range(epochs):
        z = enumerate(data_loader)
        for batch, (x, y) in z:
            pred = model(x.to(device))
            l = loss(pred, y.to(device))

            optimizer.zero_grad()
            l.backward()    
            optimizer.step()

            if batch % 100 == 0:
                print(l.item())
                loss_history.append(l.item())
    return loss_history

# Defines evaluation loop
def evaluate(model, test_loader, loss, device):
    with torch.no_grad():
        wrong = []
        size = len(test_loader.dataset)
        batches = len(test_loader)
        test_loss, correct = 0, 0
        for X, y in test_loader:
            pred = model(X.to(device))
            test_loss += loss(pred, y.to(device)).item()
            num_correct = (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()
            correct += num_correct
            #if num_correct < 1: wrong.append(y)
        return (correct / size, wrong)

# Setup and evaluation for a specific case
if torch.cuda.is_available():
    device = 'cuda'
else: device = 'cpu'

# Set data, initialize models
training_data = datasets.EMNIST(root='data', split='letters', train=True, download=True, transform=ToTensor())
test_data = datasets.EMNIST(root='data', split='letters', train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_data, batch_size=64)
data_loader = DataLoader(training_data, batch_size=64, shuffle=True)
class_model = LinearClassifier(28).to(device)
conv_model = ConvolutionalClassifier().to(device)

# Run training cycle
epochs = 100
learn_rate = 3e-3
class_optimizer = torch.optim.SGD(class_model.parameters(), lr=learn_rate)
conv_optimizer = torch.optim.SGD(conv_model.parameters(), lr=learn_rate)

# Evaluate models (ignore commented lines)
#class_loss = train(class_model, class_optimizer, data_loader, epochs, nn.CrossEntropyLoss(), device)
#class_correct, class_avg = evaluate(class_model, test_loader, nn.CrossEntropyLoss(), device)
conv_loss = train(conv_model, conv_optimizer, data_loader, epochs, nn.NLLLoss(), device)
conv_correct, conv_wrong = evaluate(conv_model, test_loader, nn.NLLLoss(), device)
#print(conv_wrong)

# Results 
#clm, = plt.plot(range(len(class_loss)), class_loss)
#clm.set_label(f'Linear model ({class_correct * 10000 // 100}% correct)')
cm, = plt.plot(range(len(conv_loss)), conv_loss)
cm.set_label(f'Convolutional model ({conv_correct* 10000 // 1 / 100}% correct)')
plt.xlabel('Iterations (hundreds)')
plt.ylabel('Error')
plt.title('Error over time')
plt.legend()
plt.show()

#plt.imshow(img.squeeze())
#plt.show()
