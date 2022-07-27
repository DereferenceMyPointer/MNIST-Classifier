import torch
from torch import nn

"""
The purpose of this file is to test output spaces
for debugging.
"""

x = torch.rand(1, 28, 28)

x = nn.Conv2d(1, 5, 5)(x)
print(x.shape)
x = nn.MaxPool2d(kernel_size=3)(x)
print(x.shape)
x = nn.Conv2d(5, 16, 3)(x)
print(x.shape)
x = nn.MaxPool2d(kernel_size=2)(x)
print(x.shape)
x = nn.Conv2d(16, 27, 3)(x)
print(x.shape)
x = x.flatten()
print(x.shape)
x = nn.Linear(27, 512)(x)
print(x.shape)
x = nn.Linear(512, 27)(x)
print(x.shape)