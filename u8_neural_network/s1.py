from pdb import set_trace
import sys 
sys.path.append('..')

import numpy as np

import torch
from torch import nn, from_numpy
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from lib.lib import Estimator

batch_size = 32
image_dim = 1*28*28

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(image_dim))
])

train_dataset = MNIST(root='../data', train=True, transform=transform, download=True)
test_dataset = MNIST(root='../data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

num_classes = len(set(train_dataset.train_labels.numpy()))

# train
model = NeuralNetwork(image_dim, 500, num_classes)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
clf = Estimator(model)
clf.compile(optimizer=optimizer, loss=criterion)
clf.fit(train_loader, nb_epoch=5, validation_data=test_loader)
score, acc, confusion = clf.evaluate(test_loader)
print('Test score:', score)
print('Test accuracy:', acc)
print(confusion[0])
print(confusion[1])


