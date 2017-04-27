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

np.set_printoptions(precision=6)

batch_size = 32
image_dim = 1*28*28
EPOCH = 10

class CNN(nn.Module):

    def __init__(self, image_scale, num_colors, num_classes, kernel=5, stride=1):
        super(CNN, self).__init__()

        # Strideが1の時のみ
        padding = int((kernel - 1) / 2)

        self.conv1 = nn.Conv2d(num_colors, 6, kernel, stride=stride, padding=padding)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(6, 16, kernel, stride=stride, padding=padding)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        num_pool = 2
        dim = int(image_scale / (2 ** num_pool))

        self.fc1   = nn.Linear(16 * dim * dim, 120)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2   = nn.Linear(120, 84)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        # [32, 1, 28, 28]
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # [32, 6, 14, 14]

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # [32, 16, 7, 7]

        x = x.view(x.size(0), -1)
        # [32, 16*7*7]

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = MNIST(root='../data', train=True, transform=transform, download=True)
test_dataset = MNIST(root='../data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

num_colors = train_dataset[0][0].size(0)
num_classes = len(set(train_dataset.train_labels.numpy()))

# train
model = CNN(28, num_colors, num_classes)
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


