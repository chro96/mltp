from pdb import set_trace
import sys 
sys.path.append('..')

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import torch
from torch import nn, from_numpy
from torch.utils.data import DataLoader, TensorDataset

from lib.lib import Estimator

batch_size = 32

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

iris = datasets.load_iris()
x, y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.4)
print("%s\n%s\n%s\n%s" % (x_train.shape, y_train.shape, x_test.shape, y_test.shape))

train = TensorDataset(from_numpy(x_train).float(), from_numpy(y_train))
train_loader = DataLoader(train, batch_size, shuffle=True)
test = TensorDataset(from_numpy(x_test).float(), from_numpy(y_test))
test_loader = DataLoader(test, batch_size, shuffle=False)

# train
model = LogisticRegression(x.shape[1], len(np.unique(y)))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()
clf = Estimator(model)
clf.compile(optimizer=optimizer, loss=criterion)
clf.fit(train_loader, nb_epoch=500, validation_data=test_loader)
score, acc, confusion = clf.evaluate(test_loader)
print('Test score:', score)
print('Test accuracy:', acc)
print(confusion[0])
print(confusion[1])
torch.save(model.state_dict(), 'pytorch_model.pth')

# Predict
model = LogisticRegression(x.shape[1], len(np.unique(y)))
model.load_state_dict(torch.load('pytorch_model.pth'))
clf = Estimator(model)
print(clf.predict_classes(from_numpy(x_test).float()))

