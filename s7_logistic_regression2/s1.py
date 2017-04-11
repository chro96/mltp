import numpy as np

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.autograd import Variable

BATCH_SIZE = 32
EPOCH = 10


def batch(tensor, batch_size):
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i+1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: (i+1) * batch_size])
        i += 1

class Estimator(object):

    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_f = loss

    def _fit(self, X_list, y_list):
        """
        train one epoch
        """
        loss_list = []
        acc_list = []
        for X, y in zip(X_list, y_list):
            y_v = Variable(torch.from_numpy(y), requires_grad=False)
            self.optimizer.zero_grad()

            y_pred = self.predict(X)
            loss = self.loss_f(y_pred, y_v)
            loss.backward()
            self.optimizer.step()

            ## for log
            loss_list.append(loss.data[0])
            classes = torch.topk(y_pred, 1)[1].data.numpy().flatten()
            acc = self._accuracy(classes, y)
            acc_list.append(acc)

        return sum(loss_list) / len(loss_list), sum(acc_list) / len(acc_list)

    def fit(self, X, y, batch_size=32, nb_epoch=10, validation_data=()):
        for t in range(1, nb_epoch + 1):
            loss, acc = self._fit(batch(X, batch_size), batch(y, batch_size))
            val_log = ''
            if validation_data:
                val_loss, val_acc = self.evaluate(validation_data[0], validation_data[1], batch_size)
                val_log = "- val_loss: %06.4f - val_acc: %06.4f" % (val_loss, val_acc)
            print("Epoch %s/%s loss: %06.4f - acc: %06.4f %s" % (t, nb_epoch, loss, acc, val_log))

    def evaluate(self, X, y, batch_size=32):
        y_pred = self.predict(X)

        y_v = Variable(torch.from_numpy(y), requires_grad=False)
        loss = self.loss_f(y_pred, y_v)

        classes = torch.topk(y_pred, 1)[1].data.numpy().flatten()
        acc = self._accuracy(classes, y)
        return (loss.data[0], acc)

    def _accuracy(self, y_pred, y):
        return sum(y_pred == y) / y.shape[0]

    def predict(self, X):
        X = Variable(torch.from_numpy(X).float())
        y_pred = self.model(X)
        return y_pred

    def predict_classes(self, X):
        return torch.topk(self.predict(X), 1)[1].data.numpy().flatten()


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)

## ======================= numpy =====================





X = np.array([[-2, 2],[-3, 0],[2, -1],[1, -4]])
y = np.array([1,1,0,0])

model = LogisticRegression(2, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

clf = Estimator(model)
clf.compile(optimizer=optimizer, loss=criterion)
clf.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=EPOCH)
print(clf.predict_classes(X))
# score, acc = clf.evaluate(X_test, y_test)
# print('Test score:', score)
# print('Test accuracy:', acc)

