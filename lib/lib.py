from pdb import set_trace
import numpy as np

import torch
from torch import nn, from_numpy
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

"""
Helper functions and classes for PyTorch
"""


def confusion_matrix(y_true, y_pred):
    n_classes = len(set(np.unique(y_true)) | set(np.unique(y_pred)))
    CM = np.zeros((n_classes, n_classes)).astype(int)
    for true, pred in zip(y_true.astype(int), y_pred.astype(int)):
        CM[true, pred] += 1
    return CM, CM / np.sum(CM, axis=1, keepdims=True)

def get_weight(data):
    """
    return weight for unbalanced data.
    data is a PyTorch dataset type
    """
    count = np.bincount([d[1] for d in data])
    return torch.from_numpy(len(data) / (len(count) * count)).float()

def train_test_split(data, test_size=0.2):
    """For PyTorch"""
    
    size = len(data)
    p = np.random.permutation(size)
    index = int(np.ceil(size * test_size))
    return p[index:], p[:index]

class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class Estimator(object):
    """PyTorchでMulticlass Classificationをする時のboilerplate code"""

    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_f = loss

    def _fit(self, loader, train=True, confusion=False):
        """train one epoch"""
        # train mode
        self.model.train()

        loss_list = []
        y_pred = np.array([])
        target = np.array([])

        for X, y in loader:
            outputs = self.predict(X)
            loss = self.loss_f(outputs, Variable(y, requires_grad=False))
            ## for log
            loss_list.append(loss.data[0])
            y_pred = np.concatenate((y_pred, torch.topk(outputs, 1)[1].data.numpy().flatten()))
            target = np.concatenate((target, y.numpy()))

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        loss = sum(loss_list) / len(loss_list)
        acc = sum(y_pred == target) / len(y_pred)

        return (loss, acc, confusion_matrix(target, y_pred)) if confusion else (loss, acc)

    def fit(self, train_loader, nb_epoch=10, validation_data=()):
        print("train...")
        for t in range(1, nb_epoch + 1):
            loss, acc = self._fit(train_loader)
            val_log = ''
            if validation_data:
                val_loss, val_acc = self._fit(validation_data, False)
                val_log = "- val_loss: %06.4f - val_acc: %06.4f" % (val_loss, val_acc)
            print("Epoch %s/%s loss: %06.4f - acc: %06.4f %s" % (t, nb_epoch, loss, acc, val_log))

    def evaluate(self, test_loader):
        return self._fit(test_loader, False, confusion=True)

    def predict(self, X):
        """X: PyTorch Tensor"""
        X = Variable(X)
        return self.model(X)

    def predict_classes(self, X):
        """X: PyTorch Tensor"""
        # eval mode
        self.model.eval()
        return torch.topk(self.predict(X), 1)[1].data.numpy().flatten()


