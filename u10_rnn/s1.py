from pdb import set_trace
import sys 
sys.path.append('..')

import numpy as np
import re
from glob import glob
import unicodedata
import string

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing import sequence

import torch
from torch import nn, from_numpy
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from lib.lib import Estimator
from models import RNN, GRU, LSTM

np.random.seed(1337)
np.set_printoptions(precision=6)

MAX_LEN = 20
EMBEDDING_SIZE = 64
BATCH_SIZE = 32
EPOCH = 20

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )

def one_hot(s):
    vec = np.zeros(n_letters)
    index = all_letters.index(s)
    vec[index] = 1
    return vec

def string2vec(string):
    return [one_hot(s) for s in string]


all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)

categories = [f.split('/')[-1].split('.')[0] for f in glob('../data/names/*.txt')]
le = LabelEncoder().fit(categories)
class_size = len(categories)

x = []
y = []
for category in categories:
    filename = '../data/names/%s.txt' % (category)
    lines = [unicodeToAscii(line) for line in open(filename).readlines()]
    x += [string2vec(line) for line in lines]
    encoded = le.transform([category])[0]
    y += [encoded for _ in range(len(lines))]

x = sequence.pad_sequences(x, maxlen=MAX_LEN)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)
print("%s\n%s\n%s\n%s" % (x_train.shape, y_train.shape, x_test.shape, y_test.shape))

train = TensorDataset(from_numpy(x_train).float(), from_numpy(y_train))
test = TensorDataset(from_numpy(x_test).float(), from_numpy(y_test))
train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

model = LSTM(n_letters, EMBEDDING_SIZE, class_size)
# model = RNN(n_letters, EMBEDDING_SIZE, class_size)
clf = Estimator(model)
clf.compile(optimizer=torch.optim.Adam(model.parameters(), lr=0.002),
            loss=nn.CrossEntropyLoss())
clf.fit(train_loader, nb_epoch=EPOCH, validation_data=test_loader)
score, acc, confusion = clf.evaluate(test_loader)
print('Test score:', score)
print('Test accuracy:', acc)
print(confusion[0])
print(confusion[1])
