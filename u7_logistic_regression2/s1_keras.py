from pdb import set_trace
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

iris = datasets.load_iris()
x, y = iris.data, iris.target
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes=num_classes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.4)
print("%s\n%s\n%s\n%s" % (x_train.shape, y_train.shape, x_test.shape, y_test.shape))

clf = Sequential()
clf.add(Dense(num_classes, input_dim=x.shape[1]))
clf.add(Activation('softmax'))

optimizer = SGD(lr=0.01)
clf.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
clf.fit(x_train, y_train, epochs=500, batch_size=32, validation_data=(x_test, y_test))
score, acc = clf.evaluate(x_test, y_test, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
