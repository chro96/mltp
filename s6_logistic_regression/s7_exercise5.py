from pdb import set_trace

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

np.random.seed(1)

class MyLogisticRegression(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.ones(X.shape[1])
        m = X.shape[0]

        for _ in range(self.n_iter):
            output = self._sigmoid(X.dot(self.w))
            errors = y - output
            self.w += self.eta / m * errors.dot(X)

            if i % 10 == 0:
                print("Error: ", sum(errors ** 2))
                print("Weights: ", self.w)
        return self

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        output = self._sigmoid(X.dot(self.w))
        return np.where(output >= .5, 1, 0)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

class LogisticRegressionOVR(object):
    """One vs Rest"""

    def __init__(self, num_classes, eta=0.1, n_iter=50):
        self.num_classes = num_classes
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.zeros((X.shape[1], self.num_classes))
        m = X.shape[0]

        for i in range(self.num_classes):
            # yを1と0だけにする
            y_copy = np.where(y == i, 1, 0)
            w = np.ones(X.shape[1])

            for j in range(self.n_iter):
                output = X.dot(w)
                errors = y_copy - self._sigmoid(output)
                w += self.eta / m * errors.dot(X)
                
                if j % 10 == 0:
                    print(sum(errors**2))
            self.w[:, i] = w

        return self

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.argmax(X.dot(self.w), axis=1)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

class LogisticRegressionSoftmax(object):
    def __init__(self, num_classes, eta=0.1, n_iter=50):
        self.num_classes = num_classes
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        y = self._one_hot(y)
        self.w = np.random.randn(X.shape[1], self.num_classes)
        m = X.shape[0]

        for i in range(self.n_iter):
            output = self._softmax(X.dot(self.w))
            errors = y - output
            self.w += self.eta / m * X.T.dot(errors)

            if i % 10 == 0:
                print(self._cross_entropy(y, output))
        return self

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.argmax(X.dot(self.w), axis=1)

    def _one_hot(self, y):
        # your code here
        pass

logi = LogisticRegressionSoftmax(3, n_iter=500)
one_hot = logi._one_hot(np.array([0,1,2]))
print(one_hot)
assert(np.array_equal(one_hot, np.array([[1,0,0],[0,1,0],[0,0,1]])))
