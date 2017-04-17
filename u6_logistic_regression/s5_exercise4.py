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
        # your code here
        pass

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

iris = datasets.load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.4)
logi = LogisticRegressionOVR(len(np.unique(iris.target)), n_iter=500)
logi.fit(x_train, y_train)

y_hat = logi.predict(x_test[:5])
print(y_hat)
assert(np.array_equal(y_hat, np.array([0, 1, 1, 0, 2])))



