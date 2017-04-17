import numpy as np

"""
このファイルはExerciseの答えだけ載せたものです。
"""

# Exercise 1

def _sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

# Exercise 2

def predict(self, X):
    X = np.insert(X, 0, 1, axis=1)
    output = self._sigmoid(X.dot(self.w))
    return np.where(output >= .5, 1, 0)

# Exercise 3

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

# Exercise 4

def predict(self, X):
    X = np.insert(X, 0, 1, axis=1)
    return np.argmax(X.dot(self.w), axis=1)

# Exercise 5

def _one_hot(self, y):
    m = y.shape[0]
    y_new = np.zeros((m, self.num_classes))
    y_new[np.arange(m), y] = 1
    return y_new

# Exercise 6

def _softmax1(self, x):
    """1サンプルづつ"""
    x -= np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

# Exercise 7

def _softmax(self, x):
    """全サンプル"""
    x -= np.max(x, axis=1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

# Exercise 8

def _cross_entropy(self, y, y_hat):
    return -np.sum(y * np.log(y_hat))


