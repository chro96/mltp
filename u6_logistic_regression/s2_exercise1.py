import numpy as np

class MyLogisticRegression(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.ones(X.shape[1])
        m = X.shape[0]
        return self

    def predict(self, X):
        return np.insert(X, 0, 1, axis=1).dot(self.w)

    def _sigmoid(self, x):
        # your code here
        pass

logi = MyLogisticRegression()
sigmoid = (logi._sigmoid(np.array([3, -1, 0])))
assert(np.allclose(sigmoid, np.array([0.952574126822, 0.26894142137, 0.5])))