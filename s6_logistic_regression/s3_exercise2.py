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
        X = np.insert(X, 0, 1, axis=1)
        # your code here
        pass

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

X = np.array([[-2, 2],[-3, 0],[2, -1],[1, -4]])
y = np.array([1,1,0,0])
logi = MyLogisticRegression().fit(X, y)
y_hat = logi.predict(X)
print(y_hat)
assert(np.array_equal(y_hat, [1,0,1,0]))