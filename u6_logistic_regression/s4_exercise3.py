import numpy as np

class MyLogisticRegression(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.ones(X.shape[1])
        m = X.shape[0]

        for _ in range(self.n_iter):
            # your code
            pass

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

X = np.array([[-2, 2],[-3, 0],[2, -1],[1, -4]])
y = np.array([1,1,0,0])
logi = MyLogisticRegression().fit(X, y)
y_hat = logi.predict(X)
print(y_hat)
assert(np.array_equal(y_hat, [1,1,0,0]))