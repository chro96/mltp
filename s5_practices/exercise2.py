from IPython import embed
import numpy as np


class MyLinearRegression(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.ones(X.shape[1])
        m = X.shape[0]
        for i in range(self.n_iter):
            # h(x)
            output = X.dot(self.w)
            # (y−h(x))
            errors = y - output
            if i % 10 == 0:
                print("Error: ", sum(errors ** 2))
                print("Weights: ", self.w)
                print("-------------------------")
            # θ := θ + (α/m)(y−h(x))X
            self.w += self.eta / m * errors.dot(X)
            return self.w

X = np.array([[0], [1], [2], [3]])
y = np.array([-1, 1, 3, 5])
regr = MyLinearRegression()
weight = regr.fit(X, y)
print(weight)
# 最初のイテレーションの値をテストする
assert(np.array_equal(weight, np.array([0.95, 1.05])))
