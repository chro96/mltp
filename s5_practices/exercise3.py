from IPython import embed
import numpy as np
from sklearn.cross_validation import train_test_split

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
        return self

    def predict(self, X):
        # dot メソッドは内積
        # h(X)=Xθ
        return np.insert(X, 0, 1, axis=1).dot(self.w)

X = np.array([[0], [1], [2], [3]])
y = np.array([-1, 1, 3, 5])
X_test = np.array([[4],[5]])
regr = MyLinearRegression(n_iter=500).fit(X, y)
y_hat = regr.predict(X_test)
print(y_hat)
assert(np.allclose(y_hat, np.array([7,9])))
