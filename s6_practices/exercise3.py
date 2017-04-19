from IPython import embed
import numpy as np

class MyLogisticRegression(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        # X = [[ 1 -2  2]
        #      [ 1 -3  0]
        #      [ 1  2 -1]
        #      [ 1  1 -4]]
        self.w = np.ones(X.shape[1])
        # self.w = [ 1.  1.  1.]
        m = X.shape[0]
        # m = [1 1 0 0]
        for i in range(self.n_iter):
            output = self._sigmoid(X.dot(self.w))
            errors = y - output
            self.w += self.eta / m * errors.dot(X)
            # θ:=θ+αm(y−h(x))X
            if i % 10 == 0:
                print("Error: ", sum(errors ** 2))
                print("Weights: ", self.w)
        return self

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        output = self._sigmoid(X.dot(self.w))
        # h(x)=g(Xθ)
        return np.where(output >= .5, 1, 0)

    def _sigmoid(self, x):
        # g(x)=1/(1+e**x)
        return 1 / (1 + np.exp(-x))

X = np.array([[-2, 2],[-3, 0],[2, -1],[1, -4]])
y = np.array([1,1,0,0])
logi = MyLogisticRegression().fit(X, y)
y_hat = logi.predict(X)
print(y_hat)
assert(np.array_equal(y_hat, [1,1,0,0]))
