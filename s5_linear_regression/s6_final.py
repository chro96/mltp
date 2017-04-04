from pdb import set_trace
import numpy as np
from bokeh.plotting import figure, output_file, show

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

class MyLinearRegression(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.ones(X.shape[1])
        m = X.shape[0]

        for i in range(self.n_iter):
            output = X.dot(self.w)
            errors = y - output
            if i % 10 == 0:
                print("Error: ", sum(errors ** 2))
                print("Weights: ", self.w)
            self.w += self.eta / m * errors.dot(X)
        return self

    def predict(self, X):
        return np.insert(X, 0, 1, axis=1).dot(self.w)

    def score(self, X, y):
        return 1 - sum((self.predict(X) - y)**2) / sum((y - np.mean(y))**2)

class MyLinearRegressionSGD(object):
    def __init__(self, eta=0.1, n_iter=50, shuffle=True):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.ones(X.shape[1])

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            for x, target in zip(X, y):
                output = x.dot(self.w)
                self.w += self.eta * (target - output) * x
        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def predict(self, X):
        return np.insert(X, 0, 1, axis=1).dot(self.w)

    def score(self, X, y):
        return 1 - sum((self.predict(X) - y)**2) / sum((y - np.mean(y))**2)


diabetes = datasets.load_diabetes()
# Use only one feature
X = diabetes.data[:, np.newaxis, 2]
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

regr = MyLinearRegressionSGD(eta=.1, n_iter=500)
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))

output_file("plot/diabetes.html")

p = figure(
   tools="pan,box_zoom,reset,save", title="Diabetes Dataset",
   x_axis_label='x', y_axis_label='y'
)

X_test_flattened = [x1 for x in X_test for x1 in x]
p.line(X_test_flattened, regr.predict(X_test), color="black")
p.circle(X_test_flattened, y_test, fill_color="blue", size=8)
show(p)



