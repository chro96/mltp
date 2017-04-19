from IPython import embed
import numpy as np

class MyLogisticRegression(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X):
        X = np.insert(X, 0, 1, axis=1)
        # X = [[ 1 -2  2]
        #      [ 1 -3  0]
        #      [ 1  2 -1]
        #      [ 1  1 -4]]
        self.w = np.ones(X.shape[1])
        # self.w = [ 1.  1.  1.]
        return self

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        # X = [[ 1 -2  2]
        #      [ 1 -3  0]
        #      [ 1  2 -1]
        #      [ 1  1 -4]]
        output = self._sigmoid(X.dot(self.w))
        # output = [ 0.73105858  0.11920292  0.88079708  0.11920292]
        # output を 1 or 0 で出力する
        return np.where(output >= .5, 1, 0)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

X = np.array([[-2, 2],[-3, 0],[2, -1],[1, -4]])
logi = MyLogisticRegression().fit(X)
y_hat = logi.predict(X)
print(y_hat)
# シグモイド曲線の上に乗ってるかどうかを見る
assert(np.array_equal(y_hat, [1,0,1,0]))
