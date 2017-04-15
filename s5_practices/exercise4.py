from IPython import embed
import numpy as np
from sklearn.cross_validation import train_test_split


class MyLinearRegressionSGD(object):
    # 確率的勾配法: 学習データの一部をランダムに選択し、選択したデータを使って勾配を求める。
    def __init__(self, eta=0.1, n_iter=50, shuffle=True):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.ones(X.shape[1])
        for _ in range(self.n_iter):
            if self.shuffle:
                # 対応する X, y の組の順番をシャッフルする
                X, y = self._shuffle(X, y)
            # self.w += self.eta / m * errors.dot(X)
            # 上のように 1 イテレートごとにサンプル全ての行列計算すると計算が多くなるから、
            # 1 イテレートごとにサンプルから無作為抽出することで計算対象の行列を小さくする
            # zip: 複数のシーケンスオブジェクトを同時にループするときに使用する
            for x, target in zip(X, y):
                # h(x)
                output = x.dot(self.w)
                # θ := θ + (α/m)(y−h(x))X
                self.w += self.eta * (target - output) * x
        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def predict(self, X):
        # dot メソッドは内積
        # h(X)=Xθ
        return np.insert(X, 0, 1, axis=1).dot(self.w)


X = np.array([[0], [1], [2], [3]])
y = np.array([-1, 1, 3, 5])
X_test = np.array([[4],[5]])
regr = MyLinearRegressionSGD(n_iter=100).fit(X, y)
y_hat = regr.predict(X_test)
print(y_hat)
assert(np.allclose(y_hat, np.array([7,9])))
