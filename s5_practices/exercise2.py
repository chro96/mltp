from IPython import embed
import numpy as np


# h(X)=Xθ を計算しましょう。
class MyLinearRegression(object):
    def __init__(self, eta=0.1, n_iter=50):
        # eta: learning rateと呼ばれるもので、どれだけ早く最適化するかを設定する。
        # これが小さすぎると学習が遅く、大きすぎると収束しなくなる。
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        # X = [[1 0]
        #      [1 1]
        #      [1 2]
        #      [1 3]]
        # shape: 大きさが 5 のベクトルはスカラー 5 に、 4×1 の行列はタプル (4, 2) となる
        self.w = np.ones(X.shape[1])
        m = X.shape[0]
        for i in range(self.n_iter):
            # numpy: dotは内積(スカラー積)をする
            # θ の値は適当に入れる
            # 最終的な目的はこの θ を見つけること。
            # その為にはまず θ を適当に選び、h(x) が y とどれだけ違うかに応じて θ を修正していく
            # θ := θ + (α/m)(y−h(x))X
            output = X.dot(self.w)
            errors = y - output
            if i % 10 == 0:
                print("Error: ", sum(errors ** 2))
                print("Weights: ", self.w)
                print("-------------------------")
            self.w += self.eta / m * errors.dot(X)
        return self.w

X = np.array([[0], [1], [2], [3]])
y = np.array([-1, 1, 3, 5])
regr = MyLinearRegression()
weight = regr.fit(X, y)
print(weight)
assert(np.array_equal(weight, np.array([1.05, 1.2])))
