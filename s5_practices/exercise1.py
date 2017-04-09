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
        x = np.insert(X, 0, 1, axis=1)
        # x = [[1 0]
        #      [1 1]
        #      [1 2]
        #      [1 3]]
        # shape: 大きさが 5 のベクトルはスカラー 5 に、 4×1 の行列はタプル (4, 2) となる
        self.w = np.ones(x.shape[1])
        for i in range(self.n_iter):
            # numpy: dotは内積(スカラー積)をする
            # θ の値は適当に入れる
            # 最終的な目的はこの θ を見つけること。
            # その為にはまず θ を適当に選び、h(x) が y とどれだけ違うかに応じて θ を修正していく
            # h(X)=Xθ
            output = x.dot(self.w)
            return output

X = np.array([[0], [1], [2], [3]])
y = np.array([-1, 1, 3, 5])
regr = MyLinearRegression()
output = regr.fit(X, y)
print(output)
assert(np.array_equal(output, np.array([1,2,3,4])))
