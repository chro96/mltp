from IPython import embed
import numpy as np
from sklearn.naive_bayes import MultinomialNB

np.set_printoptions(precision=6)

class MyMultinomialNB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        # 一番上の次元の要素数を N とする（データセットの数）
        N = X.shape[0]
        # クラスごとにデータをグルーピングする
        separated = [X[np.where(y == i)[0]] for i in np.unique(y)]
        # クラスごとの各termの要素数を計算する
        # alpha を追加することで smoothing する
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        return count

X = np.array([
    [2,1,0,0,0,0],
    [2,0,1,0,0,0],
    [1,0,0,1,0,0],
    [1,0,0,0,1,1]
])
y = np.array([0,0,0,1])
nb = MyMultinomialNB().fit(X, y)

print(nb)
assert(np.allclose(nb, np.array([[6,2,2,2,1,1],[2,1,1,1,2,2]])))
