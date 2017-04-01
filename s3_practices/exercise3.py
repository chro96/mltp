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
        # # クラスごとにデータをグルーピングする
        separated = [X[np.where(y == i)[0]] for i in np.unique(y)]
        # クラスごとの各termの要素数を計算する
        # alpha を追加することで smoothing する
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        # p(t|c) を計算する
        # numpyにおいて、行列同士の四則演算は要素同士の四則演算になる
        # numpyの行列同士の掛け算は数学的な意味での掛け算ではないことに注意
        self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T)
        return self

X = np.array([
    [2,1,0,0,0,0],
    [2,0,1,0,0,0],
    [1,0,0,1,0,0],
    [1,0,0,0,1,1]
])
y = np.array([0,0,0,1])
nb = MyMultinomialNB().fit(X, y)
print(nb.feature_log_prob_)

nb2 = MultinomialNB().fit(X, y)
print(nb2.feature_log_prob_)

assert(np.allclose(np.array(nb.feature_log_prob_), nb2.feature_log_prob_))
