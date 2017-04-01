from IPython import embed
import numpy as np
from sklearn.naive_bayes import BernoulliNB

np.set_printoptions(precision=6)

class MyMultinomialNB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        # 一番上の次元の要素数を N とする（データセットの数）
        N = X.shape[0]
        # クラスごとにデータをグルーピングする
        separated = [X[np.where(y == i)[0]] for i in np.unique(y)]
        # 各クラスの数 / データセットの数
        self.class_log_prior_ = [np.log(len(i) / N) for i in separated]
        # クラスごとの各termの要素数を計算する
        # alpha を追加することで smoothing する
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        # p(t|c) を計算する
        # numpyにおいて、行列同士の四則演算は要素同士の四則演算になる
        # numpyの行列同士の掛け算は数学的な意味での掛け算ではないことに注意
        self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T)
        return self

    def predict_log_proba(self, X):
        # p(c|d) を計算する
        # p(c|d) = p(c) * p(t1|c) * p(t2|c) * p(t3|c) * ...
        # 対数を使ってるので掛けずに足すこと
        # 対数を使ってるので乗数は掛け算すること
        return [(self.feature_log_prob_ * x).sum(axis=1) + self.class_log_prior_
                for x in X]

    def predict(self, X):
        # 確率が一番高いものを return する
        # 確率は 1 以下なので対数では負の値をとる
        return np.argmax(self.predict_log_proba(X), axis=1)

X = np.array([
    [2,1,0,0,0,0],
    [2,0,1,0,0,0],
    [1,0,0,1,0,0],
    [1,0,0,0,1,1]
])
y = np.array([0,0,0,1])
nb = MyMultinomialNB().fit(X, y)

X_test = np.array([[3,0,0,0,1,1],[0,1,1,0,1,1]])
y_hat = nb.predict(X_test)
print(y_hat)

assert(np.array_equal(y_hat, np.array([0,1])))
