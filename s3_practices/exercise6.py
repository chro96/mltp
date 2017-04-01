from IPython import embed
import numpy as np
from sklearn.naive_bayes import BernoulliNB

np.set_printoptions(precision=6)

class MyBernoulliNB(object):
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
        # smoothingはクラスの種類数で決まる
        smoothing = 2 * self.alpha
        # 分母はクラス種類ごとのデータセットの数 + smoothing で決まる
        denominator = np.array([len(i) + smoothing for i in separated])
        # それぞれの term ごとの出現確率
        self.feature_prob_ = count / denominator[np.newaxis].T
        return self

X = np.array([
    [2,1,0,0,0,0],
    [2,0,1,0,0,0],
    [1,0,0,1,0,0],
    [1,0,0,0,1,1]
])
y = np.array([0,0,0,1])

# 各termが含まれてるかだけを考慮し、何回含まれてるかは無視する。
nb = MyBernoulliNB(alpha=1).fit(np.where(X > 0, 1, 0), y)
print(nb.feature_prob_)

# 各termが含まれてるかだけを考慮し、何回含まれてるかは無視する。
nb2 = MyBernoulliNB(alpha=1).fit(np.where(X > 0, 1, 0), y)
print(nb2.feature_prob_)

assert(np.allclose(nb.feature_prob_, nb2.feature_prob_))
