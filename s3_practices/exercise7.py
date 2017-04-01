import numpy as np
from sklearn.naive_bayes import BernoulliNB

class MyBernoulliNB(object):
    def __init__(self, alpha=1.0, binarize=0.0):
        self.alpha = alpha
        self.binarize = binarize

    def fit(self, X, y):
        # 各termが含まれてるかだけを考慮し、何回含まれてるかは無視する。
        X = self._binarize_X(X)
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


    def predict_log_proba(self, X):
        # 各termが含まれてるかだけを考慮し、何回含まれてるかは無視する。
        X = self._binarize_X(X)
        # テスト対象データに出現しなかったものは (1 -  p(t2|c)) とする
        return [(np.log(self.feature_prob_) * x + \
                 np.log(1 - self.feature_prob_) * np.abs(x - 1)
                ).sum(axis=1) + self.class_log_prior_ for x in X]

    def predict(self, X):
        # 確率が一番高いものを return する
        # 確率は 1 以下なので対数では負の値をとる
        return np.argmax(self.predict_log_proba(X), axis=1)

    def _binarize_X(self, X):
        return np.where(X > self.binarize, 1, 0) if self.binarize != None else X


X = np.array([
    [2,1,0,0,0,0],
    [2,0,1,0,0,0],
    [1,0,0,1,0,0],
    [1,0,0,0,1,1]
])
y = np.array([0,0,0,1])
X_test = np.array([[3,0,0,0,1,1],[0,1,1,0,1,1]])

nb = MyBernoulliNB(alpha=1).fit(X, y)
print(nb.predict(X_test))
