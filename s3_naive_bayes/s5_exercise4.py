import numpy as np
from sklearn.naive_bayes import MultinomialNB

np.set_printoptions(precision=6)

class MyMultinomialNB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        N = X.shape[0]
        # group by class
        separated = [X[np.where(y == i)[0]] for i in np.unique(y)]
        # class prior
        self.class_log_prior_ = [np.log(len(i) / N) for i in separated]
        # count of each term
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        # log probability of each term
        self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T)
        return self

    def predict_log_proba(self, X):
        """
        p(c|d)を計算する
        log probabilityを使ってるので掛けずに足すこと
        """
        # Your code here
        pass

X = np.array([
    [2,1,0,0,0,0],
    [2,0,1,0,0,0],
    [1,0,0,1,0,0],
    [1,0,0,0,1,1]
])
y = np.array([0,0,0,1])
X_test = np.array([[3,0,0,0,1,1],[0,1,1,0,1,1]])

nb = MyMultinomialNB().fit(X, y)
proba = nb.predict_log_proba(X_test)
print(proba)

proba2 = np.array([[-8.10769, -8.906681],[-9.457617, -8.788898]])
assert(np.allclose(np.array(proba), proba2))







