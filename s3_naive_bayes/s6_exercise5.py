import numpy as np
from sklearn.naive_bayes import MultinomialNB

np.set_printoptions(precision=6)

class MyMultinomialNB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        N = X.shape[0]
        # group by class
        separated = [X[np.where(y == i)] for i in np.unique(y)]
        # class prior
        self.class_log_prior_ = [np.log(len(i) / N) for i in separated]
        # count of each term
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        # log probability of each term
        self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T)
        return self

    def predict_log_proba(self, X):
        return [(self.feature_log_prob_ * x).sum(axis=1) + self.class_log_prior_
                for x in X]

    def predict(self, X):
        # Your code here
        pass

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

