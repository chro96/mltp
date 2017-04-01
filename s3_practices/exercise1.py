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
        return self

X = np.array([
    [2,1,0,0,0,0],
    [2,0,1,0,0,0],
    [1,0,0,1,0,0],
    [1,0,0,0,1,1]
])
y = np.array([0,0,0,1])
nb = MyMultinomialNB().fit(X, y)
print(nb.class_log_prior_)

nb2 = MultinomialNB().fit(X, y)
print(nb2.class_log_prior_)

assert(np.allclose(np.array(nb.class_log_prior_), nb2.class_log_prior_))
