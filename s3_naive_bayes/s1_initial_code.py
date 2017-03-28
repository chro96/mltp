import numpy as np

np.set_printoptions(precision=6)

class MyMultinomialNB(object):
	def __init__(self, alpha=1.0):
		self.alpha = alpha

	def fit(self, X, y):
		N = X.shape[0]
		# group by class
		separated = [X[np.where(y == i)[0]] for i in np.unique(y)]
		return separated

X = np.array([
    [2,1,0,0,0,0],
    [2,0,1,0,0,0],
    [1,0,0,1,0,0],
    [1,0,0,0,1,1]
])
y = np.array([0,0,0,1])
nb = MyMultinomialNB().fit(X, y)
print(nb)