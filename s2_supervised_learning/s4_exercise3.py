import numpy as np
from sklearn.preprocessing import scale, StandardScaler

def scale_one(x):
	"""
	mean = 0, standard deviation = 1になるようにarrayを変化させる
	"""
	new = x - np.mean(x)
	return new / np.std(new)


def my_scale(X):
	"""
	各カラム(feature)がmean = 0, standard deviation = 1になるようにarrayを変化させる
	"""
	new = X - np.mean(X, axis=0)
	return new / np.std(new, axis=0)


class MyStandardScaler(object):
	def __init__(self):
		pass

	def fit(self, X):
		self.mean_ = np.mean(X, axis=0)
		self.scale_ = np.std(X - self.mean_, axis=0)
		return self

	def transform(self, X):
		"""
		self.mean_とself.scale_を元にarrayを変化させる
		"""
		# Your code here 
		pass

	def fit_transform(self, X):
		return self.fit(X).transform(X)


X = np.array([[ 1., -1.,  2.],
              [ 2.,  0.,  0.],
              [ 0.,  1., -1.]])
scaler = MyStandardScaler().fit(X)
scaled = scaler.transform(X) # Training Dataのtransform
assert(np.array_equal(scaled, my_scale(X)))

X_test = np.array([[-1.,  1.,  0.],
				   [ 1.,  0., -1.]])
scaled_test = scaler.transform(X_test) # Test Dataのtransform
print(scaled_test)
"""
[[-2.44948974  1.22474487 -0.26726124]
 [ 0.          0.         -1.06904497]]
"""












