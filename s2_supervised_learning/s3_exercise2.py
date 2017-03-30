import numpy as np
from sklearn.preprocessing import scale

def scale_one(x):
	"""
	mean = 0, standard deviation = 1になるようにarrayを変化させる
	"""
	new = x - np.mean(x)
	return new / np.std(new)


def my_scale(X):
	"""
	各カラム(feature)がmean = 0, standard deviation = 1になるようにarrayを変化させる
	ヒント：ループは要らないです。
	"""
	# Your code here
	pass

X = np.array([[ 1., -1.,  2.],
              [ 2.,  0.,  0.],
              [ 0.,  1., -1.]])
scaled = my_scale(X)
print(scaled)
assert(np.array_equal(scaled, scale(X)))

for i in range(scaled.shape[1]):
	assert(np.mean(scaled[:,i]) == 0)
	assert(np.isclose(np.std(scaled[:,i]), 1))
	