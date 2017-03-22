import numpy as np

def scale_one(x):
	"""
	mean = 0, standard deviation = 1になるようにarrayを変化させる
	np.mean(), np.std()を使って構いません。
	"""
	# Your code here
	pass

scaled = scale_one(np.array([1,2,0]))
print(scaled) # [ 0.          1.22474487 -1.22474487]
assert(np.mean(scaled) == 0)
assert(str(np.std(scaled)) == '1.0') # floatの状態だとイコールにならないのでstringに直す