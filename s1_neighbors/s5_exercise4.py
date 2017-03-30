import numpy as np
from collections import defaultdict

class MyKNeighborsClassifier(object):
	def __init__(self, n_neighbors=5, weights='uniform'):
	    self.n_neighbors = n_neighbors
	    self.weights = weights

	def fit(self, x, y):
		self.x = x
		self.y = y
		return self

	def _distance(self, data1, data2):
		"""returns Manhattan distance"""
		return sum(abs(data1 - data2))			

	def _compute_weights(self, distances):
		if self.weights == 'uniform':
			return np.ones(distances.shape[0])
		elif self.weights == 'distance':
			###### returns the distance weights (1 / distance) ########
			## distanceが0のデータが一つでもある場合は、0のweightを1にし、それ以外を0にする
			# Your code here
			return
		raise ValueError("weights not recognized: should be 'uniform' or 'distance'")

	def _predict_one(self, test):
		distances = np.array([self._distance(x, test) for x in self.x])
		top_k = np.argsort(distances)[:self.n_neighbors]
		top_k_ys = self.y[top_k]
		top_k_distances = distances[top_k]
		top_k_weights = self._compute_weights(top_k_distances)
		weights_by_class = defaultdict(float)
		for d, c in zip(top_k_weights, top_k_ys):
			weights_by_class[c] += d
		return max(weights_by_class, key=weights_by_class.get)

	def predict(self, x):
		return [self._predict_one(i) for i in x]

def main():
	neighbor = MyKNeighborsClassifier(n_neighbors=3, weights='distance')
	weights1 = neighbor._compute_weights(np.array([1,2,3,-4]))
	print(weights1)
	assert(np.array_equal(weights1, np.array([1, 1/2, 1/3, -1/4])))
	weights2 = neighbor._compute_weights(np.array([-1,0,2,3]))
	print(weights2)
	assert(np.array_equal(weights2, np.array([0,1,0,0])))	

if __name__ == '__main__': main()



