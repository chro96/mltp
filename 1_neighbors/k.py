from sklearn import datasets
import numpy as np
from collections import defaultdict
from pdb import set_trace

class KNeighborsClassifier(object):
	def __init__(self, n_neighbors=5, weights='uniform', p=2):
		self.n_neighbors = n_neighbors
		self.weights = weights
		self.p = p

	def fit(self, x, y):
		self.x = x
		self.y = y
		return self

	def _distance(self, data1, data2):
		"""1: Manhattan, 2: Euclidean"""
		if self.p == 1:
			return sum(abs(data1 - data2))			
		elif self.p == 2:
			return np.sqrt(sum((data1 - data2)**2))
		raise ValueError("p not recognized: should be 1 or 2")

	def _compute_weights(self, distances):
		if self.weights == 'uniform':
			return np.ones(distances.shape[0])
		elif self.weights == 'distance':
			"""distanceが0のデータが一つでもある場合は、0のweightを1にし、それ以外を0にする"""
			return 1 / distances if all(distances) else np.array([0 if d else 1 for d in distances])
		raise ValueError("weights not recognized: should be 'uniform' or 'distance'")

	def _predict_one(self, test):
		distances = np.array([self._distance(x, test) for x in self.x])
		top_k = np.argsort(distances)[:self.n_neighbors]
		top_k_distances = distances[top_k]
		top_k_weights = self._compute_weights(top_k_distances)
		top_k_ys = self.y[top_k]
		weights_by_class = defaultdict(float)
		for d, c in zip(top_k_weights, top_k_ys):
			weights_by_class[c] += d
		return max(weights_by_class, key=weights_by_class.get)


	def predict(self, X):
		return [self._predict_one(x) for x in X]

	def score(self, X, y):
		return sum(self.predict(X) == y) / len(y)





def main():
	X_train = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
	y_train = np.array([1,1,1,0,0,0])
	neighbor = KNeighborsClassifier().fit(X_train, y_train)
	X_test = np.array([[1, 0], [-2, -2]])
	y_test = np.array([0, 0])
	print(neighbor.score(X_test, y_test))
	# print(neighbor._predict_one(np.array([1, 0])))
	# print(neighbor._compute_weights(np.array([(1, 0), (2, 0), (3, 1)])))
	
	# print(neighbor._distance(np.array([-1, -1]),np.array([1, -2])))



if __name__ == '__main__': main()








