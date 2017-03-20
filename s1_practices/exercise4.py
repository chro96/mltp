# 今のアルゴリズムだと単純にトップKの大多数を取りますが、以下の場合はどうでしょう？
# X = np.array([[1, 1], [4, 4], [5, 5]])
# y = np.array([1,0,0])

from IPython import embed
import numpy as np
from collections import defaultdict

class MyKNeighborsClassifier(object):
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def _distance(self, data1, data2):
        return sum(abs(data1 - data2))

    def _compute_weights(self, distances):
        if self.weights == 'uniform':
            # distances.shape[0] で各次元の要素数をだす
            return np.ones(distances.shape[0])
        elif self.weights == 'distance':
            embed()
            return 1 / distances if all(distances) else np.array([0 if d else 1 for d in distances])
        raise ValueError("weights not recognized: should be 'uniform' or 'distance'")

neighbor = MyKNeighborsClassifier(n_neighbors=3, weights='distance')
weights1 = neighbor._compute_weights(np.array([1,2,3,-4]))
print(weights1)
assert(all(weights1 == np.array([1, 1/2, 1/3, -1/4])) == True)
weights2 = neighbor._compute_weights(np.array([-1,0,2,3]))
print(weights2)
assert(all(weights2 == np.array([0,1,0,0])) == True)
