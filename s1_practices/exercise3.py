# このExerciseで基本的なアルゴリズムはもう完成します！
# 先ほど計算したtop_k_weightsを元に、weightが一番大きいクラスをreturnしましょう。

from IPython import embed
import numpy as np
from collections import defaultdict

class MyKNeighborsClassifier(object):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, x, y):
        self.x = x
        self.y = y
        return self

    def _distance(self, data1, data2):
        return sum(abs(data1 - data2))

    def _compute_weights(self, distances):
        # distances.shape[0] で各次元の要素数をだす
        return np.ones(distances.shape[0])

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

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1,1,1,0,0,0])
neighbor = MyKNeighborsClassifier().fit(X, y)
output = neighbor._predict_one(np.array([1, 0]))
print(output)
assert(output == 0)
