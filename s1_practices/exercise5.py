# Euclidean Distanceを計算しましょう

from IPython import embed
import numpy as np
from collections import defaultdict

class MyKNeighborsClassifier(object):
    def __init__(self, n_neighbors=5, weights='uniform', p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self, x, y):
        self.x = x
        self.y = y
        return self

    def _distance(self, data1, data2):
        if self.p == 1:
            return sum(abs(data1 - data2))
        elif self.p == 2:
            return np.sqrt(sum((data1 - data2)**2))
        raise ValueError("p not recognized: should be 1 or 2")

neighbor = MyKNeighborsClassifier(p=2)
distance = neighbor._distance(np.array([-1, -1]),np.array([1, -2]))
print(distance)
assert(distance == np.sqrt(5))
