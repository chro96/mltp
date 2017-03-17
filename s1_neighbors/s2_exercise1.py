import numpy as np

class MyKNeighborsClassifier(object):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, x, y):
        self.x = x
        self.y = y
        return self
    
    def _distance(self, data1, data2):
        """
        Input:
            data1 - 1d array
            data2 - 1d array
        Output: Manhattan distance
        """
        # Your code here
        pass

    def _predict_one(self, test):
        distances = np.array([self._distance(x, test) for x in self.x])

    def predict(self, x):
        return [self._predict_one(i) for i in x]


neighbor = MyKNeighborsClassifier()
distance = neighbor._distance(np.array([-1, -1]), np.array([1, -2]))
print(distance)
assert(distance == 3)