import numpy as np

class MyKNeighborsClassifier(object):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, x, y):
        self.x = x
        self.y = y
        return self

    def _predict_one(self, test):
        return 1

    def predict(self, x):
        return [self._predict_one(i) for i in x]

x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1,1,1,0,0,0])
neighbor = MyKNeighborsClassifier()
neighbor.fit(x, y)
print(neighbor.predict(np.array([[1, 0], [-2, -2]])))