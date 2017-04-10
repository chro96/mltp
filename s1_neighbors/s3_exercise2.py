import numpy as np

class MyKNeighborsClassifier(object):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, x, y):
        self.x = x
        self.y = y
        return self

    def _distance(self, data1, data2):
        """returns Manhattan distance"""
        return sum(abs(data1 - data2))          

    def _compute_weights(self, distances):
        """
        computes uniform weights
        全distanceを1に変換するだけです。
        """
        # Your code here
        pass

    def _predict_one(self, test):
        distances = np.array([self._distance(x, test) for x in self.x])
        top_k = distances.argsort()[:self.n_neighbors]
        top_k_ys = self.y[top_k]
        top_k_distances = distances[top_k]
        top_k_weights = self._compute_weights(top_k_distances)

    def predict(self, x):
        return [self._predict_one(i) for i in x]

def main():
    neighbor = MyKNeighborsClassifier()
    weights = neighbor._compute_weights(np.array([1, 2, 3, -4]))
    print(weights)
    assert(np.array_equal(weights, np.array([1,1,1,1])))

if __name__ == '__main__': main()


