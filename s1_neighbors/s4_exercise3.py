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
        """computes uniform weights"""
        return np.ones(distances.shape[0])

    def _predict_one(self, test):
        distances = np.array([self._distance(x, test) for x in self.x])
        top_k = np.argsort(distances)[:self.n_neighbors]
        top_k_ys = self.y[top_k]
        top_k_distances = distances[top_k]
        top_k_weights = self._compute_weights(top_k_distances)
        ####### returns the class with the heighest weight ########
        ## [注意事項]
        ## 今はuniform weightsを使ってるのでtop_k_weightsを使わずにtop_k_ysを数えれば済むが、
        ## 今後違うweightsを使うことも考え、必ずtop_k_weightsを計算すること。
        ## また、今はクラスが２つしかないが、３つ以上の時にも対応出来るようにすること。
        # Your code here
        pass

    def predict(self, x):
        return [self._predict_one(i) for i in x]

def main():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([1,1,1,0,0,0])
    neighbor = MyKNeighborsClassifier().fit(X, y)
    output = neighbor._predict_one(np.array([1, 0]))
    print(output)
    assert(output == 0)

if __name__ == '__main__': main()



