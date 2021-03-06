import numpy as np

class MyKMeans(object):
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        if self.random_state:
            np.random.seed(self.random_state)

    def fit(self, X):
        initial = np.random.permutation(X.shape[0])[:self.n_clusters]
        self.cluster_centers_ = X[initial]
        return self

    def _nearest(self, centers, x):
        # Your code here
        pass

    def _distance(self, centers, x):
        return np.sqrt(((centers - x)**2).sum(axis=1))

X = np.array([[1,1],[1,2],[2,2],[4,5],[5,4]])
kmeans = MyKMeans(n_clusters=2, max_iter=5, random_state=1)

cluster_centers = np.array([[1,2],[2,2]])
sample = np.array([4,5])
nearest = kmeans._nearest(cluster_centers, sample)
print(nearest)
assert(nearest == 1)







