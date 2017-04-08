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

        for _ in range(self.max_iter):
            self.labels_ = np.array([self._nearest(self.cluster_centers_, x) for x in X])
            X_by_cluster = [X[np.where(self.labels_ == i)[0]] for i in range(self.n_clusters)]
            # update the clusters
            self.cluster_centers_ = [c.sum(axis=0) / len(c) for c in X_by_cluster]
        # sum of square distances from the closest cluster
        self.inertia_ = sum(((c - x)**2).sum() for c, x in zip(self.cluster_centers_, X_by_cluster))
        return self

    def _nearest(self, centers, x):
        return np.argmin(self._distance(centers, x))

    def _distance(self, centers, x):
        return np.sqrt(((centers - x)**2).sum(axis=1))


X = np.array([[1,1],[1,2],[2,2],[4,5],[5,4]])
kmeans = MyKMeans(n_clusters=2, max_iter=5, random_state=1)
kmeans.fit(X)
print(kmeans.inertia_)
assert(np.isclose(kmeans.inertia_, 2.33333333))
