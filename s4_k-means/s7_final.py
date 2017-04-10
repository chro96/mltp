import numpy as np
from pdb import set_trace

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
            X_by_cluster = [X[np.where(self.labels_ == i)] for i in range(self.n_clusters)]
            # update the clusters
            self.cluster_centers_ = [c.sum(axis=0) / len(c) for c in X_by_cluster]
        # sum of square distances from the closest cluster
        self.inertia_ = sum(((c - x)**2).sum() for c, x in zip(self.cluster_centers_, X_by_cluster))
        ## これでもいいがループ数が多い分遅い
        # self.inertia_ = sum(((self.cluster_centers_[l] - x)**2).sum() for x, l in zip(X, self.labels_))
        return self

    def _nearest(self, centers, x):
        return np.argmin(self._distance(centers, x))

    def _distance(self, centers, x):
        return np.sqrt(((centers - x)**2).sum(axis=1))

    def predict(self, X):
        return self.labels_

    def transform(self, X):
        return np.array([self._distance(self.cluster_centers_, x) for x in X])

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def score(self, X):
        return -self.inertia_


X = np.array([[1,1],[1,2],[2,2],[4,5],[5,4]])
kmeans = MyKMeans(n_clusters=2, max_iter=10, random_state=1)
kmeans.fit(X)
print(kmeans.score(X))
print(kmeans.predict(X))
print(kmeans.transform(X))



