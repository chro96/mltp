from IPython import embed
import numpy as np

class MyKMeans(object):
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        # random_state: rand() に再現性を与える
        self.random_state = random_state
        if self.random_state:
            np.random.seed(self.random_state)

    def _nearest(self, centers, x):
        # centersの中で距離が一番小さいラベルを return する
        return np.argmin(self._distance(centers, x))

    def _distance(self, centers, x):
        # ユークリッド距離を計算
        return np.sqrt(((centers - x)**2).sum(axis=1))

X = np.array([[1,1],[1,2],[2,2],[4,5],[5,4]])
kmeans = MyKMeans(n_clusters=2, max_iter=5, random_state=1)
nearest = kmeans._nearest(np.array([[1,2],[2,2]]), np.array([4,5]))
print(nearest)
assert(nearest == 1)
