from IPython import embed
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
        # イテレーションの数だけクラスターを作り直す
        # for in の return はイテレーション後の最後の値だけ返す
        for _ in range(self.max_iter):
            # X の中身がそれぞれどのクラスター(ラベル0かラベル1か)に属するかを計算しself.labels_に入れる
            self.labels_ = np.array([self._nearest(self.cluster_centers_, x) for x in X])
            # whereでラベル0かラベル1かを判定してクラスター別に振り分ける
            X_by_cluster = [X[np.where(self.labels_ == i)] for i in range(self.n_clusters)]
            # イテレート毎にクラスターの重心を計算し直す
            self.cluster_centers_ = [c.sum(axis=0) / len(c) for c in X_by_cluster]
            return self

    def _nearest(self, centers, x):
        # centersの中で距離が一番小さいラベルを return する
        return np.argmin(self._distance(centers, x))

    def _distance(self, centers, x):
        return np.sqrt(((centers - x)**2).sum(axis=1))

X = np.array([[1,1],[1,2],[2,2],[4,5],[5,4]])
kmeans = MyKMeans(n_clusters=2, max_iter=5, random_state=1).fit(X)
print(kmeans.cluster_centers_)
assert(np.allclose(np.array(kmeans.cluster_centers_), np.array([[3.6666667, 3.6666667],[1,1.5]])))
