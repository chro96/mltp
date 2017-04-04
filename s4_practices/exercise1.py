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

    def fit(self, X):
        # クラスターの初期値をn_clustersの分だけ作る
        # 今回の場合はクラスターを二つ作る
        # Xからランダムに選びself.cluster_centers_に入れる
        # permutation(x)には配列だけでなくint型整数も渡せる
        initial = np.random.permutation(X.shape[0])[:self.n_clusters]
        self.cluster_centers_ = X[initial]
        return self

X = np.array([[1,1],[1,2],[2,2],[4,5],[5,4]])
kmeans = MyKMeans(n_clusters=2, max_iter=5, random_state=1).fit(X)

print(kmeans.cluster_centers_)

# データセットが二次元でクラスターが二つなので (2, 2) となる
assert(kmeans.cluster_centers_.shape == (2,2))
# クラスターの初期値は X の中に全て含まれている
assert(all(c in X for c in kmeans.cluster_centers_))
# クラスターの初期値同士は被らない
assert(not np.array_equal(kmeans.cluster_centers_[0], kmeans.cluster_centers_[1]))
# 答え合わせ
assert(np.array_equal(kmeans.cluster_centers_, np.array([[2,2],[1,2]])))
