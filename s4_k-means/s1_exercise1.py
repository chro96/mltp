import numpy as np

class MyKMeans(object):
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        if self.random_state:
            np.random.seed(self.random_state)

    def fit(self, X):
        self.cluster_centers_ = # Your code here
        return self

X = np.array([[1,1],[1,2],[2,2],[4,5],[5,4]])
kmeans = MyKMeans(n_clusters=2, max_iter=5, random_state=1).fit(X)

print(kmeans.cluster_centers_)

# (n_clusters, D)
assert(kmeans.cluster_centers_.shape == (2,2))
# 全てXに入っている
assert(all(c in X for c in kmeans.cluster_centers_))
# 同じサンプルは入ってない
assert(not np.array_equal(kmeans.cluster_centers_[0], kmeans.cluster_centers_[1]))
# これはサンプルアウトプットなので上の３つが通ってればOKです。
assert(np.array_equal(kmeans.cluster_centers_, np.array([[2,2],[1,2]])))
