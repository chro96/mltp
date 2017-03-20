# distanceを計算したら小さい順にソートし、self.n_neighborsの数だけ取ります。
# その後_compute_weightsでweightを計算します。
# weightは上では説明しませんでしたが、各データの重要度のようなものです。
# まずはuniform weightsを計算します。
# これは上でやったのと同じ方法で、全データ同じ比重を持つという意味です。
# つまりdistanceに関わらず1をreturnすれば全データ同じ比重になります。
# 例えば top_k_distancesが[1, 2, 3, -4]だったら[1,1,1,1]をreturnします。

from IPython import embed
import numpy as np

class MyKNeighborsClassifier(object):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def _compute_weights(self, distances):
        # distances.shape[0] で各次元の要素数をだす
        # np.ones() で n*n で全ての要素が1の行列を生成する
        # 全データ同じ比重を持つ
        # 一次元配列で要素数が 4 なので distances.shape[0] は 4 が return される
        return np.ones(distances.shape[0])

# compute_weightsをテストする
neighbor = MyKNeighborsClassifier()
weights = neighbor._compute_weights(np.array([1, 2, 3, -4]))
print(weights)
assert(all(weights == np.array([1,1,1,1])) == True)
