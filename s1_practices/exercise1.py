# 最初に距離を測るメソッド_distanceを書きましょう。Manhattan Distanceを使います。

from IPython import embed
import numpy as np

class MyKNeighborsClassifier(object):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def _distance(self, data1, data2):
        # マンハッタン距離の合計を計算する
        return sum(abs(data1 - data2))

# distanceをテストする
neighbor = MyKNeighborsClassifier()
distance = neighbor._distance(np.array([-1, -1]), np.array([1, -2]))
print(distance)
assert(distance == 3)
