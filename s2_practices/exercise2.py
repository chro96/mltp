from IPython import embed
import numpy as np
from sklearn.preprocessing import scale, StandardScaler

def my_scale(X):
    # 各特徴をstandardizationする
    # axisは一次元目を 0 、二次元目を 1 とする
    new = X - np.mean(X, axis=0)
    return new / np.std(new, axis=0)

class MyStandardScaler(object):
    def fit(self, X):
        # 各特徴の平均を計算する
        self.mean_ = np.mean(X, axis=0)
        # 各特徴の標準偏差を計算する
        self.scale_ = np.std(X - self.mean_, axis=0)
        return self

    def transform(self, X):
        # 各特徴をstandardizationする
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

X = np.array([[ 1., -1.,  2.],
              [ 2.,  0.,  0.],
              [ 0.,  1., -1.]])
scaler = MyStandardScaler().fit(X)
scaled = scaler.transform(X)
# standardizationが正しく行われてるかどうか
assert(np.array_equal(scaled, my_scale(X)))

X_test = np.array([[-1.,  1.,  0.],
                   [ 1.,  0., -1.]])
scaled_test = scaler.transform(X_test)
print(scaled_test)
