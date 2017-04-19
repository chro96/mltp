from IPython import embed
import numpy as np


class LogisticRegressionSoftmax(object):
    def __init__(self, num_classes, eta=0.1, n_iter=50):
        self.num_classes = num_classes
        self.eta = eta
        self.n_iter = n_iter

    def _softmax1(self, x):
        # 指数関数は入力値が増えると、出力の増加率も大きい
        # xが大きいと指数がoverflowしてしまう
        # ので、実際にsoftmaxを使う時はx -= np.max(x)する
        x -= np.max(x)
        div_s = np.exp(x) / np.sum(np.exp(x))
        return div_s

    def _softmax(self, X):
        # 一個ずつ softmax を計算する
        return np.array([self._softmax1(s) for s in X])

logi = LogisticRegressionSoftmax(3, n_iter=500)
X = np.array(
    [[2,1,-3],
     [-6,0,-4]]
)
out = logi._softmax(X)
print(out)
assert(np.allclose(out, np.array([[0.72747516,0.26762315,0.00490169],[0.00242826,0.97962921,0.01794253]])))
