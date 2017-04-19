from IPython import embed
import numpy as np

class MyLogisticRegression(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def _sigmoid(self, x):
        # シグモイド曲線上にプロット
        return 1 / (1 + np.exp(-x))

logi = MyLogisticRegression()
sigmoid = (logi._sigmoid(np.array([3, -1, 0])))
print(sigmoid)
assert(np.allclose(sigmoid, np.array([0.952574126822, 0.26894142137, 0.5])))
