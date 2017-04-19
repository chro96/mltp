from pdb import set_trace
import numpy as np


class LogisticRegressionSoftmax(object):
    def __init__(self, num_classes, eta=0.1, n_iter=50):
        self.num_classes = num_classes
        self.eta = eta
        self.n_iter = n_iter

    def _cross_entropy(self, y, y_hat):
        # 交差エントロピー:「ある数値の軍団Aと、ある数値の軍団Bがどれくらい異なるか」
        # 学習データとモデルから出力されるデータの差異をエントロピーで表現する
        # H(p,q)=−∑xp(x)logq(x)
        return -np.sum(y * np.log(y_hat))


logi = LogisticRegressionSoftmax(3, n_iter=500)
q = [0.727,0.268,0.005]
cross_entropy = logi._cross_entropy(np.array([[1,0,0],[0,1,0],[0,0,1]]), np.array([q,q,q]))
print(cross_entropy)
assert(np.isclose(cross_entropy, 6.93391446647))
