from IPython import embed
import numpy as np


# Softmaxを使ったMulticlass Classification
class LogisticRegressionSoftmax(object):
    def __init__(self, num_classes, eta=0.1, n_iter=50):
        self.num_classes = num_classes
        self.eta = eta
        self.n_iter = n_iter

    def _one_hot(self, y):
        m = y.shape[0]
        y_new = np.zeros((m, self.num_classes))
        # y_new = [[ 0.  0.  0.]
        #          [ 0.  0.  0.]
        #          [ 0.  0.  0.]]
        y_new[np.arange(m), y] = 1
        # y_new = [[ 1.  0.  0.]
        #          [ 0.  1.  0.]
        #          [ 0.  0.  1.]]
        return y_new

logi = LogisticRegressionSoftmax(3, n_iter=500)
one_hot = logi._one_hot(np.array([0,1,2]))
# one hot エンコーディングをそれぞれ計算
# one-hotではビット列のn番目のビットがHighであればn番目の状態を表していることになる
print(one_hot)
assert(np.array_equal(one_hot, np.array([[1,0,0],[0,1,0],[0,0,1]])))
