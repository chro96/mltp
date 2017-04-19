from IPython import embed
import numpy as np


class LogisticRegressionSoftmax(object):
    def __init__(self, num_classes, eta=0.1, n_iter=50):
        self.num_classes = num_classes
        self.eta = eta
        self.n_iter = n_iter

    def _softmax(self, x):
        x -= np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

logi = LogisticRegressionSoftmax(3, n_iter=500)
X = np.array(
    [[2,1,-3],
     [-6,0,-4]]
)
out = logi._softmax(X)
print(out)
assert(np.allclose(out, np.array([[0.72747516,0.26762315,0.00490169],[0.00242826,0.97962921,0.01794253]])))
