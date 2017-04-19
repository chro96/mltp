from IPython import embed
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split

np.random.seed(1)

# Multiclass Classification
# Logistic Regressionで複数のクラスを扱う方法は２つあります。
# 一つは"One vs Rest"という手法です。
# これはLogistic Regressionに限らずどんなモデルにも使うことが出来ます。
class LogisticRegressionOVR(object):
    def __init__(self, num_classes, eta=0.1, n_iter=50):
        self.num_classes = num_classes
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.zeros((X.shape[1], self.num_classes))
        m = X.shape[0]
        # 例えばクラスが３つあったら３つのBinary Classifierを作ります。
        # １つ目では0のクラスを1とし、それ以外のクラスを0にします。
        # ２つ目、３つ目でも同じ流れで擬似的にクラスを0と1の２つだけにします。
        for i in range(self.num_classes):
            # yを1と0だけにする
            y_copy = np.where(y == i, 1, 0)
            w = np.ones(X.shape[1])
            for j in range(self.n_iter):
                output = X.dot(w)
                errors = y_copy - self._sigmoid(output)
                w += self.eta / m * errors.dot(X)
                if j % 10 == 0:
                    print(sum(errors**2))
            self.w[:, i] = w
        return self

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.argmax(X.dot(self.w), axis=1)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.4)
# テストデータ iris を準備
logi = LogisticRegressionOVR(len(np.unique(iris.target)), n_iter=500)
# クラスの数とイテレート数をセット
logi.fit(x_train, y_train)
# 学習させる
y_hat = logi.predict(x_test[:5])
# 予測する
print(y_hat)
assert(np.array_equal(y_hat, np.array([0, 1, 1, 0, 2])))
