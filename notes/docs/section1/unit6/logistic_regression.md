Logistic RegressionはClassificationのモデルです。Linear Regressionと非常に似ています。


# Logistic Regressionの説明

まずはBinary Classification（２クラス）専用のLogistic Regressionの説明をします。
`__init__`と`fit`の始めはLinear Regressionと全く同じコードです。

```python
# s1_initial_code.py

class MyLogisticRegression(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.ones(X.shape[1])
        m = X.shape[0]
        return self
```

Linear Regressionと同じ`predict`を使うとどうなるでしょうか。

```python
    def predict(self, X):
        return np.insert(X, 0, 1, axis=1).dot(self.w)

X = np.array([[-2, 2],[-3, 0],[2, -1],[1, -4]])
y = np.array([1,1,0,0])
logi = MyLogisticRegression().fit(X, y)
print(logi.predict(X))
```

```output
[ 1. -2.  2. -2.]
```

アウトプットは０か１でなければいけないので、このままではダメですよね。これを解決する為に*sigmoid function*を使います。

$$ g(x) = \frac{1}{1 + e^{-x}} $$

![Sigmoid_Function.png (28.8 kB)](https://img.esa.io/uploads/production/attachments/5475/2017/04/10/18257/99561c38-f845-4709-a9fb-76513bfc85d4.png)

sigmoid functionはxが\(\infty\)に近づくにつれ1に近づき、\(-\infty\)に近づくにつれ0に近づきます。つまりどんな値も0~1の間に収まります。これを確率と捉えることも出来ます。よって\(g(x) >= 0.5\)の場合は1、\(g(x) < 0.5\)の場合は0とします。

アップデートの数式は以下の通りです。Linear Regressionと全く同じに見えます。

\[ \theta := \theta + \frac{\alpha}{m} (y - h(x))X \]

しかしLinear Regressionでは\(h(x) = X\theta\)なのに対し、Logistic Regressionでは\(h(x) = g(X\theta)\)です。Linear Regressionでは\(h(x) = X\theta\)がどれだけターゲットの値とずれているかがエラーでした。Logistic Regressionでは\(h(x) = g(X\theta)\)、つまりSigmoidを通した値とターゲットがどれだけずれているかをエラーとします。

例えば、\(h(x)\)が0.7でターゲットが1だとすると、エラーが\(|1 - 0.7| = 0.3\)あるので、これを頑張って0にしようとします。`predict`では0.5以上あれば1とするので0.7をこれ以上1に近づけなくても精度には変わらないんですが、最適化の際は精度は全く気にせずエラーを少なくしようとします。

# Exercises

## Exercise 1

Sigmoid functionを書きましょう。

$$ g(x) = \frac{1}{1 + e^{-x}} $$

インプットがarrayなので、ループを使わずnumpyで一気に計算しましょう。

## Exercise 2

`predict`を完成させましょう。\(g(x) >= 0.5\)の場合は1、\(g(x) < 0.5\)の場合は0です。

## Exercise3

最後に`fit`を完成させましょう。Linear Regressionを参照すれば問題ないと思います。`_sigmoid`を忘れずに使いましょう。

# One vs Restを使ったMulticlass Classification

上記はBinary Classificationにしか使うことが出来ません。Logistic Regressionで複数のクラスを扱う方法は２つあります。一つは"One vs Rest"という手法です。これはLogistic Regressionに限らずどんなモデルにも使うことが出来ます。

方法はシンプルで、クラスが３つあったら３つのBinary Classifierを作ります。１つ目では0のクラスを1とし、それ以外のクラスを0にします。２つ目、３つ目でも同じ流れで擬似的にクラスを0と1の２つだけにします。

| Classifier \ クラス       | 0 | 1 | 2 |
|--------|---|---|---|
| １つ目 | 1 | 0 | 0 |
| ２つ目 | 0 | 1 | 0 |
| ３つ目 | 0 | 0 | 1 |

```python
# s5_exercise4.py

class LogisticRegressionOVR(object):
    """One vs Rest"""

    def __init__(self, num_classes, eta=0.1, n_iter=50):
        self.num_classes = num_classes
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.zeros((X.shape[1], self.num_classes))
        m = X.shape[0]

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
```

そして`predict`の時に３つのBinary Classifierを実行しアウトプットが一番高いものを採用します。

# Exercises

## Exercise 4

`predict`を書きましょう。matrix multiplicationと`np.argmax`を使えばループを全く使わずに書けます。

# Softmaxを使ったMulticlass Classification

One vs Restはクラスの数だけBinary Classifierを学習しなければならないため時間がかかります。よって実際Logistic RegressionやNeural NetworkでMulticlass Classificationをする場合はほぼ確実に*Softmax*というものを使います。

Softmaxを説明する前にOne vs Restと共通するコードを見てみましょう。勿論`fit`はまだ途中です。

```python
#s6_softmax_initial_code.py

class LogisticRegressionSoftmax(object):
    def __init__(self, num_classes, eta=0.1, n_iter=50):
        self.num_classes = num_classes
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.random.randn(X.shape[1], self.num_classes)
        m = X.shape[0]
        return self

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.argmax(X.dot(self.w), axis=1)

iris = datasets.load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.4)
logi = LogisticRegressionSoftmax(len(np.unique(iris.target)), n_iter=500)
logi.fit(x_train, y_train)
print(logi.w.shape)
print(logi.predict(x_test[:2]))
```

`self.w`のサイズはOne vs Restと同じく(5, 3)です。weightの初期値に`np.random.randn`を使ってますが、`np.zeros`でも`np.ones`でも構いません。初期値によって最適化の速度が変わったりしますが、今はそこまで気にしなくて大丈夫です。

```output
(5, 3)
[0 0]
```

以下が`fit`の完成形です。`X.dot(self.w)`のサイズは(m, クラス数)です。これにsoftmaxを当て、各クラスの確率を出します。

```python
# s7_exercise5.py

def fit(self, X, y):
    X = np.insert(X, 0, 1, axis=1)
    y = self._one_hot(y)
    self.w = np.random.randn(X.shape[1], self.num_classes)
    m = X.shape[0]

    for i in range(self.n_iter):
        output = self._softmax(X.dot(self.w))
        errors = y - output
        self.w += self.eta / m * X.T.dot(errors)

        if i % 10 == 0:
            print(self._cross_entropy(y, output))
    return self
```

以下がSoftmaxの公式です。

\[ \sigma(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}} \]

これをSigmoidと同じように線形関数`X.dot(self.w)`の後に使います。Sigmoidのインプットが数字なのに対し、Softmaxのインプットはarrayになります（コードでは全サンプル一度に計算するのでSigmoidのインプットもarrayですが、ここでは１サンプルでの話です）。

式だけ見てもピンと来ないと思うので実際に計算してみましょう。
例えば`X.dot(self.w)`の１サンプルが\([2,1,-3]\)だとします。

まずそれぞれの分子を計算しましょう。

\[
\begin{align}
e^2 &= 7.389 \\
e^1 &= 2.718 \\
e^{-3}&= 0.0498
\end{align}
\]

分母はこれらの合計です。

\[ 7.389 + 2.718 + 0.0498 = 10.157 \]

そしてそれぞれをこの合計で割れば完成です。

\[ [7.389, 2.718, 0.0498] / 10.157 = [0.727,0.268,0.005] \]

Softmaxの値は合計すると1になることが分かります。つまりこれは各クラスの確率を表しているとも言えます。「72.7％の確率でクラス０だろう」ということですね。

\[ 0.727 + 0.268 + 0.005 = 1 \]

これとターゲットの差分を縮めていくのですが、そのloss functionには*cross entropy*を使います。pがターゲット、qがsoftmax後のアウトプットとするとcross entropyは以下の通りです。

\[ H(p, q) = -\sum_x p(x)\, \log q(x) \]

qがarrayなので、pもarrayにしなければなりません。その為に*one hot encoding*という手段を使います。例えばクラスが全部で３つあるとし、クラスが0の場合は\[[1,0,0]\]、クラスが2の場合は\[[0,0,1]\]となります。

では上記のアウトプットのターゲットが0だとすると、\[p = [1,0,0]\], \[q =[0.727,0.268,0.005] \]となるのでcross entropyは

\[
\begin{align}
 H(p,q) &= - (1 \times log(0.727) + 0  \times log(0.268) + 0 \times log(0.005)) \\
&= -(1 \times -0.3188 + 0 + 0) \\
&= 0.3188
\end{align}
\]

\(p = [0,1,0]\),  \(p = [0,0,1]\)の場合も計算してみて下さい。\(p\)と\(q\)が近いほどcross entropyは小さくなります。なので\(p = [1,0,0]\)の時が一番小さくなるはずです。

# Exercises

`fit`の部分は微分積分を要するところですしSigmoidの時とほとんど変わらないので今回は既に完成されています。

## Exercise 5

`_one_hot`を書きましょう。インプットがarrayなので各数字毎にone_hot_encodingを作って下さい。

## Exercise 6

softmaxを書きましょう。まずは１サンプルづつ計算します。
xが大きいと指数が物凄くなってしまいoverflowしてしまうで、実際にsoftmaxを使う時は`x -= np.max(x)`とすることによってoverflowを防ぎます。

## Exercise 7

次に全サンプル一気に計算するsoftmaxを書きましょう。overflowの対処も書いて下さい。

## Exercise 8

`_cross_entropy`を書きましょう。全サンプル一度に計算して下さい。

# Reference

Softmaxとcross entropyの説明が分かり易いです。
[http://cs231n.github.io/linear-classify/#softmax](http://cs231n.github.io/linear-classify/#softmax)
