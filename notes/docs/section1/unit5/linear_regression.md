Linear RegressionはSupervised Learningですが、k-NNやNaive Bayesがクラスを予測するClassificationなのに対し、Linear Regressionは数値を予測するRegressionになります。不動産の価格予測や株式市場の予測がRegressionに当たります。

Linear RegressionではOptimization(最適化)を使うので、アルゴリズムの種類としてはLogistic RegressionやNeural Networkと同じです。Deep Learningを学ぶ上での土台となる理論が詰まっています。

Optimizationには微分積分を使うので今までより急激に難しくなります。

# Linear Regressionの説明

[Andrew NgのLecture notes1](http://cs229.stanford.edu/materials.html)が非常に分かり易いので、それを元に説明します。

Linear Regressionは線形関数を使ってインプット(x)からアウトプット(y)を概算します。例えば不動産の価格予測であればインプットが専有面積や間取り、アウトプットが価格となります。

以下の簡単なデータを使って説明していきます。xの特徴は幾つあっても良いのですが、グラフで説明するためこのデータのxの特徴は一つだけです。

| x | y  |
|---|----|
| 0 | -1 |
| 1 | 1  |
| 2 | 3  |
| 3 | 5  |

![plot1](/img/unit5/plot1.png)

yを概算する線形関数を求めろと人間が言われたら直感的に分かると思います。

![plot2](/img/unit5/plot2.png)

このようにグラフにすると全てのデータが直線状に並んでいます。普通のデータこんなにクリーンではないですが、説明し易いので簡単なデータを選びました。

計算すると分かる通り、この線形関数は

$$ y = 2x - 1 $$

です。

これを数式にすると以下のようになります。\(y\)と\(h(x)\)は同じ意味です。\(n\)は特徴の数です。

$$ h(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \dots + \theta_nx_n $$

今回の例ではxの特徴が一つしかないので以下のようになります。

$$ h(x) = \theta_0 + \theta_1x_1 $$

\(\theta_0 = -1\), \(\theta_1 = 2\)です。

この数式はdot productで表せるのですが、その為には\(\theta\)と\(x\)を同じ長さにする必要があります。\(\theta_0\)はインターセプトなので、\(x_0 = 1\)にします。

$$
\begin{align}
h(x) & = \theta_0x_0 + \theta_1x_1 + \theta_2x_2 + \dots + \theta_nx_n \\
          &=  \displaystyle\sum_{i=0}^{n} \theta_ix_i  \\
           &= \theta^Tx
\end{align}
$$

\(x_1 = 3\)を計算してみましょう。

$$
\begin{align}
h(x) &=  \theta^Tx \\
&= 
\begin{bmatrix}
-1 & 2
\end{bmatrix}
\cdot
\begin{bmatrix}
1 \\ 3
\end{bmatrix} \\
&= -1 \cdot 1 + 2 \cdot 3 \\
&= 5
\end{align}
$$

xは１サンプルですが、全サンプル(X)を一度に計算した方が楽なので、以下の数式を使います。mをサンプル数とするとXが\(m \times n\), \(\theta\)が\(n \times 1\)なので\(h(X)\)が\(m \times 1\)となります。

\[ h(X) = X\theta \]

例のデータを一度に計算してみましょう。

\[ 
\begin{align} h(X) &= X\theta \\ 
&= \begin{bmatrix}
1 & 0 \\
1 & 1 \\
1 & 2 \\
1 & 3 \\
\end{bmatrix} 
\cdot 
\begin{bmatrix}-1 \\ 2
\end{bmatrix} \\ 
&= 
\begin{bmatrix}
-1 \\ 1 \\ 3 \\ 5
\end{bmatrix}
\end{align}
\]


上記は最適な\(\theta\)を使って計算したものですが、目的はこの\(\theta\)を見つけることです。その為にはまず\(\theta\)を適当に選び、\(h(x)\)が\(y\)とどれだけ違うかに応じて\(\theta\)を修正していきます。

[Andrew NgのLecture notes1の4ページ](http://cs229.stanford.edu/materials.html)に最適化の説明が書かれています。微分積分が関わる部分なのでここでの説明は割愛します。\(\theta\)をアップデートする数式は以下の通りです。

$$ \theta := \theta + \frac{\alpha}{m} (y - h(x))X $$

\(\alpha\)は*learning rate*と呼ばれるもので、どれだけ早く最適化するかを設定します。これが小さすぎると学習が遅く、大きすぎるとconverge(収束)しなくなります。コードを書いて実際にアルゴリズムを実行すると理解出来ると思います。

mで割る理由は、サンプル数によるエラー値の違いを無くすためです。mがないとサンプル数に応じて\(\alpha\)を微調整することになり大変です。

# Exercises

ではコードを書いていきましょう。
`eta`はlearning rateのことです。`fit`ではまず\(x_0\)を足します。

```python
class MyLinearRegression(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        return X

X = np.array([[0], [1], [2], [3]])
y = np.array([-1, 1, 3, 5])
regr = MyLinearRegression()
print(regr.fit(X, y))
```


```Output
[[1 0]
 [1 1]
 [1 2]
 [1 3]]
```

## Exercise 1

\(h(X) = X\theta\)を計算しましょう。

## Exercise 2

\(\theta := \theta + \frac{\alpha}{m} (y - h(x))X\)を計算しましょう。`m`と`self.eta`を使って下さい。

## Exercise 3

`predict`を書きましょう。Exercise1とほとんど同じです。

## Exercise 4

Linear Regressionは完成したんですが、一つ問題があります。サンプル数が多いと計算するのに時間がかかり学習が遅いです。*Stochastic Gradient Descent*(SGD)はこれを解決する為に１サンプルづつ計算しその都度weightをアップデートします。SGDの欠点はエラーの最小値にconvergeにしないことですが、実際はそこまで大きな問題ではありません。

weightをアップデートするコードを書きましょう。１サンプルづつなのでmはもう必要ありません。

# Diabetes Dataset

[Diabetes Dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)を使ってみました。特徴が１０個ありますが、視覚化したいので１つだけ使うことにしました。是非色んな特徴で試してみて下さい。

![diabetes](/img/unit5/diabetes.png)

