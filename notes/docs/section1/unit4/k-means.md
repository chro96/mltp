今回はK-Meansという一番シンプルで有名なクラスタリングのモデルを勉強します。

以下のデータを２つのクラスターに分けたいとしましょう。

![scatter](/img/unit4/scatter.png)

まず*cluster center*(クラスターの中心値)をランダムに決めます。一番簡単な方法は、Xの中からランダムに選ぶ方法です。この場合`(1,1)`と`(1,2)`を選びました。Xと被らないように少しずらしてますが本当は同じ値です。

次に、それぞれのデータでどっちのcluster centerに近いかをEuclidean Distanceを使って計算します。この場合`(1,1)`以外の４つのデータは全て`(1,2)`の方に近いです。

![scatter2](/img/unit4/scatter2.png)

次にcluster centerをアップデートします。方法は簡単で、そのクラスターに属してるデータの平均値を取ります。`(1,1)`に関しては属してるデータが`(1,1)`しかないのでそのままです。

`(1,2)`は属してるデータが４つあります。x軸とy軸でそれぞれ平均を取るのでx軸は

$$ \frac{1+2+4+5}{4} = \frac{12}{4} = 3 $$

y軸は

$$ \frac{2+2+5+4}{4} = \frac{13}{4} = 3.25 $$

となります。

後は同じステップを繰り返すだけです。cluster centerがアップデートされたので、クラスターもそれに応じて変化したのが分かります。

![scatter3](/img/unit4/scatter3.png)

これを繰り返していくうちにcluster centerはこのようになります。

![scatter4](/img/unit4/scatter4.png)


# Exercises

今回も[scikit-learnのK-Means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)と同じAPIにします。

```python
import numpy as np

class MyKMeans(object):
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        if self.random_state:
            np.random.seed(self.random_state)
```

## Exercise 1

クラスターの初期値をn_clustersの分だけ作ります。Xからランダムに選び`self.cluster_centers_`に入れましょう。

## Exercise 2

次にEuclidean Distanceを計算します。k-NNの時に書いたので計算の方法は分かると思います。しかし今回はインプットが2d array(centers)と1d array(x)なので、xとそれぞれのクラスターの中央値とのEuclidean Distanceを計算して下さい。

numpyのbroadcastingを使えばループなしでいけます。

## Exercise 3

次に一番近いクラスターのindexをリターンするメソッドを書きましょう。`_distance`を使って下さい

## Exercise 4

やっとメインのアルゴリズムが書けます。`_nearest`を使ってxがどのクラスターに属するかを計算し`self.labels_`に入れています。
これを使ってXをクラスター毎に分けましょう。

```python
def fit(self, X):
    initial = np.random.permutation(X.shape[0])[:self.n_clusters]
    self.cluster_centers_ = X[initial]
    
    for _ in range(self.max_iter):
        self.labels_ = np.array([self._nearest(self.cluster_centers_, x) for x in X])
        X_by_cluster = # your code here
        return X_by_cluster
```


## Exercise 5

`self.cluster_centers_`をアップデートさせましょう。`X_by_cluster`の平均値を取ります。

## Exercise 6

学習には不要なのですが、どれだけ上手くクラスタリングされてるかを見るために*inertia*という指標を使います。
inertiaはそれぞれのcluster centerとそれに属してるデータの*Square Distance*の合計です。Square DistanceはEuclidean Distanceのルートを取らない版です。

例えばcluster center(1,2)とx(3,4)のSquare Distanceは

$$ (1-3)^2 + (2-4)^2 = 4 + 4 = 8 $$

です。

# Early Stopping

実際は何回目でcluster centerが動かなくなるか分からないので、max_iterを適当に選んでやります。
データが少ない場合はこれでも良いんですが、データが多い場合は無駄な学習時間を削減したいです。その為には`self.inertia_`がほとんど変化しなくなったらストップするという方法があります。
scikit-learnでは*tolerance*(tol)というパラメータがあり、どれくらい変化しなくなったら止めるかというのをここで設定することでが出来ます。

もし余裕があれば書いてみて下さい。

# 最適なクラスター数

何クラスターに分けたいというのが決まってる場合は問題ないのですが、多くの場合は幾つに分けたらいいか分からないというのが普通です。

最適なクラスター数を見つける一番簡単な方法は、クラスター数別にinertiaを見てinertiaの減少が著しいところを選びます。

以下の例だと、クラスター数が３の場合２の時よりもinertiaが大幅に減少しています。しかしそれ以上クラスター数を増やしてもそこまでinertiaが減少していないので最適なクラスター数は３と言えます。

![inertia](/img/unit4/inertia.png)

もっと複雑なテクニックもありますが、これだけ知っていれば十分でしょう。
