Naive Bayesはk-NNと同様シンプルなモデルです。よくtext classificationに使われます。

scikit-learnには[３種類のNaive Bayes](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes)があります。今回はMultinomialとBernoulliを作ります。

# Multinomial Naive Bayes

まずはMultinomial Naive Bayesについて説明します。[An Introduction to Information RetrievalのChapter13](https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html)の例を使います。

PDFはこちらから取得出来ます。
[https://nlp.stanford.edu/IR-book/](https://nlp.stanford.edu/IR-book/)

Table 13.1 にデータが載っています。これはドキュメントの種類を分類するtext classificationのデータです。training setは４つあり、それぞれのドキュメントに単語が含まれています。例えばdocument 1には"Chinese","Beijing","Chinese"という単語が含まれています。

<img width="576" alt="Screenshot 2017-03-27 12.05.41.png (71.1 kB)" src="https://img.esa.io/uploads/production/attachments/5475/2017/03/27/18257/404ba5fe-339d-408d-8112-e743bbb1e125.png">

当然実際ニュース記事や映画のレビューを分類するとなるとドキュメントに含まれる単語数はもっと多いですが、手でアルゴリズムを計算する為に小さいデータを利用しています。

一番右のカラムはクラスです。今回はクラスはこのドキュメントが中国に関するものかどうかなのでyesとnoの２種類だけです。

この４つのtraining dataを元に、document 5を予測します。

予測するには` p(c)`と`p(t|c)`の２パーツを計算します。

## p(c)

まずはp(c)を求めます。p(c)は単純にtraining setでのクラスcの割合です。
教科書では

$$ p(c) = p(c = yes) $$

$$ p(\bar{c}) = p(c = no) $$

としています。training set は全部で４つあり３つがyesなので

$$ p(c) = 3/4 $$

$$ p(\bar{c}) = 1/4 $$

となります。

## p(t|c)

次に`p(t|c)`を求めます。tは*term*の略で単語の種類のことを指します。それに対し*word*は一つ一つの単語のことを指します。例えばdocument 1ではtermは"Chinese"と"Beijing"の２つですが、wordは３つです。

`p(t|c)`は"そのクラスのドキュメントに含まれるtの総数"÷"そのクラスのドキュメントに含まれる総単語数"です。

`p(Chinese|c)`を計算してみましょう。
Chineseはdocument 1,2,3合わせて5つ含まれています。document 1,2,3の総単語数は8なので

$$ \hat{p}(Chinese|c) = 5 / 8 $$

となります。

最後にこれらを掛け合わせるのですが、一つ問題があります。training dataに一つも入っていない単語だと`p(t|c) = 0 / 8 = 0`になってしまい全体が0になります。これを防ぐために*smoothing*とうテクニックを使います。smoothingには幾つか種類がありますが、全termに1を足す*add-one smoothing*が最もシンプルな方法です。

よって`p(Chinese|c)`の場合、全体で6 terms(Chinese, Beijing, Shanghai, Macao, Tokyo, Japan)あるので分母に６を足し、分子に１を足します。

$$ \hat{p}(Chinese|c) = (5 + 1)  / (8 + 6) = 6 / 14 = 3/7 $$

本当はtraining setに出てくる全termの`p(t|c)`を`fit`時に計算するのですが、今回はtest setが１つだけなのでそれに含まれてるtermの分だけ計算します。

<img width="520" alt="Screenshot 2017-03-27 12.19.42.png (46.0 kB)" src="https://img.esa.io/uploads/production/attachments/5475/2017/03/27/18257/2e2a4134-c2ec-4e13-b985-fd425f7c2e55.png">

自分で確かめてみて下さい。

後はp258の以下の数式に従ってこれらをかけあわせます。

<img width="464" alt="Screenshot 2017-04-01 15.31.58.png (17.3 kB)" src="https://img.esa.io/uploads/production/attachments/5475/2017/04/01/18257/54441e55-089f-49b5-af5c-57a6b1b6f85b.png">

"Chinese"はdocument 5に３つ含まれてるので３乗します。先頭にp(c)が含まれているので注意しましょう。

<img width="388" alt="Screenshot 2017-03-27 12.49.52.png (22.7 kB)" src="https://img.esa.io/uploads/production/attachments/5475/2017/03/27/18257/79681dc9-6c35-4e06-9bc8-9e4416fed5e3.png">

`p(c|d5)`の方が高いので、document 5はyes、つまり中国に関するドキュメントと予測します。

上の結果を見て分かる通り、`p(c|d)`はもの凄く小さい数値になりがちです。ドキュメントが大きいと更にこれが小さくなり、floating point underflowが起こります。これを防ぐ為に、実際コードを書く時はlogを使います。（P258参照）

# Exercises

k-NNと同じく、[scikit-learnのMultinomialNB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB)と同じAPIにします。

```python
import numpy as np

np.set_printoptions(precision=6)

class MyMultinomialNB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        N = X.shape[0]
        # group by class
        separated = [X[np.where(y == i)[0]] for i in np.unique(y)]
        return separated

X = np.array([
    [2,1,0,0,0,0],
    [2,0,1,0,0,0],
    [1,0,0,1,0,0],
    [1,0,0,0,1,1]
])
y = np.array([0,0,0,1])
nb = MyMultinomialNB().fit(X, y)
print(nb)
```

Xは例のtraining setを数値に直したものです。全部で6 term(Chinese, Beijing, Shanghai, Macao, Tokyo, Japan)含まれているのでdimensionが６あります。

yは0が"yes"、1を"no"としています。

`__init__`のalphaはsmoothingのパラメータです。後で使うので今は無視して大丈夫です。

`fit`ではまずクラス毎にデータを分けます。`fit`は最終的には`self`をリターンするのですが、今は一旦無視して`separated`をリターンしています。

```
# output
[array([[2, 1, 0, 0, 0, 0],
       [2, 0, 1, 0, 0, 0],
       [1, 0, 0, 1, 0, 0]]), array([[1, 0, 0, 0, 1, 1]])]
```

## Exercise 1

`p(c)`を計算して`self.class_log_prior_`にアサインして下さい。その際logにするのを忘れないで下さい。普通に計算した後それぞれに`np.log`を使うだけです。

## Exercise 2

次に各クラス毎にそれぞれのtermをカウントします。`self.alpha`をsmoothingとして足すのを忘れないで下さい。

## Exercise 3

最後に`p(t|c)`を計算しましょう。これもlogにするのを忘れないで下さい。
これで`fit`は完成です。

## Exercise 4

次に`predict_log_proba`を書きましょう。これは`p(c|d)`のことです。
`self.feature_log_prob_`と`self.class_log_prior_`をここで使います。
log probabilityを使ってるので掛けずに足して下さい。

## Exercise 5

先程の`predict_log_proba`を使って一番高いものをリターンします。indexがクラスになってるのでindexをリターンすればOKです。

# Bernoulli Naive Bayes

Bernoulli Naive BayesはMultinomial Naive Bayesと違い各termが含まれてるかだけを考慮し、何回含まれてるかは無視します。場合によってはこちらの方が精度が高いです。

Multinomial NBではtest dataに含まれるtermの`p(t|c)`だけ使いますが、Bernoulli NBの場合はtest dataに含まれないtermを`1 - p(t|c)`として使います。

<img width="541" alt="Screenshot 2017-03-27 16.33.27.png (72.4 kB)" src="https://img.esa.io/uploads/production/attachments/5475/2017/03/27/18257/382f46ba-d572-438c-90fe-2ed853475bab.png">

<img width="494" alt="Screenshot 2017-03-28 16.21.20.png (36.3 kB)" src="https://img.esa.io/uploads/production/attachments/5475/2017/03/28/18257/8b5f5724-f7cd-45a3-85c7-b6066e8c253c.png">


# Exercises

## Exercise 6

`count`まではMultinomial Naive Bayesと同じです。`denominator`は`self.feature_prob`の分母にあたる部分なのでこれを使って`self.feature_prob`を完成させましょう。この段階ではlogにしません。

```python
class MyBernoulliNB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        N = X.shape[0]
        # group by class
        separated = [X[np.where(y == i)[0]] for i in np.unique(y)]
        # class prior
        self.class_log_prior_ = [np.log(len(i) / N) for i in separated]
        # count of each word
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha

        # number of documents in each class
        smoothing = 2 * self.alpha
        denominator = np.array([len(i) + smoothing for i in separated])
        # probability of each term
        self.feature_prob_ = # Your code here
        return self
```

## Exercise 7

`predict_log_proba`を計算しましょう。`self.feature_prob_`はここでlogにしましょう。

# binarize

これまではモデルの外で手動でデータをバイナリーにしていましたが、モデル内で自動的にバイナリーにしてくれた方が便利ですよね。
[scikit-learnのBernoulliNB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB)ではその機能が付いています。`binarize`というパラメータでスレッショルドを設定し、自動的にバイナリーにします。

```python
# s10_binarize.py

def __init__(self, alpha=1.0, binarize=0.0):
    self.alpha = alpha
    self.binarize = binarize

def fit(self, X, y):
    X = self._binarize_X(X)
    ...

def predict_log_proba(self, X):
    X = self._binarize_X(X)
    ...

def _binarize_X(self, X):
    return np.where(X > self.binarize, 1, 0) if self.binarize != None else X

nb = MyBernoulliNB(alpha=1, binarize=0.0).fit(X, y)
print(nb.predict(X_test))
```



