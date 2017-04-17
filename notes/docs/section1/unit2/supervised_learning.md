前回k-Nearest Neighborsを勉強してSupervised Learningとは何かを理解しました。
しかしデータには色んな種類があり、どんなデータでもそのままClassifierに突っ込めば勝手に予測してくれるわけではありません。

今回はIris Flower Data setよりも少しだけ複雑なデータを使い、Supervised Learningの基礎を勉強していきます。

今回使うのはHuman Resourceのデータです。従業員の情報からその人が退職するかどうかを予測するというものです。
[https://www.kaggle.com/ludobenistant/hr-analytics](https://www.kaggle.com/ludobenistant/hr-analytics)

ページの下のPreviewの"Show Column Description"というところに各特徴の簡単な説明とデータタイプが書かれています。

## カテゴリーを数字に直す

sales(所属部署)とsalaryのデータはstringで、この場合どちらもカテゴリーです。stringをそのままk-NNに突っ込むわけにはいかないので、数字に直す必要があります。

以下のsalaryのようにpandasとpython dictionaryを使って数字に変換することも出来ますし、[scikit learnの label encoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)のlabel encoderを使って自動的に数字にすることも出来ます。

```python
#s1_categorical.py
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

le = LabelEncoder()

df = pd.read_csv("HR_comma_sep.csv")

salary_mapping = {'high': 3, 'medium': 2, 'low': 1}
df["salary"] = df["salary"].map(salary_mapping)

df.sales = le.fit_transform(df.sales)
print(df.head())

y = df.left.values
col = list(df.columns)
col.remove("left")
x = df[col].values
```

## Feature Scaling

データによってレンジが違います。例えばsatisfaction_levelやlast_evaluationのレンジは0~1です。average_monthly_hoursは他の特徴に比べてレンジが大きいのが分かります。これをこのままk-NNに入れると、average_monthly_hoursの違いが他の特徴の違いに比べて大きいため、他の特徴がほとんど考慮されなくなります。

どの特徴も均等に考慮するには、各特徴を同じ大きさにする必要があります。
その処理のことを*Feature Scaling*と呼びます。
[https://en.wikipedia.org/wiki/Feature_scaling](https://en.wikipedia.org/wiki/Feature_scaling)

その中でも、平均(mean)を0にし標準偏差(standard deviation)を1になるように特徴をスケールさせるメソッドを*Standardization*と呼びます。公式は以下の通りです。

$$ x' = \frac{x - \bar{x}}{\sigma} $$

x = [1, 5, 0] だとしてx'を求めてみましょう。

$$ \bar{x} = \frac{1 + 5 + 0}{3} = 2 $$

$$ x - \bar{x} = [1-2, 5-2, 0-2] = [-1, 3, -2] $$

$$ \sigma = \sqrt{ \frac{1^2 + 3^2 + (-2)^2}{3}} = \sqrt{\frac{14}{3}} = 2.160 $$

$$ x' = \frac{[-1,3,-2]}{2.160} = [-0.463,  1.389, -0.926] $$

Feature Scalingはほぼ全てのモデルを使う際に必要です。Feature Scalingが必要ない有名なモデルにDecision Treeがあります。

## Exercise 1

まずは一つの特徴だけスケールさせる関数を書きましょう。

## Exercise 2

[scikit-learnのscale](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html#sklearn.preprocessing.scale)はXをインプットとして取るので、各特徴をスケールさせます。

## Exercise 3

実際はTraining dataをTest dataに別々にscaleするのではなく、Training dataのmeanとstandard deviationを使って両方ともスケールします。
そのためには[StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)というクラスを使います。以下が典型的な使用例です。

```python
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
```

fitの部分は既に完成しているので、transformを完成させましょう。

## Overfitting

Supervised Learningの目的は将来のデータ、つまりTest dataを予測することです。ということはtraining accuracy(training dataでの精度)が幾ら高くでも意味がありません。training accuracyを高めることを頑張り過ぎてtest accuracyが低くなってしまうことを*Overfitting*と呼びます。

k-NNの場合、k=1だとtraining accuracyは必ず1.0になります。何故なら、そのデータ自身が隣人になるからです。しかしこれではgeneralizeしていません。
k=3の方がtest accuracyが高いです。

```python
# s6_overfitting.py

...


## Overfitting
neighbor = KNeighborsClassifier(n_neighbors=1, weights='uniform', p=2)
neighbor.fit(x_train, y_train)
print(neighbor.score(x_train, y_train)) # 1.0
print(neighbor.score(x_test, y_test)) # 0.9526

## Test accuracyはこっちの方が高い
neighbor = KNeighborsClassifier(n_neighbors=3, weights='distance', p=2)
neighbor.fit(x_train, y_train)
print(neighbor.score(x_train, y_train)) # 1.0
print(neighbor.score(x_test, y_test)) # 0.9550

```

今回のデータは擬似的に作られたものでありノイズが少ないので、全体的に精度が高くてあまり面白くないですが、実際のデータだとoverfittingの問題が顕著に表れます。training accuracyとtest accuracyの差が10%以上開いていたら確実にoverfittingなので、training accuracyとtest accuracyの差を縮めるようにパラメータを調整しましょう。

モデルによってoverfittingの対処法が違います。k-NNの場合は、kが少ない程overfitします。またweightsをdistanceにすると遠くのデータの存在が弱くなるのでoverfitしがちです。

モデルに関係なくoverfitを防ぐ一番簡単な方法がデータを増やすことです。少ないデータにNeural Networkのようなキャパシティの高いモデル、つまりtraining accuracyの高いモデルを使うと、どんなにパラメータをいじってもoverfitします。
よってどうやって安く大量の教師用データを集めるかというのは非常に重要な問題です。
