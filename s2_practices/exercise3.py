# StandardScaler で、複数の特徴をStandardizationする

from IPython import embed
import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class MyStandardScaler(object):
    def fit(self, X):
        # 各特徴の平均を計算する
        self.mean_ = np.mean(X, axis=0)
        # 各特徴の標準偏差を計算する
        self.scale_ = np.std(X - self.mean_, axis=0)
        return self

    def transform(self, X):
        # 各特徴をstandardizationする
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# csvを読み込む
os.chdir(sys.path[0])
df = pd.read_csv("HR_comma_sep.csv")

# csvを操作する
le = LabelEncoder()
salary_mapping = {'high': 3, 'medium': 2, 'low': 1}
df["salary"] = df["salary"].map(salary_mapping)
df.sales = le.fit_transform(df.sales)
print(df.head())

# csvからデータを得る
y = df.left.values
col = list(df.columns)
col.remove("left")
x = df[col].values

# csvから得たデータをStandardScalerに食わせてstandardizationする
# 3割は学習用データ、7割は評価用データ
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.3)
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# K近傍法を使う
# 学習用データ・評価用データを使ってそれぞれk近傍法モデルの精度を見る
neighbor = KNeighborsClassifier(n_neighbors=3, weights='uniform', p=2)
neighbor.fit(x_train, y_train)
print(neighbor.score(x_train, y_train))
print(neighbor.score(x_test, y_test))
