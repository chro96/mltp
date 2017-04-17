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


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.3)

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

## Overfitting
neighbor = KNeighborsClassifier(n_neighbors=1, weights='uniform', p=2)
neighbor.fit(x_train, y_train)
print(neighbor.score(x_train, y_train))
print(neighbor.score(x_test, y_test))

## Test accuracyはこっちの方が高い
neighbor = KNeighborsClassifier(n_neighbors=3, weights='distance', p=2)
neighbor.fit(x_train, y_train)
print(neighbor.score(x_train, y_train))
print(neighbor.score(x_test, y_test))
