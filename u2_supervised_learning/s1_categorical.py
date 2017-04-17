import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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

neighbor = KNeighborsClassifier(n_neighbors=3, weights='uniform', p=2)

neighbor.fit(x_train, y_train)
print(neighbor.score(x_train, y_train))
print(neighbor.score(x_test, y_test))



























