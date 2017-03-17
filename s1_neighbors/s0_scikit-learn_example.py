import numpy as np
from sklearn.neighbors import KNeighborsClassifier

## Inputは2d array(N, number of features)
x_train = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
## Targetは1d array(N, )
y_train = np.array([1,1,1,0,0,0])

## モデルのinitialization.
neighbor = KNeighborsClassifier()
## fitでモデルの学習
neighbor.fit(x_train, y_train)

## predictは複数のデータを取るので2d arrayを渡す
x_test = np.array([[1, 0], [-2, -2]])
print(neighbor.predict(x_test))