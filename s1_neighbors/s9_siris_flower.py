from sklearn import datasets
from sklearn.model_selection import train_test_split
from s8_final import MyKNeighborsClassifier

iris = datasets.load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.4)
neighbor = MyKNeighborsClassifier(n_neighbors=5, weights='uniform', p=2)
neighbor.fit(x_train, y_train)
print(neighbor.score(x_train, y_train))
print(neighbor.score(x_test, y_test))