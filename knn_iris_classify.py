from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=4)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
scores = metrics.accuracy_score(y_test, y_pred)

print(scores)