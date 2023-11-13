from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import numpy as np

clf = Perceptron(tol=1e-3, random_state=0)

iris = load_iris()

X= iris.data[:, (0,1)]

y = (iris.target == 0).astype(np.int64)

clf.fit(X,y)

print(clf.score(X,y))

plt.scatter(X[:, 0], X[:, 1], c=y, s=100)
plt.xlabel("x1")
plt.ylabel("x2")


x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() +1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() +1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.show()



    









