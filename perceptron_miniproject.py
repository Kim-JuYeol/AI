from sklearn.linear_model import Perceptron
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np

clf = Perceptron(tol=1e-3, random_state=0)

#학생의 키,몸무게
X = [[160, 55], [165,48], [163,43], [170,80], [175,76], [180,70]]
X = np.array(X)
# 학생의 성별 0 = 여자 1 = 남자
y = [0,0,0,1,1,1]

clf.fit(X, y)

print(accuracy_score(clf.predict(X), y))
#데이터를 그래프 위에 표시

plt.scatter(X[:, 0], X[:, 1], c=y, s=100)
plt.xlabel("x1")
plt.ylabel("x2")


x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() +1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() +1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.show()