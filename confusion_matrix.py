import matplotlib.pyplot as plt

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

knn = KNeighborsClassifier(n_neighbors=6)

x_train, x_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2)

knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
disp = metrics.plot_confusion_matrix(knn, x_test, y_test)
plt.show()
print(f"{metrics.classification_report(y_test, y_pred)}\n")
