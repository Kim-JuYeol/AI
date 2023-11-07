import matplotlib.pylab as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

diabetes_x, diabetes_y = datasets.load_diabetes(return_X_y=True)

print(diabetes_x.data.shape)

diabetes_x_bmi = diabetes_x[:, np.newaxis, 2]
diabetes_x_age = diabetes_x[:, np.newaxis, 0]

x_train, x_test, y_train, y_test = train_test_split(diabetes_x_bmi, diabetes_y, test_size=0.1, random_state=0)
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)

plt.plot(x_test, y_pred, '.')
plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(diabetes_x_age, diabetes_y, test_size=0.1, random_state=0)
regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)

plt.plot(x_test, y_pred, '.')
plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.show()


