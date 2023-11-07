import matplotlib.pylab as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

boston_x, boston_y = datasets.load_boston(return_X_y=True)

boston_x_area = boston_x[:, np.newaxis, 2]
boston_x_crim = boston_x[:, np.newaxis, 0]
boston_x_room = boston_x[:, np.newaxis, 5]

x_train, x_test, y_train, y_test = train_test_split(boston_x_area, boston_y, test_size=0.1, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)

plt.plot(x_test, y_pred, '.')
plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.show()
# 비소매 상업 지역이 차지하고 있는 비율이 클수록 집값이 내려가는 추세

x_train, x_test, y_train, y_test = train_test_split(boston_x_crim, boston_y, test_size=0.1, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)

plt.plot(x_test, y_pred, '.')
plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.show()

#범죄율이 높을수록 집값이 내려가는 추세

x_train, x_test, y_train, y_test = train_test_split(boston_x_room, boston_y, test_size=0.1, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)

plt.plot(x_test, y_pred, '.')
plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.show()

#방의개수가 많을수록 집값이 올라가는 추세
