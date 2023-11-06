import matplotlib.pylab as plt
from sklearn import linear_model

reg = linear_model.LinearRegression()

x = [[174], [152], [138], [128], [186]]
y = [71, 55, 46, 38, 88]

reg.fit(x,y)

print(reg.predict([[167]])) # 키 167일때 예측 몸무게

plt.scatter(x,y, color='black') # 검은 점은 정답

y_pred = reg.predict(x) # 선형회귀 학습

plt.plot(x, y_pred, color = 'blue', linewidth=3)
plt.show()


