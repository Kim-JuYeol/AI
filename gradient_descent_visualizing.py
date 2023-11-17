import matplotlib.pyplot as plt
import numpy as np
import math

loss_func = lambda x: (x-3)**2 + 10

gradient = lambda x: 2*x - 6

x = 10
X = [10]
Y = [loss_func(x)]
learning_rate = 0.2
max_iterations = 100

for i in range(max_iterations):
    x = x - learning_rate * gradient(x)
    X.append(x)
    Y.append(loss_func(x))
    print("손실함수값(", x, ")=", loss_func(x))
    
print("최소값 = ", x)
x_min, x_max = min(X), max(X)
x_loss_func = np.arange(math.floor(x_min) - 1, math.ceil(x_max) + 2)

#손실함수 그리기
plt.plot(x_loss_func, loss_func(x_loss_func), color = 'blue', linewidth=3)

#최솟값 검은점으로 표시
plt.scatter(X, Y, color = 'black', s=30)
plt.show()
    