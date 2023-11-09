import numpy as np

epsilon = 0.0000001

def step_func(t):
    if t > epsilon: 
        return 1
    else:
        return 0

X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
    ])

y = np.array([0,0,0,1])
W = np.random.rand(len(X[0]))

def perceptron_fit(X, Y, epochs=10):
    global W
    eta = 0.2
    for t in range(epochs):
        print("epochs=", t, "====================")
        for i in range(len(X)):
            predict = step_func(np.dot(X[i], W))
            error = Y[i] - predict
            W += eta * error * X[i]
            print("현재 처리 입력=", X[i], "정답=", Y[i], "출력=", predict, "변경된 가중치=", W)
        print("====================")
        
def perceptron_predict(X,Y):
    global W
    for x in X:
        print(x[0], x[1], "->", step_func(np.dot(x, W)))
        

print("처음 가중치=", W)        
perceptron_fit(X, y, 6)
perceptron_predict(X, y)


