import numpy as np

def actf(x):
    return np.maximum(x,0)

inputs, hiddens, outputs = 2, 3, 1
learning_rate = 0.2

X = np.array([[1,1]])
T = np.array([[0]])

W1 = np.array([[0.80, 0.40, 0.30],
               [0.20, 0.90, 0.50]])
W2 = np.array([[0.30], [0.50], [0.90]])


def predict(x):
    layer0 = x
    Z1 = np.dot(layer0, W1) #첫번째 은닉층의 총입력 Z1 = XW1
    layer1 = actf(Z1) # layer1 = H1
    Z2 = np.dot(layer1, W2) #두번째 은닉층의 총입력 Z2 = H1(첫번째 은닉층의 출력)W2
    layer2 = actf(Z2) #layer2 = y (H2)
    return layer0, layer1, layer2

def test():
    for x, y in zip(X, T):
        x = np.reshape(x, (1,-1))
        layer0, layer1, layer2 = predict(x)
        print(x, y, layer2)
        print("오차 = " , ((y - layer2)**2)/2)
        
test()

