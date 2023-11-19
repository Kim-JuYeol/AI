import numpy as np

def actf(x):
    return np.maximum(x,0)

def actf_deriv(x):
    out_x = np.where(x > 0, 1, 0)
    return out_x
    

inputs, hiddens, outputs = 2, 3, 1
learning_rate = 0.2

X = np.array([[1,1]])
T = np.array([[0]])

W1 = np.array([[0.80, 0.40, 0.30],
               [0.20, 0.90, 0.50]])
W2 = np.array([[0.30], [0.50], [0.90]])


def predict(x):
    layer0 = x
    Z1 = np.dot(layer0, W1) #첫번째 은닉층의 총입력 Z1 = XW1 + B1
    layer1 = actf(Z1) # layer1 = H1
    Z2 = np.dot(layer1, W2) #두번째 은닉층의 총입력 Z2 = H1(첫번째 은닉층의 출력)W2 + B2
    layer2 = actf(Z2) #layer2 = y (H2)
    return layer0, layer1, layer2

def fit(epoch=10):
    global W1, W2
    for i in range(epoch):
        for x, y in zip(X, T):
            x = np.reshape(x, (1,-1))
            y = np.reshape(y, (1,-1))
            layer0, layer1, layer2 = predict(x)
            layer2_error = layer2-y
            layer2_delta = layer2_error*actf_deriv(layer2)
            layer1_error = np.dot(layer2_delta, W2.T)
            layer1_delta = layer1_error*actf_deriv(layer1)
            W2 += -learning_rate*np.dot(layer1.T, layer2_delta)
            W1 += -learning_rate*np.dot(layer0.T, layer1_delta)

def test():
    for x, y in zip(X, T):
        x = np.reshape(x, (1,-1))
        layer0, layer1, layer2 = predict(x)
        print(x, y, layer2)
        print("오차 = " , ((y - layer2)**2)/2)
        
fit()
test()
