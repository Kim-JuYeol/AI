import matplotlib.pylab as plt
import tensorflow as tf
import keras
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

#데이터 전처리
y = tf.keras.utils.to_categorical(y)

model = tf.keras.models.Sequential()

#은닉층
model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1], )))
model.add(tf.keras.layers.Dense(32, activation='relu'))
#출력층
model.add(tf.keras.layers.Dense(3))

model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

history = model.fit(X, y, epochs=100, batch_size=1)

loss = history.history['loss']
acc = history.history['accuracy']
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, acc, 'r', label='Accuracy')
plt.xlabel('epochs')
plt.ylabel('loss/acc')
plt.show()
