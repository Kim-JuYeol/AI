import matplotlib.pylab as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.datasets import boston_housing

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

model = tf.keras.models.Sequential()

#은닉층 2개
model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(13, )))
model.add(tf.keras.layers.Dense(32, activation='relu'))

model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=1)

loss = history.history['loss']
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

    