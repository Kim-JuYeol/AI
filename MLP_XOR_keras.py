import numpy as np
import tensorflow as tf
import keras

model = tf.keras.models.Sequential()

#은닉층
model.add(tf.keras.layers.Dense(units=2, input_shape=(2,), activation='sigmoid'))
#출력층
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=0.3))

model.summary()

X = np.array([[0,0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model.fit(X,y, batch_size=1, epochs=10000)

print(model.predict(X))