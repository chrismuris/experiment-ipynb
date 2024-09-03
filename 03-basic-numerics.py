import numpy as np
import scipy as scipy
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)


b = [2, 3, 4]
b.reverse()

[x ** 2 for x in b]

a_vec = np.array([1, 2, 3])
b_vec = np.array([4, 5, 6])

a_vec.dot(b_vec)

# Define a matrix using a NumPy array
C_mat = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

d_vec = C_mat @ a_vec

[d / 2 for d in d_vec]