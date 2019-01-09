import tensorflow as tf
from tensorflow.keras import layers
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from random import shuffle



def one_hot_label(img):
    label = img.split(' ')[0]
    if label == 'dog':
        ohl = np.array([1,0])
        return ohl
    elif label == 'cat':
        ohl = np.array([0,1])
        return ohl

train_data = 'C:\\Users\Jason\Documents\School\Machine Learning\project\\test\canddtrain'
train_images = []
for i in tqdm(os.listdir(train_data)):
    path = os.path.join(train_data, i)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64,64))
    train_images.append([np.array(img), one_hot_label(i)])
    shuffle(train_images)

x_train = np.array([i[0] for i in train_images]).reshape(-1,64,64,1)
y_train = np.array([i[1] for i in train_images])

test_data = 'C:\\Users\Jason\Documents\School\Machine Learning\project\\test\canddtest'
test_images =[]
for i in tqdm(os.listdir(test_data)):
    path = os.path.join(test_data, i)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64,64))
    test_images.append([np.array(img), one_hot_label(i)])

x_test = np.array([i[0] for i in test_images]).reshape(-1,64,64,1)
y_test = np.array([i[1] for i in test_images])

y_ = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
y_t = np.array([1,1,1,1,1,0,0,0,0,0])

model = tf.keras.models.Sequential([
  tf.keras.layers.InputLayer(input_shape=[64,64,1]),
  tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding ='same', activation='relu'),
  tf.keras.layers.MaxPool2D(pool_size=(2,2)),

  tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding ='same', activation='relu'),
  tf.keras.layers.MaxPool2D(pool_size=(2,2)),

  tf.keras.layers.Dropout(rate=0.25),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(rate=0.5),
  tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)
])


model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=16)
print(x_test.shape)
print(y_test.shape)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy: ', test_acc)

fig = plt.figure(figsize=(14,14))
for cnt, data in enumerate(test_images[0:22]):
    y = fig.add_subplot(6,5,cnt+1)
    img = data[0]
    data = img.reshape(1,64,64,1)
    model_out = model.predict([data])

    if np.argmax(model_out) == 1:
        str_label = 'cat'
    else:
        str_label = 'dog'

    y.imshow(img, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
