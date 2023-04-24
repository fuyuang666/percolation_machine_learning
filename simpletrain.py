import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint



path = './L=50'
training_images = []
for filename in os.listdir(path):
        img = cv2.imread(path+'/'+filename)
        training_images.append(img)
training_images = np.array(training_images, dtype='float') / 255.0

training_labels = [0]*1500+[1]*1200+[0]*600+[1]*900
training_labels = np.array(training_labels)


path1 = './L=50_'
x_val = []
for filename in os.listdir(path1):
        img = cv2.imread(path1+'/'+filename)
        x_val.append(img)
x_val = np.array(x_val, dtype='float') / 255.0


y_val = [0]*500+[1]*400+[0]*200+[1]*300
y_val = np.array(y_val)
# training the neural network
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.L2(0.3)),
                # tf.keras.layers.Dense(32, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.L2(0.6)),
                tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])


filepath = 'simple_model.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,save_best_only=True,mode='max',period=2)
callbacks_list = [checkpoint]


model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

model.fit(training_images, training_labels, batch_size=128, epochs=500, shuffle=True, validation_data=(x_val, y_val), callbacks=callbacks_list)
