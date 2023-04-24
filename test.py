import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf

#test the machine learning model
accuracy = np.zeros(14)

path = './L=50__'
test_images = []
for filename in os.listdir(path):
    img = cv2.imread(path + '/' + filename)
    test_images.append(img)
test_images = np.array(test_images, dtype='float')/255.0

# test_labels = ['0.1']*100+['0.2']*100+['0.3']*100+['0.4']*100+['0.55']*100+['0.56']*100+['0.57']*100+['0.58']*100+['0.59']*100+['0.5']*100+\
#                   ['0.61']*100+['0.62']*100+['0.63']*100+['0.64']*100+['0.65']*100+['0.66']*100+['0.6']*100+['0.7']*100+['0.8']*100+['0.9']*100
# test_labels = np.array(test_labels)
# lb = LabelBinarizer()
# test_labels = lb.fit_transform(test_labels)

test_labels = [0]*500 + [1]*400 + [0]*200 + [1]*300
test_labels = np.array(test_labels)

# model = tf.keras.models.load_model("simple_model.h5")
# accuracy = model.predict(test_images)
# train = pd.DataFrame(data=accuracy)
# train.to_csv('acc.csv', index=False)
model = tf.keras.models.load_model("simple_model.h5")
for loop in range(0, len(test_images), 100):
    eval = model.evaluate(test_images[loop:loop+100], test_labels[loop:loop+100], batch_size=64)
    accuracy[int(loop/100)] = eval[1]

print(accuracy)