import tensorflow as tf
import cv2
import numpy as np

image = cv2.imread('person.jpg')
data = cv2.resize(image, (416, 416))
data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
data = np.expand_dims(data, 0)
model = tf.keras.models.load_model('checkpoints/yolov4-416')
pred = model(data)
