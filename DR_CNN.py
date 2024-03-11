# TImport important libraries
from PIL import Image
from keras.preprocessing import image
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

### Preprocess image START ###

#Import the image(s) to be tested
path = '/mnt/usb1/images/' # Path to folder containing images in storage (e.g. 'diabetic-retinopathy-resized/resized_train/resized_train/' is what we used in the colab) 
listing = os.listdir(path)  
np.size(listing)

img_rows, img_cols = 200, 200 # input image dimensions
imgMatrix = []

for file in listing:
    baseImg = Image.open(path + file) # Base image
    img = baseImg.resize((img_rows,img_cols)) # Reduce image size to 200x200 pixels
    grayImg = img.convert('L') # Convert image to grayscale
    imgMatrix.append(np.array(grayImg)) # Append grayscaled image to a numpy matrix
imgMatrix = np.asarray(imgMatrix)
print ("Image acquired and processed.")
### Preprocess image END ###


### Display Image START ###
import matplotlib.pyplot as plt
import matplotlib

img=imgMatrix[0].reshape(img_rows,img_cols) 
plt.imshow(img)
plt.imshow(img,cmap='gray')
### Display Image END ###


### CNN Model START ###
num_filters = 32 # number of convolutional filters to use
num_pool = 2 # size of pooling area for max pooling
num_conv = 3 # convolution kernel size
num_classes = 5 # number of output classes

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD,RMSprop,Adam

model = Sequential()
model.add(Convolution2D(num_filters, num_conv, num_conv, 'valid', input_shape=(img_cols, img_rows, 1)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(num_filters, num_conv, num_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(num_pool, num_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.load_weights('/mnt/usb1/my_model.weights.h5') #Load weights gathered from the pre-trained model (Colab)
### CNN Model END ###

### Prediction START ###
import gpiod
import time

prediction = model.predict(imgMatrix) # outputs an array of size equal to the number of classes (5), predicted result is the ith index
result = 0
for idx in range(num_classes):
    if prediction[0][idx] > result:
        result = idx
# Turn on Corresponding LED to display the result
LED = 0
match result:
    case 0: 
        LED = 14
        print("No diabetic retinopathy detected (type 0).")
    case 1: 
        LED = 15
        print("Mild nonproliferative diabetic retinopathy detected (type 1).")
    case 2: 
        LED = 18
        print("Moderate nonproliferative diabetic retinopathy detected (type 2).")
    case 3: 
        LED = 23
        print("Severe nonproliferative diabetic retinopathy detected (type 3).")
    case 4: 
        LED = 24
        print("Proliferative diabetic retinopathy detected (type 4).")
    # Display an error message if none of the above classes are detected by the image
    case _:
        print("Error detected, image is assigned to unknown class/type.")

chip = gpiod.Chip('gpiochip4')
led_line = chip.get_line(LED_PIN)
led_line.request(consumer="LED",type=gpiod.LINE_REQ_DIR_OUT)

try:
    led_line.set_value(1)
    time.sleep(10)
    led_line.set_value(0)
finally:
    led_line.release()
### Prediction END ###
