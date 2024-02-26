# TODO Import important libraries (im not sure if we need to import libraries)
from PIL import Image
from keras.preprocessing import image
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

### Preprocess image START ###

#Import the image(s) to be tested and 
path = 'TODO/' # Path to folder containing images in storage (e.g. 'diabetic-retinopathy-resized/resized_train/resized_train/' is what we used in the colab) 
listing = os.listdir(path)  
np.size(listing)

img_rows, img_cols = 200, 200 # input image dimensions
immatrix = []

for file in listing:
    base = os.path.basename(path + file)
    fileName = os.path.splitext(base)[0]
    imlabel.append(trainLabels.loc[trainLabels.image==fileName, 'level'].values[0])
    im = Image.open(input + file)
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
    immatrix.append(np.array(gray).flatten())

immatrix = np.asarray(immatrix)
### Preprocess image END ###


### Display Image START ###
import matplotlib.pyplot as plt
import matplotlib

img=immatrix[0].reshape(img_rows,img_cols) 
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
    model.add(Convolution2D(num_filters, num_conv, num_conv,
                            'valid',
                            input_shape=(img_cols, img_rows, 1)))
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

model.load_weights('my_model.weights.h5') #Load weights from the pre-trained model (Colab) 
### CNN Model END ###

### Prediction START ###
result = model.predict(immatrix)
if(result == 0)
    # TODO: Code to light up LED 0
elif(result == 1)
    # TODO: Code to light up LED 1
elif(result == 2)
    # TODO: Code to light up LED 2
elif(result == 3)
    # TODO: Code to light up LED 3
elif(result == 4)
    # TODO: Code to light up LED 4
### Prediction END ###

