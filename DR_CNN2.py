# Import important libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
#import tensorflow._api.v2.compat.v1 
from keras import datasets, layers, models
#from keras.utils import Sequence
from keras.utils import Sequence, image_dataset_from_directory
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers import GlobalAveragePooling2D, ZeroPadding2D, Input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam


import time
import board
import digitalio

IMG_SIZE = 512
NUM_CLASSES = 5 # number of output classes

### Auxilary Functions START ###
import cv2
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image , (0,0) , sigmaX) ,-4 ,128)
    return image

def normalize(image):
    #Normalizes individual pixels of images
    image = tf.cast(image/255. ,tf.float32)
    return image
    
def loadImage(imgName, folderPath): 
    # name is basename of image, e.g., 16_right.jpeg
    # folderPath is path to directory of images
    image = load_ben_color(folderPath + imgName) 
    image = np.expand_dims(image, axis=0) # Convert image to tensor
    image = normalize(image)
    return image
### CNN Model START ###

from tensorflow.keras.applications.inception_v3 import InceptionV3

modelBase = InceptionV3(input_shape = (IMG_SIZE, IMG_SIZE, 3), # shape of input images (height, width, channels)
                              include_top = False,             # top layer is specific to ImageNet competition; thus, should not be used for this model
                              weights = 'imagenet'
                              )
# disable training of layers in the model to prevent overfitting data
for layer in modelBase.layers:
  layer.trainable = False
from keras.layers import BatchNormalization, multiply, Lambda

base_model = modelBase
pt_depth = base_model.output_shape[-1]
in_lay = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
pt_features = base_model(in_lay)
bn_features = BatchNormalization()(pt_features)

# here we do an attention mechanism to turn pixels in the GAP on an off

attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(Dropout(0.5)(bn_features))
attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
attn_layer = Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
attn_layer = Conv2D(1,
                    kernel_size = (1,1),
                    padding = 'valid',
                    activation = 'sigmoid')(attn_layer)

# fan it out to all of the channels
up_c2_w = np.ones((1, 1, 1, pt_depth))
up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same',
               activation = 'linear', use_bias = False, weights = [up_c2_w])
up_c2.trainable = False
attn_layer = up_c2(attn_layer)
mask_features = multiply([attn_layer, bn_features])
gap_features = GlobalAveragePooling2D()(mask_features)
gap_mask = GlobalAveragePooling2D()(attn_layer)
# to account for missing values from the attention model
gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
gap_dr = Dropout(0.25)(gap)
dr_steps = Dropout(0.25)(Dense(128, activation = 'relu')(gap_dr))
out_layer = Dense(NUM_CLASSES, activation = 'softmax')(dr_steps)
model = Model(inputs = in_lay, outputs = [out_layer])

model.compile(optimizer = 'adam', 
              loss = 'sparse_categorical_crossentropy',    
              metrics = ['sparse_categorical_accuracy'])
model.load_weights('/mnt/usb1/mymodel.weights.h5')

### CNN Model END ###


print ("Image acquired and processed.")
### Preprocess image END ###

### Prediction START ###
for classNum in range(NUM_CLASSES):
    print(f"Testing images of type {classNum}.")
    folderPath = f'/mnt/usb1/images/{classNum}/'
    for imgFile in os.listdir(folderPath):
        imgName = os.path.basename(imgFile)
        testImage = loadImage(imgName, folderPath)
        prediction = model.predict(testImage) # outputs an array of size equal to the number of classes (5), predicted result is the ith index
                
        # Turn on Corresponding LED to display the result
        LED = 0
        led0 = digitalio.DigitalInOut(board.D15)
        led0.direction = digitalio.Direction.OUTPUT
        
        led1 = digitalio.DigitalInOut(board.D18)
        led1.direction = digitalio.Direction.OUTPUT
        
        led2 = digitalio.DigitalInOut(board.D23)
        led2.direction = digitalio.Direction.OUTPUT
        
        led3 = digitalio.DigitalInOut(board.D24)
        led3.direction = digitalio.Direction.OUTPUT
        
        led4 = digitalio.DigitalInOut(board.D25)
        led4.direction = digitalio.Direction.OUTPUT

        result = 0
        for idx in range(NUM_CLASSES):
            #print(f"The probability of class {idx} is {prediction[0][idx]}") #DEBUGGING
            if prediction[0][idx] >= prediction[0][result]:
                result = idx
                
        match result:
            case 0: 
                LED = 15
                led0.value = True        
                time.sleep(3)
                led0.value = False
            case 1: 
                LED = 18
                led1.value = True
                time.sleep(3)
                led1.value = False
            case 2: 
                LED = 23
                led2.value = True
                time.sleep(3)
                led2.value = False
            case 3: 
                LED = 24
                led3.value = True
                time.sleep(3)
                led3.value = False
            case 4: 
                LED = 25
                led4.value = True
                time.sleep(3)
                led4.value = False
            # Display an error message if none of the above classes are detected by the image
            case _:
                print("Error detected, image is assigned to unknown class/type.")
                led0.value = False # light when button is pressed!
                led1.value = False
                led2.value = False
                led3.value = False
                led4.value = False
                
        print(f"    For image from class {classNum}, {imgName}: Model guesses type {result} with a probability of {prediction[0][result]}.")

#print("press the button!")



#button = digitalio.DigitalInOut(board.D4)
#button.direction = digitalio.Direction.INPUT
#button.pull = digitalio.Pull.UP

#while True:
#   led0.value = not button.value # light when button is pressed!
#  led1.value =  button.value
#    led2.value = not button.value
#    led3.value =  button.value
#    led4.value = not button.value


### Prediction END ###
