# Import important libraries
from PIL import Image
from keras.preprocessing import image
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


import time
import board
import digitalio

IMG_SIZE = 128
num_classes = 5 # number of output classes

### Preprocess image START ###
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
    image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    return image

#Import the image(s) to be tested
path = '/mnt/usb1/images/' # Path to folder containing images in storage (e.g. 'diabetic-retinopathy-resized/resized_train/resized_train/' is what we used in the colab) 
imgSet = os.listdir(path)  

imgMatrix = []

for img in imgSet:
    processedImg = load_ben_color(path + img) 
    imgMatrix.append(np.array(processedImg)) # Append grayscaled image to a numpy matrix
imgMatrix = np.asarray(imgMatrix)
print ("Image acquired and processed.")
### Preprocess image END ###


### CNN Model START ###
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.models import load_model
model = load_model('/mnt/usb1/mymodelv3.keras') # Load Colab-trained model
### CNN Model END ###


### Prediction START ###
prediction = model.predict(imgMatrix) # outputs an array of size equal to the number of classes (5), predicted result is the ith index
print(prediction) # DELETE LATER
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

print("press the button!")

led0 = digitalio.DigitalInOut(board.D14)
led0.direction = digitalio.Direction.OUTPUT

led1 = digitalio.DigitalInOut(board.D15)
led1.direction = digitalio.Direction.OUTPUT

led2 = digitalio.DigitalInOut(board.D18)
led2.direction = digitalio.Direction.OUTPUT

led3 = digitalio.DigitalInOut(board.D23)
led3.direction = digitalio.Direction.OUTPUT

led4 = digitalio.DigitalInOut(board.D24)
led4.direction = digitalio.Direction.OUTPUT

button = digitalio.DigitalInOut(board.D4)
button.direction = digitalio.Direction.INPUT
button.pull = digitalio.Pull.UP

while True:
    led0.value = not button.value # light when button is pressed!
    led1.value =  button.value
    led2.value = not button.value
    led3.value =  button.value
    led4.value = not button.value


### Prediction END ###
