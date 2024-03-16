# Import important libraries
from keras.preprocessing import image
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers import GlobalAveragePooling2D, ZeroPadding2D, Input


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
    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image , (0,0) , sigmaX) ,-4 ,128)
    return image

#Import the image(s) to be tested

path = '/mnt/usb1/images/' # Path to folder containing images in storage (e.g. 'diabetic-retinopathy-resized/resized_train/resized_train/' is what we used in the colab) 
imgSet = os.listdir(path)  
if not os.path.exists('/mnt/usb1/processedImages'):
    os.mkdir('/mnt/usb1/processedImages') # Create a directory to hold processed images    
for img in imgSet:
    processedImg = load_ben_color(path + img) 
    imgName = os.path.basename(img)
    newPath = f"/mnt/usb1/processedImages/processed{imgName}"
    cv2.imwrite(newPath, processedImg)   # Save preprocessed images to another directory ...
    
imageSet = tf.keras.utils.image_dataset_from_directory(
           directory = "/mnt/usb1/processedImages",
           labels= None,
           label_mode=None,
           class_names=None,
           color_mode='rgb',
           image_size=(IMG_SIZE, IMG_SIZE),
           shuffle=False,
           seed=None,
           validation_split=None,
           subset=None,
           interpolation='bilinear',
           follow_links=False,
           crop_to_aspect_ratio=True
)
'''
imgMatrix = []
imgMatrix = np.asarray(imgMatrix)
'''
print ("Image acquired and processed.")
### Preprocess image END ###


### CNN Model START ###
from tensorflow.keras.applications.inception_v3 import InceptionV3
modelFile = 'mymodelv3.keras'
usbPath = f"/mnt/usb1/{modelFile}"
if not os.path.exists(modelFile):
    model = tf.keras.models.load_model(usbPath) # Load Colab-trained model from usb if not already on pi
else
    model = tf.keras.models.load_model(modelFile) 

### CNN Model END ###


### Prediction START ###
prediction = model.predict(imageSet) # outputs an array of size equal to the number of classes (5), predicted result is the ith index
print(prediction) # DELETE LATER
result = 0
for idx in range(num_classes):
    if prediction[0][idx] > result:
        result = idx
# Turn on Corresponding LED to display the result

for file in os.listdir("/mnt/usb1/processedImages"): # Delete images stored in the temporary directory
    os.remove(file) 


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
