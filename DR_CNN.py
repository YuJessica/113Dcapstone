# TImport important libraries
from PIL import Image
from keras.preprocessing import image
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

#import gpiod
#import time
#chip = gpiod.Chip('/dev/gpiochip4')
import time
import board
import digitalio


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
num_classes = 5 # number of output classes

from keras.models import load_model
model = load_model('my_model.keras') # Load Colab-trained model
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


#led_line = chip.get_line(LED)
#led_line.request(consumer="LED",type=gpiod.LINE_REQ_DIR_OUT)

#try:
#    led_line.set_value(1)
#    time.sleep(10)
#   led_line.set_value(0)
#finally:
 #   led_line.release()
### Prediction END ###
