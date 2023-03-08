import cv2 as cv
import tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Input, Conv2D
import utilities as ut

# Reading the Image and converting it into GREYSCALE
img = cv.imread("images/puppy.jpg", cv.IMREAD_GRAYSCALE) 
img = cv.resize(img, (224, 224))

height, width = img.shape
print("Image Shape : ", img.shape)

cv.imshow("Image", img)
cv.waitKey(1000)

# Con2D MODEL 
model= keras.Sequential()
model.add(Conv2D(input_shape=(height, width, 1),
                 filters=64,
                 kernel_size=(3, 3))
        )
model.summary()

# Access Layer Parameter
filters, _ = model.layers[0].get_weights()
print("Filters Shape : ", filters.shape)

# Normalize the Filter
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Display Fliters
imgFilterArray = []

ind = 0
for i in range(0, 8):
    arr = []
    for j in range(0, 8):
        f = filters[:,:,:, ind]
        # f = cv.resize(f, (25, 25), interpolation=cv.INTER_NEAREST)
        ind+=1
        arr.append(f)
    imgFilterArray.append(arr)

# Stack all the imgFilterArray into single Image
stackedFltImg = ut.stackImages(30, imgFilterArray)
cv.imshow("Stacked Images", stackedFltImg)
cv.waitKey(0)