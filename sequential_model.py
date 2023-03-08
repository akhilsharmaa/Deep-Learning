import cv2 as cv
import tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Input

# Load and Preprocess Image
img = cv.imread("images/puppy.jpg")

height, width, channel = img.shape

cv.imshow("Image", img)
print("Image Shape : ", img.shape)

# Create a Sequential Model
model = keras.Sequential()
model.add(Input(shape=(height, width, channel)))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(2))
model.summary()

preprossed_img = np.array([img])
result = model(preprossed_img)

print(result.shape)
cv.waitKey(0)
