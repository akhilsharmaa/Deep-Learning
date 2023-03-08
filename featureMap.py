import cv2 as cv
import tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Input, Conv2D
import utilities as ut

img = cv.imread("images/puppy.jpg", cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (224, 224))

height, width = img.shape
print("Image Shape : ", img.shape)

cv.imshow("Image", img)
cv.waitKey(1000)

model= keras.Sequential()
model.add(Conv2D(
                input_shape=(224, 224, 1),
                filters=64, 
                kernel_size=(3, 3))
          )

model.summary()

images = np.array([img])

feature_map = model.predict([images])
print("Feature_map : ", feature_map.shape)

# Display FeatureMaps

# feature_img = feature_map[0, :, :, 0]
# plt.imshow(feature_img)
# plt.show()

for i in range(64):
    feature_img = feature_map[0, :, :, i]
    ax = plt.subplot(8, 8, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(feature_img, cmap='gray')
plt.show()
