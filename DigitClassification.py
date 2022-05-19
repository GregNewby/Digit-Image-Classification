# Greg Newby
# 5-11-22
# This program is the image classification program for handwritten digits.


import tensorflow
from keras.datasets import mnist

# Load the dataset
(img_train, lbl_train), (img_test, lbl_test) = mnist.load_data()

# reshape data image data into vectors
img_train = img_train.reshape(60000, 784).astype("float32") / 255
img_test = img_test.reshape(10000, 784).astype("float32") / 255

lbl_train = lbl_train.astype("float32")
lbl_test = lbl_test.astype("float32")

