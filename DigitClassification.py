# Greg Newby
# 5-11-22
# This program is the image classification program for handwritten digits.
import keras.models
import keras.saving.save
import numpy
import tensorflow
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
import cv2

# This function creates the model that will identify handwritten
def build_digit_CNN():

    # Load the dataset
    # MNIST is the data set of images with size of 28 pixels by 28 pixels.
    # There are 70,000 images.
    # We will use 60,000 for training images and 10,000 for test images.
    (img_train, label_train), (img_test, label_test) = mnist.load_data()

    # reshape data image data into vectors and normalize by dividing by 255 (pixel values)
    img_train = img_train.reshape(60000, 28, 28, 1).astype("float32") / 255
    img_test = img_test.reshape(10000, 28, 28, 1).astype("float32") / 255

    label_train = label_train.astype("float32")
    label_test = label_test.astype("float32")

    #One hot encoding the digit categories
    digit_classes = 10
    #print("Shape before one hot encoding: ", label_train.shape)
    lbl_train = np_utils.to_categorical(label_train, digit_classes)
    lbl_test = np_utils.to_categorical(label_test, digit_classes)
    #print("Shape after encoding: ", lbl_train.shape)

    # start building model by adding layers
    model = Sequential()

    #convolutional layer
    model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=(1,1)))

    # flatten convolutional layer
    model.add(Flatten())

    # hidden layer
    model.add(Dense(100, activation='relu'))
    # output layer
    model.add(Dense(10, activation='softmax'))

    # compiling the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # training the model for 10 epochs
    model.fit(img_train, lbl_train, batch_size=128, epochs=10, validation_data=(img_test, lbl_test))

    result = model.evaluate(img_test, lbl_test)
    print('Test loss , Test accuracy: ', result)

    # save the model
    model.save("digit_CNN.h5")

def fit_model(img_file):
    dimension = 28
    img_array = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(img_array, (dimension, dimension))
    return resized_array.reshape(1, dimension, dimension, 1).astype("float32") / 255

def classify_digit(img_file):
    model = keras.saving.save.load_model("digit_CNN.h5")
    classified = model.predict([fit_model(img_file)])
    digit = numpy.argmax(classified)
    return digit



#################### main ############################################################################################################

build_digit_CNN()

# Loop continually asks for image files to be classified until user requests to stop
stop = False
while(stop != True):
    try:
        filename = input("\nEnter the filename of the image that you would like to test: ")
        prediction = classify_digit(filename)
        print("The image in the file is: ", prediction)
    except cv2.error:
        print("FILE CANNOT BE FOUND! Please enter a valid filename.\n")


    continU = input("Press 'N' to quit or any other key to continue: ")
    if(continU == 'N' or continU == 'n'):
        stop = True


