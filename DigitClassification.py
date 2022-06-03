# Greg Newby
# 5-11-22
# This program is the image classification program for handwritten digits.
import datetime

import cv2
import keras.models
import keras.saving.save
import numpy
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import TensorBoard

# This function parses through the two partitions of the MNIST dataset and gives
# the number of digits in each class
def dataset_breakdown():
    # load dataset
    (img_train, label_train), (img_test, label_test) = mnist.load_data()

    zero = 0
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0
    six = 0
    seven = 0
    eight = 0
    nine = 0
    # add values that are in the training partition
    print("Images in Training Partition: ", label_train.shape[0])
    for x in range(label_train.shape[0]):
        if label_train[x] == 0:
            zero += 1
        elif label_train[x] == 1:
            one += 1
        elif label_train[x] == 2:
            two += 1
        elif label_train[x] == 3:
            three += 1
        elif label_train[x] == 4:
            four += 1
        elif label_train[x] == 5:
            five += 1
        elif label_train[x] == 6:
            six += 1
        elif label_train[x] == 7:
            seven += 1
        elif label_train[x] == 8:
            eight += 1
        elif label_train[x] == 9:
            nine += 1

    # add values that are in the testing partition
    print("Images in Testing Partition: ", label_test.shape[0])
    for x in range(label_test.shape[0]):
        if label_test[x] == 0:
            zero += 1
        elif label_test[x] == 1:
            one += 1
        elif label_test[x] == 2:
            two += 1
        elif label_test[x] == 3:
            three += 1
        elif label_test[x] == 4:
            four += 1
        elif label_test[x] == 5:
            five += 1
        elif label_test[x] == 6:
            six += 1
        elif label_test[x] == 7:
            seven += 1
        elif label_test[x] == 8:
            eight += 1
        elif label_test[x] == 9:
            nine += 1

   # Export the number of values in each class
    print("\nZeros in MNIST Dataset: ", zero)
    print("Ones in MNIST Dataset: ", one)
    print("Twos in MNIST Dataset: ", two)
    print("Threes in MNIST Dataset: ", three)
    print("Fours in MNIST Dataset: ", four)
    print("Fives in MNIST Dataset: ", five)
    print("Sixes in MNIST Dataset: ", six)
    print("Sevens in MNIST Dataset: ", seven)
    print("Eights in MNIST Dataset: ", eight)
    print("Nines in MNIST Dataset: ", nine)


# This function creates the model that will identify handwritten
def build_digit_CNN():
    NAME = "Digit-CNN".format((datetime.time()))

    tensorboard = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NAME))

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

    # training the model for 8 epochs
    # previously trained for 10 epochs but was over fitting and the validation accuracy started to decrease
    model.fit(img_train, lbl_train, batch_size=128, epochs=8, validation_data=(img_test, lbl_test), callbacks=[tensorboard])

    result = model.evaluate(img_test, lbl_test)
    print('\nTest loss , Test accuracy: ', result)

    # save the model
    model.save("digit_CNN.h5")

# This this takes an image file and resizes it. It returns an array that fits the model and can then be predicted
def fit_model(img_file):
    dimension = 28          # diminsions of the MNIST library files (28 pixels x 28 pixels)
    img_array = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(img_array, (dimension, dimension))
    return resized_array.reshape(1, dimension, dimension, 1).astype("float32") / 255

# This method takes a file and calls another function to fit it to a loaded model and then
# predicts what digit the original file was.
def classify_digit(img_file):
    model = keras.saving.save.load_model("digit_CNN.h5")
    classified = model.predict([fit_model(img_file)])
    digit = numpy.argmax(classified)
    return digit



#################### main ############################################################################################################

build_digit_CNN()

#Loop continually asks for image files to be classified until user requests to stop
stop = False
while(stop != True):
    try:
        branch = int(input("\n\t1 - image prediction"
                       "\n\t2 - database summary"
                       "\n\t3 - retrain model and get accuracy"
                       "\n\t4 - quit"
                       "\nEnter the number of the path you would like to follow: "))
        if branch == 1:
            try:
                filename = input("\nEnter the filename of the image that you would like to test: ")
                prediction = classify_digit(filename)
                print("The image in the file is: ", prediction)
            except cv2.error:
                print("FILE CANNOT BE FOUND! Please enter a valid filename.\n")
        elif branch == 2:
            dataset_breakdown()
        elif branch == 3:
            build_digit_CNN()
        elif branch == 4:
            stop = True
        else:
            print("Enter a value 1-4.")
    except ValueError:
        print("Incorrect value entered. Please enter one of the values listed.")



