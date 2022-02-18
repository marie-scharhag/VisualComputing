from keras.utils.np_utils import to_categorical
from mlxtend.data import loadlocal_mnist
import numpy as np
from tensorflow import keras

digit = 4

# load local dataset
images_path_train = 'data/train-images-idx3-ubyte'
labels_path_train = 'data/train-labels-idx1-ubyte'
images_path_test = 'data/t10k-images-idx3-ubyte'
labels_path_test = 'data/t10k-labels-idx1-ubyte'

# reshape Iamge arrays
trainImages, trainLabels = loadlocal_mnist(images_path_train, labels_path_train)
trainImages = trainImages.reshape(60000, 28, 28)
testImages, testLabels = loadlocal_mnist(images_path_test, labels_path_test)
testImages = testImages.reshape(10000, 28, 28)

# only 100 images per digit to train from 5 to 9
sliced_images_train = []
sliced_labels_train = []
counts = {}
for e, i in zip(trainImages, trainLabels):
    if i > digit:
        if i in counts:
            if counts[i] < 100:
                counts[i] += 1
                sliced_images_train.append(e)
                sliced_labels_train.append(i)
        else:
            counts[i] = 1
            sliced_images_train.append(e)
            sliced_labels_train.append(i)
trainImages = np.array(sliced_images_train)
trainLabels = np.array(sliced_labels_train)

# only from 5 to 9
sliced_images_test = []
sliced_labels_test = []
for e, i in zip(testImages, testLabels):
    if i > digit:
        sliced_images_test.append(e)
        sliced_labels_test.append(i)
testImages = np.array(sliced_images_test)
testLabels = np.array(sliced_labels_test)

# convert Labels to one-hot vector
trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

# resize and normalize Images
image_size = trainImages.shape[1]
input_size = image_size * image_size

trainImages = np.reshape(trainImages, [-1, input_size])
trainImages = trainImages.astype('float32')/255

testImages = np.reshape(testImages, [-1, input_size])
testImages = testImages.astype('float32')/255

# load model
model = keras.models.load_model('data/model',compile=False)
model.summary()

# freeze hiddenlayers
for i in range(len(model.layers)):
    model.layers[i].trainable=False

model.pop()
model.pop()
model.pop()
model.pop()

new_model = keras.models.Sequential()
new_model.add(model)
new_model.add(keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
new_model.add(keras.layers.BatchNormalization())
new_model.add(keras.layers.Dropout(0.2))

new_model.add(keras.layers.Dense(10, activation='softmax'))

new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

new_model.summary()

# train model
new_model.fit(trainImages, trainLabels,epochs=5)

loss, acc = new_model.evaluate(testImages, testLabels)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))
