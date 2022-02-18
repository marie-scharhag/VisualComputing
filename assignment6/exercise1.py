from keras.utils.np_utils import to_categorical
from mlxtend.data import loadlocal_mnist
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

digit = 9
outputlayer = 10

hiddenlayers = 5
hidden_units = 100

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

# # count the number of unique train labels and test Labels
# unique, counts = np.unique(trainLabels, return_counts=True)
# print("Train labels: ", dict(zip(unique, counts)))
# unique, counts = np.unique(testLabels, return_counts=True)
# print("\nTest labels: ", dict(zip(unique, counts)))
#
# # sample 25 random mnist digits from train dataset
# indexes = np.random.randint(0, trainImages.shape[0], size=25)
# images = trainImages[indexes]
# labels = trainLabels[indexes]
#
# # plot the 25 digits
# plt.figure(figsize=(5, 5))
# for i in range(len(indexes)):
#     plt.subplot(5, 5, i + 1)
#     image = images[i]
#     plt.imshow(image, cmap='gray')
#     plt.axis('off')
# plt.show()
# plt.close('all')


# just numbers from 0-4
def slice_dataset(images, labels, digits):
    sliced_images = []
    sliced_labels = []

    for e, i in zip(images, labels):
        if i <= digits:
            sliced_images.append(e)
            sliced_labels.append(i)
    return np.array(sliced_images), np.array(sliced_labels)


trainImages, trainLabels = slice_dataset(trainImages, trainLabels, digit)
testImages, testLabels = slice_dataset(testImages, testLabels, digit)

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

# define model
# inputlayer
model = keras.models.Sequential([keras.layers.Flatten(input_dim=input_size)])
# hiddenlayers
for l in range(hiddenlayers):
    model.add(keras.layers.Dense(hidden_units, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
    # Bei aktivierter Batch Normalisierung ist der Lernkurve flacher, dafuer steigt die Genauigkeit im Testing.
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))
# output layer
model.add(keras.layers.Dense(outputlayer, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
callback = keras.callbacks.ModelCheckpoint('data/cp.ckpt')
model.fit(trainImages, trainLabels, epochs=5)

model.save('data/model')

# evaluate model
loss, acc = model.evaluate(testImages, testLabels)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))
