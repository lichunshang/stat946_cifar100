import os
import pickle
import numpy as np
from keras.utils import np_utils
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import tools

dir_path = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(dir_path, "train_data")

with open(train_data_path, 'rb') as f:
    train_data = pickle.load(f)
    train_label = pickle.load(f)

train_data = tools.reshape(train_data)
print('train data shape:', train_data.shape)
print(train_data.shape[0], 'train samples')
train_data = train_data.astype('float32')
train_data /= 255.0

##### TRAINING ######
np.random.seed(2017)
batch_size = 256  # batch size
num_classes = 100  # number of classes
epochs = 250  # epoch size

train_label = np_utils.to_categorical(train_label, num_classes)

data_generator = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

model = models.Sequential()

# First convolutional layer
model.add(layers.Conv2D(300, (3, 3), padding='same', activation='relu', input_shape=train_data.shape[1:]))
model.add(layers.Dropout(0.0))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Conv2D(300, (1, 1), padding='same', activation='relu', ))

model.add(layers.Conv2D(600, (2, 2), padding='same', activation='relu', ))
model.add(layers.Dropout(0.0))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Conv2D(600, (1, 1), padding='same', activation='relu', ))

model.add(layers.Conv2D(900, (2, 2), padding='same', activation='relu', ))
model.add(layers.Dropout(0.1))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Conv2D(900, (1, 1), padding='same', activation='relu', ))

model.add(layers.Conv2D(1200, (2, 2), padding='same', activation='relu', ))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Conv2D(1200, (1, 1), padding='same', activation='relu', ))

model.add(layers.Conv2D(1500, (2, 2), padding='same', activation='relu', ))
model.add(layers.Dropout(0.3))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Conv2D(1500, (1, 1), padding='same', activation='relu', ))

model.add(layers.Conv2D(1800, (2, 2), padding='same', activation='relu', ))
model.add(layers.Dropout(0.4))
# model.add(layers.Conv2D(1800, (1, 1), padding='same', activation='relu', ))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

data_generator.fit(train_data)

# Fit the model on the batches generated by datagen.flow().
model.fit_generator(data_generator.flow(train_data, train_label,
                                        batch_size=batch_size),
                    steps_per_epoch=train_data.shape[0] // batch_size,
                    epochs=epochs)

model.save(os.path.join(dir_path, './trial_2_model.h5'))
