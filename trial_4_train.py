import os
import pickle
import numpy as np
from keras.utils import np_utils
from keras import callbacks
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import tools
from resnext import ResNext

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

########### TRAIN ############
model_file = "./trial_4_model.h5"
np.random.seed(2017)
batch_size = 128  # batch size
num_classes = 100  # number of classes
epochs = 100  # epoch size

train_label = np_utils.to_categorical(train_label, num_classes)

data_generator = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

model = ResNext((32, 32, 3), depth=29, cardinality=8, width=4, weights=None, classes=num_classes)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=1e-3),
              metrics=['accuracy'])

data_generator.fit(train_data)

lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                         cooldown=0, patience=10, min_lr=1e-6)

model_checkpoint = callbacks.ModelCheckpoint(model_file, verbose=1, monitor="val_acc", save_best_only=True, mode='auto')

train_callbacks = [lr_reducer, model_checkpoint]
model.fit_generator(data_generator.flow(train_data, train_label,
                                        batch_size=batch_size),
                    steps_per_epoch=train_data.shape[0] // batch_size,
                    callbacks=train_callbacks,
                    epochs=epochs)

model.save(os.path.join(dir_path, model_file))
