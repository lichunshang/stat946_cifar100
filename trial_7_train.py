import os
import pickle
import numpy as np
from keras import callbacks
from keras import optimizers
from keras import regularizers
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import taylor_wide_residual_network as wrn
import wide_residual_network as wrn2
import tools
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.abspath(__file__))

# train_data_path = os.path.join(dir_path, "train_data")
# with open(train_data_path, 'rb') as f:
#     train_data = pickle.load(f)
#     train_label = pickle.load(f)
#     train_data = tools.reshape(train_data)
#
# test_data_path = os.path.join(dir_path, "test_data")
# with open(test_data_path, 'rb') as f:
#     test_data = pickle.load(f)
#     test_data = tools.reshape(test_data)

# use downloaded CIFAR dataset to avoid weird reshaping problems with numpy, the test_labels are not
# used for anything though
from keras.datasets import cifar100

(train_data, train_label), (test_data, test_label_not_using) = cifar100.load_data()

print('test data shape:', test_data.shape)
test_data = test_data.astype('float32')
test_data /= 255.0

print('train data shape:', train_data.shape)
train_data = train_data.astype('float32')
train_data /= 255.0

#-----------------VISUALIZATION----------------
# plt.figure()
# fig_size = [20, 20]
# plt.rcParams["figure.figsize"] = fig_size
# for i in range(1, 101):
#     ax = plt.subplot(10, 10, i)
#     img = train_data[i, :, :, :]
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     plt.imshow(img)
#
# plt.show()
#----------------------------------------------

cifar_mean = train_data.mean(axis=(0, 1, 2), keepdims=True)
cifar_std = train_data.std(axis=(0, 1, 2), keepdims=True)
print("Mean:", cifar_mean)
print("Std:", cifar_std)
train_data = (train_data - cifar_mean) / (cifar_std + 1e-8)
test_data = (test_data - cifar_mean) / (cifar_std + 1e-8)

##############################
########### TRAIN ############
##############################
model_file = "./trial_7_model.h5"
result_filename = "trial_7_results.csv"

np.random.seed(342988509)
batch_size = 128  # batch size
num_classes = 100  # number of classes
epochs = 200  # epoch size


def schedule(epoch):
    if epoch <= 60:
        return 0.1
    if epoch <= 120:
        return 0.02
    elif epoch <= 160:
        return 0.004
    elif epoch <= 200:
        return 0.0008


train_label = np_utils.to_categorical(train_label, num_classes)

train_data_generator = ImageDataGenerator(
    # featurewise_center=False,  # set input mean to 0 over the dataset
    # samplewise_center=False,  # set each sample mean to 0
    # featurewise_std_normalization=False,  # divide inputs by std of the dataset
    # samplewise_std_normalization=False,  # divide each input by its std
    # zca_whitening=False,  # apply ZCA whitening
    # rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=4.0 / 32,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=4.0 / 32,  # randomly shift images vertically (fraction of total height)
    fill_mode='constant',
    cval=0,
    horizontal_flip=True,  # randomly flip images
    # vertical_flip=False
)

# model = wrn.build_model((32, 32, 3,), classes=100, n=4, k=10, dropout=0.3, weight_decay=0.0005, verbose=True)
model = wrn2.create_wide_residual_network((32, 32, 3,), nb_classes=num_classes, N=4, k=10, dropout=0.3)


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True),
              metrics=['accuracy'])

model.summary()

train_data_generator.fit(train_data)

learning_rate_scheduler = LearningRateScheduler(schedule, verbose=1)
train_callbacks = [learning_rate_scheduler]

model.fit_generator(train_data_generator.flow(train_data, train_label,
                                              batch_size=batch_size),
                    steps_per_epoch=train_data.shape[0] // batch_size,
                    callbacks=train_callbacks,
                    epochs=epochs, verbose=1,
                    validation_data=(test_data, np_utils.to_categorical(test_label_not_using),)
                    )

model.save_weights(os.path.join(dir_path, model_file))

prd = model.predict(train_data)

predict_result_idx = np.argmax(prd, axis=1)
label_result_idx = np.argmax(train_label, axis=1)

print("Accuracy on training data:", np.sum(predict_result_idx == label_result_idx) / len(train_data))

#################################################
########### Test on test data ###################
#################################################
print("Predicting test data!!!")
prd = model.predict(test_data)

predict_result_idx = np.argmax(prd, axis=1)

csv_out = open(os.path.join(dir_path, result_filename), "w")
csv_out.write("ids,labels\n")

for i in range(0, test_data.shape[0]):
    csv_out.write("%d,%d\n" % (i, predict_result_idx[i]))

csv_out.close()

print("CSV Saved!!!")
