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
import wide_residual_network as wrn
import tools
import matplotlib.pyplot as plt

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

# # show the data through plot
# plt.figure()  # create new figure
# fig_size = [20, 20]  # specify figure size
# plt.rcParams["figure.figsize"] = fig_size  # set figure size
#
# # Plot first 100 train image of dataset
# for i in range(1, 101):
#     ax = plt.subplot(10, 10, i)  # Specify the i'th subplot of a 10*10 grid
#     img = train_data[i, :, :, :]  # Choose i'th image from train data
#     ax.get_xaxis().set_visible(False)  # Disable plot axis.
#     ax.get_yaxis().set_visible(False)
#     plt.imshow(img)
#
# plt.show()

cifar_mean = train_data.mean(axis=(0, 1, 2), keepdims=True)
cifar_std = train_data.std(axis=(0, 1, 2), keepdims=True)
print("Mean:", cifar_mean)
print("Std:", cifar_std)
train_data = (train_data - cifar_mean) / (cifar_std + 1e-8)

########### TRAIN ############
model_file = "./trial_7_model.h5"
result_filename = "trial_7_results.csv"

np.random.seed(1912934293284)
batch_size = 128  # batch size
num_classes = 100  # number of classes
epochs = 100  # epoch size


def schedule(epoch):
    if epoch <= 30:
        return 0.1
    if epoch <= 60:
        return 0.02
    elif epoch <= 180:
        return 0.004
    elif epoch <= 100:
        return 0.0008


train_label = np_utils.to_categorical(train_label, num_classes)

data_generator = ImageDataGenerator(
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

model = wrn.create_wide_residual_network((32, 32, 3,), nb_classes=num_classes, N=4, k=10, dropout=0.3)

# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizers.Adam(lr=1e-3),
#               metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False),
              metrics=['accuracy'])

model.summary()

data_generator.fit(train_data)

# lr_reducer = callbacks.ReduceLROnPlateau(verbose=1, monitor='loss', factor=np.sqrt(0.1),
#                                          cooldown=0, patience=10, min_lr=1e-6)

# model_checkpoint = callbacks.ModelCheckpoint(model_file, verbose=1, monitor="acc", save_best_only=True, mode='auto')
learning_rate_scheduler = LearningRateScheduler(schedule, verbose=1)
train_callbacks = [learning_rate_scheduler]

model.fit_generator(data_generator.flow(train_data, train_label,
                                        batch_size=batch_size),
                    steps_per_epoch=train_data.shape[0] // batch_size,
                    callbacks=train_callbacks,
                    epochs=epochs, verbose=1)

model.save_weights(os.path.join(dir_path, model_file))

prd = model.predict(train_data)

predict_result_idx = np.argmax(prd, axis=1)
label_result_idx = np.argmax(train_label, axis=1)

print("Accuracy on training data:", np.sum(predict_result_idx == label_result_idx) / len(train_data))

########### Test on test data ###################
test_data_path = os.path.join(dir_path, "test_data")

with open(test_data_path, 'rb') as f:
    test_data = pickle.load(f)

test_data = tools.reshape(test_data)
test_data = (test_data - cifar_mean) / (cifar_std + 1e-8)

print(test_data.shape, 'test samples')
test_data = test_data.astype('float32')
test_data /= 255.0

print("Predicting test data!!!")
prd = model.predict(test_data)

predict_result_idx = np.argmax(prd, axis=1)

csv_out = open(os.path.join(dir_path, result_filename), "w")
csv_out.write("ids,labels\n")

for i in range(0, test_data.shape[0]):
    csv_out.write("%d,%d\n" % (i, predict_result_idx[i]))

csv_out.close()

print("CSV Saved!!!")
