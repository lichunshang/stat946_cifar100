import os
import pickle
import numpy as np
from keras import models
from keras.utils import np_utils
import tools

name = "trial_4"
model_filename = name + "_model.h5"

dir_path = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(dir_path, "train_data")

with open(test_data_path, 'rb') as f:
    test_data = pickle.load(f)
    test_label = pickle.load(f)

test_data = tools.reshape(test_data)
test_label = np_utils.to_categorical(test_label, 100)

print(test_data.shape, 'test samples')
test_data = test_data.astype('float32')
test_data /= 255.0

model = models.load_model(os.path.join(dir_path, model_filename))
prd = model.predict(test_data)

predict_result_idx = np.argmax(prd, axis=1)
label_result_idx = np.argmax(test_label, axis=1)

print("Accuracy:", np.sum(predict_result_idx == label_result_idx)/len(test_data))