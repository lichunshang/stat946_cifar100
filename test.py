import os
import pickle
import numpy as np
from keras import models
import tools

name = "trial_4"
model_filename = name + "_model.h5"
result_filename = name + "_results.csv"

dir_path = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(dir_path, "test_data")

with open(test_data_path, 'rb') as f:
    test_data = pickle.load(f)

test_data = tools.reshape(test_data)

print(test_data.shape, 'test samples')
test_data = test_data.astype('float32')
test_data /= 255.0

model = models.load_model(os.path.join(dir_path, model_filename))
prd = model.predict(test_data)

predict_result_idx = np.argmax(prd, axis=1)

csv_out = open(os.path.join(dir_path, result_filename), "w")
csv_out.write("ids,labels\n")

for i in range(0, test_data.shape[0]):
    csv_out.write("%d,%d\n" % (i, predict_result_idx[i]))

csv_out.close()