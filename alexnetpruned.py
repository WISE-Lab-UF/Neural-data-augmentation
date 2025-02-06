#WISE Lab
# Loading train set and test set
from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.layers import (Dense, Dropout, Flatten, Conv1D, Conv2D, GlobalMaxPooling2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation)
import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from tensorflow import keras
import keras.layers as layers
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.utils import plot_model

from keras.utils.vis_utils import model_to_dot
import time
import numpy as np
import os

from typing import List, Tuple

import pathlib
import shutil

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model, load_model

from tensorflow.python.keras.utils import layer_utils
#from tensorflow.keras.utils.vis_utils import model_to_dot
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.utils import plot_model

from tensorflow.keras.applications.imagenet_utils import preprocess_input


import scipy.misc

import tensorflow.keras.backend as K

import pickle

def load_data(path, labels, val_size, test_size):

    # Just to show on console
    for i, label in enumerate(labels):
      print(i, ' ', label)

    # for the first class
    X = np.load(path + labels[0] + '.npy')
    y = np.full(X.shape[0], fill_value= 0)

    # Split all classes among Train, Validation & Test - SET
    for i, label in enumerate(labels[1:]):
        x = np.load(path + label + '.npy')

        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=i+1))


    return (X, y)



labels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','greaterThan','comma','apostrophe','tilde','questionMark']
X, y = load_data(os.path.expanduser('~') + '/custom/dataset/augmentation/', labels=labels, val_size=0, test_size=0)


print(X.shape, y.shape)
zeros=np.zeros((10881,23,192))
X=np.concatenate((X,zeros),axis=1)
print(X.shape, y.shape)

zeros2=np.zeros((10881,224,32))
X=np.concatenate((X,zeros2),axis=2)
print(X.shape, y.shape)

test_size = 0.2
val_size = 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=45)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=45)


print(X_train.shape, ' ', y_train.shape)
print(X_val.shape, ' ', y_val.shape)
print(X_test.shape, ' ', y_test.shape)

print('\n')


# Feature dimension
channels = 1
epochs = 200
batch_size = 32

num_classes = len(labels)


# one shot hot
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
y_val_hot = to_categorical(y_val)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], channels)
X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], channels)
X_val   = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], channels)

input_shape=(X_train.shape[1], X_train.shape[2], 1)
print(X_train.shape)

pool_size = (2, 2) 
kernel_size = (3, 3)  
keras.backend.clear_session()
print(input_shape)
model = keras.Sequential()
model.add(layers.Conv2D(filters=124, kernel_size=(11, 11), 
                        strides=(4, 4), activation="relu", 
                        input_shape=(224, 224,1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(3, 3), strides= (2, 2)))
model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), 
                        strides=(1, 1), activation="relu", 
                        padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), 
                        strides=(1, 1), activation="relu", 
                        padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), 
                        strides=(1, 1), activation="relu", 
                        padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), 
                        strides=(1, 1), activation="relu", 
                        padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation="softmax"))

model.summary()

import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 2 epochs.
batch_size = 16
epochs = 500
validation_split = 0.1 # 10% of training set will be used for validation set. 

num_images = X_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_for_pruning.summary()


callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
]

model_for_pruning.fit(X_train, y_train_hot,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)


_, model_for_pruning_accuracy = model_for_pruning.evaluate(
   X_test, y_test_hot, verbose=0)


print('Pruned test accuracy:', model_for_pruning_accuracy)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
import tempfile
_, pruned_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
pruned_tflite_model = converter.convert()

_, pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(pruned_tflite_file, 'wb') as f:
  f.write(pruned_tflite_model)

print('Saved pruned TFLite model to:', pruned_tflite_file)
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_and_pruned_tflite_model = converter.convert()

_, quantized_and_pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(quantized_and_pruned_tflite_file, 'wb') as f:
  f.write(quantized_and_pruned_tflite_model)

print('Saved quantized and pruned TFLite model to:', quantized_and_pruned_tflite_file)

import numpy as np

def evaluate_model(interpreter):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on ever y image in the "test" dataset.
  prediction_digits = []
  for i, test_image in enumerate(X_test):
    if i % 1000 == 0:
      print('Evaluated on {n} results so far.'.format(n=i))
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

  print('\n')
  # Compare prediction results with ground truth labels to calculate accuracy.
  prediction_digits = np.array(prediction_digits)
  accuracy = (prediction_digits == y_test).mean()
  return accuracy

interpreter = tf.lite.Interpreter(model_content=quantized_and_pruned_tflite_model)
interpreter.allocate_tensors()

test_accuracy = evaluate_model(interpreter)

print('Pruned and quantized TFLite test_accuracy:', test_accuracy)
_, model_for_pruning_accuracy = model_for_pruning.evaluate(
   X_test, y_test_hot, verbose=0)
print('Pruned TF test accuracy:', model_for_pruning_accuracy)
model.save('jetson alexnet pruned.h5')



# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# device_name = tf.test.gpu_device_name()
# if "GPU" not in device_name:
#     print("GPU device not found")
# print('Found GPU at: {}'.format(device_name))
# start = time.time()
# with tf.device('/gpu:0'):
#     model.fit(X_train, y_train_hot, epochs=500, batch_size=16, validation_data=(X_val, y_val_hot))
# stop = time.time()
# print(f'Training on GPU took: {(stop-start)/60} minutes')
# # pickle.dump(model,open(f'resnet50 batch 16 - input size 224,224,1 - GPU','wb'))
# # load_model=pickle.load(open('resnet50 batch 32 - GPU','rb'))
# model.save('jetson resnet batch 16 input shape 224,224,1 padding -1.h5')
# model.evaluate(X_test, y_test_hot, verbose=1)
