#wiselab
# SuperFastPython.com
# example of a program that uses all cpu cores
import math
from multiprocessing import Pool
 
# define a cpu-intensive task
def task(arg):

    # Loading train set and test set
    from sklearn.model_selection import train_test_split
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    import numpy as np
    import pandas
    import os
    os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                              "intra_op_parallelism_threads=1 "
                              "inter_op_parallelism_threads=1")
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
    def convolutional_block(X: tf.Tensor, level: int, block: int, filters: List[int], s: Tuple[int,int,int]=(2, 2)) -> tf.Tensor:
        """
        Creates a convolutional block (see figure 3.1 from readme)

        Input:
            X - input tensor of shape (m, height_prev, width_prev, chan_prev)
            level - integer, one of the 5 levels that our networks is conceptually divided into (see figure 3.1 in the readme file)
                  - level names have the form: conv2_x, conv3_x ... conv5_x
            block - each conceptual level has multiple blocks (1 identity and several convolutional blocks)
                    block is the number of this block within its conceptual layer
                    i.e. first block from level 2 will be named conv2_1
            filters - a list on integers, each of them defining the number of filters in each convolutional layer
            s   - stride of the first layer;
                - a conv layer with a filter that has a stride of 2 will reduce the width and height of its input by half

        Output:
            X - tensor (m, height, width, chan)
        """

        # layers will be called conv{level}_{block}_{convlayer_number_within_block}'
        conv_name = f'conv{level}_{block}' + '_{layer}_{type}'

        # unpack number of filters to be used for each conv layer
        f1, f2, f3 = filters

        # the shortcut branch of the convolutional block
        X_shortcut = X

        # first convolutional layer
        X = Conv2D(filters=f1, kernel_size=(1, 1), strides=s, padding='valid',
                  name=conv_name.format(layer=1, type='conv'),
                  kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=1, type='bn'))(X)
        X = Activation('relu', name=conv_name.format(layer=1, type='relu'))(X)

        # second convolutional layer
        X = Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                  name=conv_name.format(layer=2, type='conv'),
                  kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=2, type='bn'))(X)
        X = Activation('relu', name=conv_name.format(layer=2, type='relu'))(X)

        # third convolutional layer
        X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                  name=conv_name.format(layer=3, type='conv'),
                  kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=3, type='bn'))(X)

        # shortcut path
        X_shortcut = Conv2D(filters=f3, kernel_size=(1, 1), strides=s, padding='valid',
                            name=conv_name.format(layer='short', type='conv'),
                            kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=conv_name.format(layer='short', type='bn'))(X_shortcut)

        # add shortcut branch to main path
        X = Add()([X, X_shortcut])

        # nonlinearity
        X = Activation('relu', name=conv_name.format(layer=3, type='relu'))(X)

        return X
    def identity_block(X: tf.Tensor, level: int, block: int, filters: List[int]) -> tf.Tensor:
        """
        Creates an identity block (see figure 3.1 from readme)

        Input:
            X - input tensor of shape (m, height_prev, width_prev, chan_prev)
            level - integer, one of the 5 levels that our networks is conceptually divided into (see figure 3.1 in the readme file)
                  - level names have the form: conv2_x, conv3_x ... conv5_x
            block - each conceptual level has multiple blocks (1 identity and several convolutional blocks)
                    block is the number of this block within its conceptual layer
                    i.e. first block from level 2 will be named conv2_1
            filters - a list on integers, each of them defining the number of filters in each convolutional layer

        Output:
            X - tensor (m, height, width, chan)
        """

        # layers will be called conv{level}_iden{block}_{convlayer_number_within_block}'
        conv_name = f'conv{level}_{block}' + '_{layer}_{type}'

        # unpack number of filters to be used for each conv layer
        f1, f2, f3 = filters

        # the shortcut branch of the identity block
        # takes the value of the block input
        X_shortcut = X

        # first convolutional layer (plus batch norm & relu activation, of course)
        X = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1),
                  padding='valid', name=conv_name.format(layer=1, type='conv'),
                  kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=1, type='bn'))(X)
        X = Activation('relu', name=conv_name.format(layer=1, type='relu'))(X)

        # second convolutional layer
        X = Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1),
                  padding='same', name=conv_name.format(layer=2, type='conv'),
                  kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=2, type='bn'))(X)
        X = Activation('relu')(X)

        # third convolutional layer
        X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1),
                  padding='valid', name=conv_name.format(layer=3, type='conv'),
                  kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=3, type='bn'))(X)

        # add shortcut branch to main path
        X = Add()([X, X_shortcut])

        # relu activation at the end of the block
        X = Activation('relu', name=conv_name.format(layer=3, type='relu'))(X)

        return X

    def ResNet50(input_size: Tuple[int,int,int], classes: int) -> Model:
        """
            Builds the ResNet50 model (see figure 4.2 from readme)

            Input:
                - input_size - a (height, width, chan) tuple, the shape of the input images
                - classes - number of classes the model must learn

            Output:
                model - a Keras Model() instance
        """

        # tensor placeholder for the model's input
        X_input = Input(input_size)

        ### Level 1 ###

        # padding
        X = ZeroPadding2D((3, 3))(X_input)

        # convolutional layer, followed by batch normalization and relu activation
        X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                  name='conv1_1_1_conv',
                  kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='conv1_1_1_nb')(X)
        X = Activation('relu')(X)

        ### Level 2 ###

        # max pooling layer to halve the size coming from the previous layer
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # 1x convolutional block
        X = convolutional_block(X, level=2, block=1, filters=[64, 64, 256], s=(1, 1))

        # 2x identity blocks
        X = identity_block(X, level=2, block=2, filters=[64, 64, 256])
        X = identity_block(X, level=2, block=3, filters=[64, 64, 256])

        ### Level 3 ###

        # 1x convolutional block
        X = convolutional_block(X, level=3, block=1, filters=[128, 128, 512], s=(2, 2))

        # 3x identity blocks
        X = identity_block(X, level=3, block=2, filters=[128, 128, 512])
        X = identity_block(X, level=3, block=3, filters=[128, 128, 512])
        X = identity_block(X, level=3, block=4, filters=[128, 128, 512])

        ### Level 4 ###
        # 1x convolutional block
        X = convolutional_block(X, level=4, block=1, filters=[256, 256, 1024], s=(2, 2))
        # 5x identity blocks
        X = identity_block(X, level=4, block=2, filters=[256, 256, 1024])
        X = identity_block(X, level=4, block=3, filters=[256, 256, 1024])
        X = identity_block(X, level=4, block=4, filters=[256, 256, 1024])
        X = identity_block(X, level=4, block=5, filters=[256, 256, 1024])
        X = identity_block(X, level=4, block=6, filters=[256, 256, 1024])

        ### Level 5 ###
        # 1x convolutional block
        X = convolutional_block(X, level=5, block=1, filters=[512, 512, 2048], s=(2, 2))
        # 2x identity blocks
        X = identity_block(X, level=5, block=2, filters=[512, 512, 2048])
        X = identity_block(X, level=5, block=3, filters=[512, 512, 2048])

        # Pooling layers
        X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

        # Output layer
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', name='fc_' + str(classes),
                  kernel_initializer=glorot_uniform(seed=0))(X)

        # Create model
        model = Model(inputs=X_input, outputs=X, name='ResNet50')

        return model

    model = ResNet50(input_size = (224,224,1), classes = num_classes)

    import tensorflow_model_optimization as tfmot

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 2 epochs.
    batch_size = 16
    epochs = 30
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
    model_for_pruning.save('jetson resnet50 pruned.h5')

# protect the entry point
if __name__ == '__main__':
    # report a message
    print('Starting task...')
    # create the process pool
    with Pool(8) as pool:
        # perform calculations
        results = pool.map(task, range(1,2))
    # report a message
    print('Done.')



