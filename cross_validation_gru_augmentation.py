# SuperFastPython.com
# example of a program that uses all cpu cores
import math
from multiprocessing import Pool
 
# define a cpu-intensive task
def task(arg):

    # Loading train set and test set
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas
    import os
    import tensorflow
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.layers import (Dense, Dropout, Flatten, Conv1D, Conv2D, GlobalMaxPooling2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation)
    from keras.models import Sequential,Model
    import tensorflow.keras.layers as layers
    from tensorflow.keras import optimizers
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

    i, j, k = X.shape[0], 23, 192
    Matrix = np.array([[[-1 for x in range(k)] for y in range(j)] for z in range(i)]) 
    print(Matrix.shape)
    X=np.concatenate((X,Matrix),axis=1)
    print(X.shape, y.shape)
    i, j, k = X.shape[0], 224, 32
    Matrix = np.array([[[-1 for x in range(k)] for y in range(j)] for z in range(i)]) 
    X=np.concatenate((X,Matrix),axis=2)
    print(X.shape, y.shape)


    test_size = 0.2
    val_size = 0.1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=45)


    X_train=X_train/15
    X_test=X_test/15


    print(X_train.shape, ' ', y_train.shape)

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


    # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], channels)
    # X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], channels)


    input_shape=(X_train.shape[1], X_train.shape[2], 1)
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input,Bidirectional,LSTM,Lambda, GRU
    from tensorflow.keras.layers import Permute,GlobalMaxPool1D,Concatenate, Dense, BatchNormalization, Dropout, GlobalAveragePooling1D
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    pool_size = (2, 2) 
    kernel_size = (3, 3)  
    keras.backend.clear_session()
    model = tf.keras.Sequential([
      tf.keras.layers.GRU(128,input_shape=(224,224)),
      tf.keras.layers.Dense(128, activation='relu',input_shape=(224, 224,1)),
      tf.keras.layers.Dropout(0.2,input_shape=(224,)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(31, activation='softmax')
    ])
    print(model.summary())
    # Compiling the model
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer="sgd",
        metrics=["accuracy"],
    )

    n_folds=5
    epochs=20
    batch_size=16

    #save the model history in a list after fitting so that we can plot later
    model_history = [] 

    for i in range(n_folds):
        print("Training on Fold: ",i+1)
        t_x, val_x, t_y, val_y = train_test_split(X_train, y_train_hot, test_size=0.1, 
                                                  random_state = np.random.randint(1,1000, 1)[0])
        model_history.append(model.fit(t_x, t_y, epochs=epochs, batch_size=batch_size, validation_split=0.1) )
        print("======="*12, "\n\n\n")
    # model.fit(X_train, y_train_hot, epochs=500, batch_size=16, validation_data=(X_val, y_val_hot))
    # pickle.dump(model,open('alexnet batch 32 input shape 224,224,1 padding -1','wb'))
    model.evaluate(X_test, y_test_hot, verbose=1)
    model.save('jetson gru batch 16 cross validation input shape 224,224,1 padding -1_augmentation_3x_10_by_50.h5')



    return sum([math.sqrt(i) for i in range(1, arg)])
 
# protect the entry point
if __name__ == '__main__':
    # report a message
    print('Starting task...')
    # create the process pool
    with Pool(32) as pool:
        # perform calculations
        results = pool.map(task, range(1,2))
    # report a message
    print('Done.')