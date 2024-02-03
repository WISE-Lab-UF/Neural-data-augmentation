import numpy as np
import os



import matplotlib.pyplot as plt
import random

def load_data(path, labels, val_size, test_size):

    # Just to show on console
    print(labels)

    # for the first class
    X = np.load(path + labels + '.npy')
    y = np.full(X.shape[0], fill_value= 0)

    # # Split all classes among Train, Validation & Test - SET
    # for i, label in enumerate(labels[1:]):
    #     x = np.load(path + label + '.npy')

    #     X = np.vstack((X, x))
    #     y = np.append(y, np.full(x.shape[0], fill_value=i+1))


    return (X, y)



labels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','greaterThan','comma','apostrophe','tilde','questionMark']
# labels=['a']
for label in labels:
    X, y = load_data(os.path.expanduser('~') + '/custom/dataset/', labels=label, val_size=0, test_size=0)


    #                           Noise Adding with fixed spike counts
    X_augmented=np.zeros((X.shape[0],201,192),dtype=int)
    for i in range(0,1):
        for aa in range(X.shape[0]):
            for bb in range(X.shape[1]):
                for cc in range(X.shape[2]):
                    X_augmented[aa][bb][cc]=X[aa][bb][cc]
        for j in range((X.shape[0])):
            random_numbers=random.sample(range(0,192),10)
            for k in random_numbers:
                random1=random.randint(0,200)
                X_augmented[j][random1][k]=int(X_augmented[j][random1][k])+1
                # print(j,k,random1)
        # print(X_augmented.shape)
        if(i==0):
            X_noise=X_augmented
        else:
            X_noise=np.concatenate((X_noise, X_augmented), axis=0)

    # print(X_noise.shape)
    


    #                               Shifting

    print(X.shape)
    for trial in range(4,5):
        X_augmented=np.zeros((X.shape[0],201,192),dtype=int)
        for i in range((X.shape[0])):
            for j in range((X.shape[1]-trial)):
                X_augmented[i][j]=X[i][j+trial]
                # for k in range(X.shape[2]):
                #     # print(X_augmented[i][j][k])
                #     X_augmented[i][j][k]=X[i][j+5][k]
                #     # print(X_augmented[i][j][k])
                
        for i in range((X.shape[0])):
            for j in range(201-trial,201):
                for k in range(X.shape[2]):
                    X_augmented[i][j][k]=-1
        if trial==4:
            X_shift=X_augmented
        else:
            X_shift=np.concatenate((X_shift, X_augmented), axis=0)
        # print(X_shift.shape)
        # print(X[0][3])
        # print(X_shift[0][0])

    # print(X[0][3])
    # print(X_shift[351][0])
    dest=os.path.expanduser('~') + '/custom/dataset/augmentation/'
    X=np.concatenate((X, X_shift), axis=0)
    X=np.concatenate((X, X_noise), axis=0)
    print(X.shape)
    np.save(dest+label+'.npy',X)
