'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

# from __future__ import print_function
# import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'

import copy
import numpy as np
import numpy.random as rng
import pandas as pd
import keras
from keras.layers import Input,merge
from keras.models import Sequential, Model
from sklearn.preprocessing import StandardScaler
from keras.layers.core import *
import pdb
# np.random.seed(0)



def euc_dist(x):
    'Merge function: euclidean_distance(u,v)'
    s = x[0] - x[1]
    output = K.sqrt(K.sum(K.square(s),axis=1,keepdims=True))
    return output

def euc_dist_shape(input_shape):
    'Merge output shape'
    shape = list(input_shape)
    outshape = (shape[0][0],1)
    return tuple(outshape)

class Linear_Metric:

    def train(self, data_pairs, similarity_labels):
        self.scaler = StandardScaler()
        self.batch_size = 10
        self.epochs = 4
        X1,X2 = zip(*data_pairs)


        number_classes = len(np.unique(similarity_labels))
        train_portion = 0.8
        input_shape = data_pairs[0][0].shape
        kernels = 100
        s_size = len(similarity_labels)
        x1_train = np.array(X1[:int(s_size * train_portion)])
        x2_train = np.array(X2[:int(s_size * train_portion)])
        x1_test = np.array(X1[int(s_size * train_portion):])
        x2_test = np.array(X2[int(s_size * train_portion):])
        y_train = keras.utils.to_categorical(similarity_labels[:int(s_size * train_portion)],number_classes)
        y_test = keras.utils.to_categorical(similarity_labels[int(s_size * train_portion):],number_classes)

        # define networks
        left_input = Input(input_shape)
        right_input = Input(input_shape)
        model = Sequential()
        model.add(Dense(kernels, activation='linear', input_shape = input_shape))
        # model.add(Dropout(0.2))
        model.add(Dense(kernels, activation='linear'))
        # model.add(Dropout(0.2))

        encoded_l = model(left_input)
        encoded_r = model(right_input)

        both = merge([encoded_l,encoded_r],mode=euc_dist,output_shape=euc_dist_shape)
        self.distance_model = Model(input=[left_input,right_input],outputs=both)

        self.distance_model.compile(loss=self.contrastive_loss, optimizer='RMSprop')

        data = np.append(x1_train, x2_train, axis=0)
        self.scaler.fit(data)
        x1_train = self.scaler.transform(x1_train)
        x2_train = self.scaler.transform(x2_train)

        self.distance_model.fit([x1_train, x2_train], y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            validation_data=([x1_test, x2_test], y_test), verbose=0)


    def transform(self, data_pairs):
        x, y = data_pairs
        x = self.scaler.transform(np.array([x]))
        y = self.scaler.transform(np.array([y]))
        distance = self.distance_model.predict([x,y])
        return distance

    def contrastive_loss(self, y_true, d):
        margin = 1
        return K.mean((1 - y_true) * 0.5 * K.square(d) + 0.5 * y_true * K.square(K.maximum(margin - d, 0)))


