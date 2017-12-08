'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding
from keras.preprocessing import sequence
from keras import backend as K
import numpy as np
import pandas as pd
np.random.seed(0)

class Deep_Metric:

    def __init__(self, input_shape, data = None):
        self.state = False if data is None else True
        self.input_shape = input_shape
        self.Sequence = False

        if self.state:
            data, self.labels = data
            train_portion = 0.7
            s_size = len(self.labels)
            self.x_train = data.iloc[:int(s_size * train_portion),:].values
            self.x_test = data.iloc[int(s_size * train_portion):,:].values
            self.y_train = self.labels[:int(s_size * train_portion)]
            self.y_test = self.labels[int(s_size * train_portion): ]
            self.batch_size = 21
            self.epochs = 20

    def train(self, mode = "deep"):
        if self.state:
            number_classes = len(np.unique(self.labels)) 
            # convert class vectors to binary class matrices
            labels = keras.utils.to_categorical(self.labels, number_classes)
            y_train = keras.utils.to_categorical(self.y_train, number_classes)
            y_test = keras.utils.to_categorical(self.y_test, number_classes)

        self.model = Sequential()

        if mode == "deep":
            self.deep()
        else:
            self.convolution()
            
        if self.state:
            self.model.add(Dense(number_classes, activation='sigmoid'))

        self.model.summary()
        if mode == "deep":
            self.model.compile(loss='categorical_crossentropy',
                        optimizer=RMSprop(),
                        metrics=['accuracy'])
        else:
            self.model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])


        if self.state:
            history =   self.model.fit(self.x_train, y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            validation_data=(self.x_test, y_test))

        score = self.model.evaluate(self.x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        # if self.state:
        
        #     history =   self.model.fit(self.x_train, y_train,
        #                     batch_size=self.batch_size,
        #                     epochs=self.epochs,
        #                     validation_data=(self.x_test, y_test))

        inp = self.model.input  
        if self.state:
            out = self.model.layers[-2].output     
        else:
            out = self.model.layers[-1].output     
        self.functor = K.function([inp]+ [K.learning_phase()], [out])           
    
    def convolution(self):
        max_features = 5000
        self.maxlen = 24
        batch_size = 32
        embedding_dims = 50
        filters = 250
        kernel_size = 24
        hidden_dims = 100

        # print('Loading data...')
        x_train = self.x_train
        y_train = self.y_train
        x_test = self.x_test
        y_test = self.y_test
    

        # print('Pad sequences (samples x time)')
        self.x_train = sequence.pad_sequences(x_train, maxlen=self.maxlen)
        self.x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
        self.Sequence = True
    

        # print('Build model...')    
        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        self.model.add(Embedding(max_features,
                            embedding_dims,
                            input_length=self.maxlen))

        # we add a Convolution1D, which will learn filters
        # word group filters of size filter_length:
        self.model.add(Conv1D(filters,
                        kernel_size,
                        padding='valid',
                        activation='relu',
                        strides=1))
        # we use max pooling:
        self.model.add(GlobalMaxPooling1D())

        # We add a vanilla hidden layer:
        self.model.add(Dense(hidden_dims, activation='relu'))

        
    def deep(self):
        kernels = 100
        self.model.add(Dense(kernels, activation='relu', input_shape=self.input_shape))
        self.model.add(Dense(kernels, activation='relu'))
    
        
    def transform(self, data):        
        test = data.reshape(1, self.input_shape[0])
        if self.Sequence:
            test = sequence.pad_sequences(test, maxlen=self.maxlen)
        layer_out = self.functor([test, 1.])
        return pd.Series(layer_out[0][0])
