'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

# from __future__ import print_function

import copy

import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Add, Merge
from keras.optimizers import RMSprop
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding
from keras.preprocessing import sequence
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.constraints import maxnorm
# from keras_diagram import ascii
np.random.seed(0)



class Deep_Metric1:


    def __init__(self, input_shape, data = None):
        self.state = False if data is None else True
        self.input_shape = input_shape
        self.Sequence = False

        if self.state:
            self.data, self.labels = data
            train_portion = 0.7
            s_size = len(self.labels)
            self.x_train = self.data.iloc[:int(s_size * train_portion),:].values
            self.x_test = self.data.iloc[int(s_size * train_portion):,:].values
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
                    
        # if self.state:
        #     history =   self.model.fit(self.x_train, y_train,
        #                     batch_size=self.batch_size,
        #                     epochs=self.epochs,
        #                     validation_data=(self.x_test, y_test))

        # score = self.model.evaluate(self.x_test, y_test, verbose=0)
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])
        
        if self.state:
            self.y_train = labels
            if mode != "deep":
                self.x_train = sequence.pad_sequences(self.data, maxlen=self.maxlen)
            else:
                self.x_train = self.data.values
            history =   self.model.fit(self.x_train, self.y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            validation_data=(self.x_train, self.y_train))
        
        score = self.model.evaluate(self.x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

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

    def merged(self, data_pairs, similarity_labels):
        
        X1 = []
        X2 = []
        for x, y in data_pairs:
            X1.append(x)
            X2.append(y)
        print(len(X1), len(X2), len(similarity_labels))
        number_classes = len(np.unique(similarity_labels)) 
        # convert class vectors to binary class matrices

        train_portion = 0.7
        input_shape = data_pairs[0][0].shape
        kernels = 100
        s_size = len(similarity_labels)
        x1_train = X1[:int(s_size * train_portion)]
        x2_train = X2[:int(s_size * train_portion)]
        x1_test = X1[int(s_size * train_portion):]
        x2_test = X2[int(s_size * train_portion):]
        y_train = similarity_labels[:int(s_size * train_portion)]
        y_test = similarity_labels[int(s_size * train_portion): ]

        labels = keras.utils.to_categorical(similarity_labels, number_classes)
        y_train = keras.utils.to_categorical(y_train, number_classes)
        y_test = keras.utils.to_categorical(y_test, number_classes)

        input1 = keras.layers.Input(shape=input_shape)
        branch1 = keras.layers.Dense(kernels, activation='relu')(input1)
        branch1 = keras.layers.Dense(kernels, activation='relu')(branch1)
        
        input2 = keras.layers.Input(shape=input_shape)
        branch2 = keras.layers.Dense(kernels, activation='relu')(input2)
        branch2 = keras.layers.Dense(kernels, activation='relu')(branch2)

        merged_vector = keras.layers.Dot(axes=1, normalize=True)([branch1, branch2])

        # merged_vector = keras.layers.Subtract()([branch1, branch2])
        # merged_vector = Merge(mode=lambda x:self.euclideanSqDistance(x[0],x[1]), output_shape=lambda inp_shp:(None,1))([branch1,branch2])

        merged_vector = keras.layers.Dense(number_classes, activation='sigmoid')(merged_vector)

        model = Model(inputs=[input1, input2], outputs=merged_vector)

        model.compile(loss='categorical_crossentropy',
                        optimizer=RMSprop(),
                        metrics=['accuracy'])


        x1_train = np.array(x1_train)
        x2_train = np.array(x2_train)
        y_train = np.array(y_train)

        x1_test = np.array(x1_test)
        x2_test = np.array(x2_test)
        y_test = np.array(y_test)

        history = model.fit([x1_train, x2_train], y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            validation_data=([x1_train, x2_train], y_train))

        score = model.evaluate([x1_test, x2_test], y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        inp1, inp2 = model.input

        functor1 = K.function([inp1]+ [K.learning_phase()], [branch1]) 
        functor2 = K.function([inp2]+ [K.learning_phase()], [branch2]) 

        layer_out1 = functor1([x1_test, 1.])
        layer_out2 = functor2([x1_test, 1.])
        print(len(layer_out1[0]), len(x1_test))
        print(len(layer_out2[0]), len(x1_test))
        

    def contrastive_loss(self, y_true, y_pred):
        margin = 1
        return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)), axis=1)
        


    def euclideanSqDistance(self, A, B):
        output = K.mean(K.square(A - B), axis=1)
        output = K.expand_dims(output, 1)
        return output