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
from keras.callbacks import ModelCheckpoint
np.random.seed(0)



class Deep_Metric:

    def train(self, data_pairs, similarity_labels):
        self.batch_size = 10
        self.epochs = 1000
        X1 = []
        X2 = []
        for x, y in data_pairs:
            X1.append(x)
            X2.append(y)
        print(len(X1), len(X2), len(similarity_labels))
        number_classes = len(np.unique(similarity_labels)) 
        # convert class vectors to binary class matrices

        train_portion = 0.8
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
        branch1 = BatchNormalization()(branch1)
        
        input2 = keras.layers.Input(shape=input_shape)
        branch2 = keras.layers.Dense(kernels, activation='relu')(input2)
        branch2 = keras.layers.Dense(kernels, activation='relu')(branch2)
        branch2 = BatchNormalization()(branch2)

        merged_vector = keras.layers.Dot(axes=1, normalize=True)([branch1, branch2])

        # merged_vector = keras.layers.Subtract()([branch1, branch2])
        # merged_vector = Merge(mode=lambda x:self.euclideanSqDistance(x[0],x[1]), output_shape=lambda inp_shp:(None,1))([branch1,branch2])

        merged_vector = keras.layers.Dense(number_classes, activation='relu')(merged_vector)

        model = Model(inputs=[input1, input2], outputs=merged_vector)

        model.compile(loss='binary_crossentropy',
                        optimizer=RMSprop(),
                        metrics=['acc'])
        
        filepath="weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]


        x1_train = np.array(x1_train)
        x2_train = np.array(x2_train)
        y_train = np.array(y_train)

        x1_test = np.array(x1_test)
        x2_test = np.array(x2_test)
        y_test = np.array(y_test)

        # x1_train = np.array(X1)
        # x2_train = np.array(X2)
        # y_train = keras.utils.to_categorical(similarity_labels, number_classes)


        history = model.fit([x1_train, x2_train], y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            callbacks=callbacks_list,
                            validation_data=([x1_train, x2_train], y_train), verbose=0)

        score = model.evaluate([x1_test, x2_test], y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        inp1, inp2 = model.input

        self.functor1 = K.function([inp1]+ [K.learning_phase()], [branch1]) 
        self.functor2 = K.function([inp2]+ [K.learning_phase()], [branch2])
          
    def transform(self, data_pairs):
        x, y = data_pairs
        x = x.reshape(1, x.shape[0])
        y = y.reshape(1, y.shape[0])
        layer_out1 = self.functor1([x, 1.])
        layer_out2 = self.functor2([y, 1.])
        x = pd.Series(layer_out1[0][0])
        y = pd.Series(layer_out2[0][0])
        return x, y

    def contrastive_loss(self, y_true, y_pred):
        margin = 1
        return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)), axis=1)
        
    def euclideanSqDistance(self, A, B):
        output = K.mean(K.square(A - B), axis=1)
        output = K.expand_dims(output, 1)
        return output

    def penalized_loss(self, branch1, branch2):
        def loss(y_true, y_pred):
            return K.mean(K.square(y_pred - y_true) - K.square(y_true - y_pred), axis=-1)
        return loss