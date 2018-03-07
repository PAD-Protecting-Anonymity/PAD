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
from keras.regularizers import l1_l2
from user_feedback import Similarity
np.random.seed(0)

class Deep_Metric:

    def __init__(self, mode = "non_linear"):
        self.mode = "linear"
        self.batch_size = 10
        self.epochs = 200

    
    def train(self, data_pairs, similarity_labels):
        self.data_pairs = data_pairs
        self.similarity_labels = similarity_labels

        if self.mode == "linear":
            self.train_for_non_linear()
        else:
            self.train_for_linear()

    def transform(self, sample_data_pairs):
        if self.mode == "linear":
            self.trans_for_non_linear()
        else:
            self.train_for_linear()

    def train_for_linear(self):
        similarity_labels = self.similarity_labels
        data_pairs = self.data_pairs

        self.clus()

        self.model = Sequential()
        self.model.add(Dense(kernels, activation='relu', input_shape=self.input_shape))
        self.model.add(Dense(kernels, activation='relu'))
    
        self.model.compile(loss='categorical_crossentropy',
            optimizer=RMSprop(),
            metrics=['accuracy'])


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


    def train_for_non_linear(self, data_pairs, similarity_labels):
        similarity_labels = self.similarity_labels
        data_pairs = self.data_pairs
        X1 = []
        X2 = []
        for x, y in data_pairs:
            X1.append(x)
            X2.append(y)
        number_classes = len(np.unique(similarity_labels)) 

        # convert class vectors to binary class matrices
        train_portion = 0.8
        input_shape = data_pairs[0][0].shape
        # kernels = 100
        kernels = input_shape[0] * 3
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
        # branch1 = BatchNormalization()(branch1)
        
        input2 = keras.layers.Input(shape=input_shape)
        branch2 = keras.layers.Dense(kernels, activation='relu')(input2)
        branch2 = keras.layers.Dense(kernels, activation='relu')(branch2)
        # branch2 = BatchNormalization()(branch2)
        # merged_vector = keras.layers.Dot(axes=1, normalize=True)([branch1, branch2])

        # merged_vector = keras.layers.Lambda(lambda x: (x[0]-x[1])**2)([branch1, branch2])

        distanceModel = Merge(mode=self.euclideanSqDistance, output_shape=[[None,1]], name='distanceMerge') ([branch1, branch2])
        distanceModel = Activation('sigmoid')(distanceModel)
        
        # merged_vector = keras.layers.Dense(number_classes, W_regularizer=reg, activation='relu')(distanceModel)

        model = Model(inputs=[input1, input2], outputs=distanceModel)

        model.summary()

        model.compile(loss=self.contrastive_loss,
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

        x1_train = np.array(X1)
        x2_train = np.array(X2)
        y_train = keras.utils.to_categorical(similarity_labels, number_classes)

        history = model.fit([x1_train, x2_train], y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            callbacks=callbacks_list,
                            validation_data=([x1_train, x2_train], y_train), verbose=0)

        score = model.evaluate([x1_test, x2_test], y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        inp1, inp2 = model.input

        func = model.layers[-1].input
        dist = model.layers[-1].output    
        self.functor1 = K.function([inp1]+ [K.learning_phase()], [func[0]]) 
        self.functor2 = K.function([inp2]+ [K.learning_phase()], [func[1]])
        self.functor3 = K.function([*model.input]+ [K.learning_phase()], [dist])

          
    def transform_for_non_linear(self, data_pairs):
        x, y = data_pairs
        x = x.reshape(1, x.shape[0])
        y = y.reshape(1, y.shape[0])
        layer_out1 = self.functor1([x, 1.])
        layer_out2 = self.functor2([y, 1.])
        # x = pd.Series(layer_out1[0][0])
        # y = pd.Series(layer_out2[0][0])
        distance = self.functor3([*[x, y], 1.])
        # print(layer_out1, layer_out2)
        print(distance)
        return distance[0].mean()

    def transform(self, data_pairs):        
        test = data_pairs.reshape(1, self.input_shape[0])
        if self.Sequence:
            test = sequence.pad_sequences(test, maxlen=self.maxlen)
        layer_out = self.functor([test, 1.])
        return pd.Series(layer_out[0][0])

    def contrastive_loss(self, y_true, y_pred):
        margin = 1
        return K.mean((1-y_true) * 0.5 * K.square(y_pred) + 0.5 * y_true * K.square(K.maximum(margin - y_pred, 0)))

    
    def penalized_loss(self, branch1, branch2):
        def loss(y_true, y_pred):
            return K.mean(K.square(y_pred - y_true) - K.square(y_true - y_pred), axis=-1)
        return loss

    def euclideanSqDistance(self, inputs):
        if (len(inputs) != 2):
            raise 'oops'

        output = K.mean(K.square(inputs[0] - inputs[1]), axis=-1)
        output = K.expand_dims(output, 1)
        return output

    def deep(self):
        kernels = 100

    def cluster(self, data_pair):
        sim = Similarity(data=data_pair)
        sim.extract_interested_attribute(interest='statistics', stat_type=interest, window=window)
        similarity_label, data_subsample = sim.cluster_for_deep(range_n_clusters=range(2,8))


    def check(self, data_pair, similarity_labels):
        
        print(self.cluster(data_pair))

            



        
