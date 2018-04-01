import copy
import numpy as np
import numpy.random as rng
import pandas as pd
import keras
from keras.layers import Input,merge
from keras.models import Sequential, Model
from sklearn.preprocessing import StandardScaler
from keras.layers.core import *
import pandas as pd
import pdb
import matplotlib.pyplot as plt
# np.random.seed(0)

def euc_dist(x):
    'Merge function: euclidean_distance(u,v)'
    s = x[0] - x[1]
    output = K.sum(K.square(s),axis=1,keepdims=True)
    return output

def euc_dist_shape(input_shape):
    'Merge output shape'
    shape = list(input_shape)
    outshape = (shape[0][0],1)
    return tuple(outshape)

class Deep_Metric:
    def train(self, x1_train,x2_train,y_train,x1_test,x2_test,y_test):
        self.scaler = StandardScaler()
        self.batch_size = 32
        self.epochs = 100
        kernels = 20
        input_shape = x1_train[0].shape
        print(input_shape)
        # define networks
        left_input = Input(input_shape)
        right_input = Input(input_shape)
        model = Sequential()
        model.add(Dense(kernels, activation='relu', input_shape = input_shape))
        model.add(Dropout(0.3))
        model.add(Dense(int(kernels/2), activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(kernels, activation='relu'))
        model.add(Dropout(0.3))
        encoded_l = model(left_input)
        encoded_r = model(right_input)
        both = merge([encoded_l,encoded_r],mode=euc_dist,output_shape=euc_dist_shape)
        self.distance_model = Model(input=[left_input,right_input],outputs=both)
        self.distance_model.compile(loss=self.contrastive_loss, optimizer='Adam')

        print('===training===')
        self.distance_model.fit([x1_train, x2_train], y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            validation_data=([x1_test, x2_test], y_test), verbose=1)


        self.d_test = self.distance_model.predict([x1_test,x2_test])
        similar_ind = np.where(y_test == 1) [0]
        dissimilar_ind = np.where(y_test == 0)[0]
        print('similar distances %s'%np.mean(self.d_test[similar_ind]))
        print('similar distances var %s' % np.std(self.d_test[similar_ind]))
        print('dissimilar distances %s' % np.mean(self.d_test[dissimilar_ind]))
        print('dissimilar distances var %s' % np.std(self.d_test[dissimilar_ind]))

    def transform(self, data_pairs):
        x, y = data_pairs
        x = x.reshape((1,len(x)))
        y = y.reshape((1,len(y)))
        distance = self.distance_model.predict([x,y])
        return distance

    def contrastive_loss(self, y_true, d):
        # 1 means simlar, and 0 means dissimilar
        margin = 1
        return K.mean(y_true*0.5*d + (1-y_true)*0.5*K.maximum(margin-d,0))



