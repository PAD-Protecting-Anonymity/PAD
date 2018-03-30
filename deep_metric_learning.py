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
from kward import Distance
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

    def train(self, data_pairs, similarity_labels_original):
        self.scaler = StandardScaler()
        self.batch_size = 32
        self.epochs = 100
        X1,X2 = zip(*data_pairs)

        similarity_labels = similarity_labels_original
        # similarity_labels = 1 - np.array(similarity_labels_original)
        train_portion = 0.5
        input_shape = data_pairs[0][0].shape
        kernels = 20
        s_size = len(similarity_labels)
        x1_train = np.array(X1[:int(s_size * train_portion)])
        x2_train = np.array(X2[:int(s_size * train_portion)])
        x1_test = np.array(X1[int(s_size * train_portion):])
        x2_test = np.array(X2[int(s_size * train_portion):])
        y_train = np.array(similarity_labels[:int(s_size * train_portion)])
        y_test = np.array(similarity_labels[int(s_size * train_portion):])
        # y_train = keras.utils.to_categorical(similarity_labels[:int(s_size * train_portion)],number_classes)
        # y_test = keras.utils.to_categorical(similarity_labels[int(s_size * train_portion):],number_classes)

        # define networks
        left_input = Input(input_shape)
        right_input = Input(input_shape)
        model = Sequential()
        model.add(Dense(kernels, activation='relu', input_shape = input_shape))
        model.add(Dropout(0.3))
        # model.add(Dense(int(kernels), activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(int(kernels/2), activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(kernels, activation='relu'))
        model.add(Dropout(0.3))
        # model.add(Dense(int(kernels/2), activation='tanh',kernel_regularizer=regularizers.l2(0.01),
        #         activity_regularizer=regularizers.l1(0.01)))
        # model.add(Dropout(0.2))


        encoded_l = model(left_input)
        encoded_r = model(right_input)

        both = merge([encoded_l,encoded_r],mode=euc_dist,output_shape=euc_dist_shape)
        self.distance_model = Model(input=[left_input,right_input],outputs=both)
        self.distance_model.compile(loss=self.contrastive_loss, optimizer='Adam')
        # prediction = Dense(1,activation='sigmoid')(both)
        # self.distance_model = Model(input=[left_input,right_input],outputs=prediction)
        # self.distance_model.compile(loss="binary_crossentropy",optimizer='Adam')
        #
        # inp1,inp2 = self.distance_model.input
        # distance_output = self.distance_model.layers[-2].output
        # self.euc_distance = K.function([*[inp1,inp2]] + [K.learning_phase()],[distance_output])

        # # normalized data
        # data = np.append(x1_train, x2_train, axis=0)
        # self.scaler.fit(data)
        # x1_train = self.scaler.transform(x1_train)
        # x2_train = self.scaler.transform(x2_train)
        # x1_test = self.scaler.transform(x1_test)
        # x2_test = self.scaler.transform(x2_test)
        #
        print('===training===')
        self.distance_model.fit([x1_train, x2_train], y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            validation_data=([x1_test, x2_test], y_test), verbose=1)


        d_test = self.distance_model.predict([x1_test,x2_test])
        stat_util = Distance()
        cols = np.arange(x1_test.shape[1])
        d_test_arr = np.zeros(x1_test.shape[0])
        x1_test_df = pd.DataFrame(x1_test)
        x2_test_df = pd.DataFrame(x2_test)

        for k in range(x1_test.shape[0]):
            d_test_arr[k] = stat_util.get_statistic_distance(x1_test_df.iloc[k,:],x2_test_df.iloc[k,:],
                                                             index=cols,mode='arrival')

        # d_train = self.distance_model.predict([x1_train, x2_train])
        # pdb.set_trace()
        # plt.scatter(d_train,d_test_arr)
        # plt.show()

        similar_ind = np.where(y_test == 1) [0]
        dissimilar_ind = np.where(y_test == 0)[0]
        # print('dissimilar distances %s'%np.mean(d_test[similar_ind]))
        # print('dissimilar distances var %s' % np.std(d_test[similar_ind]))
        # print('similar distances %s' % np.mean(d_test[dissimilar_ind]))
        # print('similar distances var %s' % np.std(d_test[dissimilar_ind]))
        print('similar distances %s'%np.mean(d_test[similar_ind]))
        print('similar distances var %s' % np.std(d_test[similar_ind]))
        print('dissimilar distances %s' % np.mean(d_test[dissimilar_ind]))
        print('dissimilar distances var %s' % np.std(d_test[dissimilar_ind]))
        print('======')
        print('similar distances %s'%np.mean(d_test_arr[similar_ind]))
        print('similar distances var %s' % np.std(d_test_arr[similar_ind]))
        print('dissimilar distances %s' % np.mean(d_test_arr[dissimilar_ind]))
        print('dissimilar distances var %s' % np.std(d_test_arr[dissimilar_ind]))
        # pdb.set_trace()


        # plt.figure()
        # plt.hist(d_test[similar_ind],label='similar')
        # plt.hist(d_test[dissimilar_ind],label='dissimilar')
        # plt.legend()
        # plt.show()
        # pdb.set_trace()

    # def transform(self,data_pairs):
    #     x, y = data_pairs
    #     x = self.scaler.transform(np.array([x]))
    #     y = self.scaler.transform(np.array([y]))
    #     distance = np.squeeze(self.euc_distance([*[x,y],0]))
    #     return np.asscalar(distance)

    def transform(self, data_pairs):
        x, y = data_pairs
        x = x.reshape((1,len(x)))
        y = y.reshape((1,len(y)))
        # x = self.scaler.transform(np.array([x]))
        # y = self.scaler.transform(np.array([y]))
        distance = self.distance_model.predict([x,y])
        # print('euc dist is %s'% np.sum(np.abs(x-y)))
        # print(distance)
        return distance

    def contrastive_loss(self, y_true, d):
        # 1 means simlar, and 0 means dissimilar
        margin = 1
        # return K.mean(y_true * 0.5 * d + (1 - y_true) * 0.5 * (-d))
        return K.mean(y_true*0.5*d + (1-y_true)*0.5*K.maximum(margin-d,0))
        # return K.mean((1 - y_true) * 0.5 * K.square(d) + 0.5 * y_true * K.square(K.maximum(margin - d, 0)))


