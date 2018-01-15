# Use scikit-learn to grid search the learning rate and momentum
import copy
import numpy
import numpy as np
import numpy.random as rng
import pandas as pd
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Dropout, Activation, Add, Merge, Input, merge
from keras.preprocessing import sequence
from keras import backend as K
from keras.callbacks import ModelCheckpoint

class Model_Tuning:
                
        def __init__(self, data_pairs, similarity_labels):
                self.input_shape = data_pairs[0][0].shape
                self.kernels = 100
                X1 = []
                X2 = []
                for x, y in data_pairs:
                        X1.append(x)
                        X2.append(y)
                self.number_classes = len(np.unique(similarity_labels)) 
                train_portion = 0.8

                kernels = 100
                s_size = len(similarity_labels)
                self.x1_train = X1[:int(s_size * train_portion)]
                self.x2_train = X2[:int(s_size * train_portion)]
                self.x1_test = X1[int(s_size * train_portion):]
                self.x2_test = X2[int(s_size * train_portion):]
                self.y_train = similarity_labels[:int(s_size * train_portion)]
                self.y_test = similarity_labels[int(s_size * train_portion): ]

                labels = keras.utils.to_categorical(similarity_labels, self.number_classes)
                self.y_train = keras.utils.to_categorical(self.y_train, self.number_classes)
                self.y_test = keras.utils.to_categorical(self.y_test, self.number_classes)

                self.x1_train = np.array(self.x1_train)
                self.x2_train = np.array(self.x2_train)
                self.y_train = np.array(self.y_train)

                self.x1_test = np.array(self.x1_test)
                self.x2_test = np.array(self.x2_test)
                self.y_test = np.array(self.y_test)



        def create_modell(self, neurons=1, learn_rate=0.01):
                left_input = Input(self.input_shape)
                right_input = Input(self.input_shape)

                model = Sequential()
                model.add(Dense(self.kernels, activation='relu', input_dim=self.input_shape[0]))
                model.add(Dense(self.kernels, activation='relu'))
                model.add(Dense(self.kernels, activation='sigmoid'))

                encoded_l = model(left_input)
                encoded_r = model(right_input)

                #merge two encoded inputs with the l1 distance between them
                L1_distance = lambda x: K.abs(x[0]-x[1])
                both = merge([encoded_l, encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
                prediction = Dense(2,activation='sigmoid')(both)
                siamese_net = Model(input=[left_input,right_input],output=prediction)
                optimizer = RMSprop()
                siamese_net.compile(loss=self.contrastive_loss, optimizer=optimizer)
                return siamese_net

        def create_model(self, neurons=1, learn_rate=0.01):
                # create model
                model = Sequential()
                model.add(Dense(neurons, input_dim=self.input_shape[0], activation='relu'))
                model.add(Dense(2,  activation='sigmoid'))
                # Compile model
                optimizer = SGD(lr=learn_rate)
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                return model


        def get_best_param(self):
                # fix random seed for reproducibility
                seed = 7
                numpy.random.seed(seed)
                # create model
                model = KerasClassifier(build_fn=self.create_modell, epochs=100, batch_size=10, verbose=0)
                # define the grid search parameters
                
                learn_rate = [0.001, 0.01, 0.1]
                neurons = [10, 5, 10, 15]
                batch_size = [10, 15, 20, 25, 30]
                epochs = [10, 50, 100, 200]

                param_grid = dict(neurons=neurons, learn_rate=learn_rate)
                grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

                X1 = self.x1_train
                X2 = self.x1_train
                
                self.maxlen = self.input_shape[0]
                x1 = sequence.pad_sequences(X1, maxlen=self.maxlen)
                x2 = sequence.pad_sequences(X2, maxlen=self.maxlen)
                print(X1.shape, X2.shape, self.y_train.shape)

                result = self.y_train#.reshape((448, 2))
                grid_result = grid.fit(*[x1, x2], result)

                # summarize results
                print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
                means = grid_result.cv_results_['mean_test_score']
                # stds = grid_result.cv_results_['std_test_score']
                params = grid_result.cv_results_['params']
                for mean, stdev, param in zip(means, stds, params):
                        print("%f (%f) with: %r" % (mean, stdev, param))

        def contrastive_loss(self, y_true, y_pred):
                margin = 1
                return K.mean((1-y_true) * 0.5 * K.square(y_pred) + 0.5 * y_true * K.square(K.maximum(margin - y_pred, 0))) 


# mt = Model_Tuning(data_pairs, similarity_labels)
# met.get_best_param()