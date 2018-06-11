# from similarity import basesimularity as BaseSimularity, simularityterms as SimularityTerms
from framework.similarity.similarityterms import SimilarityTerms
from framework.similarity.basesimilarity import BaseSimilarity
import numpy as np
import pandas as pd
import math

class DepartureSimilarity(BaseSimilarity):
    '''

    '''
    def __init__(self, data_descriptor, data_window=None,**kwargs):
        super().__init__(SimilarityTerms.ARRIVAL,data_descriptor, data_window=data_window)
        self.kwargs = kwargs

    def get_information_loss(self, data_originally, data_sanitized, **kwargs):
        stat_gt = self.get_arrival_time(data_originally)
        stat_sanitized = self.get_arrival_time(data_sanitized)
        df = stat_gt - stat_sanitized
        df = df.as_matrix()
        err_sum_sqrt = np.mean(np.absolute(df))
        return err_sum_sqrt

    def get_statistics_distance(self, sample1, sample2, **kwargs):
        if self.data_descriptor.data_window_size is None:
            if self.data_window is not None:
                stat1 = self.compute_departure_time_window(sample1,kwargs["index"],kwargs["flag"],self.data_window)
                stat2 = self.compute_departure_time_window(sample2,kwargs["index"],kwargs["flag"],self.data_window)
            else:
                stat1 = self.compute_departure_time(sample1,kwargs["index"],kwargs["flag"])
                stat2 = self.compute_departure_time(sample2,kwargs["index"],kwargs["flag"])
            dist = abs(stat1-stat2)
        else:
            slice_size = self.data_descriptor.data_window_size
            sample1 = list(sample1)
            sample2 = list(sample1)
            amount_of_slices = math.floor(len(sample1) /slice_size)
            dist = 0
            for i in range(0,amount_of_slices+1):
                index_from = slice_size*i
                index_to = slice_size*(i+1)
                if self.data_window is not None:
                    stat1 = self.compute_departure_time_window(sample1[index_from:index_to],kwargs["index"],kwargs["flag"],self.data_window)
                    stat2 = self.compute_departure_time_window(sample2[index_from:index_to],kwargs["index"],kwargs["flag"],self.data_window)
                else:
                    stat1 = self.compute_departure_time(sample1[index_from:index_to],kwargs["index"],kwargs["flag"])
                    stat2 = self.compute_departure_time(sample2[index_from:index_to],kwargs["index"],kwargs["flag"])
                dist += abs(stat1-stat2)
        return dist

    def compute_departure_time_window(self,x,index,flag,window):
        '''
        :return: departure time
                - nan indicates no arrivals in the day
                - inf indicates constant occupied in the day
                - number indicates the time when people arrive
        '''
        if isinstance(x,pd.Series):
            win_start = window[0]
            win_end = window[1]
            x = list(x[win_start:win_end])
        return self.compute_departure_time(x,index,flag)

    def compute_departure_time(self,x,index, flag):

        if isinstance(x,pd.Series):
            x = list(x)
        arr_x = np.array(x)
        if x[-1] > 0:
            if x[0] > 0 and len(arr_x[arr_x==0]) > 0:
                dep_time_ind = next((ind for ind,value in enumerate(x) if value == 0), None)
            else:
                dep_time_ind = np.inf
        elif x[-1] == 0:
            x_inv = x[::-1]
            dep_time_ind_inv = next((ind for ind, value in enumerate(x_inv) if value > 0 ), None)

            if dep_time_ind_inv is None:
                dep_time_ind = np.nan
            else:
                dep_time_ind = len(x) - dep_time_ind_inv

        # print(dep_time_ind)
        # if dep_time_ind is None:
        #     pdb.set_trace()
        if np.isinf(dep_time_ind):
            if flag == 0:
                dep_time = np.nan
            else:
                dep_time = index.max()+1  # no departure in the day
        elif np.isnan(dep_time_ind):
            if flag == 0:
                dep_time = np.nan#-1  # no arrival as well as departure in the day
            else:
                dep_time = 0
        else:
            dep_time = index[dep_time_ind]
        return dep_time

    def get_statistics(self,data):
        return self.get_arrival_time(data)

    def get_departure_time(self,flag):
        if window is None:
            departure_time = data.apply(self.compute_departure_time, axis=1,index=data.columns,flag=flag).to_frame()
        else:
            departure_time = data.apply(self.compute_departure_time_window, axis=1,index=data.columns,flag=flag, window=window).to_frame()
        return departure_time

    def get_distance(self,data):
        data_copy = data.copy()
        data_copy = data_copy.fillna(0)
        data_copy = data_copy.as_matrix()
        data_size = data_copy.shape[0]
        distance = np.empty((data_size,data_size))
        cols = data.columns
        for i in range(data_size):
            df1 = data_copy[i, :]
            for j in range(data_size):
                df2 = data_copy[j,:]
                if i > j:
                    distance[i,j] = distance[j,i]
                    continue
                elif i == j:
                    distance[i,j] = 0
                    continue
                else:
                    distance[i,j] = self.get_statistics_distance(df1,df2,index=cols,flag=1)
        return super().compute_distance(distance,data.index)