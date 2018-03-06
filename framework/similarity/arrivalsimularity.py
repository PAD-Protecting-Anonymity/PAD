# from similarity import basesimularity as BaseSimularity, simularityterms as SimularityTerms
from similarity.simularityterms import SimularityTerms
from similarity.basesimularity import BaseSimularity
from sklearn.neighbors.dist_metrics import DistanceMetric
# import basesimularity as BaseSimularity
import numpy as np
import pandas as pd

class ArrivalSimularity(BaseSimularity):

    '''

    '''
    def __init__(self, sampling_frequency,output_genelaraty, genelaraty_mode, data_window=None,**kwargs):
        super().__init__(SimularityTerms.ARRIVAL,sampling_frequency,output_genelaraty, genelaraty_mode=genelaraty_mode, data_window=data_window)
        self.kwargs = kwargs

    def get_information_loss(self, data_originally, data_sanitized, **kwargs):
        stat_gt = self.get_arrival_time(data_originally)
        stat_sanitized = self.get_arrival_time(data_sanitized)
        df = stat_gt - stat_sanitized
        df = df.as_matrix()
        err_sum_sqrt = np.mean(np.absolute(df))
        return err_sum_sqrt

    def get_statistics_distance(self, sample1, sample2, **kwargs):
        if self.data_window is not None:
            stat1 = self.compute_arrival_time_window(sample1,kwargs["index"],kwargs["flag"],self.data_window)
            stat2 = self.compute_arrival_time_window(sample2,kwargs["index"],kwargs["flag"],self.data_window)
        else:
            stat1 = self.compute_arrival_time(sample1,kwargs["index"],kwargs["flag"])
            stat2 = self.compute_arrival_time(sample2,kwargs["index"],kwargs["flag"])
        dist = abs(stat1-stat2)
        return dist
        
    def compute_arrival_time_window(self,x,index,flag,window):
        '''
        :return: arrival time
                - nan indicates no arrivals in the day
                - inf indicates constant occupied in the day
                - number indicates the time when people arrive
        '''
        if isinstance(x,pd.Series):
            win_start = window[0]
            win_end = window[1] 
            x = list(x[win_start:win_end])
        # pdb.set_trace()
        if x[0] == 0:
            arrival_time_ind = next((ind for ind, value in enumerate(x) if value > 0), np.nan)
        elif x[0] > 0:
            late_morning_dep = next((ind for ind, value in enumerate(x) if value == 0), None)
            if late_morning_dep is None:
                arrival_time_ind = np.inf
            else:
                x_sub = x[late_morning_dep:]
                arrival_time_ind_offset = next((ind for ind, value in enumerate(x_sub) if value > 0), None)
                if arrival_time_ind_offset is None:
                    arrival_time_ind = np.nan
                else:
                    arrival_time_ind = late_morning_dep + arrival_time_ind_offset
        # else:
        #     arrival_time_ind = np.nan
        # print(x)
        # print(arrival_time_ind)
        if np.isnan(arrival_time_ind): # if no arrival in the day
            if flag == 0:
                arrival_time = np.nan
            else:
                arrival_time = 0
        elif np.isinf(arrival_time_ind): # if constantly occupied in the day
            arrival_time = 0
        else:
            arrival_time = index[arrival_time_ind]
        return arrival_time

    def compute_arrival_time(self,x,index,flag):
        '''
        :return: arrival time
                - nan indicates no arrivals in the day
                - inf indicates constant occupied in the day
                - number indicates the time when people arrive
        '''
        if isinstance(x,pd.Series):
            x = list(x)
        # pdb.set_trace()
        if x[0] == 0:
            arrival_time_ind = next((ind for ind, value in enumerate(x) if value > 0), np.nan)
        elif x[0] > 0:
            late_morning_dep = next((ind for ind, value in enumerate(x) if value == 0), None)
            if late_morning_dep is None:
                arrival_time_ind = np.inf
            else:
                x_sub = x[late_morning_dep:]
                arrival_time_ind_offset = next((ind for ind, value in enumerate(x_sub) if value > 0), None)
                if arrival_time_ind_offset is None:
                    arrival_time_ind = np.nan
                else:
                    arrival_time_ind = late_morning_dep + arrival_time_ind_offset
        # else:
        #     arrival_time_ind = np.nan
        # print(x)
        # print(arrival_time_ind)
        if np.isnan(arrival_time_ind): # if no arrival in the day
            if flag == 0:
                arrival_time = np.nan
            else:
                arrival_time = 0
        elif np.isinf(arrival_time_ind): # if constantly occupied in the day
            arrival_time = 0
        else:
            arrival_time = index[arrival_time_ind]
        return arrival_time

    def get_statistics(self,data):
        return self.get_arrival_time(data)

    def get_arrival_time(self,data, window = None, flag=1):
        if window is None:
            arrival_time = data.apply(self.compute_arrival_time, axis=1,index=data.columns,flag=flag).to_frame()
        else:
            arrival_time = data.apply(self.compute_arrival_time_window, axis=1,index=data.columns,flag=flag, window=window).to_frame()            
        return arrival_time

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