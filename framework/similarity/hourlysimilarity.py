from framework.similarity.similarityterms import SimilarityTerms
from framework.similarity.basesimilarity import BaseSimilarity
import numpy as np
import pandas as pd
from framework.utilities.datadescriptor import DataDescriptorTerms
import math

class HourlySimilarity(BaseSimilarity):
    '''

    '''
    def __init__(self, data_descriptor, data_window,**kwargs):
        super().__init__(SimilarityTerms.HOURLY,data_descriptor, data_window)
        self.kwargs = kwargs
        if 'sampling_frequency' in self.kwargs:
            self.sampling_frequency = self.kwargs["sampling_frequency"]
        else:
            raise ValueError("sampling_frequency most be defined, is to be set to the samping freqency of the input samples")
        if 'mode' in self.kwargs:
            self.mode = self.kwargs["mode"]
        else:
            raise ValueError("mode most be defined, selected a HourlySimilarityMode")
        if 'time_ly' in self.kwargs:
            self.time_ly = self.kwargs["time_ly"]


    def get_information_loss(self, data_originally, data_sanitized, **kwargs):
        stat_gt = self.get_hourly(data_originally)
        stat_sanitized = self.get_hourly(data_sanitized)
        df = stat_gt - stat_sanitized
        df = df.as_matrix()
        err_sum_sqrt = np.mean(np.absolute(df))
        return err_sum_sqrt

    def get_statistics_distance(self, sample1, sample2, **kwargs):
        if self.data_descriptor.data_window_size is None:
            stat1 = self.compute_hourly(sample1,kwargs["index"],self.data_window)
            stat2 = self.compute_hourly(sample2,kwargs["index"],self.data_window)
        else:
            stat1 = self.compute_hourly_data_window_size(sample1,kwargs["index"],self.data_window,self.data_descriptor.data_window_size)
            stat2 = self.compute_hourly_data_window_size(sample2,kwargs["index"],self.data_window,self.data_descriptor.data_window_size)
        dist = np.linalg.norm(stat1-stat2)
        return dist

    def get_statistics(self,data):
        return self.get_hourly(data)

    def get_hourly(self,data):
        if self.data_descriptor.data_window_size is None:
            hourly_data = self.hourly_of_more_day(data,data.columns,self.data_window,self.data_descriptor.data_window_size)
        else:
            hourly_data = self.hourly_of_more_days(data,data.columns,self.data_window,self.data_descriptor.data_window_size)
        return hourly_data

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

    def hourly_of_more_day(self,data,index, window,data_window_size, **kwargs):
        pds = pd.DataFrame()
        for row in data.index:
            row_segment = self.compute_hourly(data.loc[row],index,window)
            pds =pds.append(pd.Series(row_segment), ignore_index=True)
        return pds

    def compute_hourly(self,x,index, window):
        if 'time_ly' not in self.kwargs:
            amount_samples_per_hour = math.floor(DataDescriptorTerms.HOUR.value /self.sampling_frequency)
        else:
            amount_samples_per_hour = math.floor(self.time_ly /self.sampling_frequency)
        if window is not None:
            win_start = window[0]
            win_end = window[1]
        else:
            win_start = 0
            win_end = len(x)-1
        amount_of_hours = math.floor((win_end - win_start)/amount_samples_per_hour)
        if isinstance(x,pd.DataFrame):
            x = list(x)
        results = []
        for i in range(0,amount_of_hours):
            hour_result = x[win_start+i*amount_samples_per_hour:win_start+(1+i)*amount_samples_per_hour]
            if 'mode' in self.kwargs:
                hour_result = self.calculate_mode(hour_result,self.kwargs["mode"])
            results.append(hour_result)
        return np.array(results)

    def compute_hourly_data_window_size(self,x,index, window,data_window_size):
        amount_of_colums = x.size
        amount_of_slices = math.floor(amount_of_colums/data_window_size)
        df = None
        for i in range(0,amount_of_slices):
            data_slice = x[data_window_size*i:data_window_size*(i+1)]
            restult =self.compute_hourly(data_slice,index,window)
            if df is not None:
                df = np.append(df,restult)
            else:
                df = restult
        df = df[:]
        return df

    def hourly_of_more_days(self,data,index, window,data_window_size, **kwargs):
        pds = pd.DataFrame()
        for row in data.index:
            hour_results = self.compute_segment_data_window_size(data.loc[row],index,window,data_window_size)
            pds =pds.append(pd.Series(hour_results), ignore_index=True)
        return pds

    def calculate_mode(self,df,mode):
        df_results = None
        if mode == HourlySimilarityModes.SUMOFHOUR:
            df_results = np.sum(df)
        elif mode == HourlySimilarityModes.MEANOFHOUR:
            df_results = np.mean(df)
        return df_results

class HourlySimilarityModes:
    MEANOFHOUR = "mean" #Use the mean of the hour
    SUMOFHOUR = "sum" #Use the sum of the hour






