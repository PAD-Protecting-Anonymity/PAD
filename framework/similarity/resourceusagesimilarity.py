from framework.similarity.similarityterms import SimilarityTerms
from framework.similarity.basesimilarity import BaseSimilarity
import numpy as np
import pandas as pd
import math

class ResourceUsageSimilarity(BaseSimilarity):
    '''

    '''
    def __init__(self, data_descriptor, data_window= None,**kwargs):
        super().__init__(SimilarityTerms.USAGE,data_descriptor, data_window)
        self.kwargs = kwargs

    def get_information_loss(self, data_originally, data_sanitized, **kwargs):
        stat_gt = self.get_statistics(data_originally)
        stat_sanitized = self.get_statistics(data_sanitized)
        df = stat_gt - stat_sanitized
        df = df.as_matrix()
        err_sum_sqrt = np.mean(np.absolute(df))
        return err_sum_sqrt

    def get_statistics_distance(self, sample1, sample2, **kwargs):
        if self.data_descriptor.data_window_size is None:
            if self.data_window is None:
                stat1 = self.compute_total_usage(sample1,kwargs["index"])
                stat2 = self.compute_total_usage(sample2,kwargs["index"])
            else:
                stat1 = self.compute_window_usage(sample1,kwargs["index"], self.data_window)
                stat2 = self.compute_window_usage(sample2,kwargs["index"], self.data_window)
        else:
            stat1 = self.compute_use_data_window_size(sample1,kwargs["index"],self.data_window,self.data_descriptor.data_window_size)
            stat2 = self.compute_use_data_window_size(sample2,kwargs["index"],self.data_window,self.data_descriptor.data_window_size)
        dist = stat1 - stat2
        return dist

    def get_statistics(self,data):
        if self.data_window is None:
            stat = self.get_use(data)
        else:
            stat = self.get_window_use(data, self.data_window)
        return stat

    def get_use(self,data):
        use_data = data.apply(self.compute_total_usage, axis=1,index=data.columns).to_frame()
        return use_data

    def get_window_use(self,data, data_window):
        use_data = data.apply(self.compute_window_usage, axis=1,index=data.columns,window=data_window).to_frame()
        return use_data

    def compute_total_usage(self,x,index):
        time_resolution = len(index)
        if isinstance(x,pd.DataFrame):
            x = list(x)
        usage = sum(x) * time_resolution
        return usage

    def compute_window_usage(self,x,index,window):
        # time_resolution = index[1]-index[0]
        if isinstance(x,pd.DataFrame):
            x = list(x)
        usage = sum(x[window[0]:window[1]]) #* time_resolution
        return usage

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
                    distance[i,j] = self.get_statistics_distance(df1,df2,index=cols)
        return super().compute_distance(distance,data.index)

    def compute_use_data_window_size(self,x,index, window,data_window_size):
        amount_of_colums = x.size
        amount_of_slices = math.floor(amount_of_colums/data_window_size)
        df = None
        for i in range(0,amount_of_slices):
            data_slice = x[data_window_size*i:data_window_size*(i+1)]
            if window is None:
                restult =self.compute_total_usage(data_slice,index)
            else:
                restult =self.compute_window_usage(data_slice,index,window)
            if df is not None:
                df = np.append(df,restult)
            else:
                df = np.array(restult)
        return df