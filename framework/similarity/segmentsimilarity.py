from framework.similarity.similarityterms import SimilarityTerms
from framework.similarity.basesimilarity import BaseSimilarity
import numpy as np
import pandas as pd
import math


class SegmentSimilarity(BaseSimilarity):
    '''

    '''
    def __init__(self, data_descriptor, data_window,**kwargs):
        super().__init__(SimilarityTerms.SEGMENT,data_descriptor, data_window)
        self.kwargs = kwargs

    def get_information_loss(self, data_originally, data_sanitized, **kwargs):
        stat_gt = self.get_segment(data_originally)
        stat_sanitized = self.get_segment(data_sanitized)
        df = stat_gt - stat_sanitized
        df = df.as_matrix()
        err_sum_sqrt = np.mean(np.absolute(df))
        return err_sum_sqrt

    def get_statistics_distance(self, sample1, sample2, **kwargs):
        if self.data_descriptor.data_window_size is None:
            stat1 = self.compute_segment(sample1,kwargs["index"],self.data_window)
            stat2 = self.compute_segment(sample2,kwargs["index"],self.data_window)
        else:
            stat1 = self.compute_segment_data_window_size(sample1,kwargs["index"],self.data_window,self.data_descriptor.data_window_size)
            stat2 = self.compute_segment_data_window_size(sample2,kwargs["index"],self.data_window,self.data_descriptor.data_window_size)
        dist = np.linalg.norm(stat1-stat2)
        return dist

    def get_statistics(self,data):
        return self.get_segment(data)

    def get_segment(self,data):
        segment_data = data.apply(self.compute_segment, axis=1,index=data.columns,window=self.data_window)
        return segment_data

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

    def compute_segment(self,x,index, window):
        win_start = window[0]
        win_end = window[1]
        if isinstance(x,pd.DataFrame):
            x = list(x)
        df = x[win_start:win_end]
        return df

    def compute_segment_data_window_size(self,x,index, window,data_window_size):
        amount_of_colums = x.size
        amount_of_slices = math.floor(amount_of_colums/data_window_size)
        df = None
        for i in range(0,amount_of_slices):
            data_slice = x[data_window_size*i:data_window_size*(i+1)]
            restult =self.compute_segment(data_slice,index,window)
            if df is not None:
                df = np.append(df,restult)
            else:
                df = np.array(restult)
        return df