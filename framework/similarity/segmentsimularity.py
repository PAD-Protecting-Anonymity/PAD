from similarity.simularityterms import SimularityTerms
from similarity.basesimularity import BaseSimularity
import numpy as np
import pandas as pd

class SegmentSimularity(BaseSimularity):
    '''

    '''
    def __init__(self, sampling_frequency,output_genelaraty, genelaraty_mode, data_window,**kwargs):
        super().__init__(SimularityTerms.ARRIVAL,sampling_frequency,output_genelaraty, genelaraty_mode=genelaraty_mode, data_window=data_window)
        self.kwargs = kwargs

    def get_information_loss(self, data_originally, data_sanitized, **kwargs):
        stat_gt = self.get_segment(data_originally)
        stat_sanitized = self.get_segment(data_sanitized)
        df = stat_gt - stat_sanitized
        df = df.as_matrix()
        err_sum_sqrt = np.mean(np.absolute(df))
        return err_sum_sqrt

    def get_statistics_distance(self, sample1, sample2, **kwargs):
        stat1 = self.compute_segment(sample1,kwargs["index"],self.data_window)
        stat2 = self.compute_segment(sample2,kwargs["index"],self.data_window)
        dist = np.linalg.norm(stat1-stat2)
        return dist

    def get_statistics(self,data):
        return self.get_segment(data)

    def get_segment(self,data, flag=1):
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
        