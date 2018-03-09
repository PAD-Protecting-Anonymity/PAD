""""""
import pandas as pd
import numpy as np

class BaseSimularity():
    """"""
    def __init__(self, simularity_type,sampling_frequency,output_frequency,data_type, data_window=None, genelaraty_mode=None,**kwargs):
        if self.__class__ == BaseSimularity:
            raise Exception('abstract class')
        self.simularity_type = simularity_type
        self.sampling_frequency = sampling_frequency
        self.output_frequency = output_frequency
        self.genelaraty_mode = genelaraty_mode
        self.data_window = data_window
        self.data_type = data_type
        
    def get_simularity_type(self):
        if (self.simularity_type is None):
            raise NotImplementedError('users must define simularity_type')
        return self.simularity_type

    def get_output_frequency(self):
        if (self.output_frequency is None):
            raise NotImplementedError('users must define output_frequency')
        return self.output_frequency

    def get_sampling_frequency(self):
        if (self.sampling_frequency is None):
            raise NotImplementedError('users must define sampling_frequency')
        return self.sampling_frequency

    def get_information_loss(self, data_originally, data_sanitized, **kwargs):
        raise NotImplementedError('users must define get_information_loss in class to use this base class')

    def get_distance(self,data):
        raise NotImplementedError('users must define get_distance in class to use this base class')

    def get_statistics_distance(self, sample1, sample2, **kwargs):
        raise NotImplementedError('users must define get_statistics_distance to use this base class')


    def get_statistics(self,data,**kwargs):
        raise NotImplementedError('users must define get_statistics to use this base class')
        
    def compute_distance(self,distance,amount_of_data):
        df = pd.DataFrame(distance)
        df.columns = amount_of_data
        df.index = amount_of_data

        distance = df
        x, y = np.meshgrid(distance.index, distance.columns)
        df = pd.DataFrame(columns=["x", "y", "distance"])
        df["x"] = y.ravel()
        df["y"] = x.ravel()
        df["distance"] = distance.as_matrix().ravel()

        df = df[df["x"] != df["y"]]
        # df = df.sort_values('distance')
        df.distance.loc[np.isnan(df.distance)] = 0
        return df