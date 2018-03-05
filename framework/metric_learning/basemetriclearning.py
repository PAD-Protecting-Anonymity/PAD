import numpy as np
import pandas as pd

class BaseMetriLearming():
    def __init__(self, **kwargs):
        raise NotImplementedError('users must define train in class to use this base class')

    def train(self, data_pairs, similarity_labels,**kwargs):
        raise NotImplementedError('users must define train in class to use this base class')

    def transform(self,data_pairs):
        raise NotImplementedError('users must define train in class to use this base class')

    def get_distance(self,data):
        raise NotImplementedError('users must define get_distance in class to use this base class')
    
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