import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors.dist_metrics import DistanceMetric

class BasemetricLearning():
    def __init__(self,metric_learning_terms, **kwargs):
        if self.__class__ == BasemetricLearning:
            raise Exception('abstract class')
        self.metric_learning_terms = metric_learning_terms

    def train(self, data_pairs, similarity_labels,**kwargs):
        raise NotImplementedError('users must define train in class to use this base class')


    def transform(self, data_pairs):
        x, y = data_pairs
        x = self.scaler.transform(np.array([x]))
        y = self.scaler.transform(np.array([y]))
        distance = self.functor3([*[x, y], 1.])
        return distance[0].mean()

    # def transform(self,data_pairs):
    #     raise NotImplementedError('users must define train in class to use this base class')

    def get_distance(self,data):
        raise NotImplementedError('users must define get_distance in class to use this base class')

    def deep_metric(self, x, y):
        # x, y = self.dm.transform((x,y))
        # dist = np.linalg.norm(x-y)
        dist = self.transform((x,y))
        # print(dist)
        return dist

    def get_distance(self,data):
        # distance = pairwise_distances(data,metric=self.deep_metric,n_jobs=-1)
        dist = DistanceMetric.get_metric(metric = 'pyfunc', func=self.deep_metric)
        distance = dist.pairwise(data)
        return self.compute_distance(distance,data.index)

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
