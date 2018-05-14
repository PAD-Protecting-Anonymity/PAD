# from similarity import basesimularity as BaseSimularity, simularityterms as SimularityTerms
from framework.similarity.similarityterms import SimilarityTerms
from framework.similarity.basesimilarity import BaseSimilarity
from sklearn.neighbors.dist_metrics import DistanceMetric
# import basesimularity as BaseSimularity
import numpy as np
import pandas as pd

class GlobalSimilarity(BaseSimilarity):
    '''
    distance_metric can
    EUCLIDEAN,
    MAHALAOBIS = "mahalanobis"
    CHEBYSHEV = "chebyshev"
    MANHATTAN = "manhattan"
    MINKOWSKI = "minkowski"
    WMINKOWSKI = "wminkowski"
    SEUCLIDEAN = "seuclidean"
    CUSTOM: add FUNC to calc the distance
    '''
    def __init__(self, data_descriptor,data_window=None,**kwargs):
        super().__init__(SimilarityTerms.GLOBAL, data_descriptor,data_window=data_window)
        if "distance_metric" in kwargs and kwargs["distance_metric"] is not None:
            self.distance_metric = kwargs["distance_metric"]
        else:
            self.distance_metric = SimilarityTerms.EUCLIDEAN
        self.kwargs = kwargs

    def get_information_loss(self, data_originally, data_sanitized, **kwargs):
        stat_gt = self.get_global(data_originally)
        stat_sanitized = self.get_global(data_sanitized)
        # print(stat_gt)
        # print(stat_sanitized)
        df = stat_gt - stat_sanitized
        df = df.as_matrix()
        err_sum_sqrt = np.mean(np.absolute(df))
        return err_sum_sqrt

    def get_statistics_distance(self, sample1, sample2, **kwargs):
        stat1 = self.get_statistics(sample1)
        stat2 = self.get_statistics(sample2)
        dist = np.linalg.norm(stat1-stat2)
        return dist

    def compute_global(self,data,index):
        return data

    def get_statistics(self,data):
        return self.get_global(data)

    def get_global(self,data):
        global_score = data.apply(self.compute_global, axis=1,index=data.columns)#.to_frame()
        return global_score

    def get_distance(self,data):
        if self.distance_metric == SimilarityTerms.EUCLIDEAN:
            dist = DistanceMetric.get_metric(self.distance_metric)
        elif self.distance_metric == SimilarityTerms.MAHALAOBIS:
            self.vi = self.kwargs['VI']
            dist = DistanceMetric.get_metric(self.distance_metric, VI=self.vi)
        elif self.distance_metric == SimilarityTerms.CHEBYSHEV:
            dist = DistanceMetric.get_metric(self.distance_metric)
        elif self.distance_metric == SimilarityTerms.MANHATTAN:
            dist = DistanceMetric.get_metric(self.distance_metric)
        elif self.distance_metric == SimilarityTerms.MINKOWSKI:
            self.p =  self.kwargs['P']
            dist = DistanceMetric.get_metric(self.distance_metric, P=self.p)
        elif self.distance_metric == SimilarityTerms.WMINKOWSKI:
            self.p =  self.kwargs['P']
            self.w =  self.kwargs['W']
            dist = DistanceMetric.get_metric(self.distance_metric, P=self.p, W=self.w)
        elif self.distance_metric == SimilarityTerms.SEUCLIDEAN:
            self.v = self.kwargs['V']
            dist = DistanceMetric.get_metric(self.distance_metric, V=self.v)
        elif self.distance_metric == SimilarityTerms.CUSTOM:
            self.func = self.kwargs['FUNC']
            dist = DistanceMetric.get_metric(metric = 'pyfunc', func=self.func)
        distance = dist.pairwise(data)
        return super().compute_distance(distance,data.index)