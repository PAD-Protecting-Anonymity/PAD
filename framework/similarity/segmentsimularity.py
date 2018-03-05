import basesimularity as BaseSimularity
from simularityterms import SimularityTerms
import numpy as np
import pandas as pd

class SegmentSimularity(BaseSimularity):

    def __init__(self, sampling_frequency,output_genelaraty, genelaraty_mode):
        super().__init__(SimularityTerms.SEGMENT,sampling_frequency,output_genelaraty, genelaraty_mode)

    def get_information_loss(self, data_originally, data_sanitized, **kwargs):
        stat_gt = self.get_global(data_originally)
        stat_sanitized = self.get_global(data_sanitized)
        df = stat_gt - stat_sanitized
        df = df.as_matrix()
        err_sum_sqrt = np.mean(np.absolute(df))
        return err_sum_sqrt

    def get_statistics_distance(self, sample1, sample2, **kwargs):
        stat1 = self.compute_global(sample1)
        stat2 = self.compute_global(sample2)
        dist = np.linalg.norm(stat1-stat2)
        return dist

    def get_global(self,data):
        global_score = data.apply(self.compute_global, axis=1,index=data.columns).to_frame()
        return global_score

    def compute_global(self,data):
        if isinstance(data,pd.DataFrame):
            data = list(data)
        return data
        