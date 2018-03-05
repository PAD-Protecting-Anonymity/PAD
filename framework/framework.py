from utilities.subsampling import Subsampling
from similarity.simularatielist import SimularatieList
from similarity.similarity import Similarity
from models.kward import K_ward
from metric_learning.linearmetric import Linear_Metric
from scipy.misc import comb

import pandas as pd

class Framwork:
    def __init__(self, data,amount_Of_Variables, anonymity_level):
        self.data = data
        self.simularatie_list = SimularatieList()
        self.amount_Of_Variables = amount_Of_Variables
        self.anonymity_level = anonymity_level
        self.subsampling = None
        self.similarity = None

    def define_data_struckture(self):
        test = 1
    
    def add_simularatie(self,simularatie):
        self.simularatie_list.add_simularatie(simularatie)

    def _find_Simularaty(self):
        # TODO: make my
        raise NotImplementedError('NotImplemented')
    
    def _find_Metric_Leaning(self,data_pair,similarity_label):
        lm = Linear_Metric()
        lm.train(data_pair, similarity_label)
        return lm

    def _find_Model(self, data):
        # TODO: make my
        raise NotImplementedError('NotImplemented')
    
    def _subsample(self,presenitizedData, sub_sampling_size=0.1, seed=None):
        self.subsampling = Subsampling(presenitizedData.sample(frac=sub_sampling_size))
        subsample_size_max = int(comb(len(self.subsampling.data), 2))
        print('total number of pairs is %s' % subsample_size_max)
        data_pair_all, data_pair_all_index = self.subsampling.uniform_sampling(subsample_size=subsample_size_max, seed=seed)
        self.similarity = Similarity(data=data_pair_all)
        self.similarity.extract_interested_attribute(self.simularatie_list.simularaties)
        similarity_label, data_subsample = self.similarity.label_via_silhouette_analysis(range_n_clusters=range(2,8))
        # similarity_label_all_series = pd.Series(similarity_label)
        # similarity_label_all_series.index = data_pair_all_index
        print('similarity balance is %s'% [sum(similarity_label),len(similarity_label)])
        return similarity_label, data_pair_all


    def _presanitized(self):
        sanitized_df = self._sanitize_data(data = self.data, distance_metric_type="init", rep_mode = "mean",
                        anonymity_level=self.anonymity_level)
        return sanitized_df

    def _sanitize_data(self,distance_metric_type, rep_mode, anonymity_level, data, **kwargs):
        k_ward = None
        if distance_metric_type == "init":
            k_ward = K_ward(data, distance_metric_type=distance_metric_type, rep_mode = rep_mode,k=anonymity_level, initProfile=self.simularatie_list)
        else:
            metric = kwargs["metric"]
            k_ward = K_ward(data, distance_metric_type=distance_metric_type, rep_mode = rep_mode,k=anonymity_level, metric=metric)
        k_ward.get_cluster()
        groups = k_ward.groups

        sanitized_df = pd.DataFrame()
        for group in groups:
            sanitized_value = group.rep.to_frame().transpose()
            keys = group.get_member_ids()
            for key in keys:
                sanitized_value.index = [key]
                sanitized_df = sanitized_df.append(sanitized_value)
        sanitized_df.columns = data.columns
        return sanitized_df

    def run(self):
        presenitized_data = self._presanitized()
        similarity_label, data_subsample = self._subsample(presenitized_data)
        model = self._find_Metric_Leaning(data_subsample,similarity_label)
        final_samitized_data = self._sanitize_data(data = self.data, distance_metric_type="metric", rep_mode = "mean",
                        anonymity_level=self.anonymity_level,metric=model)
        return final_samitized_data
        


    

        
        
        
        