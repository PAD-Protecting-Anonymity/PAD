from utilities.subsampling import Subsampling
from similarity.simularatielist import SimularatieList
from similarity.similarity import Similarity
from models.kward import K_ward
from metric_learning.linearmetric import Linear_Metric
from metric_learning.nonlineardeepmetric import NonlinearDeepMetric
from scipy.misc import comb
from utilities.outputgroupper import OutputGroupper

import pandas as pd

class Framwork:
    def __init__(self, data, anonymity_level,dataset_description=""):
        self.data = data
        self._simularatie_list = SimularatieList()
        self.dataset_descriptions = []        
        self.anonymity_level = anonymity_level
        self.subsampling = None
        self.similarity = None
        self.dataset_description = dataset_description

    def define_data_struckture(self):
        test = 1
    
    def add_data_descriptor(self,data_descriptor):
        self.dataset_descriptions.append(data_descriptor)
        if hasattr(data_descriptor, 'simularity'):
            data_descriptor.simularity.data_desciiptor = data_descriptor
            self._add_simularatie(data_descriptor.simularity)

    def _add_simularatie(self,simularatie):
        self._simularatie_list.add_simularatie(simularatie)

    def _find_Simularaty(self):
        # TODO: make my
        raise NotImplementedError('NotImplemented')
    
    def _find_Metric_Leaning(self,data_pair,similarity_label):
        nonlm = NonlinearDeepMetric()
        nonlm.train(data_pair, similarity_label)
        return nonlm

    def _find_Model(self, data):
        # TODO: make my
        raise NotImplementedError('NotImplemented')
    
    def _subsample(self,presenitizedData, sub_sampling_size=0.1, seed=None):
        self.subsampling = Subsampling(presenitizedData.sample(frac=sub_sampling_size))
        subsample_size_max = int(comb(len(self.subsampling.data), 2))
        print('total number of pairs is %s' % subsample_size_max)
        data_pair_all, data_pair_all_index = self.subsampling.uniform_sampling(subsample_size=subsample_size_max, seed=seed)
        self.similarity = Similarity(data=data_pair_all)
        self.similarity.extract_interested_attribute(self._simularatie_list.simularaties)
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
            k_ward = K_ward(data, distance_metric_type=distance_metric_type, rep_mode = rep_mode,k=anonymity_level, initProfile=self._simularatie_list)
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

    def _crate_data_description(self,data_desciptors):
        raise NotImplementedError('NotImplemented')
        

    def run(self):
        presenitized_data = self._presanitized()
        loss_presenitized=  self._simularatie_list.get_statistics_loss(presenitized_data,self.data)
        print("information loss with presenitized %s" % loss_presenitized)
        print("amount of samples presenitized_data  %s" % len(presenitized_data))
        # similarity_label, data_subsample = self._subsample(presenitized_data)
        # model = self._find_Metric_Leaning(data_subsample,similarity_label)
        # final_samitized_data = self._sanitize_data(data = self.data, distance_metric_type="metric", rep_mode = "mean",
        #                 anonymity_level=self.anonymity_level,metric=model)
        # loss_metric=  self.simularatie_list.get_statistics_loss(final_samitized_data,self.data)
        # print("information loss with nonlm metric %s" % loss_metric)
        # lm = Linear_Metric()
        # lm.train(data_subsample, similarity_label)
        # final_samitized_data = self._sanitize_data(data = self.data, distance_metric_type="metric", rep_mode = "mean",
        #                 anonymity_level=self.anonymity_level,metric=lm)
        # loss_metric=  self.simularatie_list.get_statistics_loss(final_samitized_data,self.data)
        # print("information loss with Linear_Metric metric %s" % loss_metric)
        transformed_data = OutputGroupper(self.dataset_descriptions).transform_data(presenitized_data)
        return transformed_data