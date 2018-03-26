from framework.similarity.simularatielist import SimularatieList
from framework.similarity.similarity import Similarity
from framework.models.kward import K_ward
from framework.metric_learning.linearmetric import Linear_Metric
from framework.metric_learning.nonlineardeepmetric import NonlinearDeepMetric
from scipy.misc import comb
from framework.utilities.outputgroupper import OutputGroupper
from framework.utilities.datadescriptor import DataDescriptorMetadata, DataDescriptorBase, DataDescriptorTimeSerice
from framework.similarity.basesimularity import BaseSimularity
from framework.utilities.resampler import Resampler, Subsampling
import math
import copy
from itertools import chain
import pandas as pd

import pandas as pd
class Framwork:
    def __init__(self, data, anonymity_level=5,dataset_description=None):
        self.data = data
        self._simularatie_list = SimularatieList()
        self.data_descriptors = []        
        self.anonymity_level = anonymity_level
        self.subsampling = None
        self.similarity = None
        self.dataset_description = dataset_description
        self.data_has_been_resample_data_into_blocks_of_output_rate = False
        self.amount_of_sensors =len(data.index)
        self._resampler = Resampler()

    def _get_data_for_sanitize(self):
        output_data = pd.DataFrame()
        for dd in self.data_descriptors:
            if not isinstance(dd, DataDescriptorMetadata):
                output_data = pd.concat([output_data,self.data.iloc[:,dd.data_start_index:dd.data_end_index+1]], axis=1)
        return output_data
    
    def _add_metadata_for_sanitize_data(self,sanitize_data):
        output_data = pd.DataFrame()
        for dd in self.data_descriptors:
            if isinstance(dd, DataDescriptorMetadata):
                output_data = pd.concat([output_data,self.data.iloc[:,dd.data_start_index:dd.data_end_index+1]], axis=1)
        output_data = pd.concat([output_data,sanitize_data], axis=1)
        return output_data

    def _verify_configuration_of_framework(self):
        amount_of_colums = len(self.data.columns) - 1
        if self._simularatie_list.get_amount_of_simularaties() < 1:
            raise ValueError("No data to process, use add_simularatie to add processing on the data")
        for data_descriptor in self.data_descriptors:
            data_descriptor.verify_configuration_data_descriptor_config(amount_of_colums, self.data)
    
    def _can_insture_k_anonymity(self):
        if (2*self.anonymity_level-1)*5<self.amount_of_sensors:
            return True
        return False
        
    def add_meta_data(self,data_descriptor):
        if isinstance(data_descriptor, DataDescriptorMetadata):
            self.data_descriptors.append(data_descriptor)

    def add_simularatie(self,simularatie):
        # if isinstance(simularatie, BaseSimularity):
        self._simularatie_list.add_simularatie(simularatie)
            # if isinstance(simularatie.data_descriptor, DataDescriptorBase):
        self.data_descriptors.append(simularatie.data_descriptor)

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
        max_clusters = 8
        if len(self.similarity.data_interested) < max_clusters:
            max_clusters = len(self.similarity.data_interested)
        similarity_label, data_subsample = self.similarity.label_via_silhouette_analysis(range_n_clusters=range(2,max_clusters))
        # similarity_label_all_series = pd.Series(similarity_label)
        # similarity_label_all_series.index = data_pair_all_index
        print('similarity balance is %s'% [sum(similarity_label),len(similarity_label)])
        return similarity_label, data_pair_all

    def _presanitized(self):
        sanitized_df = self._sanitize_data(data=self._get_data_for_sanitize(), distance_metric_type="init", rep_mode = "mean",
                        anonymity_level=self.anonymity_level)
        return sanitized_df

    def _sanitize_data(self,distance_metric_type, rep_mode, anonymity_level, data, **kwargs):
        k_ward = None
        if distance_metric_type == "init":
            k_ward = K_ward(data, rep_mode = rep_mode,k=anonymity_level, metric=self._simularatie_list)
        else:
            metric = kwargs["metric"]
            k_ward = K_ward(data, rep_mode = rep_mode,k=anonymity_level, metric=metric)
        k_ward.find_clusters()
        groups = k_ward.get_groups()

        sanitized_df = pd.DataFrame()
        for group in groups:
            sanitized_value = group.rep.to_frame().transpose()
            keys = group.get_member_ids()
            for key in keys:
                sanitized_value.index = [key]
                sanitized_df = sanitized_df.append(sanitized_value)
        sanitized_df.columns = data.columns
        return sanitized_df


    def generated_data_description(self):
        dd_string_out = []
        if self.dataset_description is not None:
            dd_string_out.append(self.dataset_description)
        for data_descriptor in self.data_descriptors:
            dd_string_out.append(data_descriptor.get_str_description())
        return '\n'.join(dd_string_out)

    def run(self):
        self._verify_configuration_of_framework()
        can_insture_k_anonymity = self._can_insture_k_anonymity()
        if not can_insture_k_anonymity:
            self.data, self._simularatie_list, self.data_descriptors = self._resampler.resample_data_into_blocks_of_output_rate(self.data, self.data_descriptors, self._simularatie_list)
            print("amount of samples after spilt %s" % len(self.data.index))
            print("amount of columns after spilt %s" % len(self.data.columns))
        presenitized_data = self._presanitized()
        loss_presenitized=  self._simularatie_list.get_statistics_loss(self._get_data_for_sanitize(),presenitized_data)
        print("information loss with presenitized %s" % loss_presenitized)
        print("amount of samples presenitized_data  %s" % len(presenitized_data))
        if len(presenitized_data) > 20:
            similarity_label, data_subsample = self._subsample(presenitized_data)
        else:
            similarity_label, data_subsample = self._subsample(presenitized_data, sub_sampling_size=0.20)
        model = self._find_Metric_Leaning(data_subsample,similarity_label)
        final_sanitized_data = self._sanitize_data(data = self._get_data_for_sanitize(), distance_metric_type="metric", rep_mode = "mean",
                        anonymity_level=self.anonymity_level,metric=model)
        loss_metric=  self._simularatie_list.get_statistics_loss(self._get_data_for_sanitize(),final_sanitized_data)
        print("information loss with nonlm metric %s" % loss_metric)
        # lm = Linear_Metric()
        # lm.train(data_subsample, similarity_label)
        # final_sanitized_data = self._sanitize_data(data = self._get_data_for_sanitize(), distance_metric_type="metric", rep_mode = "mean",
        #                 anonymity_level=self.anonymity_level,metric=lm)
        # loss_metric=  self._simularatie_list.get_statistics_loss(final_sanitized_data,self.data)
        # print("information loss with Linear_Metric metric %s" % loss_metric)
        if not can_insture_k_anonymity:
            transformed_data, self._simularatie_list, self.data_descriptors = self._resampler.create_timeserices_from_slices_of_data(final_sanitized_data, self._simularatie_list, self.amount_of_sensors)
            transformed_data, self.data_descriptors = OutputGroupper(self.data_descriptors).transform_data(transformed_data)
        else:
            transformed_data, self.data_descriptors = OutputGroupper(self.data_descriptors).transform_data(self._add_metadata_for_sanitize_data(final_sanitized_data))
        return transformed_data, loss_metric