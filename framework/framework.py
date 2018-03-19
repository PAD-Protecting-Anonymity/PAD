from framework.utilities.subsampling import Subsampling
from framework.similarity.simularatielist import SimularatieList
from framework.similarity.similarity import Similarity
from framework.models.kward import K_ward
from framework.metric_learning.linearmetric import Linear_Metric
from framework.metric_learning.nonlineardeepmetric import NonlinearDeepMetric
from scipy.misc import comb
from framework.utilities.outputgroupper import OutputGroupper
from framework.utilities.datadescriptor import DataDescriptorMetadata, DataDescriptorBase, DataDescriptorTimeSerice
from framework.similarity.basesimularity import BaseSimularity
import math
import copy
from itertools import chain

import pandas as pd
class Framwork:
    def __init__(self, data, anonymity_level=5,dataset_description=""):
        self.data = data
        self._simularatie_list = SimularatieList()
        self.data_descriptors = []        
        self.anonymity_level = anonymity_level
        self.subsampling = None
        self.similarity = None
        self.dataset_description = dataset_description
        self.data_has_been_resample_data_into_blocks_of_output_rate = False
        self.amount_of_sensors =len(data.index)
        
    def _verify_configuration_of_framework(self):
        amount_of_colums = len(self.data.columns) - 1
        if self._simularatie_list.get_amount_of_simularaties() < 1:
            raise ValueError("No data to process, use add_simularatie to add processing on the data")
        for data_descriptor in self.data_descriptors:
            #Test if the data decriptors are refering to data witch is not in the datastrame
            if data_descriptor.data_start_index > amount_of_colums:
                raise ValueError("refer to index wich is not in datastrame: Index %s goes out of bound for the data stream in data descriptor in data_start_index" % data_descriptor.data_start_index)
            elif data_descriptor.data_end_index > amount_of_colums:
                raise ValueError("refer to index wich is not in datastrame: Index %s goes out of bound for the data stream in data descriptor in data_end_index" % data_descriptor.data_end_index)
            if data_descriptor.data_start_index > data_descriptor.data_end_index:
                temp_holder = data_descriptor.data_start_index
                data_descriptor.data_start_index = data_descriptor.data_end_index
                data_descriptor.data_end_index = temp_holder
            if isinstance(data_descriptor, DataDescriptorTimeSerice) and data_descriptor.sampling_frequency.value > data_descriptor.output_frequency.value:
                raise ValueError("For DataDescriptorTimeSerice the sampling rate most be >= to the output frequency")

    def _can_insture_k_anonymity(self):
        if (2*self.anonymity_level-1)*5<self.amount_of_sensors:
            return True
        return False

    def _resample_data_into_blocks_of_output_rate(self):
        self.data_has_been_resample_data_into_blocks_of_output_rate = True
        output_data = pd.DataFrame()
        self.meta_datas = []
        self.row_meta_data = pd.DataFrame()
        index = 0
        for data_descriptor in self.data_descriptors: 
            if isinstance(data_descriptor, DataDescriptorMetadata):
                self.meta_datas.append(data_descriptor)
                meta_data_rows = self.data.iloc[:, data_descriptor.data_start_index:data_descriptor.data_end_index+1]
                temp_index = (data_descriptor.data_end_index-data_descriptor.data_start_index) + index
                data_descriptor.data_start_index = index
                data_descriptor.data_end_index = temp_index
                index += temp_index + 1
                meta_data_rows.columns = [range(data_descriptor.data_start_index,data_descriptor.data_end_index+1)]
                self.row_meta_data = pd.concat([self.row_meta_data,meta_data_rows], axis=1)

        #Clear framework settings
        data_descriptors = self.data_descriptors.copy()        
        self.data_descriptors = []
        _simularatie_list = copy.deepcopy(self._simularatie_list)
        self._simularatie_list = SimularatieList()
        
        index = 0
        #add meta data to strame
        # for meta_data in meta_datas:
        #     temp_index = (meta_data.data_end_index-meta_data.data_start_index) + index
        #     meta_data.data_start_index = index
        #     meta_data.data_end_index = temp_index
        #     index += temp_index + 1
        #     self.add_meta_data(meta_data)


        for simularatie in _simularatie_list.simularaties:
            data_descriptor = simularatie.data_descriptor
            if isinstance(data_descriptor, DataDescriptorTimeSerice):
                data_length = (data_descriptor.data_end_index - data_descriptor.data_start_index)
                resample_factor = math.floor(simularatie.data_descriptor.output_frequency.value / data_descriptor.sampling_frequency.value)
                amount_of_resamples = math.floor(data_length / resample_factor)
                start_index = data_descriptor.data_start_index
                data_descriptor.data_start_index = index
                index = index + resample_factor-1
                data_descriptor.data_end_index = index
                self.add_simularatie(simularatie)
                for i in range(0,amount_of_resamples):
                    loop_start_index = start_index+(i*resample_factor)
                    loop_end_index = start_index+((i+1)*resample_factor)
                    new_data = self.data.iloc[:, loop_start_index:loop_end_index]
                    new_data.columns = list(range(data_descriptor.data_start_index,data_descriptor.data_end_index+1))
                    row_data = pd.concat([new_data], axis=1)
                    output_data = output_data.append(row_data, ignore_index=True)
        return output_data

    def _create_timeserices_from_slices_of_data(self, transformed_data):
        output_data = pd.DataFrame()
        simularaties = copy.deepcopy(self._simularatie_list)
        self._simularatie_list = SimularatieList()
        self.data_descriptors = []
        index = 0
        for meta_dd in self.meta_datas:
            self.add_meta_data(meta_dd)
            index = meta_dd.data_end_index + 1
        
        amount_of_inputs = len(transformed_data.index)

        time_Serices_lenged = math.floor(amount_of_inputs / self.amount_of_sensors)

        for simularatie in simularaties.simularaties:
            simularatie.data_descriptor.data_start_index = index
            simularatie.data_descriptor.data_end_index = index + time_Serices_lenged -1
            self.add_simularatie(simularatie)
        
        for i in range(0,self.amount_of_sensors):
            new_data = transformed_data.iloc[i::self.amount_of_sensors,:]._values
            new_data = list(chain.from_iterable(new_data))
            new_data = pd.DataFrame.from_items([(i, new_data)],orient='index', columns=list(range(index,index+len(new_data))))
            row_data = pd.concat([self.row_meta_data,new_data], axis=1)
            output_data = output_data.append(row_data.iloc[i])
        print(output_data)
        return output_data
        


        
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

    def run(self):
        self._verify_configuration_of_framework()
        can_insture_k_anonymity = self._can_insture_k_anonymity()
        if not can_insture_k_anonymity:
            self.data = self._resample_data_into_blocks_of_output_rate()
        presenitized_data = self._presanitized()
        # loss_presenitized=  self._simularatie_list.get_statistics_loss(presenitized_data,self.data)
        # print("information loss with presenitized %s" % loss_presenitized)
        # print("amount of samples presenitized_data  %s" % len(presenitized_data))
        # similarity_label, data_subsample = self._subsample(presenitized_data)
        # model = self._find_Metric_Leaning(data_subsample,similarity_label)
        # final_samitized_data = self._sanitize_data(data = self.data, distance_metric_type="metric", rep_mode = "mean",
        #                 anonymity_level=self.anonymity_level,metric=model)
        # loss_metric=  self._simularatie_list.get_statistics_loss(final_samitized_data,self.data)
        # print("information loss with nonlm metric %s" % loss_metric)
        # lm = Linear_Metric()
        # lm.train(data_subsample, similarity_label)
        # final_samitized_data = self._sanitize_data(data = self.data, distance_metric_type="metric", rep_mode = "mean",
        #                 anonymity_level=self.anonymity_level,metric=lm)
        # loss_metric=  self._simularatie_list.get_statistics_loss(final_samitized_data,self.data)
        # print("information loss with Linear_Metric metric %s" % loss_metric)
        transformed_data = OutputGroupper(self.data_descriptors).transform_data(presenitized_data)
        if self.data_has_been_resample_data_into_blocks_of_output_rate:
            return self._create_timeserices_from_slices_of_data(transformed_data)
        return transformed_data