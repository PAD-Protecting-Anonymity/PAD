from framework.similarity.similaritylist import SimilarityList
from framework.similarity.similarity import Similarity
from framework.models.kward import K_ward
from .metric_learning.linearmetric import *
from .metric_learning.nonlineardeepmetric import *
from scipy.misc import comb
from framework.utilities.outputgroupper import OutputGroupper
from framework.utilities.datadescriptor import DataDescriptorMetadata, DataDescriptorBase, DataDescriptorTimeSeries
from framework.similarity.basesimilarity import BaseSimilarity
from framework.utilities.resampler import Resampler, Subsampling
from framework.utilities.configverify import Verifyerror
from framework.utilities.kanonymityutilities import KAnonymityUtilities
import math
import copy
from itertools import chain
import pandas as pd
import sys
import inspect
import random

class Framework:
    def __init__(self, data, anonymity_level=5,dataset_description=None, seed=None,rep_mode = "mean",resample_factor = 5, learning_metric=MetricLearningTerms.LINEAR, k_fold = None,output_groupper_after = True, all_data=None, all_sampling_rates = None):
        self.data = data
        self._similarity_list = SimilarityList()
        self.data_descriptors = []
        self.anonymity_level = anonymity_level
        self.subsampling = None
        self.similarity = None
        self.dataset_description = dataset_description
        self.data_has_been_resample_data_into_blocks_of_output_rate = False
        self.amount_of_sensors =None
        self.rep_mode  = rep_mode
        self._resampler = Resampler()
        self.max_clusters = 8
        self.resample_factor = resample_factor
        self.seed = seed
        self.learning_metric = learning_metric
        self.k_fold = k_fold
        self.output_groupper_after = output_groupper_after
        self.all_data = all_data
        self.all_sampling_rates = all_sampling_rates

    def _get_data_for_sanitize(self):
        output_data = pd.DataFrame()
        for dd in self.data_descriptors:
            if not isinstance(dd, DataDescriptorMetadata):
                output_data = pd.concat([output_data,self.data.iloc[:,dd.data_start_index:dd.data_end_index+1]], axis=1)
        return output_data

    def _get_data_for_model_selecting(self, percentage_of_data=0.1, amount_of_inputs=None):
        output_data = self._get_data_for_sanitize()
        if amount_of_inputs is None:
            output_data = output_data.sample(math.floor(len(output_data.index)*percentage_of_data))
        else:
            output_data = output_data.sample(amount_of_inputs)
        return output_data

    def _get_data_subsampling_k_fold(self):
        output_data = self._get_data_for_sanitize()
        amount_of_inputs = len(output_data.index)
        listOfIndex = [i for i in range(amount_of_inputs)]
        random.seed(123456)
        random.shuffle(listOfIndex)
        amount_of_samples_in_fold = math.floor(amount_of_inputs/self.k_fold[1])
        fold_elements = listOfIndex[:amount_of_samples_in_fold*self.k_fold[0]]
        fold_elements.extend(listOfIndex[amount_of_samples_in_fold*(self.k_fold[0]+1):])
        output_data = output_data.iloc[fold_elements]
        print("Amount of samples in traning set: ",len(output_data.index))
        return output_data

    def _add_metadata_for_sanitize_data(self,sanitize_data):

        output_data = pd.DataFrame()
        for dd in self.data_descriptors:
            if isinstance(dd, DataDescriptorMetadata):
                output_data = pd.concat([output_data,self.data.iloc[:,dd.data_start_index:dd.data_end_index+1]], axis=1)
        output_data = pd.concat([output_data,sanitize_data], axis=1)
        return output_data

    def add_meta_data(self,data_descriptor):
        if isinstance(data_descriptor, DataDescriptorMetadata):
            self.data_descriptors.append(data_descriptor)

    def add_similarity(self,similarity):
        self._similarity_list.add_similarity(similarity)
        self.data_descriptors.append(similarity.data_descriptor)

    def _find_all_Metric_Leanings(self):
        metrics = []
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if "framework.metric_learning." + self.learning_metric in str(obj):
                    if obj is Linear_Metric:
                        metrics.append(obj)
                    elif issubclass(obj, BasemetricLearning) and obj is not BasemetricLearning:
                        metrics.append(obj)
        return metrics

    def _find_Metric_Leaning(self,data_pair,similarity_label):
        metricNames = self._find_all_Metric_Leanings()
        metricesResults = []
        metrics = []
        if len(metricNames) == 1:
            metric = metricNames[0]()
            metric.train(data_pair, similarity_label)
            return metric
        subsample = self._get_data_for_model_selecting(percentage_of_data=0.6)

        print("Amount of samples in subsample: ", len(subsample.index))
        for metric in metricNames:
            metric = metric()
            metrics.append(metric)
            test_loss = metric.train(data_pair, similarity_label)
            final_sanitized_data = self._sanitize_data(data = subsample, distance_metric_type="metric",
                        anonymity_level=self.anonymity_level,metric=metric, rep_mode = self.rep_mode)
            loss_metric=  self._similarity_list.get_statistics_loss(subsample,final_sanitized_data)
            metricesResults.append(loss_metric)
            print("loss with subsample: " ,loss_metric, " For model: " , metric.metric_learning_terms)
        best_model_index =  metricesResults.index(min(metricesResults))
        return metrics[best_model_index]

    def _subsample(self,presensitizedData, sub_sampling_size=0.1, subsampleTrial = 0, seed=None):
        if subsampleTrial > 5:
            raise ValueError("The data is to similar to be sanitized")
        if self.k_fold is None:
            self.subsampling = Subsampling(presensitizedData.sample(frac=sub_sampling_size, random_state=seed))
        else:
            self.subsampling = Subsampling(self._get_data_subsampling_k_fold())
        subsample_size_max = int(comb(len(self.subsampling.data), 2))
        print('total number of pairs is %s' % subsample_size_max)
        data_pair_all, data_pair_all_index = self.subsampling.uniform_sampling(subsample_size=subsample_size_max, seed=seed)
        self.similarity = Similarity(data=data_pair_all)
        self.similarity.extract_interested_attribute(self._similarity_list.similarities)
        if len(self.similarity.data_interested) < self.max_clusters:
            self.max_clusters = len(self.similarity.data_interested)
        similarity_label, data_subsample = self.similarity.label_via_silhouette_analysis(range_n_clusters=range(2,self.max_clusters), seed=self.seed)
        if similarity_label == []:
            return self._subsample(presensitizedData, sub_sampling_size=sub_sampling_size+0.1, subsampleTrial = subsampleTrial+1, seed=seed)
        else:
            print('similarity balance is %s'% [sum(similarity_label),len(similarity_label)])
        return similarity_label, data_pair_all

    def _presanitized(self):
        sanitized_df = self._sanitize_data(data=self._get_data_for_sanitize(), distance_metric_type="init", rep_mode = self.rep_mode,
                        anonymity_level=self.anonymity_level)
        loss_presensitized=  self._similarity_list.get_statistics_loss(self._get_data_for_sanitize(),sanitized_df)
        print("information loss with presensitized %s" % loss_presensitized)
        print("amount of samples presensitized_data  %s" % len(sanitized_df))
        return sanitized_df , loss_presensitized

    def _sanitize_data(self,distance_metric_type , anonymity_level, data, rep_mode, **kwargs):
        k_ward = None
        if distance_metric_type == "init":
            k_ward = K_ward(data, rep_mode = rep_mode ,k=anonymity_level, metric=self._similarity_list)
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

    def anonymize(self):
        self.data = Verifyerror().verify(self.data, self._similarity_list, self.data_descriptors)
        self.amount_of_sensors = len(self.data.index)

        _can_ensure_k_anonymity = KAnonymityUtilities().can_ensure_k_anonymity(self.anonymity_level, self.amount_of_sensors)
        if self.all_data is not None and self.all_sampling_rates is not None:
            self.anonymity_level = KAnonymityUtilities().find_balance_for_k(self.anonymity_level, self.all_data,self.all_sampling_rates)
        elif not _can_ensure_k_anonymity:
            self.all_data = []
            self.all_data.append(self.data)
            self.all_sampling_rates = []
            for dd in self.data_descriptors:
                if isinstance(dd, DataDescriptorTimeSeries):
                    self.all_sampling_rates.append(dd.sampling_frequency.value)
            self.anonymity_level = KAnonymityUtilities().find_balance_for_k(self.anonymity_level, self.all_data,self.all_sampling_rates)
        _can_ensure_k_anonymity = False
        if not _can_ensure_k_anonymity:
            self.data, self._similarity_list, self.data_descriptors, resample_factor = self._resampler.resample_data_into_blocks(self.data, self.data_descriptors, self._similarity_list, self.resample_factor)
            Verifyerror().verify_after_can_not_ensure_k_anonymity(self.data, self._similarity_list)
            print('anonymity_level set to: ' + str(self.anonymity_level))
            print("amount of samples after spilt %s" % len(self.data.index))
            print("amount of columns after spilt %s" % len(self.data.columns))

        if not self.output_groupper_after:
            self.data, self.data_descriptors = OutputGroupper(self.data_descriptors).transform_data(self.data)
            print("amount of samples after output_groupper %s" % len(self.data.index))
            print("amount of columns after output_groupper %s" % len(self.data.columns))

        presensitized_data ,loss_presensitized = self._presanitized()

        similarity_label, data_pair = self._subsample(presensitized_data,seed=self.seed)

        model = self._find_Metric_Leaning(data_pair,similarity_label)
        print("Using model: ", model.metric_learning_terms)
        final_sanitized_data = self._sanitize_data(data = self._get_data_for_sanitize(), distance_metric_type="metric",anonymity_level=self.anonymity_level,metric=model, rep_mode = self.rep_mode)
        loss_metric=  self._similarity_list.get_statistics_loss(self._get_data_for_sanitize(),final_sanitized_data)
        print("information loss with", model.metric_learning_terms, "metric %s" % loss_metric)

        if not _can_ensure_k_anonymity:
            transformed_data, self._similarity_list, self.data_descriptors = self._resampler.create_timeserices_from_slices_of_data(final_sanitized_data, self._similarity_list, self.amount_of_sensors)
            if self.output_groupper_after:
                transformed_data, self.data_descriptors = OutputGroupper(self.data_descriptors).transform_data(transformed_data)
            print("information loss", loss_metric)
        else:
            transformed_data, self.data_descriptors = OutputGroupper(self.data_descriptors).transform_data(self._add_metadata_for_sanitize_data(final_sanitized_data))

        return transformed_data.sort_index(), loss_metric, self.anonymity_level