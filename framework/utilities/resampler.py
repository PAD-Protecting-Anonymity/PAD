import copy
import math
import pandas as pd
from framework.utilities.datadescriptor import DataDescriptorMetadata, DataDescriptorBase, DataDescriptorTimeSeries
from itertools import chain
from framework.similarity.similaritylist import SimilarityList
from framework.utilities.outputgroupper import DataDescriptorTerms
import itertools
import pdb
import numpy as np
import pickle


class Resampler:
    def __init__(self):
        self.meta_data = []
        self.row_meta_data = pd.DataFrame()

    def resample_data_into_blocks(self,data,data_descriptors,similarities_list, min_resample_factor):
        output_data = pd.DataFrame()

        index = 0
        for data_descriptor in data_descriptors:
            if isinstance(data_descriptor, DataDescriptorMetadata):
                self.meta_data.append(data_descriptor)
                meta_data_rows = data.iloc[:, data_descriptor.data_start_index:data_descriptor.data_end_index+1]
                temp_index = (data_descriptor.data_end_index-data_descriptor.data_start_index) + index
                data_descriptor.data_start_index = index
                data_descriptor.data_end_index = temp_index
                index += temp_index + 1
                meta_data_rows.columns = [range(data_descriptor.data_start_index,data_descriptor.data_end_index+1)]
                self.row_meta_data = pd.concat([self.row_meta_data,meta_data_rows], axis=1)
        #Clear framework settings
        _data_descriptors = []
        _similarities_list = SimilarityList()
        index = 0

        for similarity in similarities_list.similarities:
            data_descriptor = similarity.data_descriptor
            if isinstance(data_descriptor, DataDescriptorTimeSeries):
                data_length = (data_descriptor.data_end_index - data_descriptor.data_start_index)+1
                resample_factor = math.floor(similarity.data_descriptor.output_frequency.value / data_descriptor.sampling_frequency.value)
                if resample_factor < min_resample_factor:
                    resample_factor = min_resample_factor
                amount_of_resamples = math.floor(data_length / resample_factor)
                start_index = data_descriptor.data_start_index
                data_descriptor.data_start_index = index
                index = index + resample_factor-1
                data_descriptor.data_end_index = index
                _similarities_list.add_similarity(similarity)
                _data_descriptors.append(data_descriptor)
                for i in range(0,amount_of_resamples):
                    loop_start_index = start_index+(i*resample_factor)
                    loop_end_index = start_index+((i+1)*resample_factor)
                    new_data = data.iloc[:, loop_start_index:loop_end_index]
                    new_data.columns = list(range(data_descriptor.data_start_index,data_descriptor.data_end_index+1))
                    row_data = pd.concat([new_data], axis=1)
                    output_data = output_data.append(row_data, ignore_index=True)
        return output_data, _similarities_list, _data_descriptors, resample_factor

    def create_timeserices_from_slices_of_data(self, transformed_data,similarities_list, amount_of_sensors):
        output_data = pd.DataFrame()
        transformed_data = transformed_data.sort_index()
        _similarities_list = SimilarityList()
        _data_descriptors = []
        index = 0
        for meta_dd in self.meta_data:
            _data_descriptors.append(meta_dd)
            index = meta_dd.data_end_index + 1

        amount_of_inputs = len(transformed_data.index)

        time_series_lenged = math.floor(amount_of_inputs / amount_of_sensors)*len(transformed_data.columns)

        for similarity in similarities_list.similarities:
            similarity.data_descriptor.data_start_index = index
            similarity.data_descriptor.data_end_index = index + time_series_lenged -1
            _similarities_list.add_similarity(similarity)
            _data_descriptors.append(similarity.data_descriptor)

        for i in range(0,amount_of_sensors):
            # import pdb; pdb.set_trace()
            new_data = transformed_data.iloc[i::amount_of_sensors,:]._values
            new_data = list(chain.from_iterable(new_data))
            new_data = pd.DataFrame.from_items([(i, new_data)],orient='index', columns=list(range(index,index+len(new_data))))
            row_data = pd.concat([self.row_meta_data,new_data], axis=1)
            output_data = output_data.append(row_data.iloc[i])
        return output_data, _similarities_list, _data_descriptors

class Subsampling:
    def __init__(self,data):
        self.data = data # sanitized database to be sampled from
        self.pair_index_all = list(itertools.combinations(self.data.index,2))
        self.pairsize_total = len(self.pair_index_all)
        self.size_total = len(self.data)
        self.pairdata_labeled = []
        self.label = []
        self.k_already = 0
        self.subsample_status = np.zeros(self.pairsize_total)
        # self.dist_pairdata_us = []
        # self.dist_pairdata_s = []

    def reset(self):
        self.pairdata_labeled = []
        self.label = []
        self.k_already = 0
        self.subsample_status = np.zeros(self.pairsize_total)
        # self.dist_pairdata_us = []
        # self.dist_pairdata_s = []

    def get_pairdata(self,pair_subsample_index):
        pairdata = [(self.data.loc[ind1], self.data.loc[ind2]) for ind1, ind2 in pair_subsample_index]
        return pairdata

    def uniform_sampling(self,subsample_size,seed = None):
        if seed is not None:
            np.random.seed(seed)
        pair_subsample_i = np.random.choice(a=self.pairsize_total,size=subsample_size,replace=False)
        pair_subsample_index = [self.pair_index_all[i] for i in pair_subsample_i]
        pair_subsample_index.sort(key=lambda x: x[1])
        pair_subsample_index.sort(key=lambda x: x[0])
        pair_subsample = self.get_pairdata(pair_subsample_index=pair_subsample_index)
        return pair_subsample, pair_subsample_index

    def batch_uniform_sampled(self,batch_size):
        unsample_idx = [i for i in range(len(self.subsample_status)) if self.subsample_status[i] == 0]
        pair_subsample_i = np.random.choice(a=len(unsample_idx), size=batch_size, replace=False)
        next_sample_idx = [unsample_idx[i] for i in pair_subsample_i]
        pair_subsample_index = [self.pair_index_all[j] for j in next_sample_idx]
        pair_subsample = self.get_pairdata(pair_subsample_index=pair_subsample_index)
        for i in pair_subsample_i:
            self.subsample_status[unsample_idx[i]] = 1
        return pair_subsample,next_sample_idx

    def get_range_n_clusters(self,datasize,max_allowed):
        max_n_clusters = int(np.floor(datasize / 2))
        if max_n_clusters <= 2:
            range_n_clusters = [2]
        elif max_n_clusters >= max_allowed:
            range_n_clusters = range(2, max_allowed)
        else:
            range_n_clusters = range(2, max_n_clusters)
        return range_n_clusters

    def active_sampling(self,dist_metric,k_init,batch_size,seed=None):
        # at the beginning, uniformly randomly sample k_init samples for manual labeling
        if self.k_already == 0:
            np.random.seed(seed)
            next_sample_idx = np.random.choice(a=self.pairsize_total, size=k_init, replace=False).tolist()
            pairdata_idx = [self.pair_index_all[i] for i in next_sample_idx]
            pairdata_idx.sort(key=lambda x: x[1])
            pairdata_idx.sort(key=lambda x: x[0])
            self.k_already = k_init
        else:
            unsample_idx = [i for i in range(len(self.subsample_status)) if self.subsample_status[i] == 0]
            uncertainty_vec = []
            for i in unsample_idx:
                pair_index = self.pair_index_all[i]
                pair = self.get_pairdata(pair_subsample_index=[pair_index])
                uncertainty = self.get_uncertainty_entropy_deep(pair=pair[0], dist_metric=dist_metric, mu=0)
                # uncertainty = self.get_uncertainty_probability_difference(pair=pair[0], dist_metric=dist_metric, mu=0)
                uncertainty_vec.append(uncertainty)
            # pdb.set_trace()
            if len(unsample_idx) < batch_size:
                batch_size = len(unsample_idx)
            start = 0
            most_uncertain_idx = np.argsort(uncertainty_vec)[::-1][start:start + batch_size]
            next_sample_idx = [unsample_idx[i] for i in most_uncertain_idx]
            pairdata_idx = [self.pair_index_all[i] for i in next_sample_idx]
            self.k_already = self.k_already + batch_size

        for i in next_sample_idx:
            self.subsample_status[i] = 1
        pairdata = self.get_pairdata(pair_subsample_index=pairdata_idx)
        return pairdata, pairdata_idx

    def get_uncertainty_entropy_deep(self, pair, dist_metric, mu):
        dist = dist_metric.transform(pair)
        # pdb.set_trace()
        if not np.isscalar(dist):
            dist = dist[0, 0]
        prob_s = 1 / (1 + np.exp(dist - mu))
        prob_us = 1 / (1 + np.exp(-(dist - mu)))
        entropy = -prob_s * np.log(prob_s) - prob_us * np.log(prob_us)
        return entropy

    def get_uncertainty_entropy(self,pair,dist_metric,mu):
        x_diff = (pair[0] - pair[1]).values
        dist = x_diff.dot(dist_metric).dot(x_diff)
        # pdb.set_trace()]
        if not np.isscalar(dist):
            dist = dist[0,0]
        prob_s = 1/(1+np.exp(dist-mu))
        prob_us = 1/(1+np.exp(-(dist-mu)))
        entropy = -prob_s*np.log(prob_s) - prob_us*np.log(prob_us)
        return entropy

    def get_uncertainty_probability_difference(self,pair,dist_metric,mu):
        x_diff = (pair[0] - pair[1]).values
        dist = x_diff.dot(dist_metric).dot(x_diff)
        # pdb.set_trace()
        if not np.isscalar(dist):
            dist = dist[0,0]
        prob_s = 1/(1+np.exp(dist-mu))
        prob_us = 1/(1+np.exp(-(dist-mu)))
        uncertainty = np.abs(prob_s-prob_us)
        return uncertainty