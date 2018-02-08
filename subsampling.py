import itertools
import numpy as np
import pdb
import numpy as np

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
        self.dist_pairdata_us = []
        self.dist_pairdata_s = []

    def reset(self):
        self.pairdata_labeled = []
        self.label = []
        self.k_already = 0
        self.subsample_status = np.zeros(self.pairsize_total)
        self.dist_pairdata_us = []
        self.dist_pairdata_s = []


    def get_pairdata(self,pair_subsample_index):
        pairdata = [(self.data.loc[ind1], self.data.loc[ind2]) for ind1, ind2 in pair_subsample_index]
        return pairdata

    def uniform_sampling(self,subsample_size,seed):
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

    def active_sampling(self,dist_metric,k_init,batch_size,seed):
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
        # pdb.set_trace()]
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