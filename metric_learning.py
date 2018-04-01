import itertools
import cvxpy as cvx
import numpy as np
import pdb
from cvxpy import *
from helper import Miscellaneous
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
        return pair_subsample#, pair_subsample_index

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
            self.k_already = k_init
        else:
            unsample_idx = [i for i in range(len(self.subsample_status)) if self.subsample_status[i] == 0]
            uncertainty_vec = []
            for i in unsample_idx:
                pair_index = self.pair_index_all[i]
                pair = self.get_pairdata(pair_subsample_index=[pair_index])
                uncertainty = self.get_uncertainty_entropy(pair=pair[0], dist_metric=dist_metric, mu=0)
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
        self.last_returned_pairdata = pairdata
        return pairdata, pairdata_idx

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

class MetricLearning:
    def learn_with_similarity_label(self, data, label, mode, lam):
        """
        Implement the metric learning algorithm in "Distance metric learning, with application to clustering with
        side-information" by Eric P. Xing, et al. The alg learns distance metrics from similarity labels of data pairs
        """
        n_feature = data[0][0].shape[0]
        index_s = [i for i, x in enumerate(label) if x == 1]
        index_ns = [i for i, x in enumerate(label) if x == 0]
        X_s = [list(data[i][0] - data[i][1]) for i in index_s]
        X_ns = [list(data[i][0] - data[i][1]) for i in index_ns]

        # mode tells if the learned Mahalanobis distance metrics are specified by a diagonal matrix or a
        # fully-parametrized matrix
        if mode == "diag":
            x = cvx.Variable(rows=n_feature, cols=1)
            obj = 0
            for i in range(len(X_s)):
                obj = obj + sum_entries(mul_elemwise(np.square(X_s[i]), x))
            obj_neg = 0
            for i in range(len(X_ns)):
                obj_neg = obj_neg + sqrt(sum_entries(mul_elemwise(np.square(X_ns[i]), x)))
            obj = obj - cvx.log(obj_neg)
            obj = obj + lam * norm(x, 1)
            constraints = [x >= 0]
            obj_cvx = cvx.Minimize(obj)
            prob = cvx.Problem(obj_cvx, constraints)
            prob.solve(solver=SCS)
            x_mat = np.diag(x.value.transpose().tolist()[0])
            if prob.status == 'optimal' or prob.status == 'optimal_inaccurate':
                return prob.status, prob.value, x_mat
            else:
                return prob.status, np.nan, np.nan

        if mode == "full":
            lam = 1
            A = cvx.Semidef(n_feature)
            obj = 0
            for i in range(len(X_s)):
                obj = obj + cvx.quad_form(X_s[i], A)
            obj += lam * norm(A, 1)
            const = 0
            for i in range(len(X_ns)):
                const = const + cvx.sqrt(cvx.quad_form(X_ns[i], A))
            constraints = [const >= 1]
            obj_cvx = cvx.Minimize(obj)
            prob = cvx.Problem(obj_cvx, constraints)
            prob.solve(solver=MOSEK)
            if prob.status == 'optimal' or prob.status == 'optimal_inaccurate':
                return prob.status, prob.value, A.value
            else:
                return prob.status, np.nan, np.nan

    def learn_with_simialrity_label_regularization(self,data,label,lam_vec,train_portion):
        """
        Implementation of the metric learning algorithm presented in our paper with l-1 regularization
        """
        index_s = [i for i, x in enumerate(label) if x == 1]
        index_ns = [i for i, x in enumerate(label) if x == 0]
        X_s = np.array([list(data[i][0] - data[i][1]) for i in index_s])
        X_ns = np.array([list(data[i][0] - data[i][1]) for i in index_ns])
        n_feature = data[0][0].shape[0]

        misc = Miscellaneous()
        s_size = len(X_s)
        ns_size = len(X_ns)

        if s_size == 0 or ns_size == 0:
            return None

        X_s_train = X_s[:int(s_size * train_portion),:]
        X_s_test = X_s[int(s_size * train_portion):,:]
        X_ns_train = X_ns[:int(ns_size * train_portion),:]
        X_ns_test = X_ns[int(ns_size * train_portion):,:]

        ratio = []
        dist_metric = []
        while 1:
            for lam in lam_vec:
                # train
                A = cvx.Semidef(n_feature)
                obj = 0
                for i in range(int(s_size * train_portion)):
                    obj = obj + cvx.quad_form(X_s_train[i,:], A)
                obj = obj + lam * norm(A, 1)
                const = 0
                for i in range(int(ns_size * train_portion)):
                    const = const + cvx.sqrt(cvx.quad_form(X_ns_train[i,:], A))
                constraints = [const >= 1]
                obj_cvx = cvx.Minimize(obj)
                prob = cvx.Problem(obj_cvx, constraints)
                prob.solve(solver=SCS)
                dist_metric.append(A.value)

                # test
                if len(lam_vec) == 1:
                    return A.value
                dist_metric_psd = misc.PSDize(A.value)
                dist_s = np.asarray([X_s_test[i,:].dot(dist_metric_psd).dot(X_s_test[i,:]) for i in range(X_s_test.shape[0])])
                dist_s[dist_s < 0] = 0
                dist_ns = np.asarray([X_ns_test[i,:].dot(dist_metric_psd).dot(X_ns_test[i,:]) for i in range(X_ns_test.shape[0])])
                dist_ns[dist_ns < 0] = 0
                if sum(dist_s) + sum(dist_ns) == 0:
                    ratio_current = np.nan
                else:
                    ratio_current = sum(dist_s) /sum(dist_ns)
                ratio.append(ratio_current)
            if all(np.isnan(ratio)):
                best_idx = [0]
                lam_vec = [lam_vec[-1]/10]
            else:
                ratio_drop_nan = [ratio[i] for i in range(len(ratio)) if ~np.isnan(ratio[i])]
                best_idx = np.where(ratio == min(ratio_drop_nan))[0]
                print('best lambda is %s'%lam_vec[best_idx[0]])
                print(ratio)
                break
        return dist_metric[best_idx[0]]









