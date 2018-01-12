import itertools
import cvxpy as cvx
import numpy as np
import pdb
from cvxpy import *
from helper import Miscellaneous
import numpy as np


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









