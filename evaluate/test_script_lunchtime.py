import sys;
import os

sys.path.append(os.path.abspath("./"))
from helper import Utilities, PerformanceEvaluation
import pandas as pd
from user_feedback import Similarity
from scipy.misc import comb
from deep_metric_learning import Deep_Metric
import numpy as np
import pickle
from linear_metric_learning import Linear_Metric
from subsampling import Subsampling

"""
In the demo, we will showcase an example of special purpose publication.
The data user wants the published database to maximally retain the information about lunch time.
"""

# Initialization of some useful classes
util = Utilities()
pe = PerformanceEvaluation()


# mel = MetricLearning()

def evaluation_occupancy_window(n):
    anonymity_level = n
    day_profile_all = pd.read_pickle('../dataset/dataframe_all_binary.pkl')
    day_profile_all = day_profile_all.fillna(0)
    day_profile_all[day_profile_all>0] = 1
    # print(len(day_profile_all.index))
    res = 15

    day_profile = day_profile_all.iloc[90:,0::res]
    # day_profile = day_profile_all.iloc[:90,0::res] # subsample the database to improve the speed for demonstration purpose
    # day_profile2 = day_profile_all.iloc[90:,0::res] # subsample the database to improve the speed for demonstration purpose
    # s_size = len(day_profile2.index)
    # day_profile = day_profile2.iloc[int(s_size * 0.5):]

    # day_profile = day_profile_all.iloc[0::2,0::res]
    # day_profile2 = day_profile_all.iloc[1::2, 0::res]

    # step 2: data user specifies his/her interest. In the example, the data user is interested in preserving the
    # information of a segment of entire time series. In this case, he/she would also need to specify the starting and
    # ending time of the time series segment of interest.
    interest = 'segment'
    rep_mode = 'mean'
    window = [11, 15]  # window specifies the starting and ending time of the period that the data user is interested in

    # step 3: pre-sanitize the database
    sanitized_profile_best = util.sanitize_data(day_profile, distance_metric='self-defined',
                                                anonymity_level=anonymity_level, rep_mode=rep_mode,
                                                mode=interest, window=window)

    sanitized_profile_baseline = util.sanitize_data(day_profile, distance_metric='euclidean',
                                                    anonymity_level=anonymity_level, rep_mode=rep_mode)

    loss_best_metric = pe.get_information_loss(data_gt=day_profile, data_sanitized=sanitized_profile_best,
                                               window=window)

    loss_generic_metric = pe.get_information_loss(data_gt=day_profile,
                                                  data_sanitized=sanitized_profile_baseline.round(),
                                                  window=window)

    df_subsampled_from = sanitized_profile_baseline.sample(frac=1)
    subsample_size_max = int(comb(len(df_subsampled_from), 2))
    print('total number of pairs is %s' % len(df_subsampled_from))

    # step 4: sample a subset of pre-sanitized database and form the data points into pairs
    subsample_size = int(round(subsample_size_max))
    sp = Subsampling(data=df_subsampled_from)
    data_pair, data_pair_all_index = sp.uniform_sampling(subsample_size=subsample_size, seed=None)

    # User receives the data pairs and label the similarity
    sim = Similarity(data=data_pair)
    sim.extract_interested_attribute(interest=interest, window=window)
    similarity_label, data_subsample = sim.label_via_silhouette_analysis(range_n_clusters=range(2, 8))

    # step 5: PAD learns a distance metric that represents the interest of the user from the labeled data pairs
    lm = Linear_Metric()
    lm.train(data_pair, similarity_label)

    dm = Deep_Metric()
    dm.train(data_pair, similarity_label)

    # step 5: PAD learns a distance metric that represents the interest of the user from the labeled data pairs
    # lam_vec is a set of candidate lambda's for weighting the l1-norm penalty in the metric learning optimization problem.
    # The lambda that achieves lowest testing error will be selected for generating the distance metric

    # step 6: the original database is privatized using the learned metric
    sanitized_profile = util.sanitize_data(day_profile, distance_metric="deep", anonymity_level=anonymity_level,
                                           rep_mode=rep_mode, deep_model=lm)

    sanitized_profile_deep = util.sanitize_data(day_profile, distance_metric="deep", anonymity_level=anonymity_level,
                                                rep_mode=rep_mode, deep_model=dm)

    # (optionally for evaluation purpose) Evaluating the information loss of the sanitized database
    loss_learned_metric = pe.get_information_loss(data_gt=day_profile,
                                                  data_sanitized=sanitized_profile.round(),
                                                  window=window)

    loss_learned_metric_deep = pe.get_information_loss(data_gt=day_profile,
                                                       data_sanitized=sanitized_profile_deep.round(),
                                                       window=window)

    print('anonymity level %s' % anonymity_level)
    print("sampled size %s" % subsample_size)
    print("information loss with best metric %s" % loss_best_metric)
    print("information loss with generic metric %s" % loss_generic_metric)
    print("information loss with learned metric %s" % loss_learned_metric)
    print("information loss with learned metric deep  %s" % (loss_learned_metric_deep))
    return (sanitized_profile_best, sanitized_profile_baseline, sanitized_profile, sanitized_profile_deep),\
           (loss_best_metric, loss_generic_metric, loss_learned_metric, loss_learned_metric_deep), subsample_size,\
           (lm.d_test,lm.d_test_arr),(dm.d_test,dm.d_test_arr)

sanitized = {}
losses = {}
sample_sizes = []
mode = "arrival"
dist_lm = {}
dist_dm = {}
for n in [2]:#range(2,8):
    s, l, ss, dist_relation_lm, dist_relation_dm = evaluation_occupancy_window(n)
    sanitized[n] = s
    losses[n] = l
    sample_sizes.append(ss)
    dist_lm[n] = dist_relation_lm
    dist_dm[n] = dist_relation_dm



with open('../result_scripts/test_deep_lunchtime', 'wb') as f:
        pickle.dump([sanitized, losses, sample_sizes,dist_lm,dist_dm], f)

with open('./result_scripts/test_deep_lunchtime', 'rb') as f:
    sanitized, losses, sample_sizes, dist_lm, dist_dm = pickle.load(f)

import numpy as np
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(np.squeeze(dist_lm[2][0]),dist_lm[2][1])
plt.xlabel('learned distance')
plt.ylabel('ground truth distance')
plt.title('linear metric')

plt.figure()
plt.scatter(np.squeeze(dist_dm[2][0]),dist_dm[2][1])
plt.xlabel('learned distance')
plt.ylabel('ground truth distance')
plt.title('deep metric')
