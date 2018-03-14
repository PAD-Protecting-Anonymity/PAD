import sys; import os
sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("./utilities"))
sys.path.append(os.path.abspath("./kward"))
sys.path.append(os.path.abspath("./metric_learning"))
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

def evaluation_occupancy_statistics(n, mode = "arrival"):
    day_profile1 = pd.read_pickle('./dataset/dataframe_all_binary.pkl')
    day_profile1 = day_profile1.fillna(0)
    day_profile1[day_profile1>0] = 1

    res = 15

    day_profile = day_profile1.iloc[:90,0::res] # subsample the database to improve the speed for demonstration purpose
    day_profile2 = day_profile1.iloc[90:-1,0::res] # subsample the database to improve the speed for demonstration purpose

    rep_mode = 'mean'
    anonymity_level = n

    sanitized_profile_best = util.sanitize_data(day_profile, distance_metric='self-defined',
                                                anonymity_level=anonymity_level, rep_mode = rep_mode,
                                                mode=mode)

    sanitized_profile_baseline = util.sanitize_data(day_profile, distance_metric='euclidean',
                                                    anonymity_level=anonymity_level, rep_mode = rep_mode)

    loss_best_metric = pe.get_statistics_loss(data_gt=day_profile, data_sanitized=sanitized_profile_best,
                                                  mode=mode)

    loss_generic_metric = pe.get_statistics_loss(data_gt=day_profile,
                                                data_sanitized=sanitized_profile_baseline,
                                                mode = mode)

    df_subsampled_from = day_profile2.sample(frac=1)

    subsample_size_max = int(comb(len(df_subsampled_from),2))
    print('total number of pairs is %s' % len(df_subsampled_from))

    # step 4: sample a subset of pre-sanitized database and form the data points into pairs
    subsample_size = int(round(subsample_size_max))
    sp = Subsampling(data=df_subsampled_from)
    data_pair, data_pair_all_index = sp.uniform_sampling(subsample_size=subsample_size, seed = None)

    # User receives the data pairs and label the similarity
    sim = Similarity(data=data_pair)
    sim.extract_interested_attribute(interest='statistics', stat_type=mode)
    similarity_label, data_subsample = sim.label_via_silhouette_analysis(range_n_clusters=range(2,8))

    # step 5: PAD learns a distance metric that represents the interest of the user from the labeled data pairs
    lm = Linear_Metric()
    lm.train(data_pair, similarity_label)

    dm = Deep_Metric()
    dm.train(data_pair, similarity_label)

    # step 6: the original database is privatized using the learned metric
    sanitized_profile = util.sanitize_data(day_profile, distance_metric="deep",anonymity_level=anonymity_level,
                                        rep_mode=rep_mode, deep_model=lm)

    sanitized_profile_deep = util.sanitize_data(day_profile, distance_metric="deep",anonymity_level=anonymity_level,
                                        rep_mode=rep_mode, deep_model=dm)

    # (optionally for evaluation purpose) Evaluating the information loss of the sanitized database
    loss_learned_metric = pe.get_statistics_loss(data_gt=day_profile,
                                                data_sanitized=sanitized_profile.round(),
                                                mode = mode)

    loss_learned_metric_deep = pe.get_statistics_loss(data_gt=day_profile,
                                                data_sanitized=sanitized_profile_deep.round(),
                                                mode = mode)

    print('anonymity level %s' % anonymity_level)
    print("sampled size %s" % subsample_size)
    print("information loss with best metric %s" % loss_best_metric)
    print("information loss with generic metric %s" % loss_generic_metric)
    print("information loss with learned metric %s" %  loss_learned_metric)
    print("information loss with learned metric deep  %s" % (loss_learned_metric_deep))
    return (sanitized_profile_best, sanitized_profile_baseline, sanitized_profile, sanitized_profile_deep), (loss_best_metric, loss_generic_metric, loss_learned_metric, loss_learned_metric_deep), subsample_size


sanitized = {}
losses = {}
sample_sizes = []
mode = "arrival"
for n in range(2,8):
    s, l, ss = evaluation_occupancy_statistics(n, mode)
    sanitized[n] = s
    losses[n] = l
    sample_sizes.append(ss)

with open('result_scripts/loss_vs_privacy_occupancy_statistics_public_deep_%s_all.pickle'%(mode), 'wb') as f:
        pickle.dump([sanitized, losses, sample_sizes], f)

