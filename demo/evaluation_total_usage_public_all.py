import sys; import os
sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("./framework"))
sys.path.append(os.path.abspath("./kward"))
sys.path.append(os.path.abspath("./metric_learning"))
sys.path.append(os.path.abspath("./utilities"))
from helper import Utilities, PerformanceEvaluation
import pandas as pd
import numpy as np
from user_feedback import Similarity
import pickle
from scipy.misc import comb
import time
from deep_metric_learning import Deep_Metric
import pickle
from linear_metric_learning import Linear_Metric
from subsampling import Subsampling


def evaluation_total_usage(n):
    """
    In the demo, we will showcase an example of special purpose publication.
    The data user wants the published energy database to maximally retain the information about peak-time energy usage
    """

    # Initialization of some useful classes
    util = Utilities()
    pe = PerformanceEvaluation()

    # step 1: get the database to be published
    day_profile1 = pd.read_pickle('dataset/dataframe_all_energy.pkl')
    day_profile1 = day_profile1.fillna(0)
    # day_profile = day_profile.iloc[0:90,0::4] # subsample the database to improve the speed for demonstration purpose
    day_profile = day_profile1.iloc[:90,0::4] # subsample the database to improve the speed for demonstration purpose
    day_profile2 = day_profile1.iloc[90:200,0::4] # subsample the database to improve the speed for demonstration purpose
    day_profile.index = range(len(day_profile.index))
    day_profile2.index = range(len(day_profile2.index))
    rep_mode = 'mean'
    anonymity_level = n # desired anonymity level

    # step 2: data user specifies his/her interest. In the example, the data user is interested in preserving the
    # information of the cumulative energy use during peak time. In this case, he/she would also need to specify the
    # starting and ending time of the peak usage time
    interest = 'window-usage'
    window = [17,21]

    sanitized_profile_best = util.sanitize_data(day_profile, distance_metric='self-defined',
                                                anonymity_level=anonymity_level, rep_mode = rep_mode,
                                                mode=interest,window=window)

    # step 3: pre-sanitize the database
    sanitized_profile_baseline = util.sanitize_data(day_profile, distance_metric='euclidean',
                                                        anonymity_level=anonymity_level, rep_mode = rep_mode,
                                                        window=window)

    loss_best_metric = pe.get_statistics_loss(data_gt=day_profile, data_sanitized=sanitized_profile_best,
                                                  mode=interest, window=window)

    loss_generic_metric = pe.get_statistics_loss(data_gt=day_profile, data_sanitized=sanitized_profile_baseline,
                                                            mode=interest, window=window)
    # print("information loss with learned metric %s" % loss_generic_metric)

    df_subsampled_from = day_profile2.sample(frac=1)
    subsample_size_max = int(comb(len(df_subsampled_from),2))

    print('total number of pairs is %s' % subsample_size_max)
    
    # step 4: sample a subset of pre-sanitized database and form the data points into pairs
    subsample_size = int(round(subsample_size_max))
    sp = Subsampling(data=df_subsampled_from)
    data_pair, data_pair_all_index = sp.uniform_sampling(subsample_size=subsample_size, seed = None)

    # User receives the data pairs and label the similarity
    sim = Similarity(data=data_pair)
    sim.extract_interested_attribute(interest='statistics', stat_type=interest, window=window)
    similarity_label, data_subsample = sim.label_via_silhouette_analysis(range_n_clusters=range(2,8))

    # step 5: PAD learns a distance metric that represents the interest of the user from the labeled data pairs
    lm = Linear_Metric()
    lm.train(data_pair, similarity_label)
    
    dm = Deep_Metric()
    dm.train(data_pair, similarity_label)

    # step 6: the original database is privatized using the learned metric
    sanitized_profile_deep = util.sanitize_data(day_profile, distance_metric="deep",anonymity_level=anonymity_level,
                                        rep_mode=rep_mode, deep_model=dm, window=window)

    sanitized_profile = util.sanitize_data(day_profile, distance_metric="deep",anonymity_level=anonymity_level,
                                        rep_mode=rep_mode, deep_model=lm, window=window)

    # (optionally for evaluation purpose) Evaluating the information loss of the sanitized database
    loss_learned_metric_deep = pe.get_statistics_loss(data_gt=day_profile, data_sanitized=sanitized_profile_deep.round(),
                                                            mode=interest,window=window)


    loss_learned_metric = pe.get_statistics_loss(data_gt=day_profile, data_sanitized=sanitized_profile,
                                                            mode=interest,window=window)
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
for n in range(2,8):    
    s, l, ss = evaluation_total_usage(n)
    sanitized[n] = s
    losses[n] = l
    sample_sizes.append(ss)

with open('result_scripts/loss_vs_privacy_energy_usage_public_deep_all.pickle', 'wb') as f:
        pickle.dump([sanitized,losses, sample_sizes], f)
