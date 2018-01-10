import sys; import os
sys.path.append(os.path.abspath("./"))
from helper import Utilities, PerformanceEvaluation
import pandas as pd
from metric_learning import Subsampling, MetricLearning
from user_feedback import Similarity
from scipy.misc import comb
from deep_metric_learning import Deep_Metric
import numpy as np
import pickle
from linear_metric_learning import Linear_Metric

"""
In the demo, we will showcase an example of special purpose publication.
The data user wants the published database to maximally retain the information about lunch time.
"""

# Initialization of some useful classes
util = Utilities()
pe = PerformanceEvaluation()
mel = MetricLearning()

def evaluation_occupancy_statistics(n, mode = "arrival"):
    day_profile1 = pd.read_pickle('./dataset/dataframe_all_binary.pkl')
    # print(len(day_profile1))
    day_profile1 = day_profile1[(day_profile1.T != 0).any()]
    day_profile1[day_profile1>0] = 1
    # print(len(day_profile1))
    res = 15
    # day_profile = day_profile.iloc[:90,0::res]

    day_profile = day_profile1.iloc[:120,0::res] # subsample the database to improve the speed for demonstration purpose
    day_profile2 = day_profile1.iloc[120:120+52,0::res] # subsample the database to improve the speed for demonstration purpose
    # day_profile.index = range(len(day_profile.index))
    # day_profile2.index = range(len(day_profile2.index))


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

    # print("information loss with generic metric %s" % loss_generic_metric)
    df_subsampled_from = day_profile2.sample(frac=1)

    subsample_size_max = int(comb(len(df_subsampled_from),2))
    print('total number of pairs is %s' % len(df_subsampled_from))

    loss_learned_metric = {}
    loss_learned_metric_deep = {}

    random_state_vec = np.arange(5)
    for i in range(len(random_state_vec)):
        random_state = random_state_vec[i]
        np.random.seed(random_state)

        # step 4: sample a subset of pre-sanitized database and form the data points into pairs
        subsample_size = int(round(subsample_size_max/2))
        sp = Subsampling(data=df_subsampled_from)
        data_pair = sp.uniform_sampling(subsample_size=subsample_size, seed = None)

        # User receives the data pairs and label the similarity
        sim = Similarity(data=data_pair)
        sim.extract_interested_attribute(interest='statistics', stat_type=mode)
        similarity_label, data_subsample = sim.label_via_silhouette_analysis(range_n_clusters=range(2,8))

        # step 5: PAD learns a distance metric that represents the interest of the user from the labeled data pairs
        lm = Linear_Metric()
        lm.train(data_pair, similarity_label)

        dm = Deep_Metric()
        dm.train(data_pair, similarity_label)


        # dist_metric = mel.learn_with_simialrity_label_regularization(data=data_pair,
        #                                                             label=similarity_label,
        #                                                             lam_vec=[0, 0.1, 1, 10],
        #                                                             train_portion=0.8)

        # step 6: the original database is privatized using the learned metric
        sanitized_profile = util.sanitize_data(day_profile, distance_metric="deep",anonymity_level=anonymity_level,
                                            rep_mode=rep_mode, deep_model=lm)

        sanitized_profile_deep = util.sanitize_data(day_profile, distance_metric="deep",anonymity_level=anonymity_level,
                                            rep_mode=rep_mode, deep_model=dm)

        # (optionally for evaluation purpose) Evaluating the information loss of the sanitized database
        loss_learned_metric[i] = pe.get_statistics_loss(data_gt=day_profile,
                                                    data_sanitized=sanitized_profile.round(),
                                                    mode = mode)

        loss_learned_metric_deep[i] = pe.get_statistics_loss(data_gt=day_profile,
                                                    data_sanitized=sanitized_profile_deep.round(),
                                                    mode = mode)

        print('anonymity level %s' % anonymity_level)
        print('random state %s' % i)
        print("sampled size %s" % subsample_size)
        print("information loss with best metric %s" % loss_best_metric)
        print("information loss with generic metric %s" % loss_generic_metric)
        print("information loss with learned metric %s" %  loss_learned_metric[i])
        print("information loss with learned metric deep  %s" % (loss_learned_metric_deep[i]))
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

with open('result_scripts/loss_vs_privacy_occupancy_statistics_public_deep_%s.pickle'%(mode), 'wb') as f: 
        pickle.dump([sanitized, losses, sample_sizes], f)

