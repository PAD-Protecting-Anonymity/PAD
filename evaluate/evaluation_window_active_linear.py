import sys;
import os

sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("./framework"))
sys.path.append(os.path.abspath("./kward"))
sys.path.append(os.path.abspath("./metric_learning"))
from helper import Utilities, PerformanceEvaluation
import pandas as pd
from subsampling import Subsampling
from user_feedback import Similarity
from scipy.misc import comb
from deep_metric_learning import Deep_Metric
import numpy as np
import pickle
from linear_metric_learning import Linear_Metric
import pdb

# """
# In the demo, we will showcase an example of special purpose publication.
# The data user wants the published database to maximally retain the information about lunch time.
# """

# Initialization of some useful classes
util = Utilities()
pe = PerformanceEvaluation()

day_profile_all = pd.read_pickle('./dataset/dataframe_all_binary.pkl')
day_profile = day_profile_all.iloc[0:90, 0::60] # the database to be published
day_profile_metric_learn = day_profile_all.iloc[90:-1, 0::60] # the database for learning distance metric
day_profile.dropna()
day_profile_metric_learn.dropna()

rep_mode = 'mean'
anonymity_level = 2  # desired anonymity level

# step 2: data user specifies his/her interest. In the example, the data user is interested in preserving the
# information of a segment of entire time series. In this case, he/she would also need to specify the starting and
# ending time of the time series segment of interest.
interest = 'segment'
window = [11, 15]  # window specifies the starting and ending time of the period that the data user is interested in

# step 3: pre-sanitize the database
sanitized_profile_best = util.sanitize_data(day_profile, distance_metric='self-defined',
                                            anonymity_level=anonymity_level, rep_mode=rep_mode,
                                            mode=interest, window=window)

sanitized_profile_baseline = util.sanitize_data(day_profile, distance_metric='euclidean',
                                                anonymity_level=anonymity_level, rep_mode=rep_mode)

loss_best_metric = pe.get_statistics_loss(data_gt=day_profile, data_sanitized=sanitized_profile_best,
                                          mode=interest, window=window)

loss_generic_metric = pe.get_information_loss(data_gt=day_profile,
                                              data_sanitized=sanitized_profile_baseline.round(),
                                              window=window)

df_subsampled_from = day_profile_metric_learn
subsample_size_max = int(comb(len(df_subsampled_from), 2))
print('total number of pairs is %s' % len(df_subsampled_from))

## obtain ground truth similarity labels
sp = Subsampling(data=df_subsampled_from)
data_pair_all, data_pair_all_index = sp.uniform_sampling(subsample_size=subsample_size_max, seed=0)
sim = Similarity(data=data_pair_all)
sim.extract_interested_attribute(interest=interest, window=window)
similarity_label_all, class_label_all = sim.label_via_silhouette_analysis(range_n_clusters=range(2,8))
similarity_label_all_series = pd.Series(similarity_label_all)
similarity_label_all_series.index = data_pair_all_index
print('similarity balance is %s'% [sum(similarity_label_all),len(similarity_label_all)])



k_init = 50
mc_num = 5
seed_vec = np.arange(mc_num)
run_sample_size = 300

## active sampling
loss_active_all_linear = []
pairdata_all_linear = []
pairlabel_all_linear = []
for mc_i in range(len(seed_vec)):
    loss_active_mc_linear= []
    pairdata_each_mc_linear = []
    pairlabel_each_mc_linear = []
    pairdata_active_linear = []
    pairdata_label_active_linear = []
    lm = None
    sp.reset()
    k = k_init
    while k <= run_sample_size:
        pairdata, pairdata_idx = sp.active_sampling(dist_metric=lm,
                                                    k_init=k_init,
                                                    batch_size=1,
                                                    seed=seed_vec[mc_i])
        pairdata_active_linear = pairdata_active_linear + pairdata

        similarity_label = similarity_label_all_series.loc[pairdata_idx].tolist()

        pairdata_label_active_linear = pairdata_label_active_linear + similarity_label

        lm = Linear_Metric()
        lm.train(pairdata_active_linear, pairdata_label_active_linear)

        sanitized_profile = util.sanitize_data(day_profile, distance_metric="deep",
                                               anonymity_level=anonymity_level,
                                               rep_mode=rep_mode, deep_model=lm)

        # (optionally for evaluation purpose) Evaluating the information loss of the sanitized database
        loss_learned_metric = pe.get_information_loss(data_gt=day_profile,
                                                         data_sanitized=sanitized_profile.round(),
                                                         window=window)

        loss_active_mc_linear.append(loss_learned_metric)

        pairdata_each_mc_linear.append(pairdata_active_linear)
        pairlabel_each_mc_linear.append(pairdata_label_active_linear)

        print('====================')
        print('random state %s ' % mc_i)
        print('sample size %s '% k)
        print("information loss with best metric %s" % loss_best_metric)
        print("information loss with generic metric %s" % loss_generic_metric)
        print("information loss with learned metric %s" % loss_learned_metric)

        k += 1

    loss_active_all_linear.append(loss_active_mc_linear)
    pairdata_all_linear.append(pairdata_each_mc_linear)
    pairlabel_all_linear.append(pairlabel_each_mc_linear)

    with open('./result_scripts/sample_acitve_occupancy_linear.pickle', 'wb') as f:
        pickle.dump([loss_best_metric,loss_generic_metric,loss_active_all_linear,
                     k_init,subsample_size_max,run_sample_size,
                     pairdata_all_linear,pairlabel_all_linear], f)


