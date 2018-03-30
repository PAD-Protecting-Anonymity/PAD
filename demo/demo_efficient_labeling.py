# import sys, os
# sys.path.append(os.path.abspath("./"))
from helper import Utilities, PerformanceEvaluation
import pandas as pd
from metric_learning import Subsampling, MetricLearning
from user_feedback import Similarity
from scipy.misc import comb
import numpy as np
import pickle
import matplotlib.pyplot as plt


# Initialization of some useful classes
util = Utilities()
pe = PerformanceEvaluation()
mel = MetricLearning()

# get the database to be published
day_profile = pd.read_pickle('../dataset/dataframe_all_binary.pkl')
day_profile = day_profile.iloc[0::4,0::60]#day_profile.iloc[0::4,0::60]
rep_mode = 'mean'
anonymity_level = 2 # desired anonymity level
lam_vec = [1e-3,1e-2,1e-1,1,10]

# data user specifies his/her interest. In the example, the data user is interested in preserving the
# information of a segment of entire time series. In this case, he/she would also need to specify the starting and
# ending time of the time series segment of interest.
interest = 'segment'
window = [11,15] # window specifies the starting and ending time of the period that the data user is interested in

# pre-sanitize the database
sanitized_profile_baseline = util.sanitize_data(day_profile, distance_metric='euclidean',
                                                anonymity_level=anonymity_level,rep_mode = rep_mode)
loss_generic_metric = pe.get_information_loss(data_gt=day_profile,
                                              data_sanitized=sanitized_profile_baseline.round(),
                                              window=window)
print("information loss with generic metric %s" % loss_generic_metric)
df_subsampled_from = sanitized_profile_baseline.drop_duplicates().sample(frac=1)
subsample_size_max = int(comb(len(df_subsampled_from),2))
print('total number of pairs is %s' % subsample_size_max)

# # obtain ground truth similarity labels
sp = Subsampling(data=df_subsampled_from)

data_pair_all, data_pair_all_index = sp.uniform_sampling(subsample_size=subsample_size_max,seed=0)
sim = Similarity(data=data_pair_all)
sim.extract_interested_attribute(interest=interest, window=window)
similarity_label_all, class_label_all = sim.label_via_silhouette_analysis(range_n_clusters=range(2,8))
similarity_label_all_series = pd.Series(similarity_label_all)
similarity_label_all_series.index = data_pair_all_index
print('similarity balance is %s'% [sum(similarity_label_all),len(similarity_label_all)])


k_init = 10
batch_size = 20
mc_num = 5
seed_vec = np.arange(0,mc_num)

# ##################
# uniform sampling

loss_iters_unif = []

pairdata_all = []
pairlabel_all = []
dist_metric = []

for mc_i in range(mc_num):
    loss_learned_metric_unif_mc = []
    pairdata_each_mc = []
    pairlabel_each_mc = []
    dist_metric_mc = []
    k = k_init
    while k <= subsample_size_max:

        if k == k_init:
            pairdata,pairdata_idx = sp.uniform_sampling(subsample_size=k,seed=seed_vec[mc_i])
            pairdata_label = similarity_label_all_series.loc[pairdata_idx]
        else:
            pairdata, pairdata_idx = sp.uniform_sampling(subsample_size=k, seed=None)
            pairdata_label = similarity_label_all_series.loc[pairdata_idx]

        # sim = Similarity(data=pairdata)
        # sim.extract_interested_attribute(interest=interest, window=window)
        # pairdata_label,_ = sim.label_via_silhouette_analysis(range_n_clusters=range(2, 8))
        dist_metric = mel.learn_with_simialrity_label_regularization(data=pairdata,
                                                                     label=pairdata_label,
                                                                     lam_vec=lam_vec,
                                                                     train_portion=0.8)
        # _, _, dist_metric = mel.learn_with_similarity_label(pairdata, pairdata_label, "diag",lam_vec)
        if dist_metric is None:
            loss_learned_metric_unif = np.nan
        else:
            sanitized_profile_unif = util.sanitize_data(day_profile, distance_metric="mahalanobis",
                                                          anonymity_level=anonymity_level, rep_mode=rep_mode, VI=dist_metric)
            loss_learned_metric_unif = pe.get_information_loss(data_gt=day_profile,
                                                                 data_sanitized=sanitized_profile_unif.round(),
                                                                 window=window)
        loss_learned_metric_unif_mc.append(loss_learned_metric_unif)
        pairdata_each_mc.append(pairdata)
        pairlabel_each_mc.append(pairdata_label)
        print("sampled size %s" % k)
        print('k is %s' % k)
        print("information loss with uniform metric %s" % np.mean(loss_learned_metric_unif_mc))
        k += batch_size
    loss_iters_unif.append(loss_learned_metric_unif_mc)
    pairdata_all.append(pairdata_each_mc)
    pairlabel_all.append(pairlabel_each_mc)



with open('../result/loss_uniform_cv.pickle', 'wb') as f:
    pickle.dump([loss_iters_unif,k_init,subsample_size_max,batch_size,loss_generic_metric,
                 pairdata_all,pairlabel_all], f)


##################
# active learning

# actively sample a subset of pre-sanitized database




pairdata_all_active = []
pairlabel_all_active = []
loss_iters_active = []
for mc_i in range(mc_num):
    pairdata_each_mc_active = []
    pairlabel_each_mc_active = []

    pairdata_active_mc = []
    pairdata_label_active_mc = []
    loss_iters_active_mc = []
    k = k_init
    dist_metric = None
    sp.reset()
    while k <= subsample_size_max:
<<<<<<< HEAD
=======
        # pairdata, pairdata_idx = sp.active_sampling(dist_metric=dist_metric,
        #                                             k_init=k_init,
        #                                             batch_size=1)
>>>>>>> 69418cbb0576c1cb2e97585a68597ea8affb2ebf
        pairdata, pairdata_idx = sp.active_sampling(dist_metric=dist_metric,
                                                    k_init=k_init,
                                                    batch_size=1,
                                                    seed=seed_vec[mc_i])
        pairdata_active_mc = pairdata_active_mc + pairdata
        similarity_label = similarity_label_all_series.loc[pairdata_idx].tolist()
        pairdata_label_active_mc = pairdata_label_active_mc + similarity_label
        # _, _, dist_metric = mel.learn_with_similarity_label(pairdata_active_mc, pairdata_label_active_mc, "diag", lam_vec)
        dist_metric = mel.learn_with_simialrity_label_regularization(data=pairdata_active_mc,
                                                                     label=pairdata_label_active_mc,
                                                                     lam_vec=lam_vec,
                                                                     train_portion=0.8)
        sanitized_profile_active = util.sanitize_data(day_profile, distance_metric="mahalanobis",
                                               anonymity_level=anonymity_level, rep_mode=rep_mode, VI=dist_metric)

        if (k-k_init) % 1 == 0:
            loss_learned_metric_active = pe.get_information_loss(data_gt=day_profile,
                                                          data_sanitized=sanitized_profile_active.round(),
                                                          window=window)
            loss_iters_active_mc.append(loss_learned_metric_active)
            pairdata_each_mc_active.append(pairdata_active_mc)
            pairlabel_each_mc_active.append(pairdata_label_active_mc)


            print("sampled size %s" % sp.k_already)
            print('k is %s' % k)
            print("information loss with active metric %s" % loss_learned_metric_active)
        k += 1
    loss_iters_active.append(loss_iters_active_mc)
    pairdata_all_active.append(pairdata_each_mc_active)
    pairlabel_all_active.append(pairlabel_each_mc_active)



with open('../result/loss_active_cv.pickle', 'wb') as f:
    pickle.dump([loss_iters_active,k_init,subsample_size_max,batch_size,loss_generic_metric,
                 pairdata_all_active,pairlabel_all_active], f)


import numpy as np
import pickle
import matplotlib.pyplot as plt
mc_num = 5
# plot
with open('./result/loss_active_lam1_diag.pickle', 'rb') as f:
    loss_iters_active, k_init, subsample_size_max, batch_size, loss_generic_metric = \
        pickle.load(f)

with open('./result/loss_uniform_lam1_diag.pickle', 'rb') as f:
    loss_iters_unif, k_init_unif, subsample_size_max_unif, batch_size_unif, loss_generic_metric_unif = \
        pickle.load(f)

loss_iters_active_format = np.asarray(loss_iters_active)
loss_active_mean = np.mean(loss_iters_active_format,axis=0)
loss_iters_unif_format = np.asarray(loss_iters_unif)
loss_unif_mean = np.mean(loss_iters_unif,axis=0)
eval_k = np.arange(k_init,subsample_size_max+1,batch_size_unif)


plt.figure()
plt.errorbar(eval_k,loss_unif_mean,np.std(loss_iters_unif_format,axis=0),label='uniform sampling')
plt.errorbar(eval_k,loss_active_mean[eval_k-k_init],np.std(loss_iters_active_format,axis=0)[eval_k-k_init],
             label='active sampling')
# plt.errorbar(np.arange(k_init,subsample_size_max+1),loss_active_mean,np.std(loss_iters_active_format),
#              label='active sampling')
# plt.plot(np.arange(k_init,subsample_size_max+1),loss_active_mean, label='active sampling')
# for i in range(mc_num):
#     plt.plot(np.arange(k_init,subsample_size_max+1),loss_iters_active_format[i,:], label='active sample',color='red')
#     plt.plot(eval_k,loss_iters_unif_format[i,:],label='uniform sample', color='blue')
plt.plot((k_init,subsample_size_max),(loss_generic_metric,loss_generic_metric),'r--',label="generic metric")
plt.legend()
plt.show()