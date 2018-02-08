
import matplotlib.pyplot as plt
import pickle
import numpy as np
from data_statistics import OccupancyStatistics
import pandas as pd
from helper import Utilities, PerformanceEvaluation

util = Utilities()
day_profile = pd.read_pickle('dataset/dataframe_all_energy.pkl')
day_profile = day_profile.iloc[0:90,0::4]
day_profile.index = range(len(day_profile.index))
window = [17,21]
rep_mode = 'mean'
statistics_mode = 'window-usage'

with open('./result_scripts/loss_vs_privacy_energy_usage_public_deep_all.pickle','rb') as f: # Python 3: open(..., 'wb')
    data = pickle.load(f)


s, l, ss = data
print(list(s.keys()))

sanitized_profile_best_list = {}
sanitized_profile_baseline_list = {}
sanitized_profile_list = {}
sanitized_profile_deep_list = {}
for i in s.keys():
    sanitized_profile_best_list[i], sanitized_profile_baseline_list[i], sanitized_profile_list[i], sanitized_profile_deep_list[i] = s[i]


anonymity_vec = list(s.keys())
n_a = len(anonymity_vec)
tv_gn = np.empty(n_a)
tv_best = np.empty(n_a)
tv_sn = np.empty(n_a)
tv_sd = np.empty(n_a)

for i in range(len(anonymity_vec)):
    ai = anonymity_vec[i]

    stat_gt = OccupancyStatistics(day_profile)
    usage_gt = stat_gt.get_window_usage(window=window)

    sanitized_profile_baseline = sanitized_profile_baseline_list[ai]
    stat_bl = OccupancyStatistics(sanitized_profile_baseline)
    usage_bl = stat_bl.get_window_usage(window=window)

    sanitized_profile_linear = sanitized_profile_list[ai]
    stat_linear = OccupancyStatistics(sanitized_profile_linear)
    usage_linear = stat_linear.get_window_usage(window=window)

    sanitized_profile_deep = sanitized_profile_deep_list[ai]
    stat_deep = OccupancyStatistics(sanitized_profile_deep)
    usage_deep= stat_deep.get_window_usage(window=window)

    sanitized_profile_best = sanitized_profile_best_list[ai]
    stat_best = OccupancyStatistics(sanitized_profile_best)
    usage_best = stat_best.get_window_usage(window=window)

    bins = 10
    max_val = np.max([usage_gt.values,usage_best.values,usage_bl.values,usage_linear.values,usage_deep.values])

    dep_hist_gt, bin_edges_gt = np.histogram(usage_gt.values, bins=bins, range=(0, max_val),normed=True)
    dep_hist_gn, bin_edges_gn = np.histogram(usage_bl.values, bins=bins, range=(0, max_val),normed=True)
    dep_hist_sn, bin_edges_sn = np.histogram(usage_linear.values, bins=bins, range=(0, max_val),normed=True)
    dep_hist_sd, bin_edges_sd = np.histogram(usage_deep.values, bins=bins, range=(0, max_val),normed=True)
    dep_hist_best, bin_edges_best = np.histogram(usage_best.values, bins=bins, range=(0, max_val),normed=True)

    tv_gn[i] = np.linalg.norm(dep_hist_gt - dep_hist_gn, ord=1)
    tv_sn[i] = np.linalg.norm(dep_hist_gt - dep_hist_sn, ord=1)
    tv_sd[i] = np.linalg.norm(dep_hist_gt - dep_hist_sd, ord=1)
    tv_best[i] = np.linalg.norm(dep_hist_gt - dep_hist_best, ord=1)

    print('-----------------')
    print(i)
    print('generic %s' % tv_gn)
    print('linear %s' % tv_sn)
    print('deep %s' % tv_sd)
    print('best %s' % tv_best)

plt.plot(anonymity_vec,tv_gn,label='generic')
plt.plot(anonymity_vec,tv_best,label='best')
plt.plot(anonymity_vec,tv_sn,label='linear')
plt.plot(anonymity_vec,tv_sd,label='deep')
plt.legend()
plt.show()
#     bins = 20
#     fontsize = 18
#     legendsize = 12
#     plt.figure()
#     plt.hist(usage_gt,label='Ground truth metric',alpha = 0.4,normed=True,bins=bins)
#     plt.hist(usage_bl,label='Sanitized database w/ generic metric',alpha = 0.4,normed=True,bins=bins)
#     plt.xlabel('Peak hour energy usage (W)',fontsize=fontsize)
#     plt.title('Normalized histogram of peak-hour usage',fontsize=fontsize)
#     plt.legend(fontsize=legendsize)
#     plt.yticks([0,1e-4,2e-4,3e-4,3.5e-4],['0','1e-4','2e-4','3e-4',''])
#     plt.savefig("visualize/figures/histogram level %s peak energy usage with %s generic.png"%(str(anonymity_level), dataset_type))
#     # plt.show()
#
#
#
#     plt.figure()
#     plt.hist(usage_gt,label='Original database',alpha = 0.4,normed=True,bins=bins)
#     plt.hist(usage_sn,label='Sanitized database w/ linear metric',alpha = 0.4,normed=True,bins=bins)
#     plt.xlabel('Peak hour energy usage (W)',fontsize=fontsize)
#     plt.title('Normalized histogram of peak-hour usage',fontsize=fontsize)
#     plt.legend(fontsize=legendsize)
#     plt.yticks([0,1e-4,2e-4,3e-4,3.5e-4],['0','1e-4','2e-4','3e-4',''])
#     plt.savefig("visualize/figures/histogram level %s peak energy usage with %s linear.png"%(str(anonymity_level), dataset_type))
#     # plt.show()
#
#
#     plt.figure()
#     plt.hist(usage_gt,label='Original database',alpha = 0.4,normed=True,bins=bins)
#     plt.hist(usage_dn,label='Sanitized database w/ nonlinear metric',alpha = 0.4,normed=True,bins=bins)
#     plt.xlabel('Peak hour energy usage (W)',fontsize=fontsize)
#     plt.title('Normalized histogram of peak-hour usage',fontsize=fontsize)
#     plt.legend(fontsize=legendsize)
#     plt.yticks([0,1e-4,2e-4,3e-4,3.5e-4],['0','1e-4','2e-4','3e-4',''])
#     plt.savefig("visualize/figures/histogram level %s peak energy usage with %s deep.png"%(str(anonymity_level),dataset_type))
#     # plt.show()
#
# # exit()