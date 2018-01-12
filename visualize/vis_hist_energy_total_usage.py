import sys; import os
sys.path.append(os.path.abspath("./"))
import matplotlib.pyplot as plt
import pickle
import numpy as np


## visualze the distribution of usage
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

dataset_type = "public"

with open('./result_scripts/loss_vs_privacy_energy_usage_%s_deep.pickle'%(dataset_type),'rb') as f: # Python 3: open(..., 'wb')
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
ai = 2
anonymity_level = ai
# dist_metric = dist_metric_mat[ai][0][0]
stat_gt = OccupancyStatistics(day_profile)
usage_gt = stat_gt.get_window_usage(window=window)

sanitized_profile_baseline = sanitized_profile_baseline_list[ai]
stat_bl = OccupancyStatistics(sanitized_profile_baseline)
usage_bl = stat_bl.get_window_usage(window=window)

sanitized_profile = sanitized_profile_list[ai]
stat_sn = OccupancyStatistics(sanitized_profile)
usage_sn = stat_sn.get_window_usage(window=window)

sanitized_profile = sanitized_profile_deep_list[ai]
stat_dn = OccupancyStatistics(sanitized_profile)
usage_dn = stat_dn.get_window_usage(window=window)

bins = 20
fontsize = 18
legendsize = 12
plt.figure()
plt.hist(usage_gt,label='Ground truth metric',alpha = 0.4,normed=True,bins=bins)
plt.hist(usage_bl,label='Sanitized database w/ generic metric',alpha = 0.4,normed=True,bins=bins)
plt.xlabel('Peak hour energy usage (W)',fontsize=fontsize)
plt.title('Normalized histogram of peak-hour usage',fontsize=fontsize)
plt.legend(fontsize=legendsize)
plt.yticks([0,1e-4,2e-4,3e-4,3.5e-4],['0','1e-4','2e-4','3e-4',''])
plt.savefig("visualize/figures/histogram level %s peak energy usage with %s generic.png"%(str(anonymity_level), dataset_type))
# plt.show()



plt.figure()
plt.hist(usage_gt,label='Original database',alpha = 0.4,normed=True,bins=bins)
plt.hist(usage_sn,label='Sanitized database w/ linear metric',alpha = 0.4,normed=True,bins=bins)
plt.xlabel('Peak hour energy usage (W)',fontsize=fontsize)
plt.title('Normalized histogram of peak-hour usage',fontsize=fontsize)
plt.legend(fontsize=legendsize)
plt.yticks([0,1e-4,2e-4,3e-4,3.5e-4],['0','1e-4','2e-4','3e-4',''])
plt.savefig("visualize/figures/histogram level %s peak energy usage with %s linear.png"%(str(anonymity_level), dataset_type))
# plt.show()


plt.figure()
plt.hist(usage_gt,label='Original database',alpha = 0.4,normed=True,bins=bins)
plt.hist(usage_dn,label='Sanitized database w/ deep metric',alpha = 0.4,normed=True,bins=bins)
plt.xlabel('Peak hour energy usage (W)',fontsize=fontsize)
plt.title('Normalized histogram of peak-hour usage',fontsize=fontsize)
plt.legend(fontsize=legendsize)
plt.yticks([0,1e-4,2e-4,3e-4,3.5e-4],['0','1e-4','2e-4','3e-4',''])
plt.savefig("visualize/figures/histogram level %s peak energy usage with %s deep.png"%(str(anonymity_level),dataset_type))
# plt.show()

# exit()