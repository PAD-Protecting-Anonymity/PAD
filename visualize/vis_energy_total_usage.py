import matplotlib.pyplot as plt
import pickle
import numpy as np
with open('result_scripts/loss_vs_privacy_usage_SCS_5_publicdata_deep.pickle','rb') as f: # Python 3: open(..., 'wb')
    data = pickle.load(f)
# sanitized_profile_best, sanitized_profile_baseline, sanitized_profile, sanitized_profile_deep = s                                    
# loss_best_metric, loss_generic_metric, loss_learned_metric, loss_learned_metric_deep = l    

s, l, ss = data
loss_best_metric_list = {}
loss_generic_metric_list = {}
loss_learned_metric_list = {}
loss_learned_metric_deep_list = {}    
for i in l.keys():
    print(i)
    loss_best_metric_list[i], loss_generic_metric_list[i], loss_learned_metric_list[i], loss_learned_metric_deep_list[i] = l[i]
# exit()
loss_best_metric_list = list(loss_best_metric_list.values())
loss_generic_metric_list = list(loss_generic_metric_list.values())
anonymity_vec = list(l.keys())
# print(loss_generic_metric_list)
# exit()
fontsize = 18
legendsize = 12
loss_learned_metric_lists = [list(loss_learned_metric_list[i].values()) for i in loss_learned_metric_list.keys()]
loss_learned_deep_metric_lists = [list(loss_learned_metric_deep_list[i].values())for i in loss_learned_metric_deep_list.keys()]
# print(loss_learned_deep_metric_lists)
# exit()

# plt.plot(anonymity_vec,loss_best_metric_list, label='Ground truth metric',color='red')
plt.plot(anonymity_vec,loss_generic_metric_list,label='Generic metric',color='blue',linestyle='-.')
bp = plt.boxplot(loss_learned_metric_lists,positions=anonymity_vec,patch_artist=True,widths=0.1)
bp1 = plt.boxplot(loss_learned_deep_metric_lists,positions=anonymity_vec,patch_artist=True,widths=0.1)
fill_color = 'orange'
edge_color = 'orange'
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color=edge_color)

for patch in bp['boxes']:
    patch.set(facecolor=fill_color, alpha=0.5)

fill_color = 'lightgreen'
edge_color = 'lightgreen'
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp1[element], color=edge_color)

for patch in bp1['boxes']:
    patch.set(facecolor=fill_color, alpha=0.5)

plt.plot(anonymity_vec,np.mean(loss_learned_metric_lists,axis=1),label='Learned metric',color='orange',linestyle='--')
plt.plot(anonymity_vec,np.mean(loss_learned_deep_metric_lists,axis=1),label='Deep Learned metric',color='lightgreen',linestyle='--')
plt.xlabel('Anonymity level',fontsize=fontsize)
plt.ylabel('Information loss (W)',fontsize=fontsize)
plt.title('Publication specialized for peak-time consumption with Similar Dataset',fontsize=fontsize)
plt.legend(fontsize=legendsize, loc='upper left')
plt.show()

exit()

## visualze the distribution of usage
from statistics import OccupancyStatistics
import pandas as pd
from model_new import Utilities, PerformanceEvaluation

util = Utilities()
day_profile = pd.read_pickle('dataframe_all_energy.pkl')
day_profile = day_profile.iloc[0:90,0::4]
day_profile.index = range(len(day_profile.index))
window = [17,21]
rep_mode = 'mean'
statistics_mode = 'window-usage'



# sanitized_profile_baseline_list = []
# sanitized_profile_list = []
# for ai in range(len(anonymity_vec)):
#     anonymity_level = anonymity_vec[ai]
#     sanitized_profile_baseline = util.sanitize_data(day_profile, distance_metric='euclidean',
#                                                     anonymity_level=anonymity_level, rep_mode=rep_mode,
#                                                     window=window)
#     sanitized_profile_baseline_list.append(sanitized_profile_baseline)
#     sanitized_profile = util.sanitize_data(day_profile, distance_metric='mahalanobis',
#                                            anonymity_level=anonymity_level, rep_mode=rep_mode, VI=dist_metric,
#                                            window=window)
#     sanitized_profile_list.append(sanitized_profile)

# with open('./result_new/loss_vs_privacy_usage_SCS_50_sanitized_profile.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(
#         [sanitized_profile_list,sanitized_profile_baseline_list], f)


with open('./result_new/loss_vs_privacy_usage_SCS_50_sanitized_profile.pickle','rb') as f: # Python 3: open(..., 'wb')
    sanitized_profile_list, sanitized_profile_baseline_list = pickle.load(f)

ai = 3
anonymity_level = anonymity_vec[ai]
dist_metric = dist_metric_mat[ai][0][0]
stat_gt = OccupancyStatistics(day_profile)
usage_gt = stat_gt.get_window_usage(window=window)

sanitized_profile_baseline = sanitized_profile_baseline_list[ai]
stat_bl = OccupancyStatistics(sanitized_profile_baseline)
usage_bl = stat_bl.get_window_usage(window=window)

sanitized_profile = sanitized_profile_list[ai]
stat_sn = OccupancyStatistics(sanitized_profile)
usage_sn = stat_sn.get_window_usage(window=window)

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
plt.show()

plt.figure()
plt.hist(usage_gt,label='Original database',alpha = 0.4,normed=True,bins=bins)
plt.hist(usage_sn,label='Sanitized database w/ learned metric',alpha = 0.4,normed=True,bins=bins)
plt.xlabel('Peak hour energy usage (W)',fontsize=fontsize)
plt.title('Normalized histogram of peak-hour usage',fontsize=fontsize)
plt.legend(fontsize=legendsize)
plt.yticks([0,1e-4,2e-4,3e-4,3.5e-4],['0','1e-4','2e-4','3e-4',''])
plt.show()