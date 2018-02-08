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
day_profile = pd.read_pickle('dataset/dataframe_all_binary.pkl')
day_profile = day_profile.iloc[0:90,0::15]
print('row num%s' %len(day_profile.index))
ncols = len(day_profile.columns)
rep_mode = 'mean'
partial_occup_length = int(4 * 60/15)


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
day_profile = pd.read_pickle('dataset/dataframe_all_binary.pkl')
day_profile = day_profile.iloc[0:90,0::15]
print('row num%s' %len(day_profile.index))
ncols = len(day_profile.columns)
rep_mode = 'mean'
partial_occup_length = int(4 * 60/15)


dataset_type = "public"

with open('./result_scripts/loss_vs_privacy_occupancy_statistics_public_deep_departure_all.pickle','rb') as f: # Python 3: open(..., 'wb')
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

for i in anonymity_vec:
    sanitized_profile_generic = sanitized_profile_baseline_list[i]
    sanitized_profile_generic = sanitized_profile_generic.round()

    sanitized_profile = sanitized_profile_list[i]
    sanitized_profile = sanitized_profile.round()

    sanitized_profile_deep = sanitized_profile_deep_list[i]
    sanitized_profile_deep = sanitized_profile_deep.round()


    stat_gn = OccupancyStatistics(sanitized_profile_generic)
    depart_gn = stat_gn.get_departure_time(flag=0)/60
    depart_gn = depart_gn.dropna()


    stat_gt = OccupancyStatistics(day_profile)
    depart_gt = stat_gt.get_departure_time(flag=0)/60
    depart_gt = depart_gt.dropna()

    stat_sn = OccupancyStatistics(sanitized_profile)
    depart_sn = stat_sn.get_departure_time(flag=0)/60
    depart_sn = depart_sn.dropna()

    stat_sd = OccupancyStatistics(sanitized_profile_deep)
    depart_sd = stat_sd.get_departure_time(flag=0)/60
    depart_sd = depart_sd.dropna()

    fontsize = 18
    legendsize = 12

    dep_mean_gt = np.mean(depart_gt.values)
    dep_mean_gn = np.mean(depart_gn.values)
    dep_mean_sn = np.mean(depart_sn.values)
    dep_mean_sd = np.mean(depart_sd.values)

    err_gn = abs(dep_mean_gt-dep_mean_gn)/dep_mean_gt
    err_sn = abs(dep_mean_sn-dep_mean_gt)/dep_mean_gt
    err_sd = abs(dep_mean_sd-dep_mean_gt)/dep_mean_gt

    print('-----------------')
    print(i)
    print('generic %s'% err_gn)
    print('linear %s' % err_sn)
    print('deep %s' % err_sd)


plt.figure()
plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(depart_gn,alpha=0.4,label='Santized database w/ generic metric',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram for departure times',fontsize=fontsize)
plt.savefig("visualize/figures/histogram level %s departure time with %s generic.png"%(str(i), dataset_type), bbox_inches='tight',dpi=100)
# plt.xticks(np.arange(0,25,4))
# plt.show()


plt.figure()
plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(depart_sn,alpha=0.4,label='Santized database w/ linear metric',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram for departure times',fontsize=fontsize)
plt.savefig("visualize/figures/histogram level %s departure time with %s linear.png"%(str(i), dataset_type), bbox_inches='tight',dpi=100)
# plt.xticks(np.arange(0,25,4))
# plt.show()

plt.figure()
plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(depart_sd,alpha=0.4,label='Santized database w/ nonlinear metric',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram for departure times',fontsize=fontsize)
plt.savefig("visualize/figures/histogram level %s departure time with %s deep.png"%(str(i), dataset_type), bbox_inches='tight',dpi=100)
# plt.xticks(np.arange(0,25,4))
# plt.show()




##### k = 7
i = 7

sanitized_profile_generic = sanitized_profile_baseline_list[i]
sanitized_profile_generic = sanitized_profile_generic.round()

sanitized_profile = sanitized_profile_baseline_list[i]
sanitized_profile = sanitized_profile.round()
sanitized_profile_deep = sanitized_profile_deep_list[i]
sanitized_profile_deep = sanitized_profile_deep.round()

stat_gn = OccupancyStatistics(day_profile)
depart_gn = stat_gn.get_departure_time(flag=0)/60
depart_gn = depart_gn.dropna()

stat_gt = OccupancyStatistics(day_profile)
depart_gt = stat_gt.get_departure_time(flag=0)/60
depart_gt = depart_gt.dropna()

stat_sn = OccupancyStatistics(sanitized_profile)
depart_sn = stat_sn.get_departure_time(flag=0)/60
depart_sn = depart_sn.dropna()


stat_sd = OccupancyStatistics(sanitized_profile)
depart_sd = stat_sd.get_departure_time(flag=0)/60
depart_sd = depart_sd.dropna()

fontsize = 18
legendsize = 12


plt.figure()
plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(depart_gn,alpha=0.4,label='Santized database w/ generic metric',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram for departure times',fontsize=fontsize)
plt.savefig("visualize/figures/histogram level %s departure time with %s generic.png"%(str(i), dataset_type), bbox_inches='tight',dpi=100)
# plt.xticks(np.arange(0,25,4))
# plt.show()

plt.figure()
plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(depart_sn,alpha=0.4,label='Santized database w/ linear metric',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram for departure times',fontsize=fontsize)
plt.savefig("visualize/figures/histogram level %s departure time with %s linear.png"%(str(i), dataset_type), bbox_inches='tight',dpi=100)
# plt.xticks(np.arange(0,25,4))
# plt.show()

plt.figure()
plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(depart_sd,alpha=0.4,label='Santized database w/ nonlinear metric',normed=True,bins=48)
plt.hist(depart_sd,alpha=0.4,label='',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram for departure times',fontsize=fontsize)
plt.savefig("visualize/figures/histogram level %s departure time with %s deep.png"%(str(i), dataset_type), bbox_inches='tight',dpi=100)
# plt.xticks(np.arange(0,25,4))
# plt.show()et_type = "public"

with open('./result_scripts/loss_vs_privacy_occupancy_statistics_public_deep_departure_all.pickle','rb') as f: # Python 3: open(..., 'wb')
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

i = 3
sanitized_profile_generic = sanitized_profile_baseline_list[i]
sanitized_profile_generic = sanitized_profile_generic.round()

sanitized_profile = sanitized_profile_list[i]
sanitized_profile = sanitized_profile.round()

sanitized_profile_deep = sanitized_profile_deep_list[i]
sanitized_profile_deep = sanitized_profile_deep.round()


stat_gn = OccupancyStatistics(sanitized_profile_generic)
depart_gn = stat_gn.get_departure_time(flag=0)/60
depart_gn = depart_gn.dropna()


stat_gt = OccupancyStatistics(day_profile)
depart_gt = stat_gt.get_departure_time(flag=0)/60
depart_gt = depart_gt.dropna()

stat_sn = OccupancyStatistics(sanitized_profile)
depart_sn = stat_sn.get_departure_time(flag=0)/60
depart_sn = depart_sn.dropna()

stat_sd = OccupancyStatistics(sanitized_profile_deep)
depart_sd = stat_sd.get_departure_time(flag=0)/60
depart_sd = depart_sd.dropna()

fontsize = 18
legendsize = 12

dep_mean_gt = np.mean(day_profile)

plt.figure()
plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(depart_gn,alpha=0.4,label='Santized database w/ generic metric',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram for departure times',fontsize=fontsize)
plt.savefig("visualize/figures/histogram level %s departure time with %s generic.png"%(str(i), dataset_type), bbox_inches='tight',dpi=100)
# plt.xticks(np.arange(0,25,4))
# plt.show()


plt.figure()
plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(depart_sn,alpha=0.4,label='Santized database w/ linear metric',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram for departure times',fontsize=fontsize)
plt.savefig("visualize/figures/histogram level %s departure time with %s linear.png"%(str(i), dataset_type), bbox_inches='tight',dpi=100)
# plt.xticks(np.arange(0,25,4))
# plt.show()

plt.figure()
plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(depart_sd,alpha=0.4,label='Santized database w/ nonlinear metric',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram for departure times',fontsize=fontsize)
plt.savefig("visualize/figures/histogram level %s departure time with %s deep.png"%(str(i), dataset_type), bbox_inches='tight',dpi=100)
# plt.xticks(np.arange(0,25,4))
# plt.show()




##### k = 7
i = 7

sanitized_profile_generic = sanitized_profile_baseline_list[i]
sanitized_profile_generic = sanitized_profile_generic.round()

sanitized_profile = sanitized_profile_baseline_list[i]
sanitized_profile = sanitized_profile.round()
sanitized_profile_deep = sanitized_profile_deep_list[i]
sanitized_profile_deep = sanitized_profile_deep.round()

stat_gn = OccupancyStatistics(day_profile)
depart_gn = stat_gn.get_departure_time(flag=0)/60
depart_gn = depart_gn.dropna()

stat_gt = OccupancyStatistics(day_profile)
depart_gt = stat_gt.get_departure_time(flag=0)/60
depart_gt = depart_gt.dropna()

stat_sn = OccupancyStatistics(sanitized_profile)
depart_sn = stat_sn.get_departure_time(flag=0)/60
depart_sn = depart_sn.dropna()


stat_sd = OccupancyStatistics(sanitized_profile)
depart_sd = stat_sd.get_departure_time(flag=0)/60
depart_sd = depart_sd.dropna()

fontsize = 18
legendsize = 12


plt.figure()
plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(depart_gn,alpha=0.4,label='Santized database w/ generic metric',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram for departure times',fontsize=fontsize)
plt.savefig("visualize/figures/histogram level %s departure time with %s generic.png"%(str(i), dataset_type), bbox_inches='tight',dpi=100)
# plt.xticks(np.arange(0,25,4))
# plt.show()

plt.figure()
plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(depart_sn,alpha=0.4,label='Santized database w/ linear metric',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram for departure times',fontsize=fontsize)
plt.savefig("visualize/figures/histogram level %s departure time with %s linear.png"%(str(i), dataset_type), bbox_inches='tight',dpi=100)
# plt.xticks(np.arange(0,25,4))
# plt.show()

plt.figure()
plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(depart_sd,alpha=0.4,label='Santized database w/ nonlinear metric',normed=True,bins=48)
plt.hist(depart_sd,alpha=0.4,label='',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram for departure times',fontsize=fontsize)
plt.savefig("visualize/figures/histogram level %s departure time with %s deep.png"%(str(i), dataset_type), bbox_inches='tight',dpi=100)
# plt.xticks(np.arange(0,25,4))
# plt.show()