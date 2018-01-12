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

with open('./result_scripts/loss_vs_privacy_occupancy_statistics_%s_deep_departure.pickle'%(dataset_type),'rb') as f: # Python 3: open(..., 'wb')
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

i = 2
sanitized_profile = sanitized_profile_baseline_list[i]
sanitized_profile = sanitized_profile.round()

sanitized_profile_deep = sanitized_profile_deep_list[i]
sanitized_profile_deep = sanitized_profile_deep.round()

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


plt.figure()
plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(depart_sn,alpha=0.4,label='2-anonymized database',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram of departure times with linear metric',fontsize=fontsize)
plt.savefig("visualize/figures/histogram level %s departure time with %s linear.png"%(str(i), dataset_type), bbox_inches='tight',dpi=100)
# plt.xticks(np.arange(0,25,4))
# plt.show()

plt.figure()
plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(depart_sd,alpha=0.4,label='2-anonymized database',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram of departure times with deep metric',fontsize=fontsize)
plt.savefig("visualize/figures/histogram level %s departure time with %s deep.png"%(str(i), dataset_type), bbox_inches='tight',dpi=100)
# plt.xticks(np.arange(0,25,4))
# plt.show()




##### k = 7
i = 7
sanitized_profile = sanitized_profile_baseline_list[i]
sanitized_profile = sanitized_profile.round()
sanitized_profile_deep = sanitized_profile_deep_list[i]
sanitized_profile_deep = sanitized_profile_deep.round()

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
plt.hist(depart_sn,alpha=0.4,label='7-anonymized database',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram of departure times with linear metric',fontsize=fontsize)
plt.savefig("visualize/figures/histogram level %s departure time with %s linear.png"%(str(i), dataset_type), bbox_inches='tight',dpi=100)
# plt.xticks(np.arange(0,25,4))
# plt.show()

plt.figure()
plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(depart_sd,alpha=0.4,label='2-anonymized database',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram of departure  with deep metric',fontsize=fontsize)
plt.savefig("visualize/figures/histogram level %s departure time with %s deep.png"%(str(i), dataset_type), bbox_inches='tight',dpi=100)
# plt.xticks(np.arange(0,25,4))
# plt.show()