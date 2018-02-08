import sys; import os
sys.path.append(os.path.abspath("./"))
import pandas as pd
import numpy as np
from helper import Utilities, PerformanceEvaluation
import matplotlib.pyplot as plt
import pickle
from data_statistics import OccupancyStatistics
import scipy.stats

day_profile = pd.read_pickle('dataset/dataframe_all_binary.pkl')
res = 15
day_profile = day_profile.iloc[:,0::res]
ncols = len(day_profile.columns)
rep_mode = 'mean'
partial_occup_length = int(4 * 60/15)
util = Utilities()
anonymity_level_vec = np.arange(2,8)

with open('result/occup_generic_sanitized.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
   _,sanitized_profile_baseline_list = pickle.load(f)


# 2-anonymization
i = 0
print('anonymization level %s'%anonymity_level_vec[i])
sanitized_profile = sanitized_profile_baseline_list[i]
sanitized_profile = sanitized_profile.round()

stat_gt = OccupancyStatistics(day_profile)
arrival_gt = stat_gt.get_arrival_time(flag=0)/60
arrival_gt = arrival_gt.dropna()
depart_gt = stat_gt.get_departure_time(flag=0)/60
depart_gt = depart_gt.dropna()
usage_gt = stat_gt.get_total_usage()/60
usage_gt = usage_gt.dropna()

stat_sn = OccupancyStatistics(sanitized_profile)
arrival_sn = stat_sn.get_arrival_time(flag=0)/60
arrival_sn = arrival_sn.dropna()
depart_sn = stat_sn.get_departure_time(flag=0)/60
depart_sn = depart_sn.dropna()
usage_sn = stat_sn.get_total_usage()/60
usage_sn = usage_sn.dropna()

##################### Compare the estimated moments ######################

# arrival
arr_mean_gt = np.mean(arrival_gt.values)
print('arrival-ground truth-mean %s'% arr_mean_gt)
arr_mean_2 = np.mean(arrival_sn.values)
print('arrival-2 anonymized-mean %s'% arr_mean_2)
arr_std_gt = np.std(arrival_gt.values)
print('arrival-ground truth-std %s'% arr_std_gt)
arr_std_2 = np.std(arrival_sn.values)
print('arrival-2 anonymized-std %s'% arr_std_2)
arr_mode_gt = scipy.stats.mode(arrival_gt.values)
print('arrival-ground truth-mode %s'% arr_mode_gt[0])
arr_mode_2 = scipy.stats.mode(arrival_sn.values)
print('arrival-2 anonymized-mode %s'% arr_mode_2[0])


# departure
dep_mean_gt = np.mean(depart_gt.values)
print('departure-ground truth-mean %s'% dep_mean_gt)
dep_mean_2 = np.mean(depart_sn.values)
print('departure-2 anonymized-mean %s'% dep_mean_2)
dep_std_gt = np.std(depart_gt.values)
print('departure-ground truth-std %s'% dep_std_gt)
dep_std_2 = np.std(depart_sn.values)
print('departure-2 anonymized-std %s'% dep_std_2)
dep_mode_gt = scipy.stats.mode(depart_gt.values)
print('departure-ground truth-mode %s'% dep_mode_gt[0])
dep_mode_2 = scipy.stats.mode(depart_sn.values)
print('departure-2 anonymized-mode %s'% dep_mode_2[0])

# usage
use_mean_gt = np.mean(usage_gt.values)
print('usage-ground truth-mean %s'% use_mean_gt)
use_mean_2 = np.mean(usage_sn.values)
print('usage-2 anonymized-mean %s'% use_mean_2)
use_std_gt = np.std(usage_gt.values)
print('usage-ground truth-std %s'% use_std_gt)
use_std_2 = np.std(usage_sn.values)
print('usage-2 anonymized-std %s'% use_std_2)
use_mode_gt = scipy.stats.mode(usage_gt.values)
print('usage-ground truth-mode %s'% use_mode_gt[0])
use_mode_2 = scipy.stats.mode(usage_sn.values)
print('usage-2 anonymized-mode %s'% use_mode_2[0])











fontsize = 18
legendsize = 12

# Normalized histograms of arrival times
plt.figure()
plt.hist(arrival_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(arrival_sn,alpha=0.4,label='2-anonymized database',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram of arrival times',fontsize=fontsize)
plt.show()

# Normalized histograms of departure times
plt.figure()
plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(depart_sn,alpha=0.4,label='2-anonymized database',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram of departure times',fontsize=fontsize)
plt.show()

# Normalized histograms of total use time
plt.figure()
plt.hist(usage_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(usage_sn,alpha=0.4,label='2-anonymized database',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hours',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram of total usage',fontsize=fontsize)
plt.show()


# 7-anonymization
i = 5
print('anonymization level %s'%anonymity_level_vec[i])
sanitized_profile = sanitized_profile_baseline_list[i]
sanitized_profile = sanitized_profile.round()

# Normalized histograms of arrival times
stat_gt = OccupancyStatistics(day_profile)
arrival_gt = stat_gt.get_arrival_time(flag=0)/60
arrival_gt = arrival_gt.dropna()
depart_gt = stat_gt.get_departure_time(flag=0)/60
depart_gt = depart_gt.dropna()
usage_gt = stat_gt.get_total_usage()/60
usage_gt = usage_gt.dropna()

stat_sn = OccupancyStatistics(sanitized_profile)
arrival_sn = stat_sn.get_arrival_time(flag=0)/60
arrival_sn = arrival_sn.dropna()
depart_sn = stat_sn.get_departure_time(flag=0)/60
depart_sn = depart_sn.dropna()
usage_sn = stat_sn.get_total_usage()/60
usage_sn = usage_sn.dropna()

# Normalized histograms of arrival times
plt.figure()
plt.hist(arrival_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(arrival_sn,alpha=0.4,label='7-anonymized database',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram of arrival times',fontsize=fontsize)
plt.show()

# Normalized histograms of departure times
plt.figure()
plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(depart_sn,alpha=0.4,label='7-anonymized database',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram of departure times',fontsize=fontsize)
plt.show()

# Normalized histograms of total use time
plt.figure()
plt.hist(usage_gt,alpha=0.4,label='Original database',normed=True,bins=48)
plt.hist(usage_sn,alpha=0.4,label='7-anonymized database',normed=True,bins=48)
plt.legend(fontsize=legendsize)
plt.xlabel('Hours',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram of total usage',fontsize=fontsize)
plt.show()





##################### Compare the estimated moments ######################

# arrival
arr_mean_gt = np.mean(arrival_gt.values)
print('arrival-ground truth-mean %s'% arr_mean_gt)
arr_mean_2 = np.mean(arrival_sn.values)
print('arrival-2 anonymized-mean %s'% arr_mean_2)
arr_std_gt = np.std(arrival_gt.values)
print('arrival-ground truth-std %s'% arr_std_gt)
arr_std_2 = np.std(arrival_sn.values)
print('arrival-2 anonymized-std %s'% arr_std_2)
arr_mode_gt = scipy.stats.mode(arrival_gt.values)
print('arrival-ground truth-mode %s'% arr_mode_gt[0])
arr_mode_2 = scipy.stats.mode(arrival_sn.values)
print('arrival-2 anonymized-mode %s'% arr_mode_2[0])


# departure
dep_mean_gt = np.mean(depart_gt.values)
print('departure-ground truth-mean %s'% dep_mean_gt)
dep_mean_2 = np.mean(depart_sn.values)
print('departure-2 anonymized-mean %s'% dep_mean_2)
dep_std_gt = np.std(depart_gt.values)
print('departure-ground truth-std %s'% dep_std_gt)
dep_std_2 = np.std(depart_sn.values)
print('departure-2 anonymized-std %s'% dep_std_2)
dep_mode_gt = scipy.stats.mode(depart_gt.values)
print('departure-ground truth-mode %s'% dep_mode_gt[0])
dep_mode_2 = scipy.stats.mode(depart_sn.values)
print('departure-2 anonymized-mode %s'% dep_mode_2[0])


# usage
use_mean_gt = np.mean(usage_gt.values)
print('usage-ground truth-mean %s'% use_mean_gt)
use_mean_2 = np.mean(usage_sn.values)
print('usage-2 anonymized-mean %s'% use_mean_2)
use_std_gt = np.std(usage_gt.values)
print('usage-ground truth-std %s'% use_std_gt)
use_std_2 = np.std(usage_sn.values)
print('usage-2 anonymized-std %s'% use_std_2)
use_mode_gt = scipy.stats.mode(usage_gt.values)
print('usage-ground truth-mode %s'% use_mode_gt[0])
use_mode_2 = scipy.stats.mode(usage_sn.values)
print('usage-2 anonymized-mode %s'% use_mode_2[0])



