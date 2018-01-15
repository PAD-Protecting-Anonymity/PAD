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

anonymity_levels = len(anonymity_level_vec)
arr_mean = np.empty(anonymity_levels)
arr_std = np.empty(anonymity_levels)
arr_mode = np.empty(anonymity_levels)
dep_mean = np.empty(anonymity_levels)
dep_std = np.empty(anonymity_levels)
dep_mode = np.empty(anonymity_levels)
use_mean = np.empty(anonymity_levels)
use_std = np.empty(anonymity_levels)
use_mode = np.empty(anonymity_levels)

arr_mean_rela = np.empty(anonymity_levels)
arr_std_rela = np.empty(anonymity_levels)
arr_mode_rela = np.empty(anonymity_levels)
dep_mean_rela = np.empty(anonymity_levels)
dep_std_rela = np.empty(anonymity_levels)
dep_mode_rela = np.empty(anonymity_levels)
use_mean_rela = np.empty(anonymity_levels)
use_std_rela = np.empty(anonymity_levels)
use_mode_rela = np.empty(anonymity_levels)

stat_gt = OccupancyStatistics(day_profile)
arrival_gt = stat_gt.get_arrival_time(flag=0) / 60
arrival_gt = arrival_gt.dropna()
depart_gt = stat_gt.get_departure_time(flag=0) / 60
depart_gt = depart_gt.dropna()
usage_gt = stat_gt.get_total_usage() / 60
usage_gt = usage_gt.dropna()
# arrival
arr_mean_gt = np.mean(arrival_gt.values)
arr_std_gt = np.std(arrival_gt.values)
arr_mode_gt = scipy.stats.mode(arrival_gt.values)[0][0,0]
# departure
dep_mean_gt = np.mean(depart_gt.values)
dep_std_gt = np.std(depart_gt.values)
dep_mode_gt = scipy.stats.mode(depart_gt.values)[0][0,0]
# usage
use_mean_gt = np.mean(usage_gt.values)
use_std_gt = np.std(usage_gt.values)
use_mode_gt = scipy.stats.mode(usage_gt.values)[0][0,0]

for i in range(len(anonymity_level_vec)):
    anonymity = anonymity_level_vec[i]
    sanitized_profile = sanitized_profile_baseline_list[i]
    sanitized_profile = sanitized_profile.round()
    stat_sn = OccupancyStatistics(sanitized_profile)
    arrival_sn = stat_sn.get_arrival_time(flag=0)/60
    arrival_sn = arrival_sn.dropna()
    depart_sn = stat_sn.get_departure_time(flag=0)/60
    depart_sn = depart_sn.dropna()
    usage_sn = stat_sn.get_total_usage()/60
    usage_sn = usage_sn.dropna()

    arr_mean[i] = np.mean(arrival_sn.values)
    arr_std[i] = np.std(arrival_sn.values)
    arr_mode[i] = scipy.stats.mode(arrival_sn.values)[0][0,0]
    dep_mean[i] = np.mean(depart_sn.values)
    dep_std[i] = np.std(depart_sn.values)
    dep_mode[i] = scipy.stats.mode(depart_sn.values)[0][0,0]
    use_mean[i] = np.mean(usage_sn.values)
    use_std[i] = np.std(usage_sn.values)
    use_mode[i] = scipy.stats.mode(usage_sn.values)[0][0,0]

    arr_mean_rela[i] = np.abs(arr_mean_gt - np.mean(arrival_sn.values))/arr_mean_gt
    # arr_std_rela[i] = np.abs(arr_std_gt - np.std(arrival_sn.values))/arr_std_gt
    # arr_mode_rela[i] = np.abs(arr_mode_gt - scipy.stats.mode(arrival_sn.values)[0][0, 0])/arr_mode_gt
    dep_mean_rela[i] = np.abs(np.mean(depart_sn.values) - dep_mean_gt)/dep_mean_gt
    # dep_std_rela[i] = np.std(depart_sn.values)
    # dep_mode_rela[i] = scipy.stats.mode(depart_sn.values)[0][0, 0]
    use_mean_rela[i] = np.abs(np.mean(usage_sn.values) - use_mean_gt)/use_mean_gt
    # use_std_rela[i] = np.std(usage_sn.values)
    # use_mode_rela[i] = scipy.stats.mode(usage_sn.values)[0][0, 0]

print('relative error of arrival is %s' % arr_mean_rela)
print('relative error of departure is %s' % dep_mean_rela)
print('relative error of usage is %s' %use_mean_rela)

## arrival
#mean
plt.figure()
plt.plot(anonymity_level_vec,arr_mean)
plt.plot((anonymity_level_vec[0],anonymity_level_vec[-1]),(arr_mean_gt,arr_mean_gt))

# var
plt.figure()
plt.plot(anonymity_level_vec,arr_std)
plt.plot((anonymity_level_vec[0],anonymity_level_vec[-1]),(arr_std_gt,arr_std_gt))

# mode
plt.figure()
plt.plot(anonymity_level_vec,arr_mode)
plt.plot((anonymity_level_vec[0],anonymity_level_vec[-1]),(arr_mode_gt,arr_mode_gt))

## departure
#mean
plt.figure()
plt.plot(anonymity_level_vec,dep_mean)
plt.plot((anonymity_level_vec[0],anonymity_level_vec[-1]),(dep_mean_gt,dep_mean_gt))

# var
plt.figure()
plt.plot(anonymity_level_vec,dep_std)
plt.plot((anonymity_level_vec[0],anonymity_level_vec[-1]),(dep_std_gt,dep_std_gt))

# mode
plt.figure()
plt.plot(anonymity_level_vec,dep_mode)
plt.plot((anonymity_level_vec[0],anonymity_level_vec[-1]),(dep_mode_gt,dep_mode_gt))


## usage
#mean
plt.figure()
plt.plot(anonymity_level_vec,use_mean)
plt.plot((anonymity_level_vec[0],anonymity_level_vec[-1]),(use_mean_gt,use_mean_gt))

# var
plt.figure()
plt.plot(anonymity_level_vec,use_std)
plt.plot((anonymity_level_vec[0],anonymity_level_vec[-1]),(use_std_gt,use_std_gt))

# mode
plt.figure()
plt.plot(anonymity_level_vec,use_mode)
plt.plot((anonymity_level_vec[0],anonymity_level_vec[-1]),(use_mode_gt,use_mode_gt))

