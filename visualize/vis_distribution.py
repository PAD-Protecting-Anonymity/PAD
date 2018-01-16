import pickle
import matplotlib.pyplot as plt
import numpy as np
from data_statistics import OccupancyStatistics
import pandas as pd
from helper import Utilities, PerformanceEvaluation

# departure
with open('result_scripts/loss_vs_privacy_occupancy_statistics_public_deep_departure_all.pickle', 'rb') as f:
    s, losses, ss = pickle.load(f)

anonymity_vec = range(2, 8)
anonymity_n = len(anonymity_vec)

day_profile1 = pd.read_pickle('./dataset/dataframe_all_binary.pkl')
res = 15
day_profile = day_profile1.iloc[:90,0::res] # subsample the database to improve the speed for demonstration purpose

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



    # dep_mean_gt = np.mean(depart_gt.values)
    # dep_mean_gn = np.mean(depart_gn.values)
    # dep_mean_sn = np.mean(depart_sn.values)
    # dep_mean_sd = np.mean(depart_sd.values)
    #
    # err_gn = abs(dep_mean_gt-dep_mean_gn)/dep_mean_gt
    # err_sn = abs(dep_mean_sn-dep_mean_gt)/dep_mean_gt
    # err_sd = abs(dep_mean_sd-dep_mean_gt)/dep_mean_gt

    print('-----------------')
    print(i)
    print('generic %s'% err_gn)
    print('linear %s' % err_sn)
    print('deep %s' % err_sd)