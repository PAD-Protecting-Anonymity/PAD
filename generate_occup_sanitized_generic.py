import pandas as pd
import numpy as np
import pickle
from helper import Utilities



day_profile = pd.read_pickle('./dataset/dataframe_all_binary.pkl')
res = 15
day_profile = day_profile.iloc[:,0::res]
rep_mode = 'mean'
util = Utilities()

anonymity_level_vec = np.arange(2,21)

sanitized_profile_baseline_list = []
for i in range(len(anonymity_level_vec)):
    anonymity_level = anonymity_level_vec[i]
    sanitized_profile_baseline = util.sanitize_data(day_profile, distance_metric='euclidean',
                                                    anonymity_level=anonymity_level,rep_mode = rep_mode)
    sanitized_profile_baseline_list.append(sanitized_profile_baseline)
    with open('./result/occup_generic_sanitized(2-20).pickle', 'wb') as f:
        pickle.dump([day_profile,sanitized_profile_baseline_list], f)
    print(i)
