import sys; import os
sys.path.append(os.path.abspath("./"))
import pandas as pd
from helper import Utilities

util = Utilities()

# step1: get original database to be published
day_profile = pd.read_pickle('dataset/dataframe_all_binary.pkl')
print(day_profile)

# (optional) subsample the time series in each raw of the database
res = 10
day_profile = day_profile.iloc[0::5, 0::res]
#print(day_profile)
#exit()

# step2: specify the desired anonymity level
anonymity_level = 5

# util.sanitize_data will privatize the database according to the desired anonymity level
sanitized_profile = util.sanitize_data(day_profile, distance_metric='euclidean',
                                       anonymity_level=anonymity_level,rep_mode ='mean')
sanitized_profile[sanitized_profile<0.5] = 0
sanitized_profile[sanitized_profile>=0.5] = 1

sanitized_profile.to_csv("sanitized_data.csv")