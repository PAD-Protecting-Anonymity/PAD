import sys; import os
sys.path.append(os.path.abspath("./"))
from helper import Utilities, PerformanceEvaluation
import pandas as pd
from user_feedback import Similarity
from scipy.misc import comb
import numpy as np
import pickle
from subsampling import Subsampling
import pandas as pd
import matplotlib.pyplot as plt
# from deep_metric_learning import Deep_Metric
# from linear_metric_learning import Linear_Metric

"""
In the demo, we will showcase an example of special purpose publication.
The data user wants the published database to maximally retain the information about lunch time.
"""

# Initialization of some useful classes
# util = Utilities()
# pe = PerformanceEvaluation()

# def evaluation_occupancy_window(n):
#     # step 1: get the database to be published
#     day_profile = pd.read_pickle('./dataset/dataframe_all_binary.pkl')
#     day_profile = day_profile.fillna(0)
#     day_profile = day_profile.iloc[0:90,0::60]
#     rep_mode = 'mean'
#     anonymity_level = n # desired anonymity level

#     # step 2: data user specifies his/her interest. In the example, the data user is interested in preserving the
#     # information of a segment of entire time series. In this case, he/she would also need to specify the starting and
#     # ending time of the time series segment of interest.
#     interest = 'segment'
#     window = [11,15] # window specifies the starting and ending time of the period that the data user is interested in

#     # step 3: pre-sanitize the database
#     sanitized_profile_best = util.sanitize_data(day_profile, distance_metric = 'self-defined',
#                                                 anonymity_level = anonymity_level, rep_mode = rep_mode,
#                                                 mode = interest, window = window)
    
#     loss_best_metric = pe.get_information_loss(data_gt=day_profile, data_sanitized=sanitized_profile_best,
#                                                 window=window)
#     return loss_best_metric

# losses = {}
# for n in range(2,8):    
#     losses[n] = evaluation_occupancy_window(n)
    

# with open('result_scripts/window_loss.pickle', 'wb') as f: 
#         pickle.dump(losses, f)

with open('./result_scripts/window_loss.pickle','rb') as f: # Python 3: open(..., 'wb')
    data = pickle.load(f)

print(data)
sr = pd.Series()
for key in data.keys():
    sr = sr.set_value(key, data[key])
print(sr)
sr.plot()
plt.show()





