from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
# import pdb
import sys; import os
sys.path.append(os.path.abspath("./"))
from helper import Utilities, PerformanceEvaluation
from _datetime import datetime, time, timedelta
from pandas.tseries.offsets import Hour
from subsampling import Subsampling
from user_feedback import Similarity

# import pdb
# # import matplotlib.pyplot as plt
# import pickle
# from scipy.misc import comb
# import time
# from linear_metric_learning import Linear_Metric

# day_profile = pd.read_pickle('./dataset/dataframe_all_binary.pkl')
# day_profile.index = range(len(day_profile.index))
# day_profile = day_profile.iloc[0::5,:]
# print('row num%s' %len(day_profile.index))
# ncols = len(day_profile.columns)
# nrows = len(day_profile.index)
# npairs = int(comb(nrows,2))
# rep_mode = 'mean'
# statistics_mode = 'segment'
# window = [5,20]

# util = Utilities()
# # mel = ml.MetricLearning()
# lm = Linear_Metric()
# anonymity_level_vec = np.arange(2,8)
# sample_vec = np.arange(100,npairs,100)#np.concatenate((np.array([1]),np.arange(2,21,4)))
# d_vec = [24, 48, 96, 360]#np.arange(20,ncols,200)
# range_n_clusters = range(2,8)
# comp_time = np.empty((len(d_vec),len(sample_vec)))
# lam_vec = [10]
# train_portion = 0.8


# for di in range(len(d_vec)):
#     sp = Subsampling(data=day_profile.iloc[:,0:d_vec[di]])
#     print('sample dimension %s' % d_vec[di])


#     for ni in range(len(sample_vec)):
#         data_pair, data_pair_all_index = sp.uniform_sampling(subsample_size=sample_vec[ni], seed= None)
#         # pdb.set_trace()
#         print('sample number %s' % sample_vec[ni])
#         sim = Similarity(data=data_pair)
#         sim.extract_interested_attribute(interest=statistics_mode,window=window)
#         similarity_label, class_label = sim.label_via_silhouette_analysis(range_n_clusters=range_n_clusters)
#         t0 = time.time()
        
#         lm.train(data_pair, similarity_label)

#         t1 = time.time()
#         t = t1 - t0
#         comp_time[di,ni] = t
#         print('time elapsed %s' % t)

#         with open('./result_scripts/computation_issue_ml_new.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
#             pickle.dump([comp_time], f)





# visualization of computation time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.cm as cm
import pdb
import pandas as pd
from scipy.misc import comb
with open('./result_scripts/computation_issue_ml_new.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    comp_time = pickle.load(f)
comp_time = comp_time[0]

day_profile = pd.read_pickle('./dataset/dataframe_all_binary.pkl')
day_profile.index = range(len(day_profile.index))
day_profile = day_profile.iloc[0::5,:]
print('row num%s' %len(day_profile.index))
ncols = len(day_profile.columns)
nrows = len(day_profile.index)
npairs = int(comb(nrows,2))

# comp_time = comp_time[0:4,:]
sample_vec = np.arange(100,npairs,100)#np.concatenate((np.array([1]),np.arange(2,21,4)))
d_vec = [24, 48, 96, 360]#np.arange(20,ncols,200)

X,Y = np.meshgrid(d_vec,sample_vec)
fontsize = 13
fig = plt.figure()
ax = fig.gca(projection='3d')
color = 'r'
ax.plot_surface(X,Y,comp_time.transpose(),linewidth=0, antialiased=False,
                   alpha=0.3,color = color)
f = mpl.lines.Line2D([0], [0], linestyle="none", color=color, marker='o', alpha=0.3)
ax.set_xlabel('Row dimension',fontsize=fontsize)
ax.set_ylabel('Number of labeled pairs',fontsize=fontsize)
ax.set_zlabel('Computational time (s)',fontsize=fontsize)
# angle = 60
# ax.view_init(30, 150)
plt.savefig("visualize/figures/computational_time.png", bbox_inches='tight',dpi=100)
plt.show()



