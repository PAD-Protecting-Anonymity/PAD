import sys; import os
sys.path.append(os.path.abspath("./"))
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from helper import Utilities


def KNNPrediciton(data_trn,label_trn,data_tst,label_tst,n_neighbors):
    """
    KNN occupancy prediction algorithm
    :param data_trn: training feature
    :param label_trn: training label
    :param data_tst: testing feature
    :param label_tst: testing label
    :param n_neighbors: the value of K for KNN algorithm
    :return: occupancy prediction accuracy
    """
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(data_trn,label_trn)
    label_pred = neigh.predict(data_tst)
    pred_accuracy = sum([1 for i in range(len(label_pred)) if label_pred[i] == label_tst[i]])/len(label_pred)
    return pred_accuracy


day_profile = pd.read_pickle('dataset/dataframe_all_binary.pkl')
res = 15
day_profile = day_profile.iloc[:,0::res] # sample the time series data every res minutes
ncols = len(day_profile.columns)
rep_mode = 'mean'
partial_occup_length = int(4 * 60/15) # length of occupancy data that is used as features to train knn
util = Utilities()
anonymity_level_vec = np.arange(2,21)
knn_vec = np.arange(1,10)
cv_num = 5
eval_time_steps = range(partial_occup_length,ncols)
# occupancy prediction accuracy if original occupancy database is used for training
accuracy_gt_vec = np.empty((len(knn_vec),len(eval_time_steps),cv_num))
# occupancy prediction accuracy if sanitized occupancy database is used for training
accuracy_sn_vec = np.empty((len(anonymity_level_vec),len(knn_vec),len(eval_time_steps),cv_num))

# load the sanitized database
sanitized_profile_baseline_list = []
with open('result/occup_generic_sanitized(2-20).pickle', 'rb') as f:
    _,sanitized_profile_baseline_list = pickle.load(f)

for ki in range(len(knn_vec)):
    n_neighbors = knn_vec[ki]
    for i in range(len(eval_time_steps)):
        t = eval_time_steps[i]
        trn_s = t - partial_occup_length
        trn_e = t
        kf = KFold(n_splits=5, random_state=0, shuffle=False)
        X = day_profile.iloc[:, trn_s:trn_e].values
        y = day_profile.iloc[:, trn_e].values
        cvi = 0
        for train_index, test_index in kf.split(X):
            data_trn_gt, data_tst = X[train_index], X[test_index]
            label_trn_gt, label_tst = y[train_index], y[test_index]
            accuracy_gt = KNNPrediciton(data_trn_gt, label_trn_gt, data_tst, label_tst, n_neighbors)
            accuracy_gt_vec[ki, i, cvi] = accuracy_gt
            cvi += 1

for ai in range(len(anonymity_level_vec)):
    anonymity_level = anonymity_level_vec[ai]
    sanitized_profile_baseline = sanitized_profile_baseline_list[ai]
    sanitized_profile = sanitized_profile_baseline.round()

    for ki in range(len(knn_vec)):
        n_neighbors = knn_vec[ki]
        for i in range(len(eval_time_steps)):
            t = eval_time_steps[i]
            trn_s = t-partial_occup_length
            trn_e = t

            kf = KFold(n_splits=5,random_state=0, shuffle=False)
            X = day_profile.iloc[:,trn_s:trn_e].values
            y = day_profile.iloc[:,trn_e].values
            Xs = sanitized_profile.iloc[:,trn_s:trn_e].values
            ys = sanitized_profile.iloc[:,trn_e].values
            cvi = 0
            for train_index, test_index in kf.split(X):
                data_tst = X[test_index]
                label_tst = y[test_index]
                data_trn_sn = Xs[train_index]
                label_trn_sn = ys[train_index]
                accuracy_sn = KNNPrediciton(data_trn_sn,label_trn_sn,data_tst,label_tst,n_neighbors)
                accuracy_sn_vec[ai,ki,i,cvi] = accuracy_sn
                cvi += 1

        with open('result/usecase_occup_knn_anonymitylevels(2-20)_cv.pickle', 'wb') as f:
            pickle.dump(
                [accuracy_gt_vec,accuracy_sn_vec,eval_time_steps,
                 anonymity_level_vec,day_profile,sanitized_profile_baseline_list,
                 knn_vec,cv_num], f)
    print(ai)


######### visualization ###############
with open('result/usecase_occup_knn_anonymitylevels(2-20)_cv.pickle', 'rb') as f:
    accuracy_gt_vec, accuracy_sn_vec, eval_time_steps, anonymity_level_vec,\
    day_profile, sanitized_profile_baseline_list, knn_vec, cv_num\
        = pickle.load(f)
print('True data: %s' %np.mean(accuracy_gt_vec))
print('Sanitized data: %s' %np.mean(accuracy_sn_vec))


# figure that shows how the prediction accuracy varies with the anonymization level
i = 2 # the index of the value of k for knn algorithm in knn_vec
fontsize = 18
legendsize = 15
print('knn neighbor index %s'%knn_vec[i])
plt.plot(anonymity_level_vec,np.mean(accuracy_gt_vec[i,:,:])*np.ones(len(anonymity_level_vec)),marker='o',label='Original database')
plt.plot(anonymity_level_vec,np.mean(accuracy_sn_vec[:,i,:,:],axis=(1,2)),marker='o',linestyle = '--', label='Sanitized database')
plt.legend(loc=3, fontsize=legendsize)
plt.xlabel('Anonymity level', fontsize=fontsize)
plt.ylabel('Prediction accuracy', fontsize=fontsize)
plt.title('Occupancy prediction via kNN (k=3)', fontsize=fontsize)
plt.xticks(anonymity_level_vec,np.arange(2,21))
plt.ylim((0,1))
plt.show()

# zoomed-in figure
plt.figure()
plt.plot(anonymity_level_vec[0:8],np.mean(accuracy_gt_vec[i,:,:])*np.ones(len(anonymity_level_vec[0:8])),marker='o',label='Original database')
plt.plot(anonymity_level_vec[0:8],np.mean(accuracy_sn_vec[0:8,i,:,:],axis=(1,2)),marker='o',linestyle = '--', label='Sanitized database')
plt.xticks(anonymity_level_vec[0:8],np.arange(2,21))
plt.show()


