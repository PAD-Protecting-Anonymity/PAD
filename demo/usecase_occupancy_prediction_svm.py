import sys; import os
sys.path.append(os.path.abspath("./"))
import pandas as pd
from helper import Utilities, PerformanceEvaluation
from sklearn.model_selection import KFold
from sklearn import svm
import pickle
import matplotlib.pyplot as plt
import numpy as np


def SVMPrediciton(data_trn,label_trn,data_tst,label_tst):
    """
    SVM occupancy prediction algorithm
    :param data_trn: training feature
    :param label_trn: training label
    :param data_tst: testing feature
    :param label_tst: testing label
    :return: occupancy prediction accuracy
    """
    if all([label_trn[i] == label_trn[0] for i in range(len(label_trn))]):
        label_pred = label_trn[0] * np.ones(data_tst.shape[0])
        pred_accuracy = sum([1 for i in range(len(label_pred)) if label_pred[i] == label_tst[i]]) / len(label_pred)
        return pred_accuracy
    clf = svm.SVC()
    clf.fit(data_trn, label_trn)
    label_pred = clf.predict(data_tst)
    pred_accuracy = sum([1 for i in range(len(label_pred)) if label_pred[i] == label_tst[i]])/len(label_pred)
    return pred_accuracy

day_profile = pd.read_pickle('dataset/dataframe_all_binary.pkl')
res = 15
day_profile = day_profile.iloc[:,0::res]
ncols = len(day_profile.columns)
rep_mode = 'mean'
partial_occup_length = int(4 * 60/15) # length of occupancy data that is used as features to train knn
util = Utilities()
anonymity_level_vec = np.arange(2,21)
cv_num = 5
eval_time_steps = range(partial_occup_length,ncols)
accuracy_gt_vec = np.empty((len(eval_time_steps),cv_num))
accuracy_sn_vec = np.empty((len(anonymity_level_vec),len(eval_time_steps),cv_num))

# load the sanitized database
with open('result/occup_generic_sanitized(2-20).pickle', 'rb') as f:
   _,sanitized_profile_baseline_list = pickle.load(f)

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
        accuracy_gt = SVMPrediciton(data_trn_gt, label_trn_gt, data_tst, label_tst)
        accuracy_gt_vec[i, cvi] = accuracy_gt
        cvi += 1


for ai in range(len(anonymity_level_vec)):
    anonymity_level = anonymity_level_vec[ai]
    print('anonymity level %s' % anonymity_level_vec[ai])
    sanitized_profile_baseline = sanitized_profile_baseline_list[ai]
    sanitized_profile = sanitized_profile_baseline.round()
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
            accuracy_sn = SVMPrediciton(data_trn_sn,label_trn_sn,data_tst,label_tst)
            accuracy_sn_vec[ai,i,cvi] = accuracy_sn
            cvi += 1
        print("accuracy with sanitized data %s" % accuracy_sn)

    with open('result/usecase_occup_svm_anonymitylevels(2-20)_cv.pickle', 'wb') as f:
        pickle.dump(
            [accuracy_gt_vec,accuracy_sn_vec,eval_time_steps,
             anonymity_level_vec,day_profile,sanitized_profile_baseline_list,cv_num], f)


# ######## visualization ###############
with open('result/usecase_occup_svm_anonymitylevels(2-20)_cv.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    accuracy_gt_vec, accuracy_sn_vec, eval_time_steps, anonymity_level_vec,\
    day_profile, sanitized_profile_baseline_list, cv_num\
        = pickle.load(f)
print('True data: %s' % np.mean(accuracy_gt_vec))
print('Sanitized data: %s' % np.mean(accuracy_sn_vec))

# figure that shows how the prediction accuracy varies with the anonymization level
fontsize = 18
legendsize = 15
plt.figure()
plt.plot(anonymity_level_vec,np.mean(accuracy_gt_vec)*np.ones(len(anonymity_level_vec)),marker='o',label='Original database')
plt.plot(anonymity_level_vec,np.mean(accuracy_sn_vec,axis=(1,2)),marker='o',linestyle = '--', label='Sanitized database')
plt.legend(fontsize=legendsize)
plt.xlabel('Anonymity level', fontsize=fontsize)
plt.ylabel('Prediction accuracy', fontsize=fontsize)
plt.title('Occupancy prediction via SVM', fontsize=fontsize)
plt.ylim((0,1))
plt.xticks(anonymity_level_vec,np.arange(2,21))
plt.show()

# zoomed-in figure
plt.figure()
plt.plot(anonymity_level_vec[0:8],np.mean(accuracy_gt_vec)*np.ones(len(anonymity_level_vec[0:8])),marker='o',label='Original database')
plt.plot(anonymity_level_vec[0:8],np.mean(accuracy_sn_vec[0:8,:,:],axis=(1,2)),marker='o',linestyle = '--', label='Sanitized database')
plt.xticks(anonymity_level_vec[0:8],np.arange(2,21))
plt.show()


