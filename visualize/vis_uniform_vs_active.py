import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd



mc_num = 5
batch_size = 10

with open('./result_scripts/sample_uniform_occupancy.pickle', 'rb') as f:
    _, loss_generic_metric, loss_unif_all_linear, loss_unif_all_deep,\
                 k_init, subsample_size_max, pairdata_all, pairlabel_all = pickle.load(f)

with open('./result_scripts/sample_acitve_occupancy_linear.pickle', 'rb') as f:
    _, loss_generic_metric_ac_linear, loss_active_all_linear,k_init_ac_linear, subsample_size_max_ac_linear, \
    run_sample_size_ac_linear, pairdata_all_linear, pairlabel_all_linear = pickle.load(f)

with open('./result_scripts/sample_acitve_occupancy_deep.pickle', 'rb') as f:
    _, loss_generic_metric_ac_deep, loss_active_all_deep,k_init_ac_deep, subsample_size_max_ac_deep, \
    run_sample_size_ac_deep, pairdata_all_deep, pairlabel_all_deep = pickle.load(f)

# occupancy window best
with open('./result_scripts/window_loss.pickle','rb') as f: # Python 3: open(..., 'wb')
    data = pickle.load(f)

loss_best_metric = data[2]

loss_unif_all_linear = np.asarray(loss_unif_all_linear)
loss_unif_all_linear_mean = np.mean(loss_unif_all_linear,axis=0)
loss_unif_all_linear_std = np.std(loss_unif_all_linear,axis=0)
loss_unif_all_deep = np.asarray(loss_unif_all_deep)
loss_unif_all_deep_mean = np.mean(loss_unif_all_deep,axis=0)
loss_unif_all_deep_std = np.std(loss_unif_all_deep,axis=0)
loss_active_all_linear = np.asarray(loss_active_all_linear)
loss_active_all_linear_mean = np.mean(loss_active_all_linear,axis=0)
loss_active_all_linear_std = np.std(loss_active_all_linear,axis=0)
loss_active_all_deep = np.asarray(loss_active_all_deep)
loss_active_all_deep_mean = np.mean(loss_active_all_deep,axis=0)
loss_active_all_deep_std = np.std(loss_active_all_deep,axis=0)

eval_k = np.arange(k_init,run_sample_size_ac_linear+1,batch_size)

# linear metric
plt.figure()
plt.errorbar(eval_k,loss_unif_all_linear_mean,loss_unif_all_linear_std,label='uniform sampling')
plt.errorbar(eval_k,loss_active_all_linear_mean[eval_k],loss_active_all_linear_std[eval_k],
             label='active sampling')
# plt.errorbar(np.arange(k_init,subsample_size_max+1),loss_active_mean,np.std(loss_iters_active_format),
#              label='active sampling')
# plt.plot(np.arange(k_init,subsample_size_max+1),loss_active_mean, label='active sampling')
# for i in range(mc_num):
#     plt.plot(np.arange(k_init,subsample_size_max+1),loss_iters_active_format[i,:], label='active sample',color='red')
#     plt.plot(eval_k,loss_iters_unif_format[i,:],label='uniform sample', color='blue')
plt.plot((k_init,subsample_size_max),(loss_generic_metric,loss_generic_metric),'b--',label="generic metric")
plt.plot((k_init,subsample_size_max),(loss_best_metric,loss_best_metric),'r--',label="ground truth metric")
plt.legend()
plt.show()