import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd



mc_num = 5
batch_size = 10

with open('./result_scripts/sample_uniform_occupancy.pickle', 'rb') as f:
    _, loss_generic_metric, loss_unif_all_linear_list, loss_unif_all_deep,\
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
iter_finished = 5

loss_unif_all_linear = np.asarray(loss_unif_all_linear_list)[0:iter_finished,:]
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

eval_k = np.arange(k_init,251,batch_size)

# linear metric
# plt.figure()
# bp = plt.boxplot(loss_unif_all_linear[:,0:len(eval_k)],positions=eval_k,patch_artist=True,widths=1)
# fill_color = 'lightgreen'
# edge_color = 'green'
# for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
#     plt.setp(bp[element], color=edge_color)
#
# for patch in bp['boxes']:
#     patch.set(facecolor=fill_color, alpha=0.5)
# plt.plot(eval_k,loss_unif_all_linear_mean[0:len(eval_k)],label='Learned metric - uniform',
#          color='lightgreen',linestyle='--')
plt.errorbar(eval_k,loss_unif_all_linear_mean[0:len(eval_k)],loss_unif_all_linear_std[0:len(eval_k)],label='Uniform sampling',fmt='o',color='orange')

plt.errorbar(eval_k,loss_active_all_linear_mean[eval_k],loss_active_all_linear_std[eval_k],
             label='Active sampling',fmt='d',color='green')
plt.plot(eval_k,loss_unif_all_linear[0,0:len(eval_k)],'--',color='orange')
plt.plot(eval_k,loss_active_all_linear_mean[eval_k],'--',color='green')
plt.plot((k_init,251),(loss_generic_metric,loss_generic_metric),'b',label="Generic metric")
plt.plot((k_init,251),(loss_best_metric,loss_best_metric),'r',label="Ground truth metric")
plt.xlabel('Number of labeled data pairs')
plt.ylabel('Information loss')
plt.title('Comparison of sample efficiency')
plt.legend()
plt.show()


plt.errorbar(eval_k,loss_unif_all_deep_mean[0:len(eval_k)],loss_unif_all_deep_std[0:len(eval_k)],label='Uniform sampling',fmt='o',color='orange')

plt.errorbar(eval_k,loss_active_all_deep_mean[eval_k],loss_active_all_deep_std[eval_k],
             label='Active sampling',fmt='d',color='green')
plt.plot(eval_k,loss_unif_all_deep[0,0:len(eval_k)],'--',color='orange')
plt.plot(eval_k,loss_active_all_deep_mean[eval_k],'--',color='green')
plt.plot((k_init,251),(loss_generic_metric,loss_generic_metric),'b',label="Generic metric")
plt.plot((k_init,251),(loss_best_metric,loss_best_metric),'r',label="Ground truth metric")
plt.xlabel('Number of labeled data pairs')
plt.ylabel('Information loss')
plt.title('Comparison of sample efficiency')
plt.legend()
plt.show()

