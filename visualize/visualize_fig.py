import matplotlib.pyplot as plt
import pickle
import numpy as np

class Visualize:
    def infoloss_vs_level(self, title, file_path):
        with open(file_path,'rb') as f: # Python 3: open(..., 'wb')
            data = pickle.load(f)

        s, l, ss = data
        # print(s, l, ss)
        # sanitized_profile_best, sanitized_profile_baseline, sanitized_profile, sanitized_profile_deep = s                                    
        # loss_best_metric, loss_generic_metric, loss_learned_metric, loss_learned_metric_deep = l    

        # print(len(s[2][0]))
        # exit()
        # print(s)
        loss_best_metric_list = {}
        loss_generic_metric_list = {}
        loss_learned_metric_list = {}
        loss_learned_metric_deep_list = {}    
        for i in l.keys():
            # print(i)
            loss_best_metric_list[i], loss_generic_metric_list[i], loss_learned_metric_list[i], loss_learned_metric_deep_list[i] = l[i]
        # exit()
        loss_best_metric_list = list(loss_best_metric_list.values())
        loss_generic_metric_list = list(loss_generic_metric_list.values())
        anonymity_vec = list(l.keys())
        # print(loss_generic_metric_list)
        # exit()
        fontsize = 18
        legendsize = 12
        loss_learned_metric_lists = [list(loss_learned_metric_list[i].values()) for i in loss_learned_metric_list.keys()]
        loss_learned_deep_metric_lists = [list(loss_learned_metric_deep_list[i].values())for i in loss_learned_metric_deep_list.keys()]
        # print(loss_learned_deep_metric_lists)
        # exit()

        plt.plot(anonymity_vec,loss_best_metric_list, label='Ground truth metric',color='red')
        plt.plot(anonymity_vec,loss_generic_metric_list,label='Generic metric',color='blue',linestyle='-.')
        bp = plt.boxplot(loss_learned_metric_lists,positions=anonymity_vec,patch_artist=True,widths=0.1)
        bp1 = plt.boxplot(loss_learned_deep_metric_lists,positions=anonymity_vec,patch_artist=True,widths=0.1)
        fill_color = 'orange'
        edge_color = 'orange'
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=edge_color)

        for patch in bp['boxes']:
            patch.set(facecolor=fill_color, alpha=0.5)

        fill_color = 'lightgreen'
        edge_color = 'lightgreen'
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp1[element], color=edge_color)

        for patch in bp1['boxes']:
            patch.set(facecolor=fill_color, alpha=0.5)

        plt.plot(anonymity_vec,np.mean(loss_learned_metric_lists,axis=1),label='Learned metric',color='orange',linestyle='--')
        plt.plot(anonymity_vec,np.mean(loss_learned_deep_metric_lists,axis=1),label='Deep Learned metric',color='lightgreen',linestyle='--')
        plt.xlabel('Anonymity level',fontsize=fontsize)
        plt.ylabel('Information loss (W)',fontsize=fontsize)
        # plt.title(title,fontsize=fontsize-2)
        plt.legend(fontsize=legendsize, loc='upper left')
        plt.savefig("visualize/figures/%s.png"%title[28:], bbox_inches='tight',dpi=100)
        plt.close()