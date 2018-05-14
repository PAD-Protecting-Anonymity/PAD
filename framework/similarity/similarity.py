import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Similarity:
    """
    Data user: Find the similarity labels given a small set of data pairs
    """
    def __init__(self,data):
        self.pair = data
        self.pair_index = [(x[0].name, x[1].name) for x in self.pair]
        dataSubsample_rep = pd.DataFrame()
        for x in data:
            dataSubsample_rep = dataSubsample_rep.append(x[0])
            dataSubsample_rep = dataSubsample_rep.append(x[1])
        self.dataSubsample = \
            dataSubsample_rep.reset_index().drop_duplicates(subset='index',keep='first').set_index('index')
        # mapping from data index to its row # in the data frame
        dataSubsample_index = self.dataSubsample.index
        self.unique_index = dict()
        for i in range(len(self.dataSubsample)):
            self.unique_index[dataSubsample_index[i]] = i
        self.data_interested = None

    def extract_interested_attribute(self,similarities):
        data_interested = None
        for similarity in similarities:
            temp_data_interested = similarity.get_statistics(self.dataSubsample)
            if data_interested is not None:
                data_interested = data_interested + temp_data_interested
            else:
                data_interested = temp_data_interested
        self.data_interested = data_interested
        # if interest == "segment":
        #     window = kwargs['window']
        #     self.data_interested = self.dataSubsample.values[:,window[0]:window[1]]
        # elif interest == "statistics":
        #     stat_type = kwargs['stat_type']
        #     stat = OccupancyStatistics(self.dataSubsample)
        #     if stat_type == "arrival":
        #         self.data_interested = stat.get_arrival_time(flag=1)
        #     elif stat_type == "departure":
        #         self.data_interested = stat.get_departure_time(flag=1)
        #     elif stat_type == "usage":
        #         self.data_interested = stat.get_total_usage()
        #     elif stat_type == "window-usage":
        #         window = kwargs['window']
        #         self.data_interested = stat.get_window_usage(window=window)

    def label_via_silhouette_analysis(self,range_n_clusters, seed=None):
        cluster_labels = []
        silhouette_avg = []
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, random_state=seed)
            cluster_labels_current = clusterer.fit_predict(self.data_interested) #Problem, in output, sometimes only have one cluster label
            cluster_labels.append(cluster_labels_current)
            if np.sum(cluster_labels_current) == 0:
                continue
            silhouette_avg_current = silhouette_score(self.data_interested,cluster_labels_current)
            print("For n_clusters =", n_clusters,
                "The average silhouette_score is :", silhouette_avg_current)
            silhouette_avg.append(silhouette_avg_current)
        if silhouette_avg == []:
            return [], self.dataSubsample
        else:
            best_n_clusters_index = np.where(silhouette_avg == max(silhouette_avg))[0]
        best_cluster_label = cluster_labels[int(best_n_clusters_index[0])]
        similarity_label = []
        for ind1,ind2 in self.pair_index:
            if best_cluster_label[self.unique_index[ind1]] == best_cluster_label[self.unique_index[ind2]]:
                similarity_label.append(1)
            else:
                similarity_label.append(0)
        return similarity_label, self.dataSubsample
