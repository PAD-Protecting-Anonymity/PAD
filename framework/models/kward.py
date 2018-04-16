import numpy as np
import pandas as pd
from framework.models.basemodel import BaseModel
# np.random.seed(0)


class K_ward(BaseModel):
    """
    K-ward algorithm
    """
    def __init__(self, data,k,rep_mode,**kwargs):
        super().__init__(data)
        self.k = k # anonymity level
        self.group_num = 0
        self.upperbound = 2*self.k
        self.lowerbound = self.k
        # self.distance_metric_type = distance_metric_type
        # if self.distance_metric == 'self-defined':
        #     self.stat_util = Distance()
        self.rep_mode = rep_mode
        self.cards = []
        self.kwargs = kwargs

        # if distance_metric == 'mahalanobis':
        #     self.VI = kwargs['VI']
        # if distance_metric == 'deep':
        #     self.dm = kwargs['deep_model']
        # if distance_metric == 'self-defined':
        #     self.mode = kwargs['mode']
        #     if self.mode == 'window-usage':
        #         self.window = kwargs['window']
        #     elif self.mode == 'segment':
        #         self.window = kwargs['window']

    def _get_distance(self, d_profile_data):
        """
        Given the day profile, return the pairwise distance between each of the two individual series
        """
        df = self.kwargs["metric"].get_distance(d_profile_data)
        return df

    def _add_group(self,group):
        self._groups.append(group)
        self.group_num += 1

    def _initialize_groups(self):
        # boolean variable indicating if each individual has been assigned or not
        group_assign_status = dict(zip(self.data.index,np.zeros(self.datasize)))
        pairwise_dist = self._get_distance(self.data)
        max_dist = pairwise_dist[pairwise_dist["distance"] == pairwise_dist["distance"].max()]
        max_dist = max_dist.iloc[-1, :]
        extreme_points = [max_dist["x"], max_dist["y"]]
        for extreme_p in extreme_points:

            # select the neighbors of the extreme point
            df_sub = pairwise_dist[pairwise_dist["x"]== extreme_p]

            # if the neighbor has been assigned to a group, then delete
            non_assigned_ind = [i for i in range(len(df_sub)) if group_assign_status[int(df_sub["y"].iloc[i])] == 0]
            df_sub = df_sub.iloc[non_assigned_ind]

            if df_sub.shape[0] == 0:
                print("df_sub error")

            # select the k-1 nearest neighbors
            df_near_neighbors = df_sub.sort_values("distance",ascending=True).iloc[0:self.k-1]

            keys = np.concatenate([df_near_neighbors["y"].values, np.array([int(extreme_p)])])
            values = self.data.loc[keys].as_matrix()

            new_group = Group(keys,values,self.rep_mode)

            for i in range(len(keys)):
                group_assign_status[keys[i]] = 1
            self._add_group(new_group)

        for key in group_assign_status.keys():
            if group_assign_status[key] == 0:
                new_group = Group(id=[key], data=self.data.loc[key].as_matrix(), rep_mode=self.rep_mode)
                self._add_group(new_group)

        self.cards = self._get_cards()

    def _get_cards(self):
        cards = [self._groups[i].get_card() for i in range(self.group_num)]
        return cards

    def _compare_and_merge(self):
        cards = self.cards
        centroid_df = pd.DataFrame()
        for i in range(self.group_num):
            centroid_df = centroid_df.append(self._groups[i].rep,ignore_index=True)

        pairwise_dist = self._get_distance(centroid_df)
        # never merge two groups that both have size >= k
        good_keys = [i for i in range(len(cards)) if cards[i]<self.lowerbound]

        #guaratee that at least one group has cardinality <k
        pairwise_dist_good = pairwise_dist[np.logical_or(pairwise_dist['x'].isin(good_keys), pairwise_dist['y'].isin(good_keys))]

        min_dist = pairwise_dist_good[pairwise_dist_good["distance"] == pairwise_dist_good["distance"].min()]
        min_dist = min_dist.iloc[-1,:]
        group1_id = int(min_dist["x"])
        group2_id = int(min_dist["y"])
        self._merge(group_id1=group1_id,group_id2=group2_id)

    def _merge(self,group_id1,group_id2):
        group1 = self._groups[group_id1]
        group2 = self._groups[group_id2]
        group1.merge_group(group2)
        self.cards[group_id1] = group1.get_card()
        del self._groups[group_id2]
        del self.cards[group_id2]
        self.group_num -= 1

    def _replace(self,group_id,groups):
        groups_num = len(groups)
        for i in range(groups_num):
            self._groups.append(groups[i])
            self.cards.append(groups[i].get_card())
        del self._groups[group_id]
        del self.cards[group_id]
        self.group_num = self.group_num - 1 + groups_num

    def find_clusters(self):
        self._initialize_groups()
        i = 0
        while all(card >= self.lowerbound for card in self.cards) is False:
            self._compare_and_merge()

        card_status = [card >= self.upperbound for card in self.cards]
        while any(card_status):
            recurse_ids = [i for i in range(len(card_status)) if card_status[i] == True]
            recurse_id = recurse_ids[0]

            recurse_group = self._groups[recurse_id]
            recurse_df = recurse_group.get_dataframe()
            if len(recurse_df) < self.upperbound:
                print("error")
            recurse_kward = K_ward(data=recurse_df,k=self.k,
                                   rep_mode = self.rep_mode,**self.kwargs)
            recurse_kward.find_clusters()
            new_groups = recurse_kward.get_groups()
            self._replace(recurse_id,new_groups)
            card_status = [card >= self.upperbound for card in self.cards]

class Group:
    def __init__(self, id=None, data=None, rep_mode="mean"):
        self.member = {}

        if id is not None:
            if len(id) > 1:
                self.member = dict(zip(id, data))
            else:
                self.member[id[0]] = data
        self.card = len(self.member)
        self.rep_mode = rep_mode
        self.rep = self.get_rep()

    def get_members(self):
        return self.member

    def get_card(self):
        return self.card

    def get_member_ids(self):
            return list(self.member.keys())

    def get_rep(self):
        df = pd.DataFrame().from_dict(self.member, orient='index', dtype="float64")
        if self.rep_mode == 'mean':
            rep =  df.mean(axis=0)
        elif self.rep_mode == 'mean-round':
            rep = df.mean(axis=0).round()
        elif self.rep_mode == 'median':
            random_num = np.random.uniform(0,1,(1,len(df.columns)))[0]
            random_ind = []
            for i in range(len(random_num)):
                if random_num[i]>0.5:
                    random_ind.append(-0.1)
                else:
                    random_ind.append(0.1)
            rep = np.around(df.median(axis=0) + random_ind)
        elif self.rep_mode == 'max':
            rep =  df.max(axis=0)
        return rep

    def get_member_data(self, identify):
        return self.member[identify]

    def merge_group(self, group):
        new_members = group.get_members()
        self.member = {**self.member,**new_members}

        self.card = self.card + group.get_card()
        self.rep = self.get_rep()

    def get_dataframe(self):
        df = pd.DataFrame().from_dict(self.member).transpose()
        return df

# class Distance:
#     def get_statistic_distance(self,x_df1,x_df2,index,mode,**kwargs):
#         util = UtilityOccupancyStatistics()
#         if mode == "arrival":
#             stat1 = util.compute_arrival_time(x_df1,index,1)
#             stat2 = util.compute_arrival_time(x_df2,index,1)
#             dist = abs(stat1-stat2)
#         elif mode == "departure":
#             stat1 = util.compute_departure_time(x_df1, index,1)
#             stat2 = util.compute_departure_time(x_df2, index,1)
#             dist = abs(stat1-stat2)
#         elif mode == "usage":
#             stat1 = util.compute_total_usage(x_df1, index)
#             stat2 = util.compute_total_usage(x_df2, index)
#             dist = abs(stat1-stat2)
#         elif mode == "window-usage":
#             window = kwargs['window']
#             stat1 = util.compute_window_usage(x_df1, index,window)
#             stat2 = util.compute_window_usage(x_df2, index,window)
#             dist = abs(stat1-stat2)
#         elif mode == "segment":
#             window = kwargs['window']
#             stat1 = util.compute_window(x_df1, index,window)
#             stat2 = util.compute_window(x_df2, index,window)
#             dist = np.linalg.norm(stat1-stat2)
#         return dist
