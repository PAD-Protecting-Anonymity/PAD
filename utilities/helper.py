import numpy as np
import pandas as pd
import scipy as sp

from kward import K_ward
from data_statistics import OccupancyStatistics

np.random.seed(0)


class Utilities:

    def get_daily_profile_data(self, sr):
        """
        Convert time-series data into daily profiles
        """
        df = pd.DataFrame(sr)
        df['timeIndex'] = sr.index.hour * 60 + sr.index.minute
        df['dayIndex'] = sr.index.date
        df.columns.values[0] = 'occup'
        day_profile = df.pivot_table(index='dayIndex', columns='timeIndex', values='occup')
        day_profile = day_profile.reindex_axis(sorted(day_profile.columns), axis=1)
        day_profile = day_profile.dropna(axis=0)
        day_profile.index = range(len(day_profile))
        return day_profile

    def sanitize_data(self, merged_data, distance_metric, anonymity_level, rep_mode, **kwargs):
        k_ward = K_ward(merged_data, distance_metric=distance_metric, rep_mode = rep_mode,
                        k=anonymity_level, **kwargs)
        k_ward.get_cluster()
        groups = k_ward.groups

        sanitized_df = pd.DataFrame()
        for group in groups:
            sanitized_value = group.rep.to_frame().transpose()
            keys = group.get_member_ids()
            for key in keys:
                sanitized_value.index = [key]
                sanitized_df = sanitized_df.append(sanitized_value)

        sanitized_df.columns = merged_data.columns
        return sanitized_df

    def sanitize_data_deep(self, merged_data, deep_data, distance_metric, anonymity_level, rep_mode, **kwargs):
        k_ward = K_ward(deep_data, distance_metric=distance_metric, rep_mode = rep_mode,
                        k=anonymity_level, **kwargs)
        k_ward.get_cluster()
        groups = k_ward.groups


        sanitized_df = pd.DataFrame()
        for group in groups:
            sanitized_value = group.rep.to_frame().transpose()
            keys = group.get_member_ids()

            merged_data.loc[keys]
            sanitized_value = merged_data.loc[keys].mean()
            for key in keys:
                sanitized_df[key] = sanitized_value

        sanitized_df = sanitized_df.transpose()
        return sanitized_df

    def process_data(self, df_true,anonymity_level):
        """
        Process the data for subsampling purpose
        :param df_true: original database
        :param anonymity_level: anonymity level
        :return: processed database
        """
        df_shuffled = df_true.sample(frac=1).reset_index(drop=True)
        df_agg = df_shuffled.groupby(np.arange(len(df_shuffled))//anonymity_level).mean()
        return df_agg


class PerformanceEvaluation:
    def get_information_loss(self, data_gt, data_sanitized, window):
        win_start = window[0]
        win_end = window[1]
        df = data_gt.iloc[:, win_start:win_end] - data_sanitized.iloc[:, win_start:win_end]
        err = df.apply(np.square)
        err_sum = err.apply(np.sum, axis=0)
        err_sum_sqrt = err_sum.apply(np.sqrt)
        return np.mean(err_sum_sqrt)

    def get_statistics_loss(self, data_gt, data_sanitized, mode,**kwargs):
        os1 = OccupancyStatistics(data_gt)
        os2 = OccupancyStatistics(data_sanitized)
        if mode == "arrival":
            stat_gt = os1.get_arrival_time(flag=1)
            stat_sanitized = os2.get_arrival_time(flag=1)
            df = stat_gt - stat_sanitized
        elif mode == "departure":
            stat_gt = os1.get_departure_time(flag=1)
            stat_sanitized = os2.get_departure_time(flag=1)
            df = stat_gt - stat_sanitized
        elif mode == "usage":
            stat_gt = os1.get_total_usage()
            stat_sanitized = os2.get_total_usage()
            df = stat_gt - stat_sanitized
        elif mode == "window-usage":
            window = kwargs['window']
            stat_gt = os1.get_window_usage(window=window)
            stat_sanitized = os2.get_window_usage(window=window)
            df = stat_gt - stat_sanitized
        elif mode == "segment":
            window = kwargs['window']
            stat_gt = os1.get_segment(window=window)
            stat_sanitized = os2.get_segment(window=window)
            # print(stat_gt, stat_sanitized)
            df = pd.Series()
            for i in range(len(stat_gt)):
                df = df.set_value(i, np.linalg.norm(stat_gt.iloc[i,:] - stat_sanitized.iloc[i,:]))
            # print(df)
        df = df.as_matrix()
        err_sum_sqrt = np.mean(np.absolute(df))
        return err_sum_sqrt


class Miscellaneous:
    def accuracy(self,true_label,assigned_label):
        n_instances = len(true_label)
        acc_vec = [1*(1*(true_label[i]==true_label[j]) == 1*(assigned_label[i]==assigned_label[j])) \
                   for i in range(n_instances) for j in range(i+1,n_instances)]
        acc = sum(acc_vec)/(0.5*n_instances*(n_instances-1))
        return acc

    def PSDize(self,A):
        eigen_value, eigen_vector = sp.linalg.eigh(A)
        eigen_value[eigen_value < 0] = 0
        A_psd = eigen_vector.dot(np.diag(eigen_value)).dot(eigen_vector)
        return A_psd




