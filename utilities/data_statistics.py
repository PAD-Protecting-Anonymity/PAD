import pandas as pd
import numpy as np
import pdb

class OccupancyStatistics:
    '''
    Extract ocupancy statistics from the daily profile
    '''
    def __init__(self,data):
        self.data = data
        self.uil = UtilityOccupancyStatistics()

    def get_arrival_time(self,flag):
        arrival_time = self.data.apply(self.uil.compute_arrival_time, axis=1,index=self.data.columns,flag=flag).to_frame()
        return arrival_time

    def get_departure_time(self,flag):
        departure_time = self.data.apply(self.uil.compute_departure_time, axis=1,index=self.data.columns,flag=flag).to_frame()
        return departure_time

    def get_total_usage(self):
        total_usage = self.data.apply(self.uil.compute_total_usage, axis=1,index=self.data.columns).to_frame()
        return total_usage

    def get_window_usage(self,window):
        window_usage = self.data.apply(self.uil.compute_window_usage, axis=1,index=self.data.columns,window=window).to_frame()
        return window_usage

    def get_segment(self, window):
        window_usage = self.data.apply(self.uil.compute_window, axis=1,index=self.data.columns,window=window)
        return window_usage



class UtilityOccupancyStatistics:

    def compute_arrival_time(self, x, index, flag):
        '''
        :param x: daily profile
        :param index: time index (0-1440)
        :return: arrival time
                - nan indicates no arrivals in the day
                - inf indicates constant occupied in the day
                - number indicates the time when people arrive
        '''
        if isinstance(x,pd.Series):
            x = list(x)
        # pdb.set_trace()
        if x[0] == 0:
            arrival_time_ind = next((ind for ind, value in enumerate(x) if value > 0), np.nan)
        elif x[0] > 0:
            late_morning_dep = next((ind for ind, value in enumerate(x) if value == 0), None)
            if late_morning_dep is None:
                arrival_time_ind = np.inf
            else:
                x_sub = x[late_morning_dep:]
                arrival_time_ind_offset = next((ind for ind, value in enumerate(x_sub) if value > 0), None)
                if arrival_time_ind_offset is None:
                    arrival_time_ind = np.nan
                else:
                    arrival_time_ind = late_morning_dep + arrival_time_ind_offset
        # else:
        #     arrival_time_ind = np.nan
        # print(x)
        # print(arrival_time_ind)
        if np.isnan(arrival_time_ind): # if no arrival in the day
            if flag == 0:
                arrival_time = np.nan
            else:
                arrival_time = 0
        elif np.isinf(arrival_time_ind): # if constantly occupied in the day
            arrival_time = 0
        else:
            arrival_time = index[arrival_time_ind]
        return arrival_time

    def compute_departure_time(self,x,index, flag):

        if isinstance(x,pd.Series):
            x = list(x)
        arr_x = np.array(x)
        if x[-1] > 0:
            if x[0] > 0 and len(arr_x[arr_x==0]) > 0:
                dep_time_ind = next((ind for ind,value in enumerate(x) if value == 0), None)
            else:
                dep_time_ind = np.inf
        elif x[-1] == 0:
            x_inv = x[::-1]
            dep_time_ind_inv = next((ind for ind, value in enumerate(x_inv) if value > 0 ), None)

            if dep_time_ind_inv is None:
                dep_time_ind = np.nan
            else:
                dep_time_ind = len(x) - dep_time_ind_inv

        # print(dep_time_ind)
        # if dep_time_ind is None:
        #     pdb.set_trace()
        if np.isinf(dep_time_ind):
            if flag == 0:
                dep_time = np.nan
            else:
                dep_time = index.max()+1  # no departure in the day
        elif np.isnan(dep_time_ind):
            if flag == 0:
                dep_time = np.nan#-1  # no arrival as well as departure in the day
            else:
                dep_time = 0
        else:
            dep_time = index[dep_time_ind]
        return dep_time

    def compute_total_usage(self,x,index):
        time_resolution = index[1]-index[0]
        if isinstance(x,pd.DataFrame):
            x = list(x)
        usage = sum(x)* time_resolution
        return usage

    def compute_window_usage(self,x,index,window):
        time_resolution = index[1]-index[0]
        if isinstance(x,pd.DataFrame):
            x = list(x)
        usage = sum(x[window[0]:window[1]])* time_resolution
        return usage

    def compute_window(self,x,index, window):
        win_start = window[0]
        win_end = window[1]
        if isinstance(x,pd.DataFrame):
            x = list(x)
        df = x[win_start:win_end]
        return df





