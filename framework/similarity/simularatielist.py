import numpy as np

class SimularatieList:
    def __init__(self):
        self.simularaties = []
    
    def get_distance(self,data):
        distance = None
        for simularatie in self.simularaties:
            tempdistance = simularatie.get_distance(self._get_data_slice(simularatie,data))
            if distance is not None:
                distance['distance'] = distance['distance'] + tempdistance['distance']
            else:
                distance = tempdistance
        distance = distance.sort_values('distance')
        return distance

    def add_simularatie(self,simularatie):
        self.simularaties.append(simularatie)

    def get_statistics_loss(self,data_org, data_sanitized):
        err_sum_sqrt = None
        for simularatie in self.simularaties:
            temp_err_sum_sqrt = simularatie.get_information_loss(self._get_data_slice(simularatie,data_org),self._get_data_slice(simularatie,data_sanitized))
            if err_sum_sqrt is not None:
                err_sum_sqrt = err_sum_sqrt + temp_err_sum_sqrt
            else:
                err_sum_sqrt = temp_err_sum_sqrt
        return err_sum_sqrt


    def _get_data_slice(self,simularatie,data):
        data_slices = data.iloc[:,simularatie.data_desciiptor.data_start_index:simularatie.data_desciiptor.data_end_index]
        return data_slices