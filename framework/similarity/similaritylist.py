import numpy as np

class SimilarityList:
    def __init__(self):
        self.similarities = []

    def get_distance(self,data):
        distance = None
        for similarity in self.similarities:
            tempdistance = similarity.get_distance(self._get_data_slice(similarity,data))
            if distance is not None:
                distance['distance'] = distance['distance'] + tempdistance['distance']
            else:
                distance = tempdistance
        distance = distance.sort_values('distance')
        return distance

    def add_similarity(self,similarity):
        self.similarities.append(similarity)

    def get_amount_of_similarities(self):
        return len(self.similarities)

    def get_statistics_loss(self,data_org, data_sanitized):
        err_sum_sqrt = None
        for similarity in self.similarities:
            temp_err_sum_sqrt = similarity.get_information_loss(self._get_data_slice(similarity,data_org),self._get_data_slice(similarity,data_sanitized))
            if err_sum_sqrt is not None:
                err_sum_sqrt = err_sum_sqrt + temp_err_sum_sqrt
            else:
                err_sum_sqrt = temp_err_sum_sqrt
        return err_sum_sqrt


    def _get_data_slice(self,similarity,data):
        data_slices = data.iloc[:,similarity.data_descriptor.data_start_index:similarity.data_descriptor.data_end_index]
        return data_slices