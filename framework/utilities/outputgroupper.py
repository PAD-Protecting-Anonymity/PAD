import statistics
import numpy as np
import pandas as pd
import math
from similarity.simularityterms import SimularityTerms

class OutputGroupper:
    def __init__(self,simularatie_List):
        self.simularatie_List = simularatie_List

    def transform_data(self,data):
        output_data = pd.DataFrame(index=data.index)
        index_data_insert = 0
        for simularatie in self.simularatie_List.simularaties:
            input_output_factor =  simularatie.data_desciiptor.output_frequency/simularatie.data_desciiptor.sampling_frequency
            amount_of_samples_in_slice = simularatie.data_desciiptor.data_end_index - simularatie.data_desciiptor.data_start_index
            amount_of_slices = math.floor(amount_of_samples_in_slice / input_output_factor) #Floor to ensure that we do not groups for small data amounts
            for i in range(amount_of_slices):
                data_slice_index_start = int(input_output_factor*i)
                data_slice_index_start_end = int(input_output_factor*(i+1))
                data_slices = data.iloc[:,data_slice_index_start:data_slice_index_start_end]
                tm = None
                if simularatie.data_desciiptor.data_type == SimularityTerms.BOOLAEN:
                    tm = OutoutGroupperBoolean()
                elif simularatie.data_desciiptor.data_type == SimularityTerms.NUMBER:
                    tm = OutoutGroupperNumber()
                transfomred_data = tm.transform(data_slices,simularatie.data_desciiptor.genelaraty_mode)
                output_data[str(index_data_insert)] = transfomred_data
                # output_data = output_data(index_data_insert=transfomred_data)
                index_data_insert += 1
        return output_data


class OutoutGroupperTypeBase:
    def __init__(self):
        if self.__class__ == OutoutGroupperTypeBase:
            raise Exception('abstract class')

    def transform(self,data_slices, genelaraty_mode):
        data_out = None
        if genelaraty_mode == SimularityTerms.MEAN:
            data_out = self._transform_mean(data_slices)
        elif genelaraty_mode == SimularityTerms.MEDIAN:
            data_out = self._transform_median(data_slices)
        elif genelaraty_mode == SimularityTerms.MIN:
            data_out = self._transform_min(data_slices)
        elif genelaraty_mode == SimularityTerms.MAX:
            data_out = self._transform_max(data_slices)
        elif genelaraty_mode == SimularityTerms.MODE:
            data_out = self._transform_mode(data_slices)
        return data_out
    
    def _transform_mean(self,data_slices):
        raise NotImplementedError('NotImplemented')
        
    def _transform_min(self,data_slices):
        min_value = data_slices.min(axis=1)
        return min_value._values
        
    def _transform_max(self,data_slices):
        max_value = data_slices.max(axis=1)
        return max_value._values

    def _transform_median(self,data_slices):
        median = data_slices.median(axis=1)
        return median._value

    def _transform_mode(self,data_slices):
        mode = data_slices.mode(axis=1)
        return mode._values

class OutoutGroupperBoolean(OutoutGroupperTypeBase):
    def _transform_mean(self,data_slices):
        amount_Of_True = data_slices.sum(axis=1)
        return amount_Of_True/len(data_slices)

    # def _transform_median(self,data_slices):
    #     return np.median(data_slices[~np.isnan(data_slices)])

    # def _transform_min(self,data_slices):
    #     return all(data_slices)

    # def _transform_max(self,data_slices):
    #     raise any(data_slices)

class OutoutGroupperNumber(OutoutGroupperTypeBase):
    def _transform_mean(self,data_slices):
        mean = data_slices.mean(axis=1)
        return mean