import statistics
import numpy as np
import pandas as pd
import math
from utilities.datadescriptor import DataDescriptorTerms
from framework.utilities.datadescriptor import DataDescriptorBase, DataDescriptorMetadata, DataDescriptorTimeSerice

class OutputGroupper:
    def __init__(self,dataset_descriptions):
        self.dataset_descriptions = dataset_descriptions

    def transform_data(self,data):
        output_data = pd.DataFrame()
        index_data_insert = 0
        for dataset_description in self.dataset_descriptions:
            if isinstance(dataset_description, DataDescriptorTimeSerice):
                index_data_insert = self.transform_data_time_serices(data,dataset_description,index_data_insert,output_data)
            elif isinstance(dataset_description, DataDescriptorMetadata):
                index_data_insert = self.transform_data_metadata(data,dataset_description,index_data_insert,output_data)
        print(output_data)
        return output_data
        
    def transform_data_metadata(self,data,dataset_description,index_data_insert,output_data):
        data_slice_index_start = dataset_description.data_start_index
        data_slice_index_end = dataset_description.data_end_index
        data_slices = None
        if data_slice_index_start == data_slice_index_end:
            data_slices = data.iloc[:,data_slice_index_start]        
        else:
            data_slices = data.iloc[:,data_slice_index_start:data_slice_index_end]
        output_data[str(index_data_insert)] = data_slices
        index_data_insert += 1
        return index_data_insert

    def transform_data_time_serices(self,data,dataset_description,index_data_insert,output_data):
        input_output_factor =  dataset_description.output_frequency/dataset_description.sampling_frequency
        amount_of_samples_in_slice = (dataset_description.data_end_index - dataset_description.data_start_index)+1
        amount_of_slices = math.floor(amount_of_samples_in_slice / input_output_factor) #Floor to ensure that we do not groups for small data amounts
        if amount_of_slices > 1:
            for i in range(amount_of_slices):
                data_slice_index_start = int(input_output_factor*i)
                data_slice_index_end = int(input_output_factor*(i+1))
                data_slices = data.iloc[:,data_slice_index_start:data_slice_index_end]
                tm = None
                if dataset_description.data_type == DataDescriptorTerms.BOOLAEN:
                    tm = OutoutGroupperBoolean()
                elif dataset_description.data_type == DataDescriptorTerms.NUMBER:
                    tm = OutoutGroupperNumber()
                transfomred_data = tm.transform(data_slices,dataset_description.genelaraty_mode)
                output_data[str(index_data_insert)] = transfomred_data
                index_data_insert += 1
        elif amount_of_slices <= 1:
            data_slice_index_start = dataset_description.data_start_index
            data_slice_index_end = dataset_description.data_end_index
            data_slices = data.iloc[:,data_slice_index_start:data_slice_index_end]
            if dataset_description.data_type == DataDescriptorTerms.BOOLAEN:
                tm = OutoutGroupperBoolean()
            elif dataset_description.data_type == DataDescriptorTerms.NUMBER:
                tm = OutoutGroupperNumber()
            transfomred_data = tm.transform(data_slices,dataset_description.genelaraty_mode)
            output_data[str(index_data_insert)] = transfomred_data
            index_data_insert += 1
        return index_data_insert

class OutoutGroupperTypeBase:
    def __init__(self):
        if self.__class__ == OutoutGroupperTypeBase:
            raise Exception('abstract class')

    def transform(self,data_slices, genelaraty_mode):
        data_out = None
        if genelaraty_mode == DataDescriptorTerms.MEAN:
            data_out = self._transform_mean(data_slices)
        elif genelaraty_mode == DataDescriptorTerms.MEDIAN:
            data_out = self._transform_median(data_slices)
        elif genelaraty_mode == DataDescriptorTerms.MIN:
            data_out = self._transform_min(data_slices)
        elif genelaraty_mode == DataDescriptorTerms.MAX:
            data_out = self._transform_max(data_slices)
        elif genelaraty_mode == DataDescriptorTerms.MODE:
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