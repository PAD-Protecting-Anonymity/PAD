import statistics
import numpy as np
import pandas as pd
import math
from framework.utilities.datadescriptor import DataDescriptorTerms
from framework.utilities.datadescriptor import DataDescriptorBase, DataDescriptorMetadata, DataDescriptorTimeSeries
from itertools import chain

class OutputGroupper:
    def __init__(self,dataset_descriptions):
        self.dataset_descriptions = dataset_descriptions
        self.dd_string_out = []

    def sort_dataset_descriptions(self):
        sorted_dd = sorted(self.dataset_descriptions, key=lambda x: x.data_start_index)
        return sorted_dd

    def _crate_data_description(self, dataset_description, start_index, end_index):
            self.dd_string_out.append(dataset_description.get_str_description(start_index,end_index))

    def transform_data(self,data):
        output_data = pd.DataFrame()
        index_data_insert = 0
        sorted_dd = self.sort_dataset_descriptions()
        for dataset_description in sorted_dd:
            start_index = len(output_data.columns)
            if isinstance(dataset_description, DataDescriptorTimeSeries):
                output_data = self.transform_data_time_serices(data,dataset_description,output_data)
            elif isinstance(dataset_description, DataDescriptorMetadata):
                output_data = self.transform_data_metadata(data,dataset_description,output_data)
            dataset_description.data_start_index = start_index
            dataset_description.data_end_index = len(output_data.columns) -1
            # self._crate_data_description(dataset_description,  start_index, len(output_data.columns)-1)
        # print('\n'.join(self.dd_string_out))
        output_data.columns = list(range(0,len(output_data.columns)))
        return output_data, sorted_dd #, '\n'.join(self.dd_string_out)

    def transform_data_metadata(self,data,dataset_description,output_data):
        data_slice_index_start = dataset_description.data_start_index
        data_slice_index_end = dataset_description.data_end_index
        data_slices = None
        if data_slice_index_start == data_slice_index_end:
            data_slices = data.iloc[:,data_slice_index_start]
        else:
            data_slices = data.iloc[:,data_slice_index_start:data_slice_index_end]
        output_data = pd.concat([output_data,data_slices], axis=1)
        # output_data[str(index_data_insert)] = output_data data_slices
        # index_data_insert += 1
        # return index_data_insert
        return output_data

    def transform_data_time_serices(self,data,dataset_description,output_data):
        input_output_factor =  dataset_description.output_frequency.value/dataset_description.sampling_frequency.value
        amount_of_samples_in_slice = (dataset_description.data_end_index - dataset_description.data_start_index)+1
        amount_of_slices = math.floor(amount_of_samples_in_slice / input_output_factor) #Floor to ensure that we do not groups for small data amounts
        if amount_of_slices > 1 and input_output_factor != 1:
            for i in range(amount_of_slices):
                data_slice_index_start = int(input_output_factor*i)
                data_slice_index_end = int(input_output_factor*(i+1))
                data_slices = data.iloc[:,data_slice_index_start:data_slice_index_end+1]
                tm = None
                if dataset_description.data_type == DataDescriptorTerms.BOOLEAN:
                    tm = OutoutGroupperBoolean()
                elif dataset_description.data_type == DataDescriptorTerms.NUMBER:
                    tm = OutoutGroupperNumber()
                transfomred_data = tm.transform(data_slices,dataset_description.generality_mode)
                output_data = pd.concat([output_data, pd.Series(transfomred_data)], axis=1)
        else: # amount_of_slices =< 1:
            data_index_start = dataset_description.data_start_index
            data_index_end = dataset_description.data_end_index
            data_slices = data.iloc[:,data_index_start:data_index_end+1]
            output_data = pd.concat([output_data,data_slices], axis=1)
        # else: # amount_of_slices = 1 or input_output_factor = 1
        #     data_index_start = dataset_description.data_end_index
        #     data_index_end = dataset_description.data_start_index
        #     data_slices = data.iloc[:,data_index_start:data_index_end]
        #     output_data = pd.concat([output_data,data_slices], axis=1)
        return output_data

class OutoutGroupperTypeBase:
    def __init__(self):
        if self.__class__ == OutoutGroupperTypeBase:
            raise Exception('abstract class')

    def transform(self,data_slices, generality_mode):
        data_out = None
        if generality_mode == DataDescriptorTerms.MEAN:
            data_out = self._transform_mean(data_slices)
        elif generality_mode == DataDescriptorTerms.MEDIAN:
            data_out = self._transform_median(data_slices)
        elif generality_mode == DataDescriptorTerms.MIN:
            data_out = self._transform_min(data_slices)
        elif generality_mode == DataDescriptorTerms.MAX:
            data_out = self._transform_max(data_slices)
        elif generality_mode == DataDescriptorTerms.MODE:
            data_out = self._transform_mode(data_slices)
        elif generality_mode == DataDescriptorTerms.SUM:
            data_out = self._transform_sum(data_slices)
        return data_out

    def _transform_mean(self,data_slices):
        raise NotImplementedError('NotImplemented')

    def _transform_sum(self,data_slices):
        sum_value = data_slices.sum(axis=1)
        return sum_value._values

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
        return amount_Of_True/len(data_slices.columns)

class OutoutGroupperNumber(OutoutGroupperTypeBase):
    def _transform_mean(self,data_slices):
        mean = data_slices.mean(axis=1)
        return mean