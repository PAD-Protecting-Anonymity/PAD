import copy
import math
import pandas as pd
from framework.utilities.datadescriptor import DataDescriptorMetadata, DataDescriptorBase, DataDescriptorTimeSerice
from itertools import chain
from framework.similarity.simularatielist import SimularatieList

class Resampler:

    def __init__(self):
        self.meta_datas = []
        self.row_meta_data = pd.DataFrame()

    def resample_data_into_blocks_of_output_rate(self,data,data_descriptors,simularatie_list):
        output_data = pd.DataFrame()
        index = 0
        for data_descriptor in data_descriptors: 
            if isinstance(data_descriptor, DataDescriptorMetadata):
                self.meta_datas.append(data_descriptor)
                meta_data_rows = data.iloc[:, data_descriptor.data_start_index:data_descriptor.data_end_index+1]
                temp_index = (data_descriptor.data_end_index-data_descriptor.data_start_index) + index
                data_descriptor.data_start_index = index
                data_descriptor.data_end_index = temp_index
                index += temp_index + 1
                meta_data_rows.columns = [range(data_descriptor.data_start_index,data_descriptor.data_end_index+1)]
                self.row_meta_data = pd.concat([self.row_meta_data,meta_data_rows], axis=1)

        #Clear framework settings
        _data_descriptors = []
        _simularatie_list = SimularatieList()
        index = 0
        for simularatie in simularatie_list.simularaties:
            data_descriptor = simularatie.data_descriptor
            if isinstance(data_descriptor, DataDescriptorTimeSerice):
                data_length = (data_descriptor.data_end_index - data_descriptor.data_start_index)
                resample_factor = math.floor(simularatie.data_descriptor.output_frequency.value / data_descriptor.sampling_frequency.value)
                amount_of_resamples = math.floor(data_length / resample_factor)
                start_index = data_descriptor.data_start_index
                data_descriptor.data_start_index = index
                index = index + resample_factor-1
                data_descriptor.data_end_index = index
                _simularatie_list.add_simularatie(simularatie)
                _data_descriptors.append(data_descriptor)
                for i in range(0,amount_of_resamples):
                    loop_start_index = start_index+(i*resample_factor)
                    loop_end_index = start_index+((i+1)*resample_factor)
                    new_data = data.iloc[:, loop_start_index:loop_end_index]
                    new_data.columns = list(range(data_descriptor.data_start_index,data_descriptor.data_end_index+1))
                    row_data = pd.concat([new_data], axis=1)
                    output_data = output_data.append(row_data, ignore_index=True)
        return output_data, _simularatie_list, _data_descriptors

    def create_timeserices_from_slices_of_data(self, transformed_data,simularatie_list, amount_of_sensors):
        output_data = pd.DataFrame()
        _simularatie_list = SimularatieList()
        _data_descriptors = []
        index = 0
        for meta_dd in self.meta_datas:
            _data_descriptors.append(meta_dd)
            index = meta_dd.data_end_index + 1
        
        amount_of_inputs = len(transformed_data.index)

        time_Serices_lenged = math.floor(amount_of_inputs / amount_of_sensors)

        for simularatie in simularatie_list.simularaties:
            simularatie.data_descriptor.data_start_index = index
            simularatie.data_descriptor.data_end_index = index + time_Serices_lenged -1
            _simularatie_list.add_simularatie(simularatie)
            _data_descriptors.append(simularatie.data_descriptor)
            
        for i in range(0,amount_of_sensors):
            new_data = transformed_data.iloc[i::amount_of_sensors,:]._values
            new_data = list(chain.from_iterable(new_data))
            new_data = pd.DataFrame.from_items([(i, new_data)],orient='index', columns=list(range(index,index+len(new_data))))
            row_data = pd.concat([self.row_meta_data,new_data], axis=1)
            output_data = output_data.append(row_data.iloc[i])
        return output_data, _simularatie_list, _data_descriptors