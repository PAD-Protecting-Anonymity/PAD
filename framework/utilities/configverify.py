import numpy as np

class Verifyerror:
    def verify(self,data, simularatie_list, data_descriptors):
        data = self._verify_data_input(data,simularatie_list.simularaties)
        self._verify_configuration_of_framework(data,simularatie_list,data_descriptors)
        return data

    def _verify_data_input(self, data,simularaties):
        #remove samples with only nan
        for similarity in simularaties:
            all_nan = data.iloc[:,similarity.data_descriptor.data_start_index:similarity.data_descriptor.data_end_index+1].isnull().all(axis=1)
            data = data.loc[np.invert(all_nan)]
            if all_nan.any():
                removed = [i for i, x in enumerate(all_nan) if x]
                print("Removed input of {0}, since they don't have any input values".format(removed))
        if data.isnull().values.any():
            print("Replaced nan's  with 0, consider to handle this before using PAD")
        return data.fillna(0)

    def _verify_configuration_of_framework(self,data,simularatie_list, data_descriptors):
        amount_of_colums = len(data.columns) - 1
        if simularatie_list.get_amount_of_simularaties() < 1:
            raise ValueError("No data to process, use add_simularatie to add processing on the data")
        for data_descriptor in data_descriptors:
            data_descriptor.verify_configuration_data_descriptor_config(amount_of_colums, data)