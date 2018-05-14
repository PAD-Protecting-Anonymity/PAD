import numpy as np

class Verifyerror:
    def verify(self,data, similarities_list, data_descriptors):
        data = self._verify_data_input(data,similarities_list.similarities)
        self._verify_configuration_of_framework(data,similarities_list,data_descriptors)
        return data

    def _verify_data_input(self, data,similarities):
        #remove samples with only nan
        for similarity in similarities:
            all_nan = data.iloc[:,similarity.data_descriptor.data_start_index:similarity.data_descriptor.data_end_index+1].isnull().all(axis=1)
            data = data.loc[np.invert(all_nan)]
            if all_nan.any():
                removed = [i for i, x in enumerate(all_nan) if x]
                print("Removed input of {0}, since they don't have any input values".format(removed))
        if data.isnull().values.any():
            print("Replaced nan's  with 0, consider to handle this before using PAD")
        return data.fillna(0)

    def _verify_configuration_of_framework(self,data,similarities_list, data_descriptors):
        amount_of_column = len(data.columns) - 1
        if similarities_list.get_amount_of_similarities() < 1:
            raise ValueError("No data to process, use add_similarity to add processing on the data")
        for data_descriptor in data_descriptors:
            data_descriptor.verify_configuration_data_descriptor_config(amount_of_column, data)

    def verify_after_can_not_ensure_k_anonymity(self,data, similarities_list):
        amount_of_column = len(data.columns) - 1
        for similarity in similarities_list.similarities:
            if similarity.data_window is not None and max(similarity.data_window) > amount_of_column:
                raise ValueError("data_windows is larger then the amount of elements in each data slice")
            if hasattr(similarity.data_descriptor, 'data_window_size'):
                if similarity.data_descriptor.data_window_size is not None and similarity.data_descriptor.data_window_size < amount_of_column:
                    raise ValueError("data_window_size is larger then the amount of elements in each data slice")



