from enum import Enum
import numpy as np

class DataDescriptorBase:
    def __init__(self, data_descriptor_type,data_start_index,data_end_index,
            data_description=""):
        self.data_descriptor_type = data_descriptor_type
        self.data_start_index = data_start_index
        self.data_end_index = data_end_index
        self.data_description = data_description

    def get_str_description(self):
        raise NotImplementedError('users must define get_str_description in class to use this base class')

    def verify_configuration_data_descriptor_config(self,amount_of_column, data):
        """Test if the data decriptors are refering to data witch is not in the datastream
        Arguments:
            amount_of_column {int} -- the total amount of column in the datastream
            data {pandas dataframe} -- the data to be sanitized
        Raises:
            ValueError -- if config does not complie with the framework
        """
        if self.data_start_index > amount_of_column:
            raise ValueError("refer to index wish is not in datastream: Index %s goes out of bound for the data stream in data descriptor in data_start_index" % self.data_start_index)
        elif self.data_end_index > amount_of_column:
                raise ValueError("refer to index wish is not in datastream: Index %s goes out of bound for the data stream in data descriptor in data_end_index" % self.data_end_index)
        if self.data_start_index > self.data_end_index:
            temp_holder = self.data_start_index
            self.data_start_index = self.data_end_index
            self.data_end_index = temp_holder

class DataDescriptorTimeSeries(DataDescriptorBase):
    def __init__(self, sampling_frequency, generality_mode,data_type,
            data_start_index,data_end_index, output_frequency=None,
            data_window_size= None, data_description=""):
        super().__init__(DataDescriptorTerms.TIMESEICE,data_start_index,data_end_index,data_description)
        self.sampling_frequency = sampling_frequency
        if output_frequency is None:
            self.output_frequency = sampling_frequency
        else:
            self.output_frequency = output_frequency
        self.generality_mode = generality_mode
        self.data_type = data_type
        self.data_window_size = data_window_size

    def get_str_description(self):
        date_type_description = None
        if self.data_type == DataDescriptorTerms.BOOLEAN:
            if self.generality_mode == DataDescriptorTerms.MEAN:
                date_type_description = "Percentage"
            else:
                date_type_description = self.data_type.value
        else:
            date_type_description = self.data_type.value

        if self.data_description is not "":
            return "Data Description: {0} Data Type: Time Series Data Type: {1} Start Index: {2} End Index: {3} Frequency: {4} Generality Mode: {5}".format(self.data_description,date_type_description, self.data_start_index, self.data_end_index, self.output_frequency.name, self.generality_mode.value)
        return "Data Type: Time Series Data Type: {0} Start Index: {1} End Index: {2} Frequency: {3} Generality Mode: {4}".format(date_type_description, self.data_start_index, self.data_end_index, self.output_frequency.name, self.generality_mode.value)

    def verify_configuration_data_descriptor_config(self,amount_of_column, data):
        super().verify_configuration_data_descriptor_config(amount_of_column,data)
        if self.sampling_frequency.value > self.output_frequency.value:
            raise ValueError("For DataDescriptorTimeSeries the sampling rate most be >= to the output frequency")
        if (self.data_end_index - self.data_start_index) < (self.output_frequency.value / self.sampling_frequency.value):
            raise ValueError("For DataDescriptorTimeSeries, there are not enough input samples to produce an output with the selected output frequency")

        for i in range(self.data_start_index, self.data_end_index):
            if  not np.issubdtype(data.dtypes[i], np.number):
                raise ValueError("For DataDescriptorTimeSeries, data types have to be of type number, see https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html for help")

class DataDescriptorMetadata(DataDescriptorBase):
    def __init__(self,data_start_index,data_end_index=None,data_description=""):
        super().__init__(DataDescriptorTerms.METADATA,data_start_index,data_end_index,data_description)
        if data_end_index is None:
            self.data_end_index = data_start_index

    def get_str_description(self):
        if self.data_description is not "":
            return "Data Description: {0} Data Type: Meta Data, Start Index: {1} End Index: {2}".format(self.data_description,self.data_start_index, self.data_end_index)
        return "Data Type: Meta Data, Start Index: {0}  End Index: {2}".format(self.data_start_index, self.data_end_index)



class DataDescriptorTerms(Enum):
    #data_descriptor_type
    TIMESEICE = "timeserice"
    METADATA = "metadata"

    #Generality Mode
    MEAN = "mean"
    MODE = "mode"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    SUM = "sum"


    #Datatypes
    NUMBER = "number"
    BOOLEAN = "boolean"

    # EVENT = "event"
    SECOND = 1
    SECOND_2 = 2
    SECOND_5 = 5
    SECOND_20 = 20
    MINUET = 60
    MINUET_2 = 120
    MINUET_5 = 300
    MINUET_10 = 600
    QUARTER = 900
    MINUET_20 = 1200
    HALFHOUR = 1800
    HOUR = 3600
    HOUR_2 = 7200
    HOUR_4 = 14400
    HALFDAY = 43200
    DAY = 86400
    WEEK = 604800