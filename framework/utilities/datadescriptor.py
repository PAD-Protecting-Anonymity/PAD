from enum import Enum
import numpy as np

class DataDescriptorBase:
    def __init__(self, data_descriptor_type,data_start_index,data_end_index,
            data_decription=""):
        self.data_descriptor_type = data_descriptor_type
        self.data_start_index = data_start_index
        self.data_end_index = data_end_index
        self.data_decription = data_decription

    def get_str_description(self):
        raise NotImplementedError('users must define get_str_description in class to use this base class')

    def verify_configuration_data_descriptor_config(self,amount_of_colums, data):
        """Test if the data decriptors are refering to data witch is not in the datastrame
        
        Arguments:
            amount_of_colums {int} -- the total amount of colums in the datastrame
            data {pandas dataframe} -- the data to be sanitized
        Raises:
            ValueError -- if config does not complie with the framework
        """
        if self.data_start_index > amount_of_colums:
            raise ValueError("refer to index wich is not in datastrame: Index %s goes out of bound for the data stream in data descriptor in data_start_index" % self.data_start_index)
        elif self.data_end_index > amount_of_colums:
                raise ValueError("refer to index wich is not in datastrame: Index %s goes out of bound for the data stream in data descriptor in data_end_index" % self.data_end_index)
        if self.data_start_index > self.data_end_index:
            temp_holder = self.data_start_index
            self.data_start_index = self.data_end_index
            self.data_end_index = temp_holder
        

class DataDescriptorTimeSerice(DataDescriptorBase):
    def __init__(self, sampling_frequency,output_frequency, genelaraty_mode,data_type,
            data_start_index,data_end_index,
            data_window_size= None, data_decription=""):
        super().__init__(DataDescriptorTerms.TIMESEICE,data_start_index,data_end_index,data_decription)
        self.sampling_frequency = sampling_frequency
        self.output_frequency = output_frequency
        self.genelaraty_mode = genelaraty_mode
        self.data_type = data_type
        self.data_window_size = data_window_size

    def get_str_description(self):
        date_type_decription = None
        if self.data_type == DataDescriptorTerms.BOOLAEN:
            if self.genelaraty_mode == DataDescriptorTerms.MEAN:
                date_type_decription = "Percentage"
            else:
                date_type_decription = self.data_type.value
        else:
            date_type_decription = self.data_type.value

        if self.data_decription is not "":
            return "Data Decription: {0} Data Type: Time Serice Data Type: {1} Start Index: {2} End Index: {3} Frequency: {4} Genelaraty Mode: {5}".format(self.data_decription,date_type_decription, self.data_start_index, self.data_end_index, self.output_frequency.name, self.genelaraty_mode.value)
        return "Data Type: Time Serice Data Type: {0} Start Index: {1} End Index: {2} Frequency: {3} Genelaraty Mode: {4}".format(date_type_decription, self.data_start_index, self.data_end_index, self.output_frequency.name, self.genelaraty_mode.value)

    def verify_configuration_data_descriptor_config(self,amount_of_colums, data):
        super().verify_configuration_data_descriptor_config(amount_of_colums,data)
        if self.sampling_frequency.value > self.output_frequency.value:
            raise ValueError("For DataDescriptorTimeSerice the sampling rate most be >= to the output frequency")
        if (self.data_end_index - self.data_start_index) < (self.output_frequency.value / self.sampling_frequency.value):
            raise ValueError("For DataDescriptorTimeSerice, there are not enough input samples to produce an output with the selected output frequency")

        for i in range(self.data_start_index, self.data_end_index):
            if  not np.issubdtype(data.dtypes[i], np.number):
                raise ValueError("For DataDescriptorTimeSerice, data types have to be of type number, see https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html for help")

class DataDescriptorMetadata(DataDescriptorBase):
    def __init__(self,data_start_index,data_end_index=None,data_decription=""):
        super().__init__(DataDescriptorTerms.METADATA,data_start_index,data_end_index,data_decription)
        if data_end_index is None:
            self.data_end_index = data_start_index

    def get_str_description(self):
        if self.data_decription is not "":
            return "Data Decription: {0} Data Type: Meta Data, Start Index: {1} End Index: {2}".format(self.data_decription,self.data_start_index, self.data_end_index)
        return "Data Type: Meta Data, Start Index: {0}  End Index: {2}".format(self.data_start_index, self.data_end_index)    



class DataDescriptorTerms(Enum):
    #data_descriptor_type
    TIMESEICE = "timeserice"
    METADATA = "metadata"

    #Genelaraty Mode
    MEAN = "mean"
    MODE = "mode"
    MEDIAN = "median"    
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    

    #Datatypes
    NUMBER = "number"
    BOOLAEN = "boolean"

    #Genelaraty
    # EVENT = "event"
    SECOND = 1
    SECOND_5 = 5    
    SECOND_20 = 20
    MINUE = 60
    MINUE_5 = 300    
    QUARETER = 900
    MINUE_20 = 1200
    HALFHOUR = 1800
    HOUR = 3600
    DAY = 86400
    WEEK = 604800