from enum import Enum

class DataDescriptorBase:
    def __init__(self, data_descriptor_type,data_start_index,data_end_index,
            data_decription=""):
        self.data_descriptor_type = data_descriptor_type
        self.data_start_index = data_start_index
        self.data_end_index = data_end_index
        self.data_decription = data_decription

    def get_str_description(self, output_start_index, output_end_index):
        raise NotImplementedError('users must define get_str_description in class to use this base class')

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

    def get_str_description(self, output_start_index, output_end_index):
        date_type_decription = None
        if self.data_type == DataDescriptorTerms.BOOLAEN:
            if self.genelaraty_mode == DataDescriptorTerms.MEAN:
                date_type_decription = "percentage"
            else:
                date_type_decription = self.data_type.value
        else:
            date_type_decription = self.data_type.value

        if self.data_decription is not "":
            return "Data Decription: {0} Data Type: Time Serice Data Type: {1} Start Index: {2} End Index: {3} Frequency: {4} Genelaraty Mode: {5}".format(self.data_decription,date_type_decription, output_start_index, output_end_index, self.output_frequency.name, self.genelaraty_mode.value)
        return "Data Type: Time Serice Data Type: {0} Start Index: {1} End Index: {2} Frequency: {3} Genelaraty Mode: {4}".format(date_type_decription, output_start_index, output_end_index, self.output_frequency.name, self.genelaraty_mode.value)

class DataDescriptorMetadata(DataDescriptorBase):
    def __init__(self,data_start_index,data_end_index=None,data_decription=""):
        super().__init__(DataDescriptorTerms.METADATA,data_start_index,data_end_index,data_decription)
        if data_end_index is None:
            self.data_end_index = data_start_index
    def get_str_description(self, output_start_index, output_end_index):
        if self.data_decription is not "":
            return "Data Decription: {0} Data Type: Meta Data, Start Index: {1} End Index: {2}".format(self.data_decription,output_start_index, output_end_index)
        return "Data Type: Meta Data, Start Index: {0}  End Index: {2}".format(output_start_index, output_end_index)



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
    MINUE = 60
    QUARETER = 900
    HALFHOUR = 1800
    HOUR = 3600
    DAY = 86400
    