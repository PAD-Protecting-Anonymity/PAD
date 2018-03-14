class DataDescriptorBase:
    def __init__(self, data_descriptor_type,data_start_index,data_end_index,
            data_decription=""):
        self.data_descriptor_type = data_descriptor_type
        self.data_start_index = data_start_index
        self.data_end_index = data_end_index
        self.data_decription = data_decription

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

class DataDescriptorMetadata(DataDescriptorBase):
    def __init__(self,data_start_index,data_end_index=None,data_decription=""):
        super().__init__(DataDescriptorTerms.METADATA,data_start_index,data_end_index,data_decription)
        if data_end_index is None:
            self.data_end_index = data_start_index

class DataDescriptorTerms:
    #data_descriptor_type
    TIMESEICE = "timeserice"
    METADATA = "metadata"

    #Genelaraty Mode
    MEAN = "mean"
    MODE = "mode"
    MEDIAN = "median"    
    MIN = "min"
    MAX = "max"

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
    