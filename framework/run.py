import sys;
import os

sys.path.insert(0,os.path.abspath("./"))
# sys.path.append(os.path.abspath("/framework/simulraty"))
from framework.framework import Framwork
from framework.similarity.arrivalsimularity import ArrivalSimularity
from framework.similarity.segmentsimularity import SegmentSimularity
from framework.similarity.globalsimularity import GlobalSimularity 
from framework.similarity.simularityterms import SimularityTerms
from framework.utilities.datadescriptor import DataDescriptorMetadata,DataDescriptorTimeSerice,DataDescriptorTerms
import pandas as pd

data = pd.read_pickle('./dataset/dataframe_all_binary.pkl')
# data = data.iloc[0:90, 0::60] # the database to be published
data = data.iloc[0:20, 0::19] # the database to be published
data = data.fillna(0)
print("amount of samples %s" % len(data))
print("amount of columns %s" % len(data.columns))
anonymity_level = 5

framework = Framwork(data,anonymity_level)

sampling_frequency = DataDescriptorTerms.MINUE
output_genelaraty = DataDescriptorTerms.QUARETER
genelaraty_mode = DataDescriptorTerms.MEAN
data_type = DataDescriptorTerms.BOOLAEN
data_window = 15

# dd = DataDescriptorTimeSerice(sampling_frequency,output_genelaraty,genelaraty_mode,data_type,0,30,data_window_size=data_window)
# dd1 = DataDescriptorTimeSerice(sampling_frequency,output_genelaraty,genelaraty_mode,data_type,31,61,data_window_size=data_window)
dd = DataDescriptorTimeSerice(sampling_frequency,output_genelaraty,genelaraty_mode,data_type,0,73)
# dd1 = DataDescriptorTimeSerice(sampling_frequency,output_genelaraty,genelaraty_mode,data_type,31,61)

segmentedData = SegmentSimularity(dd,[10,15])
# segmentedData1 = SegmentSimularity(dd1,[10,15])

# dd1 = DataDescriptorTimeSerice(sampling_frequency,output_genelaraty,genelaraty_mode,data_type,61,74,segmentedData1)
metaData = DataDescriptorMetadata(75, data_decription="Meta Data")
metaData1 = DataDescriptorMetadata(74, data_decription="Meta Data")


framework.add_simularatie(segmentedData)
# framework.add_simularatie(segmentedData1)
framework.add_meta_data(metaData)
framework.add_meta_data(metaData1)
out = framework.run()
print(out)