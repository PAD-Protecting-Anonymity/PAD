import sys;
import os

sys.path.insert(0,os.path.abspath("./"))
# sys.path.append(os.path.abspath("/framework/simulraty"))
from framework.framework import Framwork
from framework.similarity import arrivalsimularity
from framework.similarity.segmentsimularity import SegmentSimularity
from framework.similarity import globalsimularity
from framework.similarity.simularityterms import SimularityTerms
from framework.utilities.datadescriptor import DataDescriptor
import pandas as pd

data = pd.read_pickle('./dataset/dataframe_all_binary.pkl')
# data = data.iloc[0:90, 0::60] # the database to be published
data = data.iloc[0:20, 0:60] # the database to be published
data = data.fillna(0)
print("amount of samples %s" % len(data))
print("amount of columns %s" % len(data.columns))
amount_Of_Variables = 1
anonymity_level = 2

framework = Framwork(data,amount_Of_Variables,anonymity_level)

sampling_frequency = SimularityTerms.MINUE
output_genelaraty = SimularityTerms.QUARETER
genelaraty_mode = SimularityTerms.MODE
data_type = SimularityTerms.NUMBER

dd = DataDescriptor(sampling_frequency,output_genelaraty,genelaraty_mode,data_type,0,30)
dd1 = DataDescriptor(sampling_frequency,output_genelaraty,genelaraty_mode,data_type,31,60)

segmentedData = SegmentSimularity(dd,[10,15])
segmentedData1 = SegmentSimularity(dd1,[10,15])

framework.add_simularatie(segmentedData)
framework.add_simularatie(segmentedData1)
framework.run()