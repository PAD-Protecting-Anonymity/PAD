import sys;
import os
import numpy as np

sys.path.insert(0,os.path.abspath("./"))
# sys.path.append(os.path.abspath("/framework/simulraty"))
from framework.framework import Framwork
from framework.similarity.arrivalsimularity import ArrivalSimularity
from framework.similarity.segmentsimularity import SegmentSimularity
from framework.similarity.globalsimularity import GlobalSimularity 
from framework.similarity.simularityterms import SimularityTerms
from framework.utilities.datadescriptor import DataDescriptorMetadata,DataDescriptorTimeSerice,DataDescriptorTerms
import pandas as pd
import pickle

# ann5Data, dist7, loss7 = pickle.load( open( "./results/Ann7Data.pickle", "rb" ))
# ann2Data, dist2, loss2 = pickle.load( open( "./results/Ann2Data.pickle", "rb" ))

# data = pd.read_pickle("./dataset/PreprocedHamiltonData_HamiltonData_presence.pkl")
# data = pd.read_csv("./dataset/PreprocedHamiltonData_HamiltonData_lux.csv")
# data = pd.read_csv("./dataset/PreprocedHamiltonData_HamiltonData_air_temp.csv")
data = pd.read_csv("./dataset/PreprocedHamiltonData_HamiltonData_air_hum.csv")
# data = pd.read_csv("./dataset/PreprocedHamiltonData_HamiltonData_presence.csv")
# data = pd.read_pickle('./dataset/dataframe_all_binary.pkl')
# data = data.iloc[0:90, 0::60] # the database to be published
data = data.iloc[:,0:50] # the database to be published
# data = data.fillna(0)

data = data.infer_objects()

print("amount of samples %s" % len(data.index))
print("amount of columns %s" % len(data.columns))
anonymity_level = 5

framework = Framwork(data,anonymity_level)

sampling_frequency = DataDescriptorTerms.SECOND_20
output_genelaraty = DataDescriptorTerms.MINUE
genelaraty_mode = DataDescriptorTerms.MEAN
data_type = DataDescriptorTerms.NUMBER
data_window = 15

# dd = DataDescriptorTimeSerice(sampling_frequency,output_genelaraty,genelaraty_mode,data_type,0,30,data_window_size=data_window)
# dd1 = DataDescriptorTimeSerice(sampling_frequency,output_genelaraty,genelaraty_mode,data_type,31,61,data_window_size=data_window)
dd = DataDescriptorTimeSerice(sampling_frequency,genelaraty_mode,data_type,1,len(data.columns)-1, output_frequency=output_genelaraty)
# dd1 = DataDescriptorTimeSerice(sampling_frequency,output_genelaraty,genelaraty_mode,data_type,31,61)

# segmentedData = GlobalSimularity(dd)
segmentedData = SegmentSimularity(dd,[10,15])

# dd1 = DataDescriptorTimeSerice(sampling_frequency,output_genelaraty,genelaraty_mode,data_type,61,74,segmentedData1)
metaData = DataDescriptorMetadata(0, data_decription="Meta Data")

framework.add_simularatie(segmentedData)
# framework.add_simularatie(segmentedData1)
framework.add_meta_data(metaData)
out , loss_metric = framework.run()
print(out)
print(framework.generated_data_description())

pickle.dump([out, framework.generated_data_description(), loss_metric], open('./results/FWResult1.pickle', "wb"))