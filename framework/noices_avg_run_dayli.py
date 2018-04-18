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
from framework.similarity.resourceusagesimularity import ResourceUsageSimularity
from framework.utilities.datadescriptor import DataDescriptorMetadata,DataDescriptorTimeSerice,DataDescriptorTerms
import pandas as pd
import pickle

data = pd.read_csv("./dataset/Preprocessed_noices_avg_dayli.csv")

data = data.iloc[:,0:289]

data = data.infer_objects()

print("amount of samples %s" % len(data.index))
print("amount of columns %s" % len(data.columns))
anonymity_level = 5
rep_mode = "mean"

framework = Framwork(data,anonymity_level,rep_mode=rep_mode)

sampling_frequency = DataDescriptorTerms.MINUE_5
output_genelaraty = DataDescriptorTerms.MINUE_5
genelaraty_mode = DataDescriptorTerms.MEAN
data_type = DataDescriptorTerms.NUMBER

data_window = 288
segment = [96,192]
dd = DataDescriptorTimeSerice(sampling_frequency,genelaraty_mode,data_type,1,len(data.columns)-1, output_frequency=output_genelaraty)

segmentedData = SegmentSimularity(dd, data_window=segment)

metaData = DataDescriptorMetadata(0, data_decription="Meta Data")

framework.add_simularatie(segmentedData)
framework.add_meta_data(metaData)
out , loss_metric = framework.run()

pickle.dump([out, framework.generated_data_description(), loss_metric,anonymity_level], open('./results/PAD_results_noices_avg_dayli.pickle', "wb"))