import sys;
import os

sys.path.insert(0,os.path.abspath("./"))
# sys.path.append(os.path.abspath("/framework/simulraty"))
from framework.framework import Framwork
from framework.similarity import globalsimularity
from framework.similarity import simularityterms

# from framework. import *
import pandas as pd

data = pd.read_pickle('./dataset/dataframe_all_binary.pkl')
data = data.iloc[0:90, 0::2] # the database to be published
data = data.fillna(0)
amount_Of_Variables = 1
anonymity_level = 2

framework = Framwork(data,amount_Of_Variables,anonymity_level)

sampling_frequency = simularityterms.SimularityTerms.MINUE
output_genelaraty = simularityterms.SimularityTerms.QUARETER
genelaraty_mode = simularityterms.SimularityTerms.MEAN

test = globalsimularity.GlobalSimularity(sampling_frequency,output_genelaraty,genelaraty_mode)
framework.add_simularatie(test)
# framework.add_simularatie(test)
framework.run()