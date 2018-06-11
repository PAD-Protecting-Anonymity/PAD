import sys;
import os
import numpy as np
import math
import time

sys.path.insert(0,os.path.abspath("./"))
from framework.framework import Framework
from framework.similarity.arrivalsimilarity import ArrivalSimilarity
from framework.similarity.segmentsimilarity import SegmentSimilarity
from framework.similarity.globalsimilarity import GlobalSimilarity
from framework.similarity.similarityterms import SimilarityTerms
from framework.similarity.resourceusagesimilarity import ResourceUsageSimilarity
from framework.similarity.hourlysimilarity import HourlySimilarity, HourlySimilarityModes
from framework.metric_learning.metriclearningterms import MetricLearningTerms
from framework.utilities.datadescriptor import DataDescriptorMetadata,DataDescriptorTimeSeries,DataDescriptorTerms
import pandas as pd
import pickle



for i in range(0,7):
    all_data = []
    all_samplingRates = []
    data = pd.read_csv("./dataset/preprocessed_line_counter.csv")
    data_Noices = pd.read_csv("./dataset/Preprocessed_noices_avg_4s.csv")

    data_presence = pd.read_csv("./dataset/Preprocssed_HamiltonData_presence.csv")
    data = data.iloc[0::2,0:10081] # the database to be published
    data_presence = data_presence.iloc[0:11,0:30241] # the database to be published
    data_Noices = data_Noices.iloc[:,0:2017]
    all_data.append(data)
    all_data.append(data_presence)
    all_data.append(data_Noices)

    data = data.infer_objects()

    all_samplingRates.append(DataDescriptorTerms.MINUET.value)
    all_samplingRates.append(DataDescriptorTerms.SECOND_20.value)
    all_samplingRates.append(DataDescriptorTerms.MINUET_5.value)
    print("amount of samples %s" % len(data.index))
    print("amount of columns %s" % len(data.columns))
    anonymity_level = 5
    rep_mode = "mean"
    min_resample_factor = math.floor(DataDescriptorTerms.DAY.value / DataDescriptorTerms.MINUET.value)
    # seed = 123456
    k_fold = [i,7]

    framework = Framework(data,anonymity_level,rep_mode=rep_mode, resample_factor = min_resample_factor, learning_metric=MetricLearningTerms.LINEAR,k_fold = k_fold)
    sampling_frequency = DataDescriptorTerms.MINUET
    output_generality = DataDescriptorTerms.MINUET
    generality_mode = DataDescriptorTerms.SUM
    data_type = DataDescriptorTerms.NUMBER

    data_window = 1440
    amountOfSamplesInHour = 60
    segment = [480,960]

    dd = DataDescriptorTimeSeries(sampling_frequency,generality_mode,data_type,1,len(data.columns)-1, output_frequency=output_generality)

    # segmentSimilarity = SegmentSimilarity(dd,segment)
    segmentSimilarity = HourlySimilarity(dd,segment, sampling_frequency=sampling_frequency.value, mode=HourlySimilarityModes.SUMOFHOUR)

    metaData = DataDescriptorMetadata(0, data_description="Location of the sensor")

    framework.add_similarity(segmentSimilarity)
    framework.add_meta_data(metaData)

    start = time.clock()

    out , loss_metric, anonymity_level = framework.anonymize()
    timeTaken = time.clock() - start
    pickle.dump([out, framework.generated_data_description(), loss_metric,anonymity_level, timeTaken], open('./results/line count/day/PAD_results_line_counter_sum_Day_LINEAR_fold_'+str(i)+'.pickle', "wb"))
    print("Time: " + str(timeTaken))