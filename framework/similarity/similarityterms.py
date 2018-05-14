"""Constants used to defined types of Similarity"""

class SimilarityTerms:
    #Types of simularity in time series
    ARRIVAL = "arrival" #e.g. used for finding first true in boolian time series
    DEPARTURE = "departure" #e.g. used for finding first false in boolian time series
    USAGE = "usage" #e.g. used for power use
    WINDOW_USAGE = "window-usage"  #e.g. used for power use in window
    SEGMENT = "segment" #used for window
    GLOBAL = "global" #use all of the time series
    TOTAL = "total" #used to sum a time series counter
    WINDOW_TOTAL = "window-total"  #used to sum a time series counter in window

    #Global distance_metric
    EUCLIDEAN = "euclidean"
    MAHALAOBIS = "mahalanobis"
    CHEBYSHEV = "chebyshev"
    MANHATTAN = "manhattan"
    MINKOWSKI = "minkowski"
    WMINKOWSKI = "wminkowski"
    SEUCLIDEAN = "seuclidean"
    CUSTOM = "custom"