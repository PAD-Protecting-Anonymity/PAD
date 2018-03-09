"""Constants used to definde types of Simularity"""

class SimularityTerms:
    #Types of simularity in time series
    ARRIVAL = "arrival" #e.g. used for finding first true in boolian time series
    DEPARTURE = "departure" #e.g. used for finding first false in boolian time series
    USAGE = "usage" #e.g. used for power use
    WINDOW_USAGE = "window-usage"  #e.g. used for power use in window
    SEGMENT = "segment" #used for window
    GLOBAL = "global" #use all of the time series
    TOTAL = "total" #used to sum a time series counter
    WINDOW_TOTAL = "window-total"  #used to sum a time series counter in window

    #Genelaraty
    # EVENT = "event"
    SECOND = 1
    MINUE = 60
    QUARETER = 900
    HALFHOUR = 1800
    HOUR = 3600
    DAY = 86400

    #Genelaraty Mode
    MEAN = "mean"
    MODE = "mode"
    MEDIAN = "median"    
    MIN = "min"
    MAX = "max"

    #Datatypes
    NUMBER = "number"
    BOOLAEN = "boolean"

    #Global distance_metric
    EUCLIDEAN = "euclidean"
    MAHALAOBIS = "mahalanobis"
    CHEBYSHEV = "chebyshev"
    MANHATTAN = "manhattan"
    MINKOWSKI = "minkowski"
    WMINKOWSKI = "wminkowski"
    SEUCLIDEAN = "seuclidean"
    CUSTOM = "custom"
    
    
    
    
    
