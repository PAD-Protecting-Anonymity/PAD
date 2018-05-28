import math
from framework.utilities.datadescriptor import DataDescriptorTerms
import pandas as pd

class KAnonymityUtilities:
    def can_ensure_k_anonymity(self,anonymity_level,amount_of_inputs):
            if (2*anonymity_level-1)<amount_of_inputs:
                return True
            return False

    def find_balance_for_k(self,anonymity_level,data, sampling_rate):
        anonymity_levels = []
        for i in range(0,len(data)):
            amount_of_input = len(data[i].index)
            length_of_inputs = len(data[i].columns)
            if self.can_ensure_k_anonymity(anonymity_level,amount_of_input):
                # anonymity_levels.append(anonymity_level)
                continue
            resample_factor = math.floor(DataDescriptorTerms.DAY.value / sampling_rate[i])
            if not self.can_ensure_k_anonymity(anonymity_level,amount_of_input*resample_factor):
                raise ValueError("The amount of data input is not lang enough to anonymize")
            amount_of_daily = math.floor(length_of_inputs/resample_factor)
            anonymity_level_new = anonymity_level + math.ceil(math.log(amount_of_daily,3))
            anonymity_levels.append(anonymity_level_new)
        if anonymity_levels == []:
            anonymity_levels.append(anonymity_level)
        return min(anonymity_levels)

    # def can_ensure_k_anonymity_new(self,anonymity_level,amount_of_inputs):
    #         if (2*anonymity_level-1)*amount_of_inputs<amount_of_inputs:
    #             return True
    #         return False

# def balance(n, l, k):
#     if n > 2*k - 1:
#         d = 1
#         return d, k
#     else:
#         n_d = n
#         k_d = k
#         d = 1
#         while n_d < 2*k_d:
#             n_d = n_d * d
#             k_d = k + math.ceil(math.log(d))
#             d += 1
#         return d, k_d

# def balance_daily(n, l, k):
#     day = 1440
#     d = l/day
#     k_d = k + math.ceil(math.log(d))
#     return d, k_d

# def balance_k(n, l, k):
#     pass

# def find_balance_for_k(anonymity_level,data, sampling_rate):
#         anonymity_levels = []
#         for i in range(0,len(data)):
#             amount_of_input = len(data[i].index)
#             length_of_inputs = len(data[i].columns)
#             if amount_of_input > 2*anonymity_level - 1:
#                 # anonymity_levels.append(anonymity_level)
#                 continue
#             resample_factor = math.floor(86400 / sampling_rate[i])
#             if not amount_of_input*resample_factor > 2*anonymity_level - 1:
#                 raise ValueError("The amount of data input is not lang enough to anonymize")
#             amount_of_daily = math.floor(length_of_inputs/resample_factor)
#             anonymity_level_new = anonymity_level + math.ceil(math.log(amount_of_daily))
#             anonymity_levels.append(anonymity_level_new)
#         return min(anonymity_levels)

# anonymity_level = 5
# data = []
# sampling_rate = [60,300,20]
# data.append(pd.read_csv("./dataset/preprocessed_line_counter.csv"))
# data.append(pd.read_csv("./dataset/Preprocessed_noices_avg_4s.csv"))
# # data.append(pd.read_csv("./dataset/Preprocessed_noices_avg_4s - changed.csv"))
# data.append(pd.read_csv("./dataset/Preprocssed_HamiltonData_presence.csv"))

# print(find_balance_for_k(anonymity_level, data, sampling_rate))
# pir_sensor = [10080, 4320, 10080]
# count_sensor = [10080, 4320, 10080]
# noise_sensor = [10080, 4320, 10080]
# for i in pir_sensor:
#     print(balance_daily(n,i,k))

# print(balance(n,l,k))

